# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""context_attention for RDNA3 (gfx1100) — prefill/context-stage scaled
dot-product attention on packed variable-length sequences (Triton).

Computes, per sequence i (delimited by ``b_start_loc`` / ``b_seq_len``):

    o[start:end] = SDPA(q[start:end], k[start:end], v[start:end], is_causal)

where the packed tensors are laid out as
``[total_tokens, num_heads, head_dim]``. The output is
``[total_tokens, num_heads, head_dim]`` in float32.

For float16 and bfloat16 inputs both products run as WMMA on the input
type with fp32 accumulation, and the softmax weights are rounded to that
type before the second one. float32 inputs stay in fp32: gfx1100 has no
TF32 matrix instructions, so their products run on FMA units.

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator (no
``F.scaled_dot_product_attention``, no ``_compiled`` handles, no
``torch.ops.*`` attention) — the whole computation is done inside the
Triton kernel.
"""

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
_DOT_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


@triton.jit
def _attn_block(
    acc,
    l_i,
    m_i,
    q,
    k_ptr,
    v_ptr,
    base,
    stride_t,
    n0,
    offs_m,
    offs_d,
    d_mask,
    seq_len,
    qk_scale,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MASK: tl.constexpr,
    DOT_DTYPE: tl.constexpr,
):
    """One key/value block of the online softmax (scores in base 2).

    ``MASK=False`` is only used for blocks whose keys are all inside the
    sequence and, when causal, not after any query row of the block.
    """
    n_off = n0 + tl.arange(0, BLOCK_N)
    n_mask = n_off < seq_len
    k_ptrs = k_ptr + base + n_off[None, :] * stride_t + offs_d[:, None]
    v_ptrs = v_ptr + base + n_off[:, None] * stride_t + offs_d[None, :]
    if MASK:
        k = tl.load(k_ptrs, mask=d_mask[:, None] & n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    else:
        k = tl.load(k_ptrs, mask=d_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=d_mask[None, :], other=0.0)

    # Scores: [BLOCK_M, BLOCK_N], already multiplied by log2(e).
    qk = tl.dot(q, k.to(DOT_DTYPE)) * qk_scale

    if MASK:
        if IS_CAUSAL:
            attn_mask = (offs_m[:, None] >= n_off[None, :]) & n_mask[None, :]
        else:
            attn_mask = n_mask[None, :]
        qk = tl.where(attn_mask, qk, -1.0e30)

    # Online softmax update.
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
    p = tl.math.exp2(qk - m_ij[:, None])
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(DOT_DTYPE), v.to(DOT_DTYPE))
    return acc, l_i, m_ij


@triton.jit
def _context_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    b_start_loc_ptr,
    b_seq_len_ptr,
    qk_scale,
    num_heads,
    grid_m,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DOT_DTYPE: tl.constexpr,
):
    """Flash-attention style kernel for packed prefill.

    The grid is::

        grid = (cdiv(max_input_len, BLOCK_M), num_heads, num_seq)

    Query blocks are the fastest grid axis, so the programs of one
    (sequence, head) run together and share its K/V in cache.  A program
    handles the query blocks ``pid_m, pid_m + grid_m, ...`` of its sequence,
    counted from the last block (the longest one when causal), so rows beyond
    ``max_input_len`` are still computed without reading the sequence lengths
    on the host.

    ``BLOCK_D`` is the next power of two >= ``head_dim``; the head dimension
    is loaded with a mask so that non-power-of-two head dims (e.g. 72/80/96)
    are supported.  Padded head columns are read as 0, so they contribute
    nothing to the score (q=0) nor to the output accumulator (v=0).
    """
    pid_m = tl.program_id(0)
    head_id = tl.program_id(1)
    seq_id = tl.program_id(2)

    start = tl.load(b_start_loc_ptr + seq_id)
    seq_len = tl.load(b_seq_len_ptr + seq_id)
    num_m = tl.cdiv(seq_len, BLOCK_M)

    # Offset to the start of this sequence in the packed [total_tokens, H, D]
    # layout. Token stride = num_heads * head_dim, head stride = head_dim.
    stride_t = num_heads * head_dim
    base = start * stride_t + head_id * head_dim
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    for it in range(pid_m, num_m, grid_m):
        m_block = num_m - 1 - it
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < seq_len
        q = tl.load(
            q_ptr + base + offs_m[:, None] * stride_t + offs_d[None, :],
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        q = q.to(DOT_DTYPE)

        # Running statistics for the online softmax.
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Key blocks before n_full need no mask: when causal they end at or
        # before the first query row of the block, otherwise inside seq_len.
        # The loop stops after the last key a query row of the block can see.
        if IS_CAUSAL:
            n_end = tl.minimum(seq_len, (m_block + 1) * BLOCK_M)
            n_full = (m_block * BLOCK_M) // BLOCK_N * BLOCK_N
        else:
            n_end = seq_len
            n_full = seq_len // BLOCK_N * BLOCK_N
        for n0 in range(0, n_full, BLOCK_N):
            acc, l_i, m_i = _attn_block(
                acc,
                l_i,
                m_i,
                q,
                k_ptr,
                v_ptr,
                base,
                stride_t,
                n0,
                offs_m,
                offs_d,
                d_mask,
                seq_len,
                qk_scale,
                BLOCK_N,
                IS_CAUSAL,
                False,
                DOT_DTYPE,
            )
        for n0 in range(n_full, n_end, BLOCK_N):
            acc, l_i, m_i = _attn_block(
                acc,
                l_i,
                m_i,
                q,
                k_ptr,
                v_ptr,
                base,
                stride_t,
                n0,
                offs_m,
                offs_d,
                d_mask,
                seq_len,
                qk_scale,
                BLOCK_N,
                IS_CAUSAL,
                True,
                DOT_DTYPE,
            )

        # Normalize and store.  Only write the real `head_dim` columns.
        out = acc / l_i[:, None]
        tl.store(
            o_ptr + base + offs_m[:, None] * stride_t + offs_d[None, :],
            out.to(tl.float32),
            mask=m_mask[:, None] & d_mask[None, :],
        )


def context_attention(
    q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
):
    """Prefill/context-stage scaled dot-product attention on packed
    variable-length sequences.

    Args:
        q, k, v: packed tensors of shape ``[total_tokens, num_heads, head_dim]``
            (float32 / bfloat16 / float16).  ``total_tokens`` is the sum of the
            per-sequence lengths given by ``b_seq_len``.
        b_start_loc: ``[B]`` int tensor, start offset (in tokens) of each
            sequence.
        b_seq_len:   ``[B]`` int tensor, length (in tokens) of each sequence.
        max_input_len: reserved for kernel scratch allocation (unused in the
            reference).  Used here only to size the M grid; longer sequences
            are still computed in full.
        is_causal: whether to apply a causal attention mask per sequence.

    Returns:
        ``[total_tokens, num_heads, head_dim]`` float32 tensor.
    """
    total_tokens, num_heads, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)

    B = b_seq_len.shape[0]
    # Ensure int32 offsets for the kernel.
    if b_start_loc.dtype != torch.int32:
        b_start_loc = b_start_loc.to(torch.int32)
    if b_seq_len.dtype != torch.int32:
        b_seq_len = b_seq_len.to(torch.int32)

    out = torch.empty(
        (total_tokens, num_heads, head_dim),
        device=q.device,
        dtype=torch.float32,
    )

    # BLOCK_D must be a power of two (Triton's arange / dot require it).  Use
    # the next power of two >= head_dim; the kernel masks the head-dim axis so
    # the padded columns do not affect the result.
    block_d = 1 << (head_dim - 1).bit_length()

    # Short sequences have few query blocks per (sequence, head): smaller
    # blocks give more programs and compute fewer masked scores.
    num_warps = 4
    if q.dtype != torch.float32:
        if block_d <= 64:
            BLOCK_M, BLOCK_N = 64, 64
        else:
            BLOCK_M, BLOCK_N = (32, 32) if max_input_len <= 128 else (64, 32)
    elif block_d <= 64:
        BLOCK_M, BLOCK_N = (32, 16) if max_input_len <= 128 else (32, 32)
    elif block_d <= 128 and max_input_len > 512:
        # At BLOCK_D 128 with fp32 dots, 64x32 blocks on 8 warps and 32x16
        # blocks sometimes write a wrong output block (Triton 3.6, ROCm 7.2).
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 16, 16
        num_warps = 4 if block_d <= 128 else 2

    grid_m = max(1, triton.cdiv(max_input_len, BLOCK_M))
    grid = (grid_m, num_heads, B)
    _context_attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        b_start_loc,
        b_seq_len,
        scale * _LOG2E,
        num_heads,
        grid_m,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
        IS_CAUSAL=is_causal,
        DOT_DTYPE=_DOT_DTYPE.get(q.dtype, tl.float32),
        num_warps=num_warps,
        num_stages=1,
    )
    return out


__all__ = ["context_attention"]
