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

"""fused_moe_router_tensorcore for RDNA3 (gfx1100).

Same computation as the PyTorch reference:

  logits = x.float() @ router_weight.float().t()        # [M, E]
  if softcap != 0: logits = tanh(logits/cap)*cap
  if bias is not None: logits = logits + bias
  probs = softmax(logits, dim=-1)
  topk_ids = top-k (k <= 2) of logits; topk_weights = probs at topk_ids

Up to 8 tokens the logits come from an fp32 FMA kernel and above that from
a tiled tl.dot kernel; a per-row Triton epilogue does the rest. Both match
the generic version except the tiled GEMM: gfx1100 has no TF32, so the
generic version's fp32 dots run on FMA units. Here float16 and bfloat16
inputs use WMMA on the input type, where every product is exact in fp32.
Each K block is its own dot starting from zero, and the blocks are summed
in fp32: chaining all K blocks through the WMMA accumulator loses enough
accuracy to fail the fp32 weight tolerance from M = 64 up. Other inputs are
converted to fp32 first. Launch configurations are fixed by shape instead
of autotuned, and K is never split, so every call gives the same bits.
"""

import torch
import triton
import triton.language as tl


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def _block_e_for(E):
    be = _next_pow2(E)
    if be < 16:
        be = 16
    return be


@triton.jit
def _fma_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    E,
    K,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wk,
    stride_om,
    stride_oe,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)
    row = tl.program_id(1)

    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = offs_e < E
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k_block = k_start * BLOCK_K + offs_k
        k_mask = offs_k_block < K
        xv = tl.load(
            x_ptr + row * stride_xm + offs_k_block * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr
            + offs_e[:, None] * stride_we
            + offs_k_block[None, :] * stride_wk,
            mask=e_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)

    tl.store(out_ptr + row * stride_om + offs_e * stride_oe, acc, mask=e_mask)


@triton.jit
def _router_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    DOT_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # The expert tiles of one row block run next to each other, so each row
    # block of x is read from memory once.
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ones = tl.full((BLOCK_M, BLOCK_N), 1.0, tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k_start * BLOCK_K + offs_k) < K
        a = tl.load(
            x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0
        )
        b = tl.load(
            w_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0.0
        )
        if DOT_FP32:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        part = tl.dot(a, b.T, input_precision="ieee", out_dtype=tl.float32)
        # tl.fma instead of + keeps Triton from folding the sum into the
        # accumulator of the dot.
        acc = tl.fma(part, ones, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = (
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["M", "TOPK", "SCAP_FLAG", "HAS_BIAS"],
)
@triton.heuristics({"BLOCK_E": lambda args: _block_e_for(args["E"])})
@triton.jit
def _router_topk_kernel(
    logits_ptr,
    bias_ptr,
    w_out_ptr,
    id_out_ptr,
    M,
    E,
    stride_lm,
    stride_le,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    scap_val,
    SCAP_FLAG: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_E)
    valid = offs < E

    logit = tl.load(
        logits_ptr + row * stride_lm + offs * stride_le, mask=valid, other=0.0
    ).to(tl.float32)

    if SCAP_FLAG != 0:
        # tanh(x) = 2*sigmoid(2x) - 1
        logit = scap_val * (2.0 * tl.sigmoid(2.0 * logit / scap_val) - 1.0)

    if HAS_BIAS:
        b = tl.load(bias_ptr + offs, mask=valid, other=0.0).to(tl.float32)
        logit = logit + b

    sel = tl.where(valid, logit, -float("inf"))

    # Softmax over all experts; the top-k weights are not renormalized.
    m = tl.max(sel, axis=0)
    e = tl.exp(sel - m)
    probs = e / tl.sum(e, axis=0)

    # tl.argmax returns the lowest index of the max, like torch.topk.
    k_offs = tl.arange(0, TOPK)
    w_out = tl.zeros((TOPK,), dtype=tl.float32)
    ix_out = tl.zeros((TOPK,), dtype=tl.int32)
    cur = sel
    for j in tl.static_range(TOPK):
        i = tl.argmax(cur, axis=0)
        hit = offs == i
        v = tl.sum(tl.where(hit, probs, 0.0), axis=0)
        slot = k_offs == j
        w_out = tl.where(slot, v, w_out)
        ix_out = tl.where(slot, i.to(tl.int32), ix_out)
        cur = tl.where(hit, -float("inf"), cur)

    tl.store(w_out_ptr + row * stride_wm + k_offs * stride_wk, w_out)
    tl.store(id_out_ptr + row * stride_im + k_offs * stride_ik, ix_out)


def _gemm_config(M, E, dot_fp32):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for M > 8."""
    if dot_fp32:
        if M * E < 32768:
            return 16, 16, 64, 2, 1
        return 64, 16, 32, 2, 1
    if M <= (512 if E <= 64 else 128):
        return 32, 16, 256, 8, 3
    if M <= 1024 or E <= 64:
        return 32, 16, 128, 2, 3
    return 16, 64, 128, 4, 3


def fused_moe_router_tensorcore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    M, K = x.shape
    E, _ = router_weight.shape

    device = x.device
    use_bias = correction_bias is not None
    use_softcap = moe_softcapping != 0 and moe_softcapping is not None

    topk_weights = torch.empty((M, topk), dtype=torch.float32, device=device)
    topk_ids = torch.empty((M, topk), dtype=torch.int32, device=device)

    x_c = x if x.is_contiguous() else x.contiguous()
    w_c = (
        router_weight
        if router_weight.is_contiguous()
        else router_weight.contiguous()
    )
    bias_ptr = (
        correction_bias
        if (not use_bias or correction_bias.is_contiguous())
        else correction_bias.contiguous()
    )

    logits = torch.empty((M, E), dtype=torch.float32, device=device)

    if M <= 8:
        # Bandwidth bound: E-tiles over programs, fp32 FMA.
        if M == 1:
            block_e, block_k, n_warps, n_stages = 4, 2048, 16, 3
        else:
            block_e, block_k, n_warps, n_stages = 4, 2048, 4, 2
        grid_g = (triton.cdiv(E, block_e), M)
        _fma_gemm_kernel[grid_g](
            x_c,
            w_c,
            logits,
            M,
            E,
            K,
            x_c.stride(0),
            x_c.stride(1),
            w_c.stride(0),
            w_c.stride(1),
            logits.stride(0),
            logits.stride(1),
            BLOCK_E=block_e,
            BLOCK_K=block_k,
            num_warps=n_warps,
            num_stages=n_stages,
        )
    else:
        dot_fp32 = (
            x_c.dtype not in (torch.float16, torch.bfloat16)
            or w_c.dtype != x_c.dtype
        )
        block_m, block_n, block_k, n_warps, n_stages = _gemm_config(
            M, E, dot_fp32
        )
        grid_g = (triton.cdiv(E, block_n), triton.cdiv(M, block_m))
        _router_gemm_kernel[grid_g](
            x_c,
            w_c,
            logits,
            M,
            E,
            K,
            x_c.stride(0),
            x_c.stride(1),
            w_c.stride(0),
            w_c.stride(1),
            logits.stride(0),
            logits.stride(1),
            DOT_FP32=dot_fp32,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=n_warps,
            num_stages=n_stages,
        )

    _router_topk_kernel[(M,)](
        logits,
        bias_ptr,
        topk_weights,
        topk_ids,
        M,
        E,
        logits.stride(0),
        logits.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        float(moe_softcapping) if use_softcap else 0.0,
        SCAP_FLAG=1 if use_softcap else 0,
        HAS_BIAS=1 if use_bias else 0,
        TOPK=topk,
    )

    return topk_weights, topk_ids


__all__ = ["fused_moe_router_tensorcore"]
