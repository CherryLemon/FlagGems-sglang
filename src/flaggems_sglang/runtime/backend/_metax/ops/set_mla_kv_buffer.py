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

import torch  # noqa: F401
import triton
import triton.language as tl

__all__ = ["set_mla_kv_buffer"]


@triton.jit
def _copy_words(src, dst, words, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < words
    value = tl.load(src + offsets, mask=mask)
    tl.store(dst + offsets, value, mask=mask)


@triton.jit
def _copy_strided(
    src,
    dst,
    rows,
    width,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (row < rows) & (col < width)
    value = tl.load(src + row * SRC_S0 + col * SRC_S1, mask=mask)
    tl.store(dst + row * DST_S0 + col * DST_S1, value, mask=mask)


@triton.jit
def _scatter_words(
    dst,
    loc,
    nope,
    rope,
    rows,
    NOPE_WORDS: tl.constexpr,
    ROPE_WORDS: tl.constexpr,
    DST_S0: tl.constexpr,
    LOC_S0: tl.constexpr,
    NOPE_S0: tl.constexpr,
    ROPE_S0: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    row_mask = row < rows
    target = tl.load(loc + row * LOC_S0, mask=row_mask, other=0).to(tl.int32)
    nope_col = tl.arange(0, NOPE_BLOCK)
    rope_col = tl.arange(0, ROPE_BLOCK)
    nope_mask = row_mask[:, None] & (nope_col[None, :] < NOPE_WORDS)
    rope_mask = row_mask[:, None] & (rope_col[None, :] < ROPE_WORDS)
    nope_value = tl.load(
        nope + row[:, None] * NOPE_S0 + nope_col[None, :], mask=nope_mask
    )
    rope_value = tl.load(
        rope + row[:, None] * ROPE_S0 + rope_col[None, :], mask=rope_mask
    )
    tl.store(
        dst + target[:, None] * DST_S0 + nope_col[None, :],
        nope_value,
        mask=nope_mask,
    )
    tl.store(
        dst + target[:, None] * DST_S0 + NOPE_WORDS + rope_col[None, :],
        rope_value,
        mask=rope_mask,
    )


@triton.jit
def _scatter_values(
    dst,
    loc,
    nope,
    rope,
    rows,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    LOC_S0: tl.constexpr,
    NOPE_S0: tl.constexpr,
    NOPE_S1: tl.constexpr,
    ROPE_S0: tl.constexpr,
    ROPE_S1: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    row_mask = row < rows
    target = tl.load(loc + row * LOC_S0, mask=row_mask, other=0).to(tl.int64)
    nope_col = tl.arange(0, NOPE_BLOCK)
    rope_col = tl.arange(0, ROPE_BLOCK)
    nope_mask = row_mask[:, None] & (nope_col[None, :] < NOPE)
    rope_mask = row_mask[:, None] & (rope_col[None, :] < ROPE)
    nope_value = tl.load(
        nope + row[:, None] * NOPE_S0 + nope_col[None, :] * NOPE_S1,
        mask=nope_mask,
    )
    rope_value = tl.load(
        rope + row[:, None] * ROPE_S0 + rope_col[None, :] * ROPE_S1,
        mask=rope_mask,
    )
    tl.store(
        dst + target[:, None] * DST_S0 + nope_col[None, :] * DST_S1,
        nope_value,
        mask=nope_mask,
    )
    tl.store(
        dst + target[:, None] * DST_S0 + (NOPE + rope_col[None, :]) * DST_S1,
        rope_value,
        mask=rope_mask,
    )


def set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    out = kv_buffer
    rows = kv_buffer.shape[0]  # noqa: F841
    width = kv_buffer.shape[1]
    count = kv_buffer.numel()
    packed_copy = (
        kv_buffer.element_size() == 2
        and kv_buffer.stride(1) == 1
        and kv_buffer.stride(0) == width
        and kv_buffer.storage_offset() % 2 == 0
        and out.stride(1) == 1
        and out.stride(0) == width
        and out.storage_offset() % 2 == 0
        and count % 2 == 0
    )
    n = loc.numel()
    nope_width = cache_k_nope.shape[-1]
    rope_width = cache_k_rope.shape[-1]
    packed_scatter = (
        packed_copy
        and cache_k_nope.dtype == kv_buffer.dtype
        and cache_k_rope.dtype == kv_buffer.dtype
        and nope_width % 2 == 0
        and rope_width % 2 == 0
        and cache_k_nope.stride(1) == 1
        and cache_k_rope.stride(1) == 1
        and cache_k_nope.stride(0) == nope_width
        and cache_k_rope.stride(0) == rope_width
        and cache_k_nope.storage_offset() % 2 == 0
        and cache_k_rope.storage_offset() % 2 == 0
    )
    if n:
        if packed_scatter:
            nope_words = nope_width // 2
            rope_words = rope_width // 2
            nope_block = triton.next_power_of_2(nope_words)
            rope_block = triton.next_power_of_2(rope_words)
            _scatter_words[(triton.cdiv(n, 8),)](
                triton.reinterpret(out, tl.uint32),
                loc,
                triton.reinterpret(cache_k_nope, tl.uint32),
                triton.reinterpret(cache_k_rope, tl.uint32),
                n,
                nope_words,
                rope_words,
                out.stride(0) // 2,
                loc.stride(0),
                cache_k_nope.stride(0) // 2,
                cache_k_rope.stride(0) // 2,
                8,
                nope_block,
                rope_block,
                num_warps=8,
            )
        else:
            nope_block = triton.next_power_of_2(nope_width)
            rope_block = triton.next_power_of_2(rope_width)
            _scatter_values[(triton.cdiv(n, 4),)](
                out,
                loc,
                cache_k_nope,
                cache_k_rope,
                n,
                nope_width,
                rope_width,
                out.stride(0),
                out.stride(1),
                loc.stride(0),
                cache_k_nope.stride(0),
                cache_k_nope.stride(1),
                cache_k_rope.stride(0),
                cache_k_rope.stride(1),
                4,
                nope_block,
                rope_block,
                num_warps=8,
            )
    return out
