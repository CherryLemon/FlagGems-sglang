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
def _copy_flat(src, dst, count, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    value = tl.load(src + offset, mask=mask)
    tl.store(dst + offset, value, mask=mask)


@triton.jit
def _copy_rows(
    src,
    dst,
    rows,
    WIDTH: tl.constexpr,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = (row < rows) & (col < WIDTH)
    value = tl.load(src + row * SRC_S0 + col * SRC_S1, mask=mask)
    tl.store(dst + row * DST_S0 + col * DST_S1, value, mask=mask)


@triton.jit
def _scatter_row_segment(
    dst,
    loc,
    src,
    START: tl.constexpr,
    WIDTH: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    LOC_S0: tl.constexpr,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    target = tl.load(loc + row * LOC_S0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    value = tl.load(src + row * SRC_S0 + col * SRC_S1, mask=mask)
    tl.store(dst + target * DST_S0 + (START + col) * DST_S1, value, mask=mask)


@triton.jit
def _scatter_flat_segment(
    dst,
    loc,
    src,
    count,
    START: tl.constexpr,
    WIDTH: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    LOC_S0: tl.constexpr,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    row = offset // WIDTH
    col = offset - row * WIDTH
    target = tl.load(loc + row * LOC_S0, mask=mask, other=0).to(tl.int64)
    value = tl.load(src + row * SRC_S0 + col * SRC_S1, mask=mask)
    tl.store(dst + target * DST_S0 + (START + col) * DST_S1, value, mask=mask)


def set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    out = kv_buffer
    n = loc.numel()
    nope_width = cache_k_nope.shape[-1]
    rope_width = cache_k_rope.shape[-1]
    if n:
        if n < 64:
            _scatter_row_segment[(n,)](
                out,
                loc,
                cache_k_nope,
                0,
                nope_width,
                out.stride(0),
                out.stride(1),
                loc.stride(0),
                cache_k_nope.stride(0),
                cache_k_nope.stride(1),
                triton.next_power_of_2(nope_width),
            )
            _scatter_row_segment[(n,)](
                out,
                loc,
                cache_k_rope,
                nope_width,
                rope_width,
                out.stride(0),
                out.stride(1),
                loc.stride(0),
                cache_k_rope.stride(0),
                cache_k_rope.stride(1),
                triton.next_power_of_2(rope_width),
            )
        else:
            nope_count = n * nope_width
            rope_count = n * rope_width
            _scatter_flat_segment[(triton.cdiv(nope_count, 32768),)](
                out,
                loc,
                cache_k_nope,
                nope_count,
                0,
                nope_width,
                out.stride(0),
                out.stride(1),
                loc.stride(0),
                cache_k_nope.stride(0),
                cache_k_nope.stride(1),
                32768,
            )
            _scatter_flat_segment[(triton.cdiv(rope_count, 32768),)](
                out,
                loc,
                cache_k_rope,
                rope_count,
                nope_width,
                rope_width,
                out.stride(0),
                out.stride(1),
                loc.stride(0),
                cache_k_rope.stride(0),
                cache_k_rope.stride(1),
                32768,
            )
    return out
