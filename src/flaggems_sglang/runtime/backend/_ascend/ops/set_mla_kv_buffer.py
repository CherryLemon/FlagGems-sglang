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


@triton.jit(do_not_specialize=("src", "dst"))
def _copy_dense(src, dst, N: tl.constexpr):
    BLOCK: tl.constexpr = 4096
    programs = tl.num_programs(0)
    chunks = tl.cdiv(N, BLOCK * programs)
    start = tl.program_id(0) * chunks * BLOCK
    for index in range(chunks):
        offset = start + index * BLOCK + tl.arange(0, BLOCK)
        if N % (BLOCK * programs) == 0:
            value = tl.load(src + offset)
            tl.store(dst + offset, value)
        else:
            mask = offset < N
            value = tl.load(src + offset, mask=mask)
            tl.store(dst + offset, value, mask=mask)


@triton.jit
def _copy_strided(
    src,
    dst,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    for row in range(tl.program_id(0), ROWS, PROGRAMS):
        value = tl.load(src + row * SRC_S0 + col * SRC_S1, mask=mask)
        tl.store(dst + row * DST_S0 + col * DST_S1, value, mask=mask)


@triton.jit
def _scatter_rows(
    dst,
    loc,
    nope,
    rope,
    ROWS: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    LOC_S0: tl.constexpr,
    NOPE_S0: tl.constexpr,
    NOPE_S1: tl.constexpr,
    ROPE_S0: tl.constexpr,
    ROPE_S1: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    col = tl.arange(0, BLOCK)
    for row in range(tl.program_id(0), ROWS, PROGRAMS):
        target = tl.load(loc + row * LOC_S0).to(tl.int64)
        nope_value = tl.load(
            nope + row * NOPE_S0 + col * NOPE_S1, mask=col < NOPE
        )
        rope_value = tl.load(
            rope + row * ROPE_S0 + col * ROPE_S1, mask=col < ROPE
        )
        tl.store(
            dst + target * DST_S0 + col * DST_S1, nope_value, mask=col < NOPE
        )
        tl.store(
            dst + target * DST_S0 + (NOPE + col) * DST_S1,
            rope_value,
            mask=col < ROPE,
        )


def set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    out = kv_buffer
    rows = kv_buffer.shape[0]  # noqa: F841
    width = kv_buffer.shape[1]  # noqa: F841
    count = kv_buffer.numel()  # noqa: F841
    n = loc.numel()
    nope_width = cache_k_nope.shape[-1]
    rope_width = cache_k_rope.shape[-1]
    if n and nope_width + rope_width:
        programs = min(n, 40)
        block = triton.next_power_of_2(max(nope_width, rope_width))
        _scatter_rows.run(
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
            programs,
            block,
            grid=(programs,),
            warmup=False,
            enable_auto_bind_sub_block=False,
        )
    return out
