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

"""Benchmark for kvcache/concat_and_cast_mha_k."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/kvcache/concat_and_cast_mha_k.
SHAPES = [(t, 32, 128, 64) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 32, 128, 64) for t in (4096,)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, num_heads, nope_dim, rope_dim = shape
    g = torch.Generator(device=device).manual_seed(0)
    k = torch.zeros(
        num_tokens,
        num_heads,
        nope_dim + rope_dim,
        dtype=cur_dtype,
        device=device,
    )
    k_nope = torch.randn(
        num_tokens, num_heads, nope_dim, device=device, generator=g
    ).to(cur_dtype)
    k_rope = torch.randn(
        num_tokens, 1, rope_dim, device=device, generator=g
    ).to(cur_dtype)
    yield k, k_nope, k_rope


@pytest.mark.concat_and_cast_mha_k
def test_perf_concat_and_cast_mha_k():
    bench = OpBenchmark(
        op_name="concat_and_cast_mha_k",
        torch_op=get_reference("concat_and_cast_mha_k"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, num_heads, nope_dim, rope_dim",
    )
    bench.set_gems(flaggems_sglang.concat_and_cast_mha_k)
    bench.run()
