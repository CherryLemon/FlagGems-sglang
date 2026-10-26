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

"""Benchmark for attention/fused_qkvzba_split_reshape_cat."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/attention/fused_qkvzba_split_reshape_cat.
SHAPES = [(batch, 16, 32, 128, 128) for batch in (1, 8, 64, 512)]
MORE_SHAPES = [(batch, 16, 32, 128, 128) for batch in (4096,)]


def _input_fn(shape, cur_dtype, device):
    batch, nqk, nv, head_qk, head_v = shape
    g = torch.Generator(device=device).manual_seed(9)
    r = nv // nqk
    mixed_qkvz = torch.randn(
        batch,
        nqk * (2 * head_qk + 2 * r * head_v),
        device=device,
        generator=g,
    ).to(cur_dtype)
    mixed_ba = torch.randn(batch, nqk * 2 * r, device=device, generator=g).to(
        cur_dtype
    )
    yield (
        mixed_qkvz.contiguous(),
        mixed_ba.contiguous(),
        nqk,
        nv,
        head_qk,
        head_v,
    )


@pytest.mark.fused_qkvzba_split_reshape_cat
def test_perf_fused_qkvzba_split_reshape_cat():
    bench = OpBenchmark(
        op_name="fused_qkvzba_split_reshape_cat",
        torch_op=get_reference("fused_qkvzba_split_reshape_cat"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="batch, num_heads_qk, num_heads_v, head_qk, head_v",
    )
    bench.set_gems(flaggems_sglang.fused_qkvzba_split_reshape_cat)
    bench.run()
