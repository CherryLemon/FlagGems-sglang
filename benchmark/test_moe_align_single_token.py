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

"""Benchmark for moe/moe_align_single_token."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/moe/moe_align_single_token.
SHAPES = [(k, 256, block) for k in (1, 4, 8) for block in (16, 64)]
MORE_SHAPES = [(k, 256, 128) for k in (1, 4, 8)]


def _input_fn(shape, cur_dtype, device):
    top_k, num_experts, block_size = shape
    # The experts of the single token are distinct, which randperm guarantees.
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    g = torch.Generator(device="cpu").manual_seed(0)
    experts = torch.randperm(num_experts, generator=g)[:top_k].to(device)
    yield experts.reshape(1, top_k).to(torch.int32), block_size


@pytest.mark.moe_align_single_token
def test_perf_moe_align_single_token():
    bench = OpBenchmark(
        op_name="moe_align_single_token",
        torch_op=get_reference("moe_align_single_token"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="top_k, num_experts, block_size",
    )
    bench.set_gems(flaggems_sglang.moe_align_single_token)
    bench.run()
