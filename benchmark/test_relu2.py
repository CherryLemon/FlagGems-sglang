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

"""Benchmark for activation_norm/relu2."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/activation_norm/relu2.
SHAPES = [
    (bs, hidden) for bs in (1, 8, 64, 512) for hidden in (2048, 4096, 8192)
]
MORE_SHAPES = [(bs, hidden) for bs in (4096,) for hidden in (2048, 4096, 8192)]


def _input_fn(shape, cur_dtype, device):
    num_tokens, hidden = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        num_tokens, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    yield (x,)


@pytest.mark.relu2
def test_perf_relu2():
    bench = OpBenchmark(
        op_name="relu2",
        torch_op=get_reference("relu2"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="num_tokens, hidden",
    )
    bench.set_gems(flaggems_sglang.relu2)
    bench.run()
