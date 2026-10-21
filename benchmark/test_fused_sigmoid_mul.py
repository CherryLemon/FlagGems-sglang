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

import pytest
import torch
from flaggems_sglang.reference import get_reference

import flaggems_sglang

from .attri_util import FUSED_SIGMOID_MUL_SHAPES
from .op_benchmark import OpBenchmark


def _input_fn(shape, cur_dtype, device):
    tokens, heads, dim, transposed = shape
    attn = torch.randn((tokens, heads * dim), device=device, dtype=cur_dtype)
    if transposed:
        gate = torch.randn(
            (tokens, dim, heads), device=device, dtype=cur_dtype
        ).transpose(1, 2)
    else:
        gate = torch.randn(
            (tokens, heads * dim), device=device, dtype=cur_dtype
        )
    yield attn, gate


@pytest.mark.fused_sigmoid_mul
def test_perf_fused_sigmoid_mul():
    bench = OpBenchmark(
        op_name="fused_sigmoid_mul",
        torch_op=get_reference("fused_sigmoid_mul"),
        input_fn=_input_fn,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
        shapes=FUSED_SIGMOID_MUL_SHAPES,
        shape_desc="tokens, heads, head_dim, transposed_gate",
    )
    bench.set_gems(flaggems_sglang.fused_sigmoid_mul)
    bench.run()
