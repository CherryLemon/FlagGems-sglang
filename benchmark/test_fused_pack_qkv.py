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

"""Benchmark for diffusion/fused_pack_qkv."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/diffusion/fused_pack_qkv.
SHAPES = [(1, s, 16, 128) for s in (8, 128, 1024)]
MORE_SHAPES = [(1, s, 16, 128) for s in (8192,)]

_KEEP = 0.6


def _input_fn(shape, cur_dtype, device):
    bs, seq, num_heads, head_dim = shape
    g = torch.Generator(device=device).manual_seed(9)

    def r():
        return torch.randn(
            bs, seq, num_heads, head_dim, device=device, generator=g
        ).to(cur_dtype)

    total = bs * seq
    n_valid = max(1, int(total * _KEEP))
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    indices = (
        torch.randperm(
            total, generator=torch.Generator(device="cpu").manual_seed(9)
        )[:n_valid]
        .to(torch.int64)
        .to(device)
    )
    yield r(), r(), r(), indices


@pytest.mark.fused_pack_qkv
def test_perf_fused_pack_qkv():
    bench = OpBenchmark(
        op_name="fused_pack_qkv",
        torch_op=get_reference("fused_pack_qkv"),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="bs, seq, num_heads, head_dim",
    )
    bench.set_gems(flaggems_sglang.fused_pack_qkv)
    bench.run()
