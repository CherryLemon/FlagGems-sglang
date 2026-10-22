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

"""Benchmark for mamba/state_scatter_with_mask."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from .op_benchmark import OpBenchmark

# Shapes match kernel-comp-baseline/problems/mamba/state_scatter_with_mask.
SHAPES = [(8, 512, requests, 4, 16, 128) for requests in (8, 64)]
MORE_SHAPES = [(8, 512, requests, 4, 16, 128) for requests in (256,)]


def _input_fn(shape, cur_dtype, device):
    layers, cache, requests, draft, s0, s1 = shape
    g = torch.Generator(device=device).manual_seed(9)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(9)
    dst = torch.randn(layers, cache, s0, s1, device=device, generator=g).to(
        cur_dtype
    )
    src = torch.randn(
        layers, requests, draft, s0, s1, device=device, generator=g
    ).to(cur_dtype)
    dst_idx = torch.randperm(cache, device="cpu", generator=perm_g)[
        :requests
    ].to(device=device, dtype=torch.int32)
    step = torch.randint(
        0,
        draft,
        (requests,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    yield dst.contiguous(), src.contiguous(), dst_idx, step


@pytest.mark.state_scatter_with_mask
def test_perf_state_scatter_with_mask():
    bench = OpBenchmark(
        op_name="state_scatter_with_mask",
        torch_op=get_reference("state_scatter_with_mask"),
        input_fn=_input_fn,
        dtypes=[torch.float32],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="layers, cache, requests, draft, state0, state1",
    )
    bench.set_gems(flaggems_sglang.state_scatter_with_mask)
    bench.run()
