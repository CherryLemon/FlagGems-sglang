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

"""Complete-wrapper Graph comparisons against frozen configured native GEMM."""

from functools import partial

import pytest
import torch

import flaggems_sglang as gems
from flaggems_sglang.runtime.backend._nvidia.hopper.ops._dsv41_fp8_configs import (
    _H100_MIDRANGE_FP8_CONFIGS,
)
from tests.dsv41_helpers import fp8_inputs, fp8_reference

from .op_benchmark import OpBenchmark

CORE = [
    (6, 1792, 5120),
    (24, 1792, 5120),
    (48, 1792, 5120),
    (96, 5120, 4096),
    (24, 4608, 5120),
    (48, 5120, 2304),
]
ALL = sorted(_H100_MIDRANGE_FP8_CONFIGS)


def _inputs(shape, cur_dtype, device):
    yield fp8_inputs(*shape)


@pytest.mark.dsv41_block_fp8_matmul
@pytest.mark.parametrize("reference", ["native", "torch"])
def test_perf_dsv41_block_fp8_matmul(reference):
    baseline = (
        partial(gems.dsv41_block_fp8_matmul, tuned=False)
        if reference == "native"
        else fp8_reference
    )
    bench = OpBenchmark(
        op_name="dsv41_block_fp8_matmul_" + reference,
        torch_op=baseline,
        input_fn=_inputs,
        dtypes=[torch.bfloat16],
        shapes=CORE,
        more_shapes=ALL if reference == "native" else [],
        shape_desc="M,N,K",
    )
    bench.set_gems(gems.dsv41_block_fp8_matmul)
    bench.run()
