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

"""mHC full stats + Sinkhorn chain: existing TileLang vs exact fusion."""

import pytest
import torch

import flaggems_sglang as gems
from tests.dsv41_helpers import mhc_inputs

from .op_benchmark import OpBenchmark


def _inputs(shape, cur_dtype, device):
    yield mhc_inputs(*shape)


def baseline(x, fn, scale, base):
    return gems.hc_split_sinkhorn(
        gems.hc_mix_stats(x, fn, 1e-6, tuned=False), scale, base
    )


def candidate(x, fn, scale, base):
    if x.shape[0] in (36, 40, 42, 48, 50, 60, 80, 84, 96):
        return gems.hc_mix_stats_sinkhorn_exact(
            x, fn, scale, base, 4, 20, 1e-6, 1e-6
        )
    return gems.hc_split_sinkhorn(gems.hc_mix_stats(x, fn, 1e-6), scale, base)


@pytest.mark.dsv41_mhc
def test_perf_dsv41_mhc():
    bench = OpBenchmark(
        op_name="dsv41_mhc_native",
        torch_op=baseline,
        input_fn=_inputs,
        dtypes=[torch.bfloat16],
        shapes=[
            (m,)
            for m in [
                5,
                6,
                10,
                12,
                15,
                18,
                20,
                24,
                25,
                30,
                35,
                36,
                40,
                42,
                48,
                50,
                60,
                70,
                72,
                80,
                84,
                96,
                100,
                120,
            ]
        ],
        shape_desc="M,K20480,HC4",
    )
    bench.set_gems(candidate)
    bench.run()
