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

from . import conftest as cfg


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1), (3, 5, 7), (1, 32, 128), (17, 32, 128), (256, 32, 128)],
)
@pytest.mark.parametrize(
    "layout", ["flat", "transpose_gate", "sliced", "transpose_attn"]
)
@pytest.mark.fused_sigmoid_mul
def test_fused_sigmoid_mul(shape, dtype, layout):
    tokens, heads, dim = shape
    hidden = heads * dim
    attn = torch.randn((tokens, hidden), device=cfg.device, dtype=dtype)
    if layout == "flat":
        gate = torch.randn((tokens, hidden), device=cfg.device, dtype=dtype)
    elif layout == "transpose_gate":
        gate = torch.randn(
            (tokens, dim, heads), device=cfg.device, dtype=dtype
        ).transpose(1, 2)
    elif layout == "sliced":
        attn = torch.randn(
            (tokens, hidden * 2), device=cfg.device, dtype=dtype
        )[:, ::2]
        gate = torch.randn(
            (tokens, heads, dim * 2), device=cfg.device, dtype=dtype
        )[..., ::2]
    else:
        attn = torch.randn((hidden, tokens), device=cfg.device, dtype=dtype).T
        gate = torch.randn(
            (tokens, heads, dim), device=cfg.device, dtype=dtype
        )
    saved_attn = attn.clone()
    saved_gate = gate.clone()
    expected = get_reference("fused_sigmoid_mul")(attn, gate)
    actual = flaggems_sglang.fused_sigmoid_mul(attn, gate)
    rtol = {torch.float16: 1e-3, torch.bfloat16: 1e-2, torch.float32: 1e-5}[
        dtype
    ]
    assert actual.shape == attn.shape and actual.dtype == dtype
    assert actual.device == attn.device
    assert actual.data_ptr() not in (attn.data_ptr(), gate.data_ptr())
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=1e-5)
    torch.testing.assert_close(attn, saved_attn, rtol=0, atol=0)
    torch.testing.assert_close(gate, saved_gate, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0, 128), (3, 0), (0, 0)])
@pytest.mark.fused_sigmoid_mul
def test_fused_sigmoid_mul_empty(shape):
    attn = torch.empty(shape, device=cfg.device, dtype=torch.bfloat16)
    result = flaggems_sglang.fused_sigmoid_mul(attn, torch.empty_like(attn))
    assert result.shape == attn.shape and result.dtype == attn.dtype
    assert result.device == attn.device


@pytest.mark.fused_sigmoid_mul
def test_fused_sigmoid_mul_registered():
    assert "fused_sigmoid_mul" in flaggems_sglang.all_registered_ops()
