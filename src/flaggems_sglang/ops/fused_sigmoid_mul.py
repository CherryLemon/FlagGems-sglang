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

"""Task 97: FP32 sigmoid and multiplication fused with stride-aware reads."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_kernel(
    A,
    G,
    O,
    N: tl.constexpr,
    HIDDEN: tl.constexpr,
    DIM: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    token = i // HIDDEN
    hidden = i % HIDDEN
    av = tl.load(A + token * A0 + hidden * A1, valid, other=0).to(tl.float32)
    gv = tl.load(
        G + token * G0 + (hidden // DIM) * G1 + (hidden % DIM) * G2,
        valid,
        other=0,
    ).to(tl.float32)
    value = av * (1.0 / (1.0 + tl.exp(-gv)))
    tl.store(O + i, value, valid)


def fused_sigmoid_mul(attn_output, gate):
    tokens, hidden = attn_output.shape
    out = torch.empty(
        (tokens, hidden), dtype=attn_output.dtype, device=attn_output.device
    )
    if out.numel() == 0:
        return out
    if gate.ndim == 3:
        dim = gate.shape[2]
        g0, g1, g2 = gate.stride()
    else:
        dim = hidden
        g0, g2 = gate.stride()
        g1 = 0
    block = 1024
    _gate_kernel[(triton.cdiv(out.numel(), block),)](
        attn_output,
        gate,
        out,
        out.numel(),
        hidden,
        dim,
        *attn_output.stride(),
        g0,
        g1,
        g2,
        block,
        num_warps=4,
    )
    return out


__all__ = ["fused_sigmoid_mul"]
