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

"""Contiguous batch/token and head/dim offsets avoid quotient/remainder arithmetic."""

import torch
import triton
import triton.language as tl


@triton.jit
def _offset(
    token, column, S: tl.constexpr, D: tl.constexpr, STRIDES: tl.constexpr
):
    if STRIDES[0] == S * STRIDES[1] and STRIDES[2] == D * STRIDES[3]:
        return token * STRIDES[1] + column * STRIDES[3]
    return (
        (token // S) * STRIDES[0]
        + (token % S) * STRIDES[1]
        + (column // D) * STRIDES[2]
        + (column % D) * STRIDES[3]
    )


@triton.jit
def _pack(
    Q,
    K,
    V,
    Index,
    Out,
    N: tl.constexpr,
    TOKENS: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    IS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tiles = tl.cdiv(H * D, BLOCK)
    pid = tl.program_id(0).to(tl.int64)
    row = pid // tiles
    column = (pid % tiles) * BLOCK + tl.arange(0, BLOCK)
    i = row * (H * D) + column
    valid = (column < H * D) & (i < N)
    token = tl.load(Index + row * IS).to(tl.int64)
    token = tl.where(token < 0, token + TOKENS, token)
    q = tl.load(Q + _offset(token, column, S, D, QS), valid, other=0)
    k = tl.load(K + _offset(token, column, S, D, KS), valid, other=0)
    v = tl.load(V + _offset(token, column, S, D, VS), valid, other=0)
    tl.store(Out + i, q, valid)
    tl.store(Out + N + i, k, valid)
    tl.store(Out + 2 * N + i, v, valid)


def fused_pack_qkv(q, k, v, indices):
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, v must have the same [B,S,H,D] shape")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, v must have the same dtype")
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be a one-dimensional int32/int64 tensor")
    if any(t.device != q.device for t in (k, v, indices)):
        raise ValueError("all inputs must be on the same device")
    shape = (indices.numel(), q.shape[2], q.shape[3])
    storage = torch.empty((3, *shape), dtype=q.dtype, device=q.device)
    outputs = storage.unbind(0)
    n = outputs[0].numel()
    if n:
        block = min(
            8192, max(256, triton.next_power_of_2(q.shape[2] * q.shape[3]))
        )
        _pack[
            (indices.numel() * triton.cdiv(q.shape[2] * q.shape[3], block),)
        ](
            q,
            k,
            v,
            indices,
            storage,
            n,
            q.shape[0] * q.shape[1],
            q.shape[1],
            q.shape[2],
            q.shape[3],
            q.stride(),
            k.stride(),
            v.stride(),
            indices.stride(0),
            block,
        )
    return outputs


__all__ = ["fused_pack_qkv"]
