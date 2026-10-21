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

import torch
import triton
import triton.language as tl

__all__ = ["recompute_w_u"]


@triton.jit
def _recompute(
    Kp,
    Vp,
    Bp,
    Gp,
    Ap,
    Wp,
    Up,
    T: tl.constexpr,
    H: tl.constexpr,
    HG: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    BS: tl.constexpr,
    GS: tl.constexpr,
    AS: tl.constexpr,
    US: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    LARGE: tl.constexpr,
):
    tile = tl.program_id(0)
    q = tl.program_id(1)
    if LARGE:
        q = q.to(tl.int64)
    h = q % H
    c = q // H % (T // BT)
    b = q // (H * (T // BT))
    mr = tile // tl.cdiv(tl.maximum(DK, DV), BN) * BM + tl.arange(0, BM)
    nn = tile % tl.cdiv(tl.maximum(DK, DV), BN) * BN + tl.arange(0, BN)
    rr = tl.arange(0, BR)
    if LARGE:
        mr = mr.to(tl.int64)
        nn = nn.to(tl.int64)
        rr = rr.to(tl.int64)
    a = tl.load(
        Ap
        + b * AS[0]
        + (c * BT + mr[:, None]) * AS[1]
        + h * AS[2]
        + rr[None, :] * AS[3],
        (mr[:, None] < BT) & (rr[None, :] < BT),
        0,
    )
    beta = tl.load(
        Bp + b * BS[0] + (c * BT + rr) * BS[1] + h * BS[2], rr < BT, 0
    ).to(tl.float32)
    gate = tl.exp(
        tl.load(
            Gp + b * GS[0] + (c * BT + rr) * GS[1] + h * GS[2], rr < BT, 0
        ).to(tl.float32)
    )
    v = tl.load(
        Vp
        + b * VS[0]
        + (c * BT + rr[:, None]) * VS[1]
        + h * VS[2]
        + nn[None, :] * VS[3],
        (rr[:, None] < BT) & (nn[None, :] < DV),
        0,
    )
    k = tl.load(
        Kp
        + b * KS[0]
        + (c * BT + rr[:, None]) * KS[1]
        + (h // (H // HG)) * KS[2]
        + nn[None, :] * KS[3],
        (rr[:, None] < BT) & (nn[None, :] < DK),
        0,
    )
    vb = (v.to(tl.float32) * beta[:, None]).to(v.dtype)
    kb = ((k.to(tl.float32) * beta[:, None]) * gate[:, None]).to(k.dtype)
    u = tl.dot(a, vb)
    w = tl.dot(a, kb)
    tl.store(
        Wp + ((b * T + c * BT + mr[:, None]) * H + h) * DK + nn[None, :],
        w,
        (mr[:, None] < BT) & (nn[None, :] < DK),
    )
    tl.store(
        Up
        + b * US[0]
        + (c * BT + mr[:, None]) * US[1]
        + h * US[2]
        + nn[None, :] * US[3],
        u,
        (mr[:, None] < BT) & (nn[None, :] < DV),
    )


@triton.jit
def _direct(
    Kp,
    Vp,
    Bp,
    Gp,
    Ap,
    Wp,
    Up,
    T: tl.constexpr,
    H: tl.constexpr,
    HG: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    BS: tl.constexpr,
    GS: tl.constexpr,
    AS: tl.constexpr,
    US: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    q = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    h = q % H
    c = q // H % (T // BT)
    b = q // (H * (T // BT))
    mr = tile // tl.cdiv(tl.maximum(DK, DV), BN)
    nn = tile % tl.cdiv(tl.maximum(DK, DV), BN) * BN + tl.arange(0, BN)
    rr = tl.arange(0, BR).to(tl.int64)
    nn = nn.to(tl.int64)
    a = tl.load(
        Ap + b * AS[0] + (c * BT + mr) * AS[1] + h * AS[2] + rr * AS[3],
        rr < BT,
        0,
    ).to(tl.float32)
    beta = tl.load(
        Bp + b * BS[0] + (c * BT + rr) * BS[1] + h * BS[2], rr < BT, 0
    ).to(tl.float32)
    gate = tl.exp(
        tl.load(
            Gp + b * GS[0] + (c * BT + rr) * GS[1] + h * GS[2], rr < BT, 0
        ).to(tl.float32)
    )
    v = tl.load(
        Vp
        + b * VS[0]
        + (c * BT + rr[:, None]) * VS[1]
        + h * VS[2]
        + nn[None, :] * VS[3],
        (rr[:, None] < BT) & (nn[None, :] < DV),
        0,
    )
    k = tl.load(
        Kp
        + b * KS[0]
        + (c * BT + rr[:, None]) * KS[1]
        + (h // (H // HG)) * KS[2]
        + nn[None, :] * KS[3],
        (rr[:, None] < BT) & (nn[None, :] < DK),
        0,
    )
    vb = (v.to(tl.float32) * beta[:, None]).to(v.dtype).to(tl.float32)
    kb = (
        ((k.to(tl.float32) * beta[:, None]) * gate[:, None])
        .to(k.dtype)
        .to(tl.float32)
    )
    w = tl.sum(a[:, None] * kb, 0)
    u = tl.sum(a[:, None] * vb, 0)
    tl.store(Wp + ((b * T + c * BT + mr) * H + h) * DK + nn, w, nn < DK)
    tl.store(
        Up + b * US[0] + (c * BT + mr) * US[1] + h * US[2] + nn * US[3],
        u,
        nn < DV,
    )


def recompute_w_u(k, v, beta, g_cumsum, A, cu_seqlens):
    assert cu_seqlens is None
    B, T, Hg, K = k.shape
    H, V = v.shape[-2:]
    BT = A.shape[-1]
    w = torch.empty((B, T, H, K), dtype=k.dtype, device=k.device)
    u = torch.empty_like(v)
    if B * T * H == 0:
        return w, u
    if BT % 16 != 0:
        _direct[(B * (T // BT) * H, BT * triton.cdiv(max(K, V), 32))](
            k,
            v,
            beta,
            g_cumsum,
            A,
            w,
            u,
            T,
            H,
            Hg,
            K,
            V,
            BT,
            k.stride(),
            v.stride(),
            beta.stride(),
            g_cumsum.stride(),
            A.stride(),
            u.stride(),
            32,
            triton.next_power_of_2(BT),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return w, u
    contiguous = (
        k.is_contiguous()
        and v.is_contiguous()
        and beta.is_contiguous()
        and g_cumsum.is_contiguous()
        and A.is_contiguous()
    )
    if not contiguous:
        _direct[(B * (T // BT) * H, BT * triton.cdiv(max(K, V), 32))](
            k,
            v,
            beta,
            g_cumsum,
            A,
            w,
            u,
            T,
            H,
            Hg,
            K,
            V,
            BT,
            k.stride(),
            v.stride(),
            beta.stride(),
            g_cumsum.stride(),
            A.stride(),
            u.stride(),
            32,
            triton.next_power_of_2(BT),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return w, u
    large = B * T * H * max(K, V, BT) >= 2147483648
    bm = 64
    bn = 128 if B * (T // BT) * H >= 128 and max(K, V) > 64 else 64
    _recompute[
        (triton.cdiv(BT, bm) * triton.cdiv(max(K, V), bn), B * (T // BT) * H)
    ](
        k,
        v,
        beta,
        g_cumsum,
        A,
        w,
        u,
        T,
        H,
        Hg,
        K,
        V,
        BT,
        k.stride(),
        v.stride(),
        beta.stride(),
        g_cumsum.stride(),
        A.stride(),
        u.stride(),
        bm,
        bn,
        max(16, triton.next_power_of_2(BT)),
        large,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return w, u
