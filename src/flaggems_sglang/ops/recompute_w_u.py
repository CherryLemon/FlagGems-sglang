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
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    KS3: tl.constexpr,
    VS0: tl.constexpr,
    VS1: tl.constexpr,
    VS2: tl.constexpr,
    VS3: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BS2: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    AS3: tl.constexpr,
    US0: tl.constexpr,
    US1: tl.constexpr,
    US2: tl.constexpr,
    US3: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    tile = tl.program_id(1)
    q = tl.program_id(0).to(tl.int64)
    h = q % H
    c = q // H % (T // BT)
    b = q // (H * (T // BT))
    mr = tile // tl.cdiv(tl.maximum(DK, DV), BN) * BM + tl.arange(0, BM)
    mr = mr.to(tl.int64)
    nn = tile % tl.cdiv(tl.maximum(DK, DV), BN) * BN + tl.arange(0, BN)
    nn = nn.to(tl.int64)
    rr = tl.arange(0, BR)
    rr = rr.to(tl.int64)
    a = tl.load(
        Ap
        + b * AS0
        + (c * BT + mr[:, None]) * AS1
        + h * AS2
        + rr[None, :] * AS3,
        (mr[:, None] < BT) & (rr[None, :] < BT),
        0,
    )
    beta = tl.load(
        Bp + b * BS0 + (c * BT + rr) * BS1 + h * BS2, rr < BT, 0
    ).to(tl.float32)
    gate = tl.exp(
        tl.load(Gp + b * GS0 + (c * BT + rr) * GS1 + h * GS2, rr < BT, 0).to(
            tl.float32
        )
    )
    v = tl.load(
        Vp
        + b * VS0
        + (c * BT + rr[:, None]) * VS1
        + h * VS2
        + nn[None, :] * VS3,
        (rr[:, None] < BT) & (nn[None, :] < DV),
        0,
    )
    k = tl.load(
        Kp
        + b * KS0
        + (c * BT + rr[:, None]) * KS1
        + (h // (H // HG)) * KS2
        + nn[None, :] * KS3,
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
        + b * US0
        + (c * BT + mr[:, None]) * US1
        + h * US2
        + nn[None, :] * US3,
        u,
        (mr[:, None] < BT) & (nn[None, :] < DV),
    )


@triton.jit
def _compact(
    Kp,
    Vp,
    Bp,
    Gp,
    Ap,
    Wp,
    Up,
    H: tl.constexpr,
    HG: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    SPLIT: tl.constexpr,
    INDEX64: tl.constexpr,
):
    q = tl.program_id(0)
    if INDEX64:
        q = q.to(tl.int64)
    tile = tl.program_id(1)
    h = q % H
    t = q // H * BT
    if SPLIT:
        ntiles: tl.constexpr = tl.cdiv(DK, BN) + tl.cdiv(DV, BN)
    else:
        ntiles: tl.constexpr = tl.cdiv(max(DK, DV), BN)
    ni = tile % ntiles
    mr = tile // ntiles * BM + tl.arange(0, BM)
    rr = tl.arange(0, BR)
    a = tl.load(
        Ap + ((t + mr[:, None]) * H + h) * BT + rr[None, :],
        (mr[:, None] < BT) & (rr[None, :] < BT),
        0,
    )
    beta = tl.load(Bp + (t + rr) * H + h, rr < BT, 0).to(tl.float32)
    if SPLIT:
        is_k = ni < tl.cdiv(DK, BN)
        nn = tl.where(is_k, ni, ni - tl.cdiv(DK, BN)) * BN + tl.arange(0, BN)
        if is_k:
            gate = tl.exp(
                tl.load(Gp + (t + rr) * H + h, rr < BT, 0).to(tl.float32)
            )
            k = tl.load(
                Kp
                + ((t + rr[:, None]) * HG + h // (H // HG)) * DK
                + nn[None, :],
                (rr[:, None] < BT) & (nn[None, :] < DK),
                0,
            )
            x = ((k.to(tl.float32) * beta[:, None]) * gate[:, None]).to(
                k.dtype
            )
        else:
            v = tl.load(
                Vp + ((t + rr[:, None]) * H + h) * DV + nn[None, :],
                (rr[:, None] < BT) & (nn[None, :] < DV),
                0,
            )
            x = (v.to(tl.float32) * beta[:, None]).to(v.dtype)
        y = tl.dot(a, x)
        if is_k:
            tl.store(
                Wp + ((t + mr[:, None]) * H + h) * DK + nn[None, :],
                y,
                (mr[:, None] < BT) & (nn[None, :] < DK),
            )
        else:
            tl.store(
                Up + ((t + mr[:, None]) * H + h) * DV + nn[None, :],
                y,
                (mr[:, None] < BT) & (nn[None, :] < DV),
            )
    else:
        nn = ni * BN + tl.arange(0, BN)
        gate = tl.exp(
            tl.load(Gp + (t + rr) * H + h, rr < BT, 0).to(tl.float32)
        )
        k = tl.load(
            Kp + ((t + rr[:, None]) * HG + h // (H // HG)) * DK + nn[None, :],
            (rr[:, None] < BT) & (nn[None, :] < DK),
            0,
        )
        v = tl.load(
            Vp + ((t + rr[:, None]) * H + h) * DV + nn[None, :],
            (rr[:, None] < BT) & (nn[None, :] < DV),
            0,
        )
        kb = ((k.to(tl.float32) * beta[:, None]) * gate[:, None]).to(k.dtype)
        vb = (v.to(tl.float32) * beta[:, None]).to(v.dtype)
        w = tl.dot(a, kb)
        u = tl.dot(a, vb)
        tl.store(
            Wp + ((t + mr[:, None]) * H + h) * DK + nn[None, :],
            w,
            (mr[:, None] < BT) & (nn[None, :] < DK),
        )
        tl.store(
            Up + ((t + mr[:, None]) * H + h) * DV + nn[None, :],
            u,
            (mr[:, None] < BT) & (nn[None, :] < DV),
        )


@triton.jit
def _prepare(
    Kp,
    Vp,
    Bp,
    Gp,
    Ap,
    Xk,
    Xv,
    Aa,
    H: tl.constexpr,
    HG: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    q = tl.program_id(0)
    h = q % H
    t = q // H * BT
    r = tl.arange(0, BR)
    n = tl.arange(0, BN)
    beta = tl.load(Bp + (t + r) * H + h, r < BT, 0).to(tl.float32)
    gate = tl.exp(tl.load(Gp + (t + r) * H + h, r < BT, 0).to(tl.float32))
    k = tl.load(
        Kp + ((t + r[:, None]) * HG + h // (H // HG)) * DK + n[None, :],
        (r[:, None] < BT) & (n[None, :] < DK),
        0,
    )
    v = tl.load(
        Vp + ((t + r[:, None]) * H + h) * DV + n[None, :],
        (r[:, None] < BT) & (n[None, :] < DV),
        0,
    )
    kb = ((k.to(tl.float32) * beta[:, None]) * gate[:, None]).to(k.dtype)
    vb = (v.to(tl.float32) * beta[:, None]).to(v.dtype)
    tl.store(
        Xk + (q * BT + r[:, None]) * DK + n[None, :],
        kb,
        (r[:, None] < BT) & (n[None, :] < DK),
    )
    tl.store(
        Xv + (q * BT + r[:, None]) * DV + n[None, :],
        vb,
        (r[:, None] < BT) & (n[None, :] < DV),
    )
    a = tl.load(
        Ap + ((t + r[:, None]) * H + h) * BT + r[None, :],
        (r[:, None] < BT) & (r[None, :] < BT),
        0,
    )
    tl.store(
        Aa + (q * BT + r[:, None]) * BT + r[None, :],
        a,
        (r[:, None] < BT) & (r[None, :] < BT),
    )


@triton.jit
def _packed_dot(
    Xk,
    Xv,
    Aa,
    Wp,
    Up,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    q = tl.program_id(0)
    ni = tl.program_id(1)
    h = q % H
    t = q // H * BT
    r = tl.arange(0, BR)
    m = tl.arange(0, BM)
    is_k = ni < tl.cdiv(DK, BN)
    n = tl.where(is_k, ni, ni - tl.cdiv(DK, BN)) * BN + tl.arange(0, BN)
    a = tl.load(
        Aa + (q * BT + m[:, None]) * BT + r[None, :],
        (m[:, None] < BT) & (r[None, :] < BT),
        0,
    )
    if is_k:
        x = tl.load(
            Xk + (q * BT + r[:, None]) * DK + n[None, :],
            (r[:, None] < BT) & (n[None, :] < DK),
            0,
        )
    else:
        x = tl.load(
            Xv + (q * BT + r[:, None]) * DV + n[None, :],
            (r[:, None] < BT) & (n[None, :] < DV),
            0,
        )
    y = tl.dot(a, x)
    if is_k:
        tl.store(
            Wp + ((t + m[:, None]) * H + h) * DK + n[None, :],
            y,
            (m[:, None] < BT) & (n[None, :] < DK),
        )
    else:
        tl.store(
            Up + ((t + m[:, None]) * H + h) * DV + n[None, :],
            y,
            (m[:, None] < BT) & (n[None, :] < DV),
        )


def recompute_w_u(k, v, beta, g_cumsum, A, cu_seqlens):
    assert cu_seqlens is None
    B, T, Hg, K = k.shape
    H, V = v.shape[-2:]
    BT = A.shape[-1]
    w = torch.empty((B, T, H, K), dtype=k.dtype, device=k.device)
    u = torch.empty_like(v)
    q = B * (T // BT) * H
    if q == 0:
        return w, u
    if (
        torch.version.hip is not None
        and q >= 512
        and BT == 64
        and K <= 128
        and V <= 128
        and k.is_contiguous()
        and v.is_contiguous()
        and beta.is_contiguous()
        and g_cumsum.is_contiguous()
        and A.is_contiguous()
    ):
        xk = torch.empty((q, BT, K), dtype=k.dtype, device=k.device)
        xv = torch.empty((q, BT, V), dtype=v.dtype, device=v.device)
        aa = torch.empty((q, BT, BT), dtype=A.dtype, device=A.device)
        _prepare[(q,)](
            k,
            v,
            beta,
            g_cumsum,
            A,
            xk,
            xv,
            aa,
            H,
            Hg,
            K,
            V,
            BT,
            128,
            64,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        _packed_dot[(q, triton.cdiv(K, 128) + triton.cdiv(V, 128))](
            xk,
            xv,
            aa,
            w,
            u,
            H,
            K,
            V,
            BT,
            64,
            128,
            64,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return w, u
    if (
        k.is_contiguous()
        and v.is_contiguous()
        and beta.is_contiguous()
        and g_cumsum.is_contiguous()
        and A.is_contiguous()
    ):
        bm, split = 64, (torch.version.hip is not None or q >= 16)
        if q >= 512:
            bn, nw = 128, 8
        else:
            bn, nw = 64, 4
        if split:
            nt = triton.cdiv(K, bn) + triton.cdiv(V, bn)
        else:
            nt = triton.cdiv(max(K, V), bn)
        _compact[(q, triton.cdiv(BT, bm) * nt)](
            k,
            v,
            beta,
            g_cumsum,
            A,
            w,
            u,
            H,
            Hg,
            K,
            V,
            BT,
            bm,
            bn,
            max(16, triton.next_power_of_2(BT)),
            split,
            B * T * H * max(K, V, BT) >= 2147483648,
            num_warps=nw,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        _recompute[(q, triton.cdiv(BT, 64) * triton.cdiv(max(K, V), 64))](
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
            *k.stride(),
            *v.stride(),
            *beta.stride(),
            *g_cumsum.stride(),
            *A.stride(),
            *u.stride(),
            64,
            64,
            max(16, triton.next_power_of_2(BT)),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return w, u
