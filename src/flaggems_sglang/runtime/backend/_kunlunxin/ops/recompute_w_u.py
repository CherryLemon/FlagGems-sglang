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
def _prepare(
    Kp,
    Vp,
    Bp,
    Gp,
    Xw,
    Xu,
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
    BN: tl.constexpr,
    BR: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64) if WIDE else tl.program_id(0)
    i = pid * BN + tl.arange(0, BN)
    nk = i % DK
    nv = i % DV
    rk = i // DK
    rv = i // DV
    hk = rk % H
    hv = rv % H
    tk = rk // H % T
    tv = rv // H % T
    bk = rk // (H * T)
    bv = rv // (H * T)
    beta_k = tl.load(Bp + bk * BS0 + tk * BS1 + hk * BS2, i < BR * DK, 0)
    beta_v = tl.load(Bp + bv * BS0 + tv * BS1 + hv * BS2, i < BR * DV, 0)
    gate = tl.exp(tl.load(Gp + bk * GS0 + tk * GS1 + hk * GS2, i < BR * DK, 0))
    v = tl.load(
        Vp + bv * VS0 + tv * VS1 + hv * VS2 + nv * VS3, i < BR * DV, 0
    ).to(tl.float32)
    k = tl.load(
        Kp + bk * KS0 + tk * KS1 + (hk // (H // HG)) * KS2 + nk * KS3,
        i < BR * DK,
        0,
    ).to(tl.float32)
    vb = (v * beta_v).to(tl.bfloat16, fp_downcast_rounding="rtne")
    kb = ((k * beta_k) * gate).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(Xw + i, kb, i < BR * DK)
    tl.store(Xu + i, vb, i < BR * DV)


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
    WIDE: tl.constexpr = False,
):
    tile = tl.program_id(1)
    q = tl.program_id(0).to(tl.int64) if WIDE else tl.program_id(0)
    h = q % H
    c = q // H % (T // BT)
    b = q // (H * (T // BT))
    mr = tile // tl.cdiv(tl.maximum(DK, DV), BN) * BM + tl.arange(0, BM)
    nn = tile % tl.cdiv(tl.maximum(DK, DV), BN) * BN + tl.arange(0, BN)
    rr = tl.arange(0, BR)
    if WIDE:
        mr = mr.to(tl.int64)
        nn = nn.to(tl.int64)
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
    vb = tl.load(
        Vp + ((b * T + c * BT + rr[:, None]) * H + h) * DV + nn[None, :],
        (rr[:, None] < BT) & (nn[None, :] < DV),
        0,
    )
    kb = tl.load(
        Kp + ((b * T + c * BT + rr[:, None]) * H + h) * DK + nn[None, :],
        (rr[:, None] < BT) & (nn[None, :] < DK),
        0,
    )
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
def _prepare_one(
    P,
    Bp,
    Gp,
    X,
    T: tl.constexpr,
    H: tl.constexpr,
    HG: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    KW: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64) if WIDE else tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    n = i % D
    row = i // D
    h = row % H
    t = row // H % T
    b = row // (H * T)
    ih = h // (H // HG) if KW else h
    x = tl.load(P + b * S0 + t * S1 + ih * S2 + n * S3, i < N, 0).to(
        tl.float32
    )
    beta = tl.load(Bp + b * B0 + t * B1 + h * B2, i < N, 0)
    z = x * beta
    if KW:
        gate = tl.exp(tl.load(Gp + b * G0 + t * G1 + h * G2, i < N, 0))
        z = z * gate
    tl.store(X + i, z.to(tl.bfloat16, fp_downcast_rounding="rtne"), i < N)


@triton.jit
def _compute_one(
    Ap,
    Xp,
    Op,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    A3: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    O3: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    q = tl.program_id(0).to(tl.int64) if WIDE else tl.program_id(0)
    tile = tl.program_id(1)
    h = q % H
    c = q // H % (T // BT)
    b = q // (H * (T // BT))
    m = tile // tl.cdiv(D, BN) * BM + tl.arange(0, BM)
    n = tile % tl.cdiv(D, BN) * BN + tl.arange(0, BN)
    r = tl.arange(0, BR)
    if WIDE:
        m = m.to(tl.int64)
        n = n.to(tl.int64)
        r = r.to(tl.int64)
    a = tl.load(
        Ap + b * A0 + (c * BT + m[:, None]) * A1 + h * A2 + r[None, :] * A3,
        (m[:, None] < BT) & (r[None, :] < BT),
        0,
    )
    x = tl.load(
        Xp + ((b * T + c * BT + r[:, None]) * H + h) * D + n[None, :],
        (r[:, None] < BT) & (n[None, :] < D),
        0,
    )
    y = tl.dot(a, x)
    tl.store(
        Op + b * O0 + (c * BT + m[:, None]) * O1 + h * O2 + n[None, :] * O3,
        y,
        (m[:, None] < BT) & (n[None, :] < D),
    )


@triton.jit
def _reduce_one(
    Ap,
    Xp,
    Op,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    N: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    A3: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    O2: tl.constexpr,
    O3: tl.constexpr,
    BR: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64) if WIDE else tl.program_id(0)
    i = pid * 16 + tl.arange(0, 16)
    n = i % D
    row = i // D
    h = row % H
    t = row // H % T
    b = row // (H * T)
    c = t // BT
    r = tl.arange(0, BR)
    if WIDE:
        r = r.to(tl.int64)
    a = tl.load(
        Ap
        + b[:, None] * A0
        + t[:, None] * A1
        + h[:, None] * A2
        + r[None, :] * A3,
        (i[:, None] < N) & (r[None, :] < BT),
        0,
    ).to(tl.float32)
    x = tl.load(
        Xp
        + ((b[:, None] * T + c[:, None] * BT + r[None, :]) * H + h[:, None])
        * D
        + n[:, None],
        (i[:, None] < N) & (r[None, :] < BT),
        0,
    ).to(tl.float32)
    y = tl.sum(a * x, 1).to(tl.bfloat16, fp_downcast_rounding="rtne")
    ptr = Op + b * O0 + t * O1 + h * O2 + n * O3
    tl.store(ptr, y, i < N)


def recompute_w_u(k, v, beta, g_cumsum, A, cu_seqlens):
    assert cu_seqlens is None
    B, T, Hg, K = k.shape
    H, V = v.shape[-2:]
    BT = A.shape[-1]
    w = torch.empty((B, T, H, K), dtype=k.dtype, device=k.device)
    u = torch.empty_like(v)
    if B * T * H == 0:
        return w, u
    wide = B * T * H * max(K, V) >= 2147483648 or any(
        sum((d - 1) * st for d, st in zip(x.shape, x.stride())) >= 2147483648
        for x in (k, v, beta, g_cumsum, A, u)
    )
    bm, bn = 64, min(128, K & -K, V & -V)
    xw = torch.empty((B * T * H * K,), dtype=k.dtype, device=k.device)
    xu = torch.empty((B * T * H * V,), dtype=v.dtype, device=v.device)
    reduce_rows = (
        K % 16 != 0
        or V % 16 != 0
        or bool(BT & (BT - 1))
        or BT < 32
        or min(K, V) < 32
        or A.stride(-1) != 1
        or u.stride(-1) != 1
        or (BT == 64 and K == V and K == 64 and B * T * H <= 4096)
    )
    if K != V or reduce_rows:
        _prepare_one[(triton.cdiv(B * T * H * K, 4096),)](
            k,
            beta,
            g_cumsum,
            xw,
            T,
            H,
            Hg,
            K,
            B * T * H * K,
            True,
            *k.stride(),
            *beta.stride(),
            *g_cumsum.stride(),
            4096,
            WIDE=wide,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _prepare_one[(triton.cdiv(B * T * H * V, 4096),)](
            v,
            beta,
            g_cumsum,
            xu,
            T,
            H,
            Hg,
            V,
            B * T * H * V,
            False,
            *v.stride(),
            *beta.stride(),
            *g_cumsum.stride(),
            4096,
            WIDE=wide,
            num_warps=4,
            enable_fp_fusion=False,
        )
        if reduce_rows:
            _reduce_one[(triton.cdiv(B * T * H * K, 16),)](
                A,
                xw,
                w,
                T,
                H,
                K,
                BT,
                B * T * H * K,
                *A.stride(),
                *w.stride(),
                triton.next_power_of_2(BT),
                WIDE=wide,
                num_warps=4,
            )
            _reduce_one[(triton.cdiv(B * T * H * V, 16),)](
                A,
                xu,
                u,
                T,
                H,
                V,
                BT,
                B * T * H * V,
                *A.stride(),
                *u.stride(),
                triton.next_power_of_2(BT),
                WIDE=wide,
                num_warps=4,
            )
            return w, u
        _compute_one[
            (B * (T // BT) * H, triton.cdiv(BT, 64) * triton.cdiv(K, 128))
        ](
            A,
            xw,
            w,
            T,
            H,
            K,
            BT,
            *A.stride(),
            *w.stride(),
            64,
            128,
            max(16, triton.next_power_of_2(BT)),
            WIDE=wide,
            num_warps=4,
        )
        _compute_one[
            (B * (T // BT) * H, triton.cdiv(BT, 64) * triton.cdiv(V, 128))
        ](
            A,
            xu,
            u,
            T,
            H,
            V,
            BT,
            *A.stride(),
            *u.stride(),
            64,
            128,
            max(16, triton.next_power_of_2(BT)),
            WIDE=wide,
            num_warps=4,
        )
        return w, u
    prep_block = 8192 if B * T * H * K >= 262144 else 4096
    _prepare[(triton.cdiv(B * T * H * max(K, V), prep_block),)](
        k,
        v,
        beta,
        g_cumsum,
        xw,
        xu,
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
        prep_block,
        B * T * H,
        WIDE=wide,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _recompute[
        (B * (T // BT) * H, triton.cdiv(BT, bm) * triton.cdiv(max(K, V), bn))
    ](
        xw,
        xu,
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
        bm,
        bn,
        max(16, triton.next_power_of_2(BT)),
        WIDE=wide,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return w, u
