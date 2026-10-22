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

"""fused_moe_router_cudacore for RDNA3 (gfx1100).

The kernels of the generic version, with two changes. With at most 32
experts one program holds an experts-by-hidden tile of up to 32768 fp32
values, which the generic version runs on a single warp; the number of warps
here grows with the tile, and with the number of experts when softcap is
used. With more than 32 experts a program of the logits kernel computes 1, 2
or 4 tokens and can sum the products element-wise over the hidden loop
before reducing them, with launch parameters by batch size and element size.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _router_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    BS: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    STRIDE_X_B: tl.constexpr,
    STRIDE_X_H: tl.constexpr,
    STRIDE_W_E: tl.constexpr,
    STRIDE_W_H: tl.constexpr,
    CAP: tl.constexpr,
    HAS_CAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    e_live = offs_e < E
    t0 = pid_m * BLOCK_M
    t1 = t0 + 1
    t2 = t0 + 2
    t3 = t0 + 3
    acc0 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    even = (H % BLOCK_H == 0) and (BS % BLOCK_M == 0)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        if even:
            w = tl.load(
                w_ptr
                + offs_e[:, None] * STRIDE_W_E
                + offs_h[None, :] * STRIDE_W_H,
                mask=e_live[:, None],
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(x_ptr + t0 * STRIDE_X_B + offs_h * STRIDE_X_H).to(
                tl.float32
            )
            x1 = tl.load(x_ptr + t1 * STRIDE_X_B + offs_h * STRIDE_X_H).to(
                tl.float32
            )
            x2 = tl.load(x_ptr + t2 * STRIDE_X_B + offs_h * STRIDE_X_H).to(
                tl.float32
            )
            x3 = tl.load(x_ptr + t3 * STRIDE_X_B + offs_h * STRIDE_X_H).to(
                tl.float32
            )
        else:
            h_live = offs_h < H
            w = tl.load(
                w_ptr
                + offs_e[:, None] * STRIDE_W_E
                + offs_h[None, :] * STRIDE_W_H,
                mask=e_live[:, None] & h_live[None, :],
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(
                x_ptr + t0 * STRIDE_X_B + offs_h * STRIDE_X_H,
                mask=(t0 < BS) & h_live,
                other=0.0,
            ).to(tl.float32)
            x1 = tl.load(
                x_ptr + t1 * STRIDE_X_B + offs_h * STRIDE_X_H,
                mask=(t1 < BS) & h_live,
                other=0.0,
            ).to(tl.float32)
            x2 = tl.load(
                x_ptr + t2 * STRIDE_X_B + offs_h * STRIDE_X_H,
                mask=(t2 < BS) & h_live,
                other=0.0,
            ).to(tl.float32)
            x3 = tl.load(
                x_ptr + t3 * STRIDE_X_B + offs_h * STRIDE_X_H,
                mask=(t3 < BS) & h_live,
                other=0.0,
            ).to(tl.float32)
        acc0 += tl.sum(w * x0[None, :], axis=1)
        acc1 += tl.sum(w * x1[None, :], axis=1)
        acc2 += tl.sum(w * x2[None, :], axis=1)
        acc3 += tl.sum(w * x3[None, :], axis=1)
    if HAS_CAP:
        acc0 = (2.0 * tl.sigmoid(2.0 * (acc0 / CAP)) - 1.0) * CAP
        acc1 = (2.0 * tl.sigmoid(2.0 * (acc1 / CAP)) - 1.0) * CAP
        acc2 = (2.0 * tl.sigmoid(2.0 * (acc2 / CAP)) - 1.0) * CAP
        acc3 = (2.0 * tl.sigmoid(2.0 * (acc3 / CAP)) - 1.0) * CAP
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=e_live, other=0.0).to(
            tl.float32
        )
        acc0 += bias
        acc1 += bias
        acc2 += bias
        acc3 += bias
    logits0 = tl.where(e_live, acc0, NEG)
    top0 = tl.max(logits0, axis=0)
    inv0 = 1.0 / tl.sum(tl.exp(logits0 - top0), axis=0)
    rem0 = logits0
    for slot in range(0, TOPK):
        sel = tl.max(rem0, axis=0)
        expert = tl.min(tl.where(rem0 == sel, offs_e, BLOCK_E), axis=0)
        tl.store(ids_ptr + t0 * TOPK + slot, expert, mask=t0 < BS)
        tl.store(
            weights_ptr + t0 * TOPK + slot,
            tl.exp(sel - top0) * inv0,
            mask=t0 < BS,
        )
        rem0 = tl.where(offs_e == expert, NEG, rem0)

    logits1 = tl.where(e_live, acc1, NEG)
    top1 = tl.max(logits1, axis=0)
    inv1 = 1.0 / tl.sum(tl.exp(logits1 - top1), axis=0)
    rem1 = logits1
    for slot in range(0, TOPK):
        sel = tl.max(rem1, axis=0)
        expert = tl.min(tl.where(rem1 == sel, offs_e, BLOCK_E), axis=0)
        tl.store(ids_ptr + t1 * TOPK + slot, expert, mask=t1 < BS)
        tl.store(
            weights_ptr + t1 * TOPK + slot,
            tl.exp(sel - top1) * inv1,
            mask=t1 < BS,
        )
        rem1 = tl.where(offs_e == expert, NEG, rem1)

    logits2 = tl.where(e_live, acc2, NEG)
    top2 = tl.max(logits2, axis=0)
    inv2 = 1.0 / tl.sum(tl.exp(logits2 - top2), axis=0)
    rem2 = logits2
    for slot in range(0, TOPK):
        sel = tl.max(rem2, axis=0)
        expert = tl.min(tl.where(rem2 == sel, offs_e, BLOCK_E), axis=0)
        tl.store(ids_ptr + t2 * TOPK + slot, expert, mask=t2 < BS)
        tl.store(
            weights_ptr + t2 * TOPK + slot,
            tl.exp(sel - top2) * inv2,
            mask=t2 < BS,
        )
        rem2 = tl.where(offs_e == expert, NEG, rem2)

    logits3 = tl.where(e_live, acc3, NEG)
    top3 = tl.max(logits3, axis=0)
    inv3 = 1.0 / tl.sum(tl.exp(logits3 - top3), axis=0)
    rem3 = logits3
    for slot in range(0, TOPK):
        sel = tl.max(rem3, axis=0)
        expert = tl.min(tl.where(rem3 == sel, offs_e, BLOCK_E), axis=0)
        tl.store(ids_ptr + t3 * TOPK + slot, expert, mask=t3 < BS)
        tl.store(
            weights_ptr + t3 * TOPK + slot,
            tl.exp(sel - top3) * inv3,
            mask=t3 < BS,
        )
        rem3 = tl.where(offs_e == expert, NEG, rem3)


@triton.jit
def _add_products(
    acc, w, x_ptrs, t_live, h_live, EVEN: tl.constexpr, DEFER: tl.constexpr
):
    if EVEN:
        x = tl.load(x_ptrs)
    else:
        x = tl.load(x_ptrs, mask=t_live & h_live, other=0.0)
    p = w * x.to(tl.float32)[None, :]
    if not DEFER:
        p = tl.sum(p, axis=1)
    return acc + p


@triton.jit
def _finish(
    acc,
    bias,
    CAP: tl.constexpr,
    HAS_CAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DEFER: tl.constexpr,
):
    if DEFER:
        acc = tl.sum(acc, axis=1)
    if HAS_CAP:
        acc = (2.0 * tl.sigmoid(2.0 * (acc / CAP)) - 1.0) * CAP
    if HAS_BIAS:
        acc += bias
    return acc


@triton.jit
def _logits_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    logits_ptr,
    BS: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    STRIDE_X_B: tl.constexpr,
    STRIDE_X_H: tl.constexpr,
    STRIDE_W_E: tl.constexpr,
    STRIDE_W_H: tl.constexpr,
    CAP: tl.constexpr,
    HAS_CAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DEFER: tl.constexpr,
):
    # BLOCK_M tokens (1, 2 or 4) per program. With DEFER the products are
    # summed element-wise over the hidden loop and reduced once after it.
    offs_e = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    e_live = offs_e < E
    t0 = tl.program_id(0) * BLOCK_M
    even = (H % BLOCK_H == 0) and (E % BLOCK_E == 0) and (BS % BLOCK_M == 0)
    if DEFER:
        acc0 = tl.zeros((BLOCK_E, BLOCK_H), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_E, BLOCK_H), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_E, BLOCK_H), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_E, BLOCK_H), dtype=tl.float32)
    else:
        acc0 = tl.zeros((BLOCK_E,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_E,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_E,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        h_live = offs_h < H
        w_ptrs = (
            w_ptr + offs_e[:, None] * STRIDE_W_E + offs_h[None, :] * STRIDE_W_H
        )
        if even:
            w = tl.load(w_ptrs).to(tl.float32)
        else:
            w = tl.load(
                w_ptrs, mask=e_live[:, None] & h_live[None, :], other=0.0
            ).to(tl.float32)
        xp = x_ptr + t0 * STRIDE_X_B + offs_h * STRIDE_X_H
        acc0 = _add_products(acc0, w, xp, t0 < BS, h_live, even, DEFER)
        if BLOCK_M > 1:
            acc1 = _add_products(
                acc1, w, xp + STRIDE_X_B, t0 + 1 < BS, h_live, even, DEFER
            )
        if BLOCK_M > 2:
            acc2 = _add_products(
                acc2, w, xp + 2 * STRIDE_X_B, t0 + 2 < BS, h_live, even, DEFER
            )
            acc3 = _add_products(
                acc3, w, xp + 3 * STRIDE_X_B, t0 + 3 < BS, h_live, even, DEFER
            )
    bias = 0.0
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=e_live, other=0.0).to(
            tl.float32
        )
    out = logits_ptr + t0 * E + offs_e
    logits = _finish(acc0, bias, CAP, HAS_CAP, HAS_BIAS, DEFER)
    tl.store(out, logits, mask=e_live & (t0 < BS))
    if BLOCK_M > 1:
        logits = _finish(acc1, bias, CAP, HAS_CAP, HAS_BIAS, DEFER)
        tl.store(out + E, logits, mask=e_live & (t0 + 1 < BS))
    if BLOCK_M > 2:
        logits = _finish(acc2, bias, CAP, HAS_CAP, HAS_BIAS, DEFER)
        tl.store(out + 2 * E, logits, mask=e_live & (t0 + 2 < BS))
        logits = _finish(acc3, bias, CAP, HAS_CAP, HAS_BIAS, DEFER)
        tl.store(out + 3 * E, logits, mask=e_live & (t0 + 3 < BS))


@triton.jit
def _topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NEG: tl.constexpr,
):
    token = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    e_live = offs_e < E
    logits = tl.load(logits_ptr + token * E + offs_e, mask=e_live, other=NEG)
    logits = tl.where(e_live, logits, NEG)
    row_max = tl.max(logits, axis=0)
    inv = 1.0 / tl.sum(tl.exp(logits - row_max), axis=0)
    remaining = logits
    for slot in range(0, TOPK):
        selected = tl.max(remaining, axis=0)
        expert = tl.min(
            tl.where(remaining == selected, offs_e, BLOCK_E), axis=0
        )
        tl.store(ids_ptr + token * TOPK + slot, expert)
        tl.store(
            weights_ptr + token * TOPK + slot, tl.exp(selected - row_max) * inv
        )
        remaining = tl.where(offs_e == expert, NEG, remaining)


def _next_pow2(value):
    result = 1
    while result < value:
        result *= 2
    return result


# _logits_kernel launches with more than 32 experts, for router weights of 2
# and of 4 bytes per element: up to a batch size, (tokens per program,
# deferred reduction, experts per program, hidden block, warps).
_LOGITS_LAUNCHES = {
    2: (
        (8, 1, False, 4, 2048, 8),
        (64, 2, True, 8, 1024, 4),
        (768, 2, True, 8, 512, 2),
        (None, 2, True, 32, 256, 4),
    ),
    4: (
        (8, 1, True, 4, 2048, 8),
        (256, 2, True, 8, 512, 4),
        (None, 4, True, 32, 128, 8),
    ),
}


def _logits_launch(bs, hidden, element_size):
    """(block_m, defer, block_e, block_h, num_warps)."""
    # Hidden blocks of at least 32 lanes x 16 bytes, and at most one warp per
    # such row, so that a warp covers one expert row of the tile: with warps
    # over several expert rows, some launches (among them the generic one with
    # a small hidden size and softcap) gave wrong, run-to-run varying logits
    # on gfx1100.
    row = 512 // element_size
    launches = _LOGITS_LAUNCHES[4 if element_size >= 4 else 2]
    _, block_m, defer, block_e, block_h, num_warps = next(
        launch for launch in launches if launch[0] is None or bs <= launch[0]
    )
    block_h = max(min(block_h, _next_pow2(hidden)), row)
    num_warps = min(num_warps, block_e * block_h // row)
    return block_m, defer, block_e, block_h, num_warps


def fused_moe_router_cudacore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    x = x.contiguous()
    weights = router_weight.contiguous()
    bs, hidden = x.shape
    experts = weights.shape[0]
    topk = int(topk)
    cap = float(moe_softcapping)
    has_cap = cap != 0.0
    has_bias = correction_bias is not None
    bias_arg = correction_bias if has_bias else x
    sx = x.stride()
    sw = weights.stride()
    neg = -3.0e38
    if _next_pow2(experts) <= 32:
        topk_weights = torch.empty(
            (bs, topk), dtype=torch.float32, device=x.device
        )
        topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
        sx = x.stride()
        sw = router_weight.stride()
        block_e = _next_pow2(experts)
        block_h = 32768 // block_e
        if block_h > hidden:
            block_h = _next_pow2(hidden)
        if block_h < 16:
            block_h = 16
        block_m = 4
        # At most 128 fp32 tile values per lane, and 64 for a 32 x 1024
        # tile: on gfx1100 one warp holding 32 x 512 spills 2056 B of
        # registers per lane, and 32 x 1024 still spills 188 B on 8 warps.
        num_warps = max(1, block_e * block_h // 4096)
        if block_e == 32 and block_h == 1024:
            num_warps = 16
        # With softcap, fewer than block_e // 4 warps run 1.5-3x slower once
        # the tile reaches 4096 values and a program has more than 16 logits.
        rows = min(bs, block_m)
        if (
            has_cap
            and block_e >= 16
            and rows * block_e > 16
            and block_e * block_h >= 4096
        ):
            num_warps = max(num_warps, block_e // 4)
        _router_kernel[((bs + block_m - 1) // block_m,)](
            x,
            router_weight,
            correction_bias if has_bias else x,
            topk_weights,
            topk_ids,
            bs,
            hidden,
            experts,
            topk,
            sx[0],
            sx[1],
            sw[0],
            sw[1],
            cap,
            has_cap,
            has_bias,
            -3.0e38,
            block_e,
            block_h,
            block_m,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        logits = torch.empty(
            (bs, experts), dtype=torch.float32, device=x.device
        )
        topk_weights = torch.empty(
            (bs, topk), dtype=torch.float32, device=x.device
        )
        topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
        block_m, defer, block_e, block_h, num_warps = _logits_launch(
            bs, hidden, weights.element_size()
        )
        grid_n = (experts + block_e - 1) // block_e
        grid_m = (bs + block_m - 1) // block_m
        _logits_kernel[(grid_m, grid_n)](
            x,
            weights,
            bias_arg,
            logits,
            bs,
            hidden,
            experts,
            sx[0],
            sx[1],
            sw[0],
            sw[1],
            cap,
            has_cap,
            has_bias,
            block_e,
            block_h,
            block_m,
            defer,
            num_warps=num_warps,
            num_stages=1,
        )

        _topk_kernel[(bs,)](
            logits,
            topk_weights,
            topk_ids,
            experts,
            topk,
            _next_pow2(experts),
            neg,
            num_warps=1,
            num_stages=1,
        )
    return topk_weights, topk_ids


__all__ = ["fused_moe_router_cudacore"]
