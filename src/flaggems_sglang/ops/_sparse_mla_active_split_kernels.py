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

"""BF16 selected-KV split and active-head merge, SM90 target.

The selected index list is partitioned by position, never reselected. Partial
statistics contain KV only. The learnable sink is added once by the merge.
"""

import triton
import triton.language as tl


@triton.jit
def _active_head_sparse_mla_partials(
    Q,
    KV,
    INDICES,
    LENGTHS,
    PART_O,
    PART_M,
    PART_L,
    SKV,
    SM_SCALE: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    ACTIVE_HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    HAVE_LENGTHS: tl.constexpr,
    SPLITS: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    pid = tl.program_id(0)
    query = pid // SPLITS
    split = pid % SPLITS
    heads = tl.arange(0, BH)
    head_valid = heads < ACTIVE_HEADS
    dims = tl.arange(0, 256)
    rope_dims = tl.arange(0, 64)
    columns = split * BK + tl.arange(0, BK)
    length = tl.load(LENGTHS + query) if HAVE_LENGTHS else TOPK
    position_valid = (columns < TOPK) & (columns < length)
    slots = tl.load(INDICES + query * TOPK + columns, position_valid, other=-1)
    valid = position_valid & (slots >= 0) & (slots < SKV)
    safe_slots = tl.where(valid, slots, 0).to(tl.int64)

    q_base = Q + query * Q_STRIDE + heads[:, None] * 576
    q0 = tl.load(q_base + dims[None, :], head_valid[:, None], other=0.0)
    q1 = tl.load(q_base + 256 + dims[None, :], head_valid[:, None], other=0.0)
    qr = tl.load(
        q_base + 512 + rope_dims[None, :], head_valid[:, None], other=0.0
    )
    kv_base = KV + safe_slots[None, :] * KV_STRIDE
    k0 = tl.load(kv_base + dims[:, None], valid[None, :], other=0.0)
    k1 = tl.load(kv_base + 256 + dims[:, None], valid[None, :], other=0.0)
    kr = tl.load(kv_base + 512 + rope_dims[:, None], valid[None, :], other=0.0)
    logits = tl.dot(q0, k0, out_dtype=tl.float32)
    logits = tl.dot(q1, k1, logits, out_dtype=tl.float32)
    logits = tl.dot(qr, kr, logits, out_dtype=tl.float32)
    logits *= SM_SCALE
    cell_valid = head_valid[:, None] & valid[None, :]
    logits = tl.where(cell_valid, logits, float("-inf"))
    maximum = tl.max(logits, axis=1)
    safe_maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    probability = tl.exp(logits - safe_maximum[:, None])
    probability = tl.where(cell_valid, probability, 0.0)
    normalizer = tl.sum(probability, axis=1)
    probability_bf16 = probability.to(tl.bfloat16)
    stat = (query * SPLITS + split) * ACTIVE_HEADS + heads
    tl.store(PART_M + stat, maximum, head_valid)
    tl.store(PART_L + stat, normalizer, head_valid)
    out = PART_O + stat[:, None] * 512 + dims[None, :]
    # Store one DV half before producing the other: no cross-tile reduction
    # changes, and the two FP32 output accumulators need not stay live together.
    accum0 = tl.dot(probability_bf16, tl.trans(k0), out_dtype=tl.float32)
    tl.store(out, accum0, head_valid[:, None])
    accum1 = tl.dot(probability_bf16, tl.trans(k1), out_dtype=tl.float32)
    tl.store(out + 256, accum1, head_valid[:, None])


@triton.jit
def _active_head_sparse_mla_merge(
    PART_O,
    PART_M,
    PART_L,
    SINK,
    OUT,
    MAXIMUM,
    LSE,
    ACTIVE_HEADS: tl.constexpr,
    HAVE_SINK: tl.constexpr,
    SPLITS: tl.constexpr,
    BDV: tl.constexpr,
):
    query_head = tl.program_id(0)
    dim_block = tl.program_id(1)
    query = query_head // ACTIVE_HEADS
    head = query_head % ACTIVE_HEADS
    split = tl.arange(0, SPLITS)
    dims = dim_block * BDV + tl.arange(0, BDV)
    stat = (query * SPLITS + split) * ACTIVE_HEADS + head
    maximum = tl.load(PART_M + stat)
    normalizer = tl.load(PART_L + stat)

    # Global KV-only statistics. Empty partials have m=-inf,l=0. A NaN
    # normalizer is not silently converted into an empty row.
    have_keys = tl.sum((normalizer != 0.0).to(tl.int32), axis=0) != 0
    global_max = tl.max(maximum, axis=0)
    safe_global_max = tl.where(global_max == float("-inf"), 0.0, global_max)
    kv_weights = tl.exp(maximum - safe_global_max)
    kv_sum = tl.sum(normalizer * kv_weights, axis=0)
    safe_kv_sum = tl.where(kv_sum == 0.0, 1.0, kv_sum)
    original_lse = safe_global_max + tl.log(safe_kv_sum)
    original_lse = tl.where(have_keys, original_lse, float("inf"))

    sink = tl.load(SINK + head) if HAVE_SINK else float("-inf")
    positive_infinite_sink = sink == float("inf")
    sink_for_math = tl.where(positive_infinite_sink, float("-inf"), sink)
    # A single global sink term. It never entered PART_L/PART_O.
    denominator_base = tl.maximum(global_max, sink_for_math)
    safe_base = tl.where(
        denominator_base == float("-inf"), 0.0, denominator_base
    )
    weights = tl.exp(maximum - safe_base)
    denominator = tl.sum(normalizer * weights, axis=0)
    if HAVE_SINK:
        denominator += tl.exp(sink_for_math - safe_base)
    safe_denominator = tl.where(denominator == 0.0, 1.0, denominator)
    values = tl.load(PART_O + stat[:, None] * 512 + dims[None, :])
    numerator = tl.sum(values * weights[:, None], axis=0)
    output = numerator / safe_denominator
    output = tl.where(have_keys & ~positive_infinite_sink, output, 0.0)
    tl.store(OUT + query_head * 512 + dims, output.to(tl.bfloat16))
    if dim_block == 0:
        tl.store(MAXIMUM + query_head, global_max)
        tl.store(LSE + query_head, original_lse)
