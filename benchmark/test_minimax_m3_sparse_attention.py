# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standard operator benchmark with an unfused Torch reference baseline."""

import os

import flaggems_sglang
import pytest
import torch
from flag_attn.testing.minimax_sparse_paged import (
    make_sparse_paged_inputs,
    sparse_paged_reference,
)

from .op_benchmark import OpBenchmark


def _inputs(shape, dtype, device):
    batch, qlen, context, page_size = shape
    a = make_sparse_paged_inputs(
        batch, qlen, context, page_size, dtype, device=device
    )
    partial, lse = flaggems_sglang.minimax_m3_sparse_decode_workspace(
        a["q"], 1, 16
    )
    yield (
        a["q"],
        a["k_cache"].view(-1, 1, 128),
        a["v_cache"].view(-1, 1, 128),
        a["topk_idx"],
        a["block_table"],
        a["cu_seqlens_q"],
        a["seq_lens"],
        a["prefix_lens"],
        qlen,
        a["sm_scale"],
        a["output"],
        page_size,
        partial,
        lse,
    )


def _reference(
    q,
    k,
    v,
    topk,
    pages,
    cu,
    seq,
    prefix,
    qlen,
    scale,
    out,
    page_size,
    partial,
    lse,
):
    del partial, lse
    return sparse_paged_reference(
        q,
        k.unflatten(0, (-1, page_size)),
        v.unflatten(0, (-1, page_size)),
        topk,
        pages,
        cu,
        seq,
        prefix,
        qlen,
        scale,
        out,
        page_size=page_size,
    )


@pytest.mark.skipif(
    os.getenv("FLAGGEMS_RUN_SPARSE_GPU_TESTS") != "1",
    reason="GPU regression deferred",
)
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.minimax_m3_sparse_attn
def test_minimax_m3_sparse_attention(decode):
    def candidate(
        q,
        k,
        v,
        topk,
        pages,
        cu,
        seq,
        prefix,
        qlen,
        scale,
        out,
        page_size,
        partial,
        lse,
    ):
        if decode:
            return flaggems_sglang.minimax_m3_sparse_attn_decode(
                q,
                k,
                v,
                topk,
                pages,
                seq,
                scale,
                out,
                qlen,
                page_size=page_size,
                workspace=(partial, lse),
            )
        return flaggems_sglang.minimax_m3_sparse_attn(
            q,
            k,
            v,
            topk,
            pages,
            cu,
            seq,
            prefix,
            qlen,
            scale,
            out,
            page_size=page_size,
        )

    shapes = (
        [
            (b, 1, c, p)
            for b in (1, 4, 64)
            for c in (4096, 16384, 65536)
            for p in (1, 128)
        ]
        if decode
        else [
            (1, q, c, p)
            for q in (1, 128, 512)
            for c in (4096, 16384, 65536)
            for p in (1, 128)
        ]
    )
    bench = OpBenchmark(
        op_name=f"minimax_m3_sparse_{'decode' if decode else 'prefill'}",
        torch_op=_reference,
        input_fn=_inputs,
        dtypes=[torch.bfloat16],
        shapes=shapes,
        shape_desc="batch, query_len, context_len, page_size",
    )
    bench.set_gems(candidate)
    bench.run()
