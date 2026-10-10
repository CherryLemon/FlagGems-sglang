# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU adapter tests; GPU checks are opt-in during development."""

import ast
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def adapter():
    source = (
        Path(__file__).parents[1]
        / "src/flaggems_sglang/ops/minimax_m3_sparse_attention.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sglang_sparse_adapter_test", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dispatch(monkeypatch):
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return args[10] if len(args) >= 11 else args[7]

    monkeypatch.setitem(
        sys.modules,
        "flag_attn",
        SimpleNamespace(
            minimax_m3_sparse_attn_paged=record,
            minimax_m3_sparse_attn_decode_paged=record,
            minimax_m3_sparse_decode_workspace=record,
        ),
    )
    return calls


@pytest.mark.parametrize("page_size", [1, 16, 128])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_decode_uses_no_copy_token_slot_views(
    adapter, dispatch, page_size, noncontiguous
):
    q = torch.empty(4, 8, 128)
    k = torch.empty(512, 1, 256 if noncontiguous else 128)
    v = torch.empty_like(k)
    if noncontiguous:
        k, v = k[..., ::2], v[..., ::2]
    out = torch.empty_like(q)
    scratch = (object(), object())
    result = adapter.minimax_m3_sparse_attn_decode(
        q,
        k,
        v,
        object(),
        object(),
        object(),
        output=out,
        decode_query_len=2,
        page_size=page_size,
        workspace=scratch,
    )
    args, kwargs = dispatch[0]
    assert result is out
    assert args[1].shape == (512 // page_size, page_size, 1, 128)
    assert (
        args[1].data_ptr() == k.data_ptr()
        and args[2].data_ptr() == v.data_ptr()
    )
    assert args[1].stride(-1) == k.stride(-1)
    assert args[7] is out and args[8] == 2
    assert args[6] == 128**-0.5
    assert kwargs["workspace"] is scratch and kwargs["page_size"] == page_size


def test_prefill_forwards_original_metadata(adapter, dispatch):
    q = torch.empty(4, 8, 128)
    k, v = torch.empty(512, 1, 128), torch.empty(512, 1, 128)
    topk, pages, cu, seq, prefix = [object() for _ in range(5)]
    out = torch.empty_like(q)
    assert (
        adapter.minimax_m3_sparse_attn(
            q, k, v, topk, pages, cu, seq, prefix, 2, output=out, k_scale=0.5
        )
        is out
    )
    args, kwargs = dispatch[0]
    assert args[3:8] == (topk, pages, cu, seq, prefix)
    assert args[8] == 2 and args[10] is out
    assert kwargs["k_scale"] == 0.5


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"page_size": 0}, ValueError),
        ({"page_size": 3}, ValueError),
        ({"page_size": True}, ValueError),
        ({"block_size_k": 64}, NotImplementedError),
        ({"sink": object()}, NotImplementedError),
    ],
)
def test_unsupported_contract_rejected(adapter, dispatch, kwargs, error):
    with pytest.raises(error):
        adapter.minimax_m3_sparse_attn_decode(
            torch.empty(2, 8, 128),
            torch.empty(512, 1, 128),
            torch.empty(512, 1, 128),
            None,
            None,
            None,
            **kwargs,
        )
    assert not dispatch


def test_cache_shape_mismatch_rejected(adapter, dispatch):
    with pytest.raises(ValueError, match="matching"):
        adapter.minimax_m3_sparse_attn_decode(
            torch.empty(2, 8, 128),
            torch.empty(512, 1, 128),
            torch.empty(256, 1, 128),
            None,
            None,
            None,
        )
    assert not dispatch


def test_actual_registrar_discovers_public_functions(adapter):
    source = (
        Path(__file__).parents[1]
        / "src/flaggems_sglang/runtime/op_registrar.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "OpRegistrar"
    )
    # Execute the existing class without importing the device-detection module.
    scope = {
        "inspect": inspect,
        "Optional": __import__("typing").Optional,
        "Dict": dict,
        "List": list,
        "Callable": __import__("typing").Callable,
    }
    exec(
        compile(
            ast.Module(body=[cls], type_ignores=[]), "registrar_class", "exec"
        ),
        scope,
    )
    registrar = object.__new__(scope["OpRegistrar"])
    registrar._resolved = {}
    registrar._merge_from_module(adapter)
    assert set(registrar.names()) == set(adapter.__all__)
    assert all(
        registrar.get(name) is getattr(adapter, name)
        for name in adapter.__all__
    )


@pytest.mark.skipif(
    os.getenv("FLAGGEMS_RUN_SPARSE_GPU_TESTS") != "1",
    reason="GPU regression explicitly deferred",
)
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("page_size", [1, 16, 128])
def test_gpu_against_torch_reference(decode, page_size):
    import flaggems_sglang
    from flag_attn.testing.minimax_sparse_paged import (
        make_sparse_paged_inputs,
        sparse_paged_reference,
    )

    a = make_sparse_paged_inputs(2, 3, 257, page_size, torch.bfloat16)
    reference = sparse_paged_reference(**{**a, "output": None})
    k = a["k_cache"].view(-1, 1, 128)
    v = a["v_cache"].view(-1, 1, 128)
    if decode:
        result = flaggems_sglang.minimax_m3_sparse_attn_decode(
            a["q"],
            k,
            v,
            a["topk_idx"],
            a["block_table"],
            a["seq_lens"],
            output=a["output"],
            decode_query_len=3,
            page_size=page_size,
        )
    else:
        result = flaggems_sglang.minimax_m3_sparse_attn(
            a["q"],
            k,
            v,
            a["topk_idx"],
            a["block_table"],
            a["cu_seqlens_q"],
            a["seq_lens"],
            a["prefix_lens"],
            3,
            output=a["output"],
            page_size=page_size,
        )
    assert result is a["output"]
    torch.testing.assert_close(result, reference, rtol=0.02, atol=0.02)
