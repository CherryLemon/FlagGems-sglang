# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real operator test declarations without allocating their full grids."""

import importlib
from pathlib import Path

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference_candidate import (
    is_reference_test,
    reference_test_mode,
)

ROOT = Path(__file__).resolve().parents[2]
OPERATORS = sorted(p.stem[5:] for p in (ROOT / "tests").glob("test_*.py"))
DICT_OPERATORS = [
    op
    for op in OPERATORS
    if op not in {"mrope_fused", "softcap_inplace_logits"}
]


@pytest.mark.parametrize("operator", OPERATORS)
def test_every_operator_exposes_an_explicit_reference(operator):
    module = importlib.import_module(f"tests.test_{operator}")
    assert is_reference_test(getattr(module, f"test_{operator}"))


@pytest.mark.parametrize("operator", DICT_OPERATORS)
@pytest.mark.parametrize("enabled", [False, True])
def test_real_declarations_route_and_compare_independent_bundles(
    operator, enabled, monkeypatch
):
    module = importlib.import_module(f"tests.test_{operator}")
    value = [1]
    calls = []

    def operation(kind, left, right):
        assert left is right
        assert left is not value
        calls.append(kind)
        left.append(2)
        return left

    def reference(**inputs):
        return operation("reference", **inputs)

    def candidate(**inputs):
        return operation("candidate", **inputs)

    def compare(actual, expected):
        calls.append("compare")
        assert actual == expected == [1, 2]
        assert actual is not expected

    def resolve(name):
        assert name == operator
        return reference

    monkeypatch.setattr(module, "get_reference", resolve)
    monkeypatch.setattr(
        module,
        "_CASE_FACTORIES",
        [lambda: dict(left=value, right=value, check=compare)],
    )
    monkeypatch.setattr(flaggems_sglang, operator, candidate, raising=False)
    with reference_test_mode(enabled, operator) as state:
        getattr(module, f"test_{operator}")(0)
    assert calls == [
        "reference",
        "reference" if enabled else "candidate",
        "compare",
    ]
    assert value == [1]
    assert state.candidate_calls == state.completed_cases == 1
    assert state.reference_calls == (2 if enabled else 1)


@pytest.mark.parametrize("enabled", [False, True])
def test_inplace_declaration_keeps_original_input_and_comparison(
    enabled, monkeypatch
):
    module = importlib.import_module("tests.test_softcap_inplace_logits")
    original = torch.tensor([[2.0, -5.0]])
    seen = []

    def reference(full_logits, final_logit_softcapping):
        seen.append("reference")
        return full_logits / final_logit_softcapping

    def candidate(full_logits, final_logit_softcapping):
        seen.append("candidate")
        return full_logits.div_(final_logit_softcapping)

    monkeypatch.setattr(module, "get_reference", lambda name: reference)
    monkeypatch.setattr(
        module,
        "_CASE_FACTORIES",
        [lambda: dict(full_logits=original, final_logit_softcapping=2.0)],
    )
    monkeypatch.setattr(flaggems_sglang, "softcap_inplace_logits", candidate)
    with reference_test_mode(enabled) as state:
        module.test_softcap_inplace_logits(0)
    torch.testing.assert_close(original, torch.tensor([[2.0, -5.0]]))
    assert seen == ["reference", "reference" if enabled else "candidate"]
    assert state.completed_cases == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_mrope_declaration_preserves_both_output_checks(enabled, monkeypatch):
    module = importlib.import_module("tests.test_mrope_fused")
    monkeypatch.setattr(module.cfg, "device", "cpu")
    original_reference = module.mrope_fused_ref
    calls = []

    def reference(*inputs):
        calls.append("reference")
        return original_reference(*inputs)

    def candidate(*inputs):
        calls.append("candidate")
        return original_reference(*inputs)

    monkeypatch.setattr(module, "mrope_fused_ref", reference)
    monkeypatch.setattr(flaggems_sglang, "mrope_fused", candidate)
    with reference_test_mode(enabled) as state:
        module.test_mrope_fused(module.CORRECTNESS_CASES[0])
    assert calls == ["reference", "reference" if enabled else "candidate"]
    assert state.completed_cases == 1

    def incorrect(*inputs):
        q, k = original_reference(*inputs)
        return q, k + 100

    monkeypatch.setattr(flaggems_sglang, "mrope_fused", incorrect)
    with pytest.raises(AssertionError):
        module.test_mrope_fused(module.CORRECTNESS_CASES[0])
