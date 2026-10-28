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

"""Explicit reference contracts through shared helpers and real pytest hooks."""

import importlib.util
import inspect
import json
import math
import sys
import textwrap
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def contract():
    # The public contract module is intentionally usable without a device
    # runtime. CLI integration below loads the normal package and conftests.
    source = ROOT / "src/flaggems_sglang/reference_candidate.py"
    spec = importlib.util.spec_from_file_location(
        "_sglang_reference_contract_test", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _compare(actual, expected):
    assert actual == expected


def _forbidden(*args, **kwargs):
    raise AssertionError("the original candidate was called")


def test_default_and_reference_candidate_selection(contract):
    @contract.reference_test
    def case():
        return contract.ReferenceCase([1, 2], _forbidden, sum, _compare)

    with pytest.raises(AssertionError, match="original candidate"):
        case()
    with contract.reference_test_mode(True, "explicit-node") as state:
        case()
    assert (state.reference_calls, state.candidate_calls) == (2, 1)
    with pytest.raises(AssertionError, match="original candidate"):
        case()


@pytest.mark.parametrize("enabled", [False, True])
def test_independent_bundles_preserve_aliases_and_mutations(contract, enabled):
    original = [1]

    def mutate(bundle):
        assert bundle[0] is bundle[1]
        bundle[0].append(2)
        return bundle

    def compare(actual, expected):
        assert actual == expected == ([1, 2], [1, 2])
        assert actual[0] is not expected[0]

    @contract.reference_test
    def case():
        return contract.ReferenceCase(
            (original, original), mutate, mutate, compare
        )

    with contract.reference_test_mode(enabled):
        case()
    assert original == [1]


def test_explicit_reference_inputs_and_custom_clone(contract):
    snapshots = []

    def clone(bundle):
        snapshots.append(tuple(bundle))
        return list(bundle)

    @contract.reference_test
    def case(value):
        return contract.ReferenceCase(
            [value],
            _forbidden,
            sum,
            _compare,
            reference_inputs=[float(value)],
            clone=clone,
        )

    assert list(inspect.signature(case).parameters) == ["value"]
    with contract.reference_test_mode(True):
        case(2)
    assert snapshots == [(2.0,), (2,)]


def test_nested_mode_is_restored_after_an_exception(contract):
    @contract.reference_test
    def case():
        return contract.ReferenceCase([1], _forbidden, sum, _compare)

    with contract.reference_test_mode(True):
        with contract.reference_test_mode(False):
            with pytest.raises(AssertionError, match="original candidate"):
                case()
        case()
    with pytest.raises(AssertionError, match="original candidate"):
        case()


@pytest.mark.parametrize(
    "field", ["candidate", "reference", "compare", "clone"]
)
def test_missing_callable_is_an_explicit_contract_error(contract, field):
    values = dict(inputs=[1], candidate=sum, reference=sum, compare=_compare)
    values[field] = None
    with pytest.raises(contract.ReferenceContractError, match=field):
        contract.ReferenceCase(**values)


@pytest.mark.parametrize("value", [None, {}, sum])
def test_factory_cannot_hide_a_missing_reference_contract(contract, value):
    @contract.reference_test
    def case():
        return value

    with pytest.raises(contract.ReferenceContractError, match="ReferenceCase"):
        case()


def test_reference_mode_still_executes_the_comparator(contract):
    def compare(actual, expected):
        raise AssertionError("the original comparison executed")

    @contract.reference_test
    def case():
        return contract.ReferenceCase([1], _forbidden, sum, compare)

    with contract.reference_test_mode(True):
        with pytest.raises(
            AssertionError, match="original comparison executed"
        ):
            case()


@pytest.fixture
def reference_project(pytester):
    pytester.makeini(
        f"[pytest]\npythonpath = . {ROOT / 'src'}\n"
        "markers = reference_probe: explicit reference contract checks\n"
    )
    for name in ("tests", "benchmark"):
        directory = pytester.path / name
        directory.mkdir()
        for path in (ROOT / name).iterdir():
            if not path.is_file() or path.name.startswith("test_"):
                continue
            if name == "tests" and path.name not in {
                "__init__.py",
                "conftest.py",
            }:
                continue
            (directory / path.name).symlink_to(path)
    return ROOT


def _accuracy(pytester, source=None):
    path = "tests/test_reference_probe.py"
    if source is None:
        source = """
            import pytest
            import flaggems_sglang
            from flaggems_sglang.reference_candidate import (
                ReferenceCase, reference_test,
            )

            def compare(actual, expected):
                assert actual == expected

            @pytest.mark.reference_probe
            @pytest.mark.parametrize('value', [1, 2])
            @reference_test
            def test_candidate(value):
                return ReferenceCase(
                    (value, 2),
                    lambda args: flaggems_sglang.reference_probe(*args),
                    lambda args: sum(args),
                    compare,
                )
        """
    (pytester.path / path).write_text(textwrap.dedent(source))
    return path


def _benchmark(
    pytester, *, reference="reference", extra="", outer=False, backward=False
):
    path = "benchmark/test_reference_probe.py"
    parametrization = (
        "@pytest.mark.parametrize('offset', [0, 10], ids=['first', 'second'])\n"
        if outer
        else ""
    )
    argument = "offset" if outer else ""
    offset = "offset" if outer else "0"
    source = f"""
        import json
        from pathlib import Path
        import pytest
        import torch
        import flaggems_sglang
        from benchmark.cases import BenchmarkCasePlan
        from benchmark.op_benchmark import OpBenchmark

        def reference(value):
            with Path('reference-calls.jsonl').open('a') as output:
                output.write(json.dumps(value) + '\\n')
            return value

        def forbidden(*args, **kwargs):
            raise AssertionError('the bound candidate was called')

        @pytest.mark.reference_probe
        {parametrization.rstrip()}
        def test_candidate({argument}):
            {extra or 'pass'}
            bench = OpBenchmark(
                op_name='reference_probe', torch_op={reference},
                shapes=[(1,), (2,)], shape_desc='n',
                dtypes=[torch.float32],
                is_backward={backward!r},
                case_fn=lambda shape, dtype: [BenchmarkCasePlan(
                    shape={{'value': shape}}, params={{'offset': {offset}}},
                    builder_args=(shape[0] + {offset},),
                )],
                build_inputs_fn=lambda plan, dtype, device: (
                    plan.builder_args[0],
                ),
            )
            bench.set_gems_from_module(flaggems_sglang)
            bench.run()
    """
    (pytester.path / path).write_text(textwrap.dedent(source))
    return path


def _run(pytester, *paths, enabled=True, options=(), report="result.json"):
    destination = pytester.path / report
    command = [*paths, "-q", "--record=json", f"--output={destination}"]
    if enabled:
        command.append("--ref-as-candidate")
    command.extend(options)
    result = pytester.runpytest_subprocess(*command, timeout=90)
    data = (
        json.loads(destination.read_text()) if destination.exists() else None
    )
    return result, data


def _rows(data):
    return [
        row
        for detail in data["reference_probe"]["details"]
        for row in detail["result"]
    ]


def _bench_options(*extra):
    return (
        "--level=core",
        "--mode=operator",
        "--warmup=0",
        "--iter=1",
        *extra,
    )


def test_real_accuracy_hooks_record_each_reference_candidate(
    pytester, reference_project
):
    path = _accuracy(pytester)
    result, data = _run(pytester, path, options=("-m", "reference_probe"))
    result.assert_outcomes(passed=2)
    assert result.ret == 0
    assert len(data) == 2
    assert all(row["candidate_source"] == "reference" for row in data.values())
    assert all(row["reference_calls"] == 2 for row in data.values())
    assert all(row["reference_candidate_calls"] == 1 for row in data.values())


def test_normal_execution_and_external_overrides_remain_available(
    pytester, reference_project
):
    path = _accuracy(pytester)
    result, _ = _run(pytester, path, enabled=False, report="ordinary.json")
    result.assert_outcomes(failed=2)
    candidate = pytester.makepyfile(candidate="def run(a, b): return a + b\n")
    result, data = _run(
        pytester,
        path,
        enabled=False,
        options=("--override", f"reference_probe:{candidate}:run"),
        report="override.json",
    )
    result.assert_outcomes(passed=2)
    assert result.ret == 0
    assert all(
        row.get("candidate_source") != "reference" for row in data.values()
    )
    config = pytester.path / "override-config.json"
    config.write_text(
        json.dumps(
            {
                "overrides": {
                    "reference_probe": {
                        "file": str(candidate),
                        "function": "run",
                    }
                }
            }
        )
    )
    result, _ = _run(
        pytester,
        path,
        enabled=False,
        options=(f"--override-config={config}",),
        report="configured.json",
    )
    result.assert_outcomes(passed=2)
    assert result.ret == 0


def test_missing_declaration_fails_only_when_enabled_and_selected(
    pytester, reference_project
):
    path = _accuracy(pytester)
    legacy = "tests/test_legacy_reference.py"
    (pytester.path / legacy).write_text("def test_legacy(): pass\n")
    result, _ = _run(pytester, legacy, enabled=False, report="ordinary.json")
    result.assert_outcomes(passed=1)
    result, _ = _run(pytester, path, legacy, report="missing.json")
    assert result.ret != 0
    output = result.stdout.str() + "\n" + result.stderr.str()
    assert "test_legacy_reference.py::test_legacy" in output
    assert "predefined reference" in output.lower()
    result, _ = _run(
        pytester,
        path,
        legacy,
        options=("-m", "reference_probe"),
        report="selected.json",
    )
    result.assert_outcomes(passed=2, deselected=1)
    assert result.ret == 0


@pytest.mark.parametrize(
    "kind", ["bad_factory", "empty_execution", "counter_only"]
)
def test_a_declaration_must_execute_an_explicit_reference_contract(
    pytester, reference_project, kind
):
    if kind == "bad_factory":
        source = """
            from flaggems_sglang.reference_candidate import reference_test
            @reference_test
            def test_candidate():
                reference = lambda args: args
        """
    elif kind == "empty_execution":
        source = """
            from flaggems_sglang.reference_candidate import (
                ReferenceCase, reference_test,
            )
            @reference_test
            def declared():
                return ReferenceCase([1], sum, sum, lambda a,b: None)
            def test_candidate():
                pass
            test_candidate.__dict__.update(declared.__dict__)
        """
    else:
        source = """
            from flaggems_sglang.reference_candidate import (
                ReferenceCase, reference_test, select_candidate,
            )
            @reference_test
            def declared():
                return ReferenceCase([1], sum, sum, lambda a,b: None)
            def test_candidate():
                operation = select_candidate(sum, sum, label='counter-only-node')
                operation([1])
            test_candidate.__dict__.update(declared.__dict__)
        """
    path = _accuracy(pytester, source)
    result, _ = _run(pytester, path)
    assert result.ret != 0
    assert (
        "ReferenceCase" in result.stdout.str()
        or "candidate call" in result.stdout.str()
        or "reference contract" in result.stdout.str().lower()
    )


@pytest.mark.parametrize("conversion", ["skip", "xfail", "fixture_skip"])
def test_contract_errors_cannot_be_hidden_by_skip_or_xfail(
    pytester, reference_project, conversion
):
    source = """
        import pytest
        from flaggems_sglang.reference_candidate import (
            ReferenceCase, ReferenceContractError, reference_test,
        )

        def compare(actual, expected):
            assert actual == expected

        def good_case():
            return ReferenceCase([1], sum, sum, compare)

        @reference_test
        def test_good():
            return good_case()
    """
    if conversion == "fixture_skip":
        source += """
        @pytest.fixture
        def bad_case():
            try:
                return ReferenceCase([1], sum, None, compare)
            except ReferenceContractError:
                pytest.skip('fixture hid its missing reference')

        @reference_test
        def test_bad(bad_case):
            return bad_case
        """
    else:
        source += f"""
        @reference_test
        def completed():
            return good_case()

        def test_bad():
            completed()
            try:
                ReferenceCase([1], sum, None, compare)
            except ReferenceContractError:
                pytest.{conversion}('test hid its missing reference')
        test_bad.__dict__.update(completed.__dict__)
        """
    path = _accuracy(pytester, source)
    result, data = _run(pytester, path)
    assert result.ret != 0
    assert data[path + "::test_good"]["result"] == "passed"
    assert data[path + "::test_bad"]["result"] != "passed"
    output = result.stdout.str().lower()
    assert "referencecontracterror" in output or "reference contract" in output


@pytest.mark.parametrize("selection", ["empty", "all_skip"])
def test_empty_or_all_skipped_reference_validation_cannot_pass(
    pytester, reference_project, selection
):
    source = "value = 1\n"
    if selection == "all_skip":
        source = """
            import pytest
            from flaggems_sglang.reference_candidate import reference_test
            @reference_test
            def test_candidate():
                pytest.skip('capability absent')
        """
    path = _accuracy(pytester, source)
    result, _ = _run(pytester, path)
    assert result.ret != 0


def test_benchmark_times_the_explicit_reference_for_every_case(
    pytester, reference_project
):
    path = _benchmark(pytester)
    result, data = _run(pytester, path, options=_bench_options())
    result.assert_outcomes(passed=1)
    assert result.ret == 0
    rows = _rows(data)
    assert len(rows) == 2
    assert len({row["case_id"] for row in rows}) == 2
    assert all(row["candidate_source"] == "reference" for row in rows)
    assert all(
        isinstance(row[name], (int, float))
        and math.isfinite(row[name])
        and row[name] > 0
        for row in rows
        for name in ("latency", "latency_base")
    )
    assert (
        pytester.path / "reference-calls.jsonl"
    ).read_text().splitlines() == [
        "1",
        "1",
        "2",
        "2",
    ]


def test_exact_replay_preserves_outer_parameters(pytester, reference_project):
    path = _benchmark(pytester, outer=True)
    selected = path + "::test_candidate[second]::core::float32::1"
    result, data = _run(
        pytester,
        path,
        options=_bench_options(f"--case-id={selected}"),
    )
    assert result.ret == 0
    assert [row["case_id"] for row in _rows(data)] == [selected]
    assert (
        pytester.path / "reference-calls.jsonl"
    ).read_text().splitlines() == [
        "12",
        "12",
    ]


def test_bare_reference_selection_is_not_evidence_of_benchmark_timing(
    pytester, reference_project
):
    path = "benchmark/test_bare_reference.py"
    (pytester.path / path).write_text(
        textwrap.dedent(
            """
            import pytest
            from flaggems_sglang.reference_candidate import select_candidate

            @pytest.mark.reference_probe
            def test_candidate():
                operation = select_candidate(sum, sum)
                assert operation([1, 2]) == 3
            """
        )
    )
    result, data = _run(pytester, path, options=_bench_options())
    assert result.ret != 0
    assert "timing" in result.stdout.str().lower()
    assert not data.get("reference_probe", {}).get("details")


@pytest.mark.parametrize(
    "problem", ["missing", "metric", "zero_timer", "backward"]
)
def test_benchmark_cannot_claim_reference_timing_without_real_invocation(
    pytester, reference_project, problem
):
    extra = ""
    options = _bench_options()
    if problem == "zero_timer":
        extra = (
            "import benchmark.performance_utils as base; "
            "base.do_bench_latency = lambda *args, **kwargs: 1.0"
        )
        options = ("--level=core", "--mode=kernel", "--warmup=0", "--iter=1")
    elif problem == "metric":
        options = _bench_options("--metrics=latency_base")
    path = _benchmark(
        pytester,
        reference="None" if problem == "missing" else "reference",
        extra=extra,
        backward=problem == "backward",
    )
    result, _ = _run(pytester, path, options=options)
    assert result.ret != 0
    if problem == "missing":
        assert "predefined reference" in result.stdout.str().lower()
    elif problem == "metric":
        assert "candidate latency metric" in result.stdout.str()
    elif problem == "backward":
        output = result.stdout.str().lower()
        assert "backward" in output
        assert "unsupported" in output or "not support" in output
    else:
        assert "candidate" in result.stdout.str().lower()
    if problem != "zero_timer":
        assert not (pytester.path / "reference-calls.jsonl").exists()


@pytest.mark.parametrize(
    "method", ["run", "get_latency", "_measure_input", "_candidate_call"]
)
def test_custom_benchmark_execution_cannot_bypass_reference_contract(
    monkeypatch, method
):
    from benchmark import conftest, performance_utils
    from benchmark.op_benchmark import OpBenchmark
    from flaggems_sglang import reference_candidate as reference

    monkeypatch.setattr(performance_utils, "Config", conftest.BenchConfig())
    custom = type("Custom", (OpBenchmark,), {method: _forbidden})
    with reference.reference_test_mode(True, "custom-node"):
        with pytest.raises(
            (ValueError, reference.ReferenceContractError), match="custom"
        ):
            custom(
                op_name="reference_probe",
                torch_op=sum,
                shapes=[(1,)],
                shape_desc="n",
                input_fn=lambda shape, dtype, device: [(1,)],
            )


@pytest.mark.parametrize(
    "conflict",
    [
        ("--dummy-candidate=reference_probe",),
        ("--override=reference_probe:missing.py:run",),
        ("--override-config=missing.json",),
        ("--reference-only",),
        ("--preflight-only",),
        ("--list-cases",),
        ("--query",),
        ("--parallel=2",),
        ("--collect-only",),
    ],
)
def test_reference_mode_rejects_conflicting_execution_modes(
    pytester, reference_project, conflict
):
    path = _benchmark(pytester)
    result, _ = _run(pytester, path, options=conflict)
    assert result.ret != 0
    assert not (pytester.path / "reference-calls.jsonl").exists()


def test_mixed_session_uses_the_same_mode_for_both_suites(
    pytester, reference_project
):
    accuracy = _accuracy(pytester)
    benchmark = _benchmark(pytester)
    result, data = _run(
        pytester, accuracy, benchmark, options=_bench_options()
    )
    result.assert_outcomes(passed=3)
    assert result.ret == 0
    accuracy_rows = [
        row for key, row in data.items() if key.startswith("tests/")
    ]
    assert len(accuracy_rows) == 2
    assert all(row["reference_candidate_calls"] == 1 for row in accuracy_rows)
    assert all(row["candidate_source"] == "reference" for row in _rows(data))


def test_repeated_pytest_sessions_restore_default_candidate_selection(
    pytester, reference_project
):
    path = _accuracy(pytester)
    script = pytester.makepyfile(
        repeat=f"""
        import sys
        sys.path.insert(0, {str(reference_project / 'src')!r})
        import pytest
        assert pytest.main([{path!r}, '--ref-as-candidate', '-q']) == 0
        assert pytest.main([{path!r}, '-q']) != 0
        """
    )
    result = pytester.run(sys.executable, str(script), timeout=180)
    assert result.ret == 0
