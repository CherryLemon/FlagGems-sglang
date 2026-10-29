# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Reference as candidate

`--ref-as-candidate` checks that a selected correctness test or benchmark can
execute with its explicitly declared reference on the candidate side. It is off
by default. Existing operator tests, external overrides and dummy checks retain
their usual commands and behavior. This design follows FlagGems PR #7630; that
unmerged PR is not a runtime dependency.

```sh
pytest -q tests/test_silu_and_mul.py -m silu_and_mul --ref-as-candidate \
  --record json --output silu-reference-correctness.json
pytest -q benchmark/test_silu_and_mul.py -m silu_and_mul --level core \
  --ref-as-candidate --record json --output silu-reference-timing.json
```

## Declare a correctness reference

All 125 operator correctness tests declare their references through this
contract, including the in-place softcap and tuple-output mrope tests. Existing
parameter grids, reference dispatch, comparison tolerances and candidate skips
are retained. To add another test, decorate a parametrized factory:

```python
from flaggems_sglang.reference_candidate import ReferenceCase, reference_test


@reference_test
def test_example():
    return ReferenceCase(
        inputs={"value": [1, 2]},
        candidate=lambda inputs: candidate(**inputs),
        reference=lambda inputs: explicitly_defined_reference(**inputs),
        compare=lambda actual, expected: check(actual, expected),
    )
```

Each operation receives one input bundle. Keep candidate lookup inside the
callable so external overrides still work and reference mode does not need a
public candidate export. Merely naming a local variable `reference` does not
expose the contract. Selecting an unconverted test reports `No predefined
reference` and its pytest node ID. Missing/non-callable references, invalid
factory returns and missing candidate-side execution are errors.

The runner clones both bundles before either operation. Optional
`reference_inputs` supplies CPU/upcast inputs for the expected side; the default
is `inputs`. The default clone is `copy.deepcopy`. Supply `clone` explicitly when
view/storage/gradient relationships need specialized handling. It must preserve
within-bundle aliases while making the two executions independent. Return all
observable mutated buffers and outputs, and retain shape, device, dtype,
mutation and alias checks in `compare`. The framework does not guess an ABI,
move outputs between devices or adjust dtypes.

Without the flag the wrapper runs the real candidate and original comparator.
With the flag it runs reference on both independently cloned input bundles;
correctness JSON identifies `candidate_source: reference`, `reference_calls` and
`reference_candidate_calls` per node.

## Benchmark contract

Standard benchmarks explicitly declare their reference as callable `torch_op`.
This option also uses that callable on the candidate side through the same shared
selection and invocation tracking as correctness. It retains the original input
builders, case IDs, selected level, timer, warmup and metrics. `--case-id` replays
one exact case. A real candidate-side call is required for every timing record;
only then is `candidate_source: reference` recorded.

This initial mode supports forward benchmarks only. `is_backward=True` is
rejected explicitly: its preparation forward pass cannot prove that the timed
backward callable ran. Ordinary backward benchmarks are unchanged without the
flag.

A missing/non-callable reference, a custom execution/timing override or metrics
without candidate latency is an explicit error in this mode. The ordinary
behavior of those benchmarks remains unchanged. `OpBenchmark` skips eager public
candidate lookup in reference mode, so an unavailable candidate export is not
needed. No same-named Torch function or implicit reference is invented.

External override/config, dummy, reference-only, preflight, profiling, listing,
query, collection-only, marker collection and parallel execution cannot be
combined with this mode. Catalog listing is a separate command. Empty selection
and an all-skip run do not count as successful validation.

Reference-vs-reference proves harness executability, not that the reference is
independently correct or that an optimized candidate works. Continue to use
dummy injection to validate the external override route. These timings are
functional diagnostics, not published H800 performance baselines.
