<!--
Copyright 2026 FlagOS Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# List and replay benchmark cases

The existing `flaggems.benchmark-case-list/v2` API separates a lightweight
case plan from Tensor construction. `silu_and_mul` is the first operator in
this repository connected to this API. Other benchmarks using only the
legacy `input_fn` still run normally, but do not support case listing or
exact selection until they provide a planner and builder.

## List cases

```shell
pytest -q benchmark/test_silu_and_mul.py --level core --list-cases --output /tmp/silu-core.json
pytest -q benchmark/test_silu_and_mul.py --list-cases --output /tmp/silu-comprehensive.json
```

Core contains the original five `SHAPES`; the default comprehensive level
also contains the original ten `MORE_SHAPES`. Both retain the original
`torch.bfloat16` dtype. Listing calls the case planner but not the Tensor
builder, reference, candidate, or timer. It still imports the normal package
and therefore requires its regular Torch/vendor runtime environment.

Listing does not require a candidate export on the current backend. The
entrypoints defer public candidate lookup via `set_gems_from_module` in
listing mode, and existing missing-implementation skip guards bypass only
that mode. Ordinary execution retains its original binding, errors and skips.
This does not migrate the remaining legacy input providers to the Case API.

The JSON contains the actual input shape, dtype, dimension parameters and
seed. For example, the third core case describes `hidden_states` with shape
`[64, 8192]`, `bs=64`, `d=4096`, and `seed=0`.

## Replay a selected case

Use the exact ID from the catalog and the same level and source revision:

```shell
pytest -q benchmark/test_silu_and_mul.py --level core --case-id 'benchmark/test_silu_and_mul.py::test_perf_silu_and_mul::core::bfloat16::2' --record json --output /tmp/silu-selected.json
```

The builder materializes only the selected case, and the ordinary benchmark
executes its actual reference/candidate timers. Unknown or unexecuted IDs
fail the session. IDs include the pytest node, level, dtype and ordinal;
outer pytest parametrization therefore gets separate node-scoped IDs.
These IDs are not content hashes. Preserve the source revision, invocation,
and actual outer parameter values with a catalog before using it as an
acceptance artifact; changing source or case order can change its meaning.

Listing rejects empty providers and duplicate IDs within a provider or across
the session's published benchmark lists. A listing session that publishes no
cases also fails. Consumers must check the pytest exit status as well as the
catalog; a failed run may retain partial output for diagnosis.

## Connect another operator

Use the existing `GenericBenchmark`/`OpBenchmark` arguments:

- `case_fn(shape, dtype)` yields `BenchmarkCasePlan` objects containing the
  actual Tensor shapes, non-Tensor configuration in `params`, and private
  `builder_args`. It must not allocate inputs.
- `build_inputs_fn(plan, dtype, device)` constructs exactly one input tuple
  using the original input-generation logic. `builder_args` are private and
  are not serialized into the public catalog.
- Keep the original shapes, their order, dtypes, seeds and reference/candidate
  bindings. Do not mix the planner/builder pair with `input_fn`.

The `silu_and_mul` migration preserves the original seeded float32 generation
followed by conversion to bfloat16. Default benchmark invocation still runs
all the same core or comprehensive workloads; this change adds case IDs to
their timing records without changing the timed operator.
