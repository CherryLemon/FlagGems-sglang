# Hy4-preview active-head sparse MLA candidate

`flaggems_sglang.sparse_mla_active_split` is an explicit numerical operator.
Registering it does not select it for a serving model. The integration must opt
in before capture and retain its native fallback for unsupported metadata.

The current support domain is inference on H100/SM90, BF16 query
`[SQ, 64, 576]` with `SQ` equal to 1 or 4, four logical active heads, BF16 KV
`[physical_slots, 1, 576]`, INT32 physical indices `[SQ, 1, 2048]`, FP32 sink
`[64]`, and INT32 selected-position lengths `[SQ]`. All inputs must have the
documented contiguous strides, zero storage offset, matching CUDA device, and
no gradient, conjugate, or negative view state. Missing sink/lengths, other
devices/shapes/dtypes, compilation, and gradient mode raise `NotImplementedError`.
The caller may inspect the module's `metadata` helper before choosing this op.

Indices are physical KV slots. Invalid indices and positions beyond each row's
length contribute nothing; duplicate selected slots retain their multiplicity.
The operator partitions the 2048 selected positions into 32 groups of 64, runs
two Triton kernels, and adds the sink to the denominator once in the merge.
It returns `(output, maximum, lse)` with shapes `[SQ, 4, 512]`, `[SQ, 4]`,
and `[SQ, 4]`. `maximum` and `lse` describe KV-only logits. Empty rows return
zero output, `maximum=-inf`, and `lse=+inf`; a positive infinite sink gives
zero output. Q/K/V inputs in the qualified tests are finite.

No input is mutated. Optional FP32 partial buffers must have separate storage
from one another and from every input. Callers that capture a CUDA graph must
keep the original input, partial-buffer, and output owners alive through its
final replay. Numerical launch errors propagate to the caller.

The Triton numerical function bodies are migrated without an AST change from
the independently tested local `flag_gems` candidate. Package migration and
the narrower support guard require fresh tests before qualification:

```shell
pytest -q tests/test_sparse_mla_active_split.py
pytest -q benchmark/test_sparse_mla_active_split.py --level core --iter 1 --warmup 1
```

Correctness tests use independent CPU FP64 mathematics, the existing output
gate `atol=5e-4, rtol=2**-7`, ten repeated calls, unchanged-input checks, and
A/B/A mutable-input capture/replay at both supported query sizes. Component
benchmark results do not establish full-model numerical or performance
qualification. The current real 78-layer TP16/EP16 Hy4-preview serving trial
failed strict batch/teacher-forced score comparison; no full-model speedup is
claimed by this operator draft.
