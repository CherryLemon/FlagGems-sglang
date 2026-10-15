# DeepSeek V4.1 SM90 inference operators

This opt-in operator port carries the validated H100 decode schedules into
FlagGems-sglang. The numerical kernels are derived from SGLang's V4.1 serving
implementation (base `77f7fdc0fcdadcf5ce0e13f239137bf9b165151d`) plus the
subsequent H100 optimization work. It supersedes the earlier, broader port
proposed in [FlagGems #6684](https://github.com/flagos-ai/FlagGems/pull/6684)
for these three decode operator families. The earlier configurable Hopper
GEMM work is recorded in [FlagGems #6631](https://github.com/flagos-ai/FlagGems/pull/6631).
Source attribution is retained. This is a backport with modifications, not
a claim that the tuned kernels are unmodified official SGLang code.

## Numerical APIs

The APIs are registered in the NVIDIA Hopper tier. They are forward inference
operators, validated on H100/SM90. They do not import SGLang and contain no
Torch compute fallback. Torch allocations and documented dtype conversions
remain part of their original wrappers.

| Family | Public API | Preserved contract and changes |
|---|---|---|
| FP8 dense GEMM | `dsv41_block_fp8_matmul(A, B, As, Bs, output_dtype=BF16, tuned=True)` | Contiguous E4M3FN `A[M,K]`, `B[N,K]`; numeric FP32 `As[M,K/32]` and `Bs[ceil(N/32),K/32]`, including strided scales; K32 scaling and FP32 Split-K partials/reduction retained. Exact `(M,N,K)` schedules select row/column tiles, swap order, stages and static metadata; unlisted shapes retain the frozen configured native path. BF16/FP32 outputs; empty M and tail N supported. Encoded UE8M0 scales and FP16 output are rejected by this API. |
| Packed FP4 index scoring | `fp4_index_logits_decode`, `fp4_index_logits_req_to_token`, `fp4_index_logits_candidate_blocks` | BF16 queries `[B,32,128]`, packed E2M1 payload and finite UE8M0 per-32 scales in the native paged byte layout. Request-map addressing, ratio 1/2, BF16 score materializations, candidate maxima/lens and masked `-inf` outputs retained. The semantic `query_group_size=6` hint enables six-query K reuse and finite persistent schedules; it is never inferred from row count. Device request IDs and lengths are read on every replay. |
| Predecessor-pre mHC | `hc_mix_stats`, `hc_combine`, `hc_split_sinkhorn`, `hc_mix_stats_sinkhorn_exact` | Preserve FP32 mixing weights and deterministic K slicing (K20480: 80 slices, K64, TF32x3). Exact fusion combines the final statistics reduction and original TileLang Sinkhorn, with explicit rounding boundaries and the predecessor BF16 combine. Fusion supports HC4 and M36/40/42/48/50/60/80/84/96; other serving rows retain separate operators. |

FP4 callers own valid physical slot mappings, page storage and capacities.
Only the explicitly supplied six-query semantic grouping is specialized;
the kernel still validates per-replay group membership. No routing, expert
placement, communication, model weight or KV allocator implementation is
part of this operator port.

The exact mHC path requires **TileLang 0.1.12**. Its import/compilation is lazy:
installing or importing FlagGems-sglang does not require TileLang unless that
entry is invoked. The approximate Triton Sinkhorn alternative and unrelated
FP8 prefill conversion helpers are outside this port.

## Optional SGLang adapter

The `flaggems_dsv41` general-plugin entry point is disabled by default.
Enable it only on the reviewed source revision:

```bash
FLAGGEMS_SGLANG_DSV41=1 \
SGLANG_PLUGINS=flaggems_dsv41 \
DNN_VENDOR=nvidia ARCH=sm_90 \
python -m sglang.launch_server ...
```

The lightweight entry-point module imports no Torch or CUDA at plugin
discovery time. In each scheduler process it first runs native process
configuration, selects that scheduler's explicit GPU, then imports and
installs the operators before model construction and Graph capture.

The adapter checks three complete native operator-file SHA256 identities
before replacing anything. Unknown SGLang versions fail explicitly; source
compatibility is not inferred from a version string. It rebinds already
imported SGLang aliases by original function identity. FP8 redirects only
reviewed finite numeric-FP32-scale shapes; packed scales, FP16 output,
Torch compilation and unlisted shapes retain the native entry. Uninstall
restores only bindings still owned by this adapter.

The standalone APIs can be used independently of this narrow serving
adapter. Extending the adapter to another SGLang source revision requires
reviewing native argument/numerical contracts and updating the source pins.

## Validation and benchmarks

Run from this repository with its dependencies and `PYTHONPATH=src`:

```bash
python -m pytest -q tests/test_dsv41_block_fp8_matmul.py \
  tests/test_dsv41_fp4_indexer.py tests/test_dsv41_mhc.py \
  tests/test_dsv41_plugin.py

# Standard repository benchmark entry points (eager timing).
python -m pytest -q -s benchmark/test_dsv41_block_fp8_matmul.py \
  benchmark/test_dsv41_fp4_indexer.py benchmark/test_dsv41_mhc.py \
  --mode kernel --level core --warmup 25 --iter 100 --record json \
  --output dsv41-eager.json

# Two-round full CUDA Graph comparison, preserving every measured shape.
python benchmark/dsv41_h100_graph.py --full --rounds 2 --rep-ms 100 \
  --out dsv41-graph.json
```

Tests compare FP8 against an independent FP32 Torch block-scale reference
and retain **byte-exact native regression gates**, including signed-zero and
NaN payload bytes. The byte-exact FP8 regression matrix uses the model's
UE8M0-derived power-of-two scales represented as numeric FP32 tensors; it
does not promise byte equality for arbitrary numeric scale distributions.
FP8 tensor-core native
accumulation already differs from FP32 Torch; the independent semantic
comparison uses rtol=.01/atol=.01, tighter than the earlier port's
.02/.25 reference check. This semantic tolerance does not replace the
bitwise native-vs-tuned comparison. Changed-input Graph tests refresh
activations, weights, scales and device request/length metadata. FP4 page,
ratio, odd-row, candidate-consumer and empty-width boundaries are covered.
mHC validates the predecessor combine independently and compares exact fused
Sinkhorn outputs against the original separated chain, including changing
Graph inputs and repeated execution.

The Graph report uses 25 eager warmups, median
`triton.testing.do_bench_cudagraph` latency with 100 ms repetition budgets,
alternating implementation order and reversed shape order in round 2.
Two-round mean latencies form each shape's ratio. Summary geometric means
weight each measured shape equally; minima and maxima are retained.
FP8/mHC use deterministic synthetic values and serving-derived physical
shapes. FP4 uses retained C32 target-verify inputs (B48), with complete
six-query groups repeated or selected for other row counts. Complete KV
pages are duplicated and physical slots relocated for added requests.
Compacting request-map row order preserves the original queries, packed
payloads, scales and lengths. Only B48 is an actual captured batch; the
other sizes are explicitly labelled as derived. Raw activations are not
published.

The FP4 Graph capacity is 655627 full-token slots, while the visible
sequence is approximately 128K tokens. Page size is 64. Ratio 1 measures
source scoring and candidate maxima (`candidate_block_size=8`,
`write_logits=False`); ratio 2 materializes dense scores over width 327813.
This capacity and mode distinction matters: a smaller 128K output width
does not reproduce the serving Graph's masking/output work. The public
benchmark defaults to synthetic inputs with the same capacity, per-request
distinct pages and modes. `--fixtures DIR` uses the optional SHA-pinned
retained-input supplement; it is not required for the public synthetic run.

mHC covers the union of target (6 tokens/request) and draft
(5 tokens/request) rows for the serving DP-local batch sizes
1/2/3/4/5/6/7/8/10/12/14/16/20. The baseline includes native statistics
geometry followed by a separate TileLang Sinkhorn; the candidate includes
the finite geometry improvements and exact fusion where eligible.

The complete public call is captured, including output allocations and
FP8 Split-K reduction. JIT, input creation and Python replay submission are
excluded. FP8 excludes activation quantization before GEMM; FP4 times source
scoring/candidate maxima rather than the entire downstream Top-K and
attention pipeline. mHC times the complete statistics plus Sinkhorn chain.
These measurements do not establish an end-to-end serving speedup or
whole-model numerical equivalence. Full-model serving/quality results are
reported separately with their complete deployment configuration.

Fresh measurement tables and numerical counts are recorded with the tested
source revision in the accompanying performance report.

## Fresh H100 Graph measurements

Torch 2.13.0+cu129, Triton 3.7.1, TileLang 0.1.12; one isolated H100 80GB. Each shape passes exact native regression before timing. The table uses equal-shape geometric means over the retained two rounds.

| Operator scope | Shapes | Geometric mean speedup | Minimum | Maximum |
|---|---:|---:|---:|---:|
| FP8 GEMM wrapper | 162 | 1.410x | 1.042x | 3.795x |
| FP4 index scoring, serving Graph capacity | 26 | 2.751x | 1.678x | 3.069x |
| mHC statistics + Sinkhorn chain | 24 | 1.084x | 1.000x | 1.285x |

All per-shape, per-round baseline and candidate latencies are retained in
[the machine-readable report](dsv41-h100-benchmark.json). This combines the
complete FP8/FP4 two-round run with the complete 24-row mHC supplement;
execution script identities and methodology are recorded separately.
There are 212 shapes and 424 timed comparisons. No-change mHC rows remain
in the aggregate, including measurement noise around 1.0x. Minima/maxima
above use two-round mean latencies, not a selected best round.

The final correctness run passed 162 tests, including the adapter's four
CPU lifecycle/source-identity tests. The four standard eager benchmark
entry points also passed. The fixture request-map remap and grouped
candidate are checked against the original retained B48 inputs byte for
byte. The regular eager entry points are a separate timing mode and are
not substituted for these Graph results. Earlier smaller-output-width FP4
diagnostics are excluded from this report.
