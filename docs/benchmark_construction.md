# Benchmark Construction

`benchmark_v3` is designed as a three-level ladder over planning depth rather
than as a single mixed chemistry dataset.

- `Level A`: local reaction understanding
- `Level B`: single-step retrosynthesis
- `Level C`: route-level synthesis planning

This structure reflects the benchmark's main hypothesis: model failures become
more severe as the planning horizon grows, even within the same broad chemistry
domain.

## Public Repository Scope

This public repository keeps the paper-facing benchmark artifacts and the
metadata required to understand how they were assembled:

- `benchmark/level_a_eval.jsonl`
- `benchmark/level_b_eval.jsonl`
- `benchmark/level_c_eval.jsonl`
- `benchmark/benchmark_v3_eval.jsonl`
- `benchmark/levels_manifest.yaml`
- `benchmark/paper_eval_manifest.yaml`
- `external_sources/` provenance manifests

Large intermediate pools, parser outputs, rescue overlays, run directories, and
historical analysis scratchpads are intentionally excluded from version control.

## Why `benchmark_v3` Replaced the Old Mixed Benchmark

The earlier `benchmark/benchmark_v1_0.jsonl` was useful as an internal pilot,
but it mixed several task families:

- general organic chemistry questions
- reaction understanding
- synthesis planning
- mass-spectrometry structure tasks

That format was not suitable for controlled capability analysis because it did
not provide a clean axis of difficulty. `benchmark_v3` replaces it with a ladder
that isolates reasoning depth.

`benchmark_v1_0.jsonl` is retained only as a source/compatibility layer for a
small number of `Level C` route-design records that remain part of the public
materialized benchmark.

## Source Normalization Strategy

The benchmark was not derived from a single upstream dataset. Instead, the
construction workflow normalizes heterogeneous sources into one internal schema.

High-level process:

1. catalog candidate sources in `external_sources/`
2. mark each source as directly usable, transformed, or provenance-only
3. normalize source examples into a common benchmark row format
4. materialize deterministic paper-facing eval subsets

This keeps benchmark semantics stable even when source formats differ.

## Level A Sources

`Level A` covers local reaction reasoning. Publicly documented source families:

- `PMechDB`
- `USPTO-50K`
- `ChEMU 2020`
- `CHORISO`
- `WEAVE2`

Representative task subtypes:

- `reaction_center_identification`
- `mechanistic_classification`
- `transformation_classification`
- `reagent_role_identification`
- `condition_role_identification`

## Level B Sources

`Level B` covers one-step disconnection reasoning. Primary sources:

- `ORDerly retrosynthesis`
- `PaRoutes selected_reactions_all`

Representative task subtypes:

- `immediate_precursor_prediction`
- `immediate_precursor_with_disconnection`

## Level C Sources

`Level C` covers route-level synthesis planning. The public benchmark keeps this
layer conservative and provenance-explicit:

- `PaRoutes` reference routes (`n1` and `n5`)
- selected route-design tasks sourced from `benchmark_v1_0.jsonl`

Only tasks that genuinely require synthesis planning are retained. Forward
multi-step execution and unrelated product-prediction items are excluded.

## Reproducibility Contract

For external work, treat the following as the source of truth:

- `benchmark/benchmark_v3_eval.jsonl` for full benchmark execution
- `benchmark/level_a_eval.jsonl`, `benchmark/level_b_eval.jsonl`,
  `benchmark/level_c_eval.jsonl` for per-level analysis
- `benchmark/levels_manifest.yaml` and `benchmark/paper_eval_manifest.yaml`
  for composition rules
- `docs/benchmark_ladder.md` for semantic interpretation
- `docs/benchmark_taxonomy.md` for reporting overlays

The repository is therefore published as a clean benchmark and runtime package,
not as a dump of every intermediate artifact used during dataset construction.
