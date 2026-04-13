# Benchmark Directory

This directory contains the public benchmark artifacts needed to reproduce the
paper-facing evaluation workflow.

## Primary Files

- `benchmark_v3_eval.jsonl`
  - materialized end-to-end evaluation file used by the runtime
- `level_a_eval.jsonl`
  - Level A evaluation subset
- `level_b_eval.jsonl`
  - Level B evaluation subset
- `level_c_eval.jsonl`
  - Level C evaluation subset

## Metadata and Documentation

- `levels_manifest.yaml`
  - documents the generated large-pool layer used during benchmark construction
- `paper_eval_manifest.yaml`
  - documents the paper-facing subset composition
- `LEVELS.md`
  - semantic overview of the ladder
- `PAPER_EVALS.md`
  - compact description of eval-subset balancing

## Legacy Compatibility

- `benchmark_v1_0.jsonl`
  - retained only as a source/compatibility artifact for selected `Level C`
    route-design records

The public repository does not track the generated large benchmark pools
(`level_a.jsonl`, `level_b.jsonl`, `level_c.jsonl`). Public evaluation should
use `benchmark_v3_eval.jsonl` or the per-level `*_eval.jsonl` files.
