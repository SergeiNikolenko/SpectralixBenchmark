# Benchmark Ladder

This repository uses a three-level benchmark ladder for chemistry reasoning.

For the detailed construction history, source mapping, inclusion/exclusion policy, and build rationale, see `docs/benchmark_construction.md`.

## Levels

## Level A - Reaction Understanding

Primary objective:

- interpret local reaction behavior

Typical task subtypes:

- `reaction_center_identification`
- `mechanistic_classification`
- `transformation_classification`
- `reagent_role_identification`
- `condition_role_identification`

Data files:

- pool: `benchmark/level_a.jsonl`
- paper eval subset: `benchmark/level_a_eval.jsonl`

## Level B - Single-Step Retrosynthesis

Primary objective:

- propose immediate precursors and local disconnection logic

Typical task subtypes:

- `immediate_precursor_prediction`
- `immediate_precursor_with_disconnection`

Data files:

- pool: `benchmark/level_b.jsonl`
- paper eval subset: `benchmark/level_b_eval.jsonl`

## Level C - Multi-Step Synthesis Planning

Primary objective:

- plan route-level synthesis with multi-step structure

Typical task subtype:

- `reference_route_planning`
- `route_design`

Data files:

- pool: `benchmark/level_c.jsonl`
- paper eval subset: `benchmark/level_c_eval.jsonl`

## Source and Size Truth

Use these manifests as source of truth:

- `benchmark/levels_manifest.yaml`
- `benchmark/paper_eval_manifest.yaml`

## Operational Use

- For large-scale agent workloads, use pool files (`level_a.jsonl`, `level_b.jsonl`, `level_c.jsonl`).
- For reproducible paper comparisons and rapid checks, use eval subsets (`*_eval.jsonl`).
- For baseline viability checks, use smoke reports under:
  - `benchmark/smoke_reports/`
