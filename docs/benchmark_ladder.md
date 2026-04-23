# Benchmark Ladder

This repository uses a three-level benchmark ladder for chemistry reasoning.

For the detailed construction history, source mapping, inclusion/exclusion policy, and build rationale, see `docs/benchmark_construction.md`.

## Levels

## Level A - Local Reaction Reasoning

Primary objective:

- interpret local reaction behavior

Typical task subtypes:

- `reaction_center_identification`
- `mechanistic_classification`
- `transformation_classification`

Suggested paper-facing subtracks:

- `A1` bond-change localization
- `A2` mechanistic inference
- `A3` selectivity and stereochemistry reasoning

Data files:

- public eval subset: `benchmark/level_a_eval.jsonl`

## Level B - Single-Step Disconnection Reasoning

Primary objective:

- propose chemically plausible immediate precursors and local disconnection logic

Typical task subtypes:

- `immediate_precursor_prediction`
- `immediate_precursor_with_disconnection`

Suggested paper-facing subtracks:

- `B1` precursor proposal
- `B2` disconnection justification
- `B3` constraint-aware retrosynthesis

Scoring contract:

- the documented source precursor set is one acceptable reference route, not the
  only correct answer
- chemically plausible one-step alternatives receive credit when they can reach
  the same target at the requested planning depth
- multistep plans, earlier-stage building blocks, and chemically implausible
  disconnections are penalized

Data files:

- public eval subset: `benchmark/level_b_eval.jsonl`

## Level C - Route-Level Synthesis Planning

Primary objective:

- plan route-level synthesis with multi-step structure

Typical task subtypes:

- `reference_route_planning`
- `route_design`

Suggested paper-facing subtracks:

- `C1` route completion or ranking
- `C2` reference-route planning
- `C3` open route design

Data files:

- public eval subset: `benchmark/level_c_eval.jsonl`

## Auxiliary Suite G - Procedure Grounding

Purpose:

- keep chemistry IE and role-grounding tasks available without mixing them into the
  core planning-depth score

Current task subtypes remapped into `G` for reporting:

- `reagent_role_identification`
- `condition_role_identification`

Important compatibility note:

- the current legacy runtime still keeps these rows under legacy `level = A`
- benchmark taxonomy reporting remaps them into auxiliary suite `G`
- existing completed runs can be backfilled into this structure without rerunning

## Source and Size Truth

Use these manifests as source of truth:

- `benchmark/levels_manifest.yaml`
- `benchmark/paper_eval_manifest.yaml`
- taxonomy overlay: `spectralix_benchmark/evaluation/benchmark_taxonomy.py`

`levels_manifest.yaml` documents the generated large-pool layer used during
benchmark construction. Those pool files are not tracked in the public
repository; the public runtime entrypoint remains `benchmark/benchmark_v3_eval.jsonl`.

## Operational Use

- For reproducible paper comparisons and rapid checks, use eval subsets (`*_eval.jsonl`).
- For end-to-end benchmark runs, use the materialized file `benchmark/benchmark_v3_eval.jsonl`.
- For baseline viability checks, run ad hoc smoke subsets against `benchmark/benchmark_v3_eval.jsonl`
  and keep generated outputs outside the tracked repository tree.
