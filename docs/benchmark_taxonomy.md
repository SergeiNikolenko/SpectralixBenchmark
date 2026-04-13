# Benchmark Taxonomy

This document defines the paper-facing taxonomy overlay introduced on top of the
legacy `benchmark_v3` evaluation contract.

## Purpose

The benchmark taxonomy separates three axes that were previously entangled:

- planning horizon
- task mode
- difficulty proxies

It also moves chemistry IE style role-grounding tasks out of the core depth score.

## Core Suites

- `A` — Local Reaction Reasoning
- `B` — Single-Step Disconnection Reasoning
- `C` — Route-Level Synthesis Planning

## Auxiliary Suite

- `G` — Procedure Grounding

`G` is reported separately and is not part of the main depth macro score.

## Current mapping

- `reaction_center_identification` -> `A1`
- `transformation_classification` -> `A1`
- `mechanistic_classification` -> `A2`
- `reagent_role_identification` -> `G1`
- `condition_role_identification` -> `G2`
- `immediate_precursor_prediction` -> `B1`
- `immediate_precursor_with_disconnection` -> `B2`
- `reference_route_planning` -> `C2`
- `route_design` -> `C3`

## Metrics

Primary paper-facing aggregate:

- `macro_depth_quality_score = mean(A_quality, B_quality, C_quality)`
- `macro_depth_end_to_end_score = mean(A_e2e, B_e2e, C_e2e)`

Auxiliary reporting:

- `auxiliary_grounding_quality_score = G_quality`
- `auxiliary_grounding_end_to_end_score = G_e2e`

## Compatibility

- Legacy `level` and `answer_type` fields remain unchanged for runtime compatibility.
- Existing completed runs are not rerun.
- Existing completed runs are migrated in place:
  - `llm_judge_output.jsonl` is enriched with taxonomy fields
  - `metrics.json` is rewritten in benchmark taxonomy format
  - `summary.json` / `summary.csv` become benchmark taxonomy summaries
