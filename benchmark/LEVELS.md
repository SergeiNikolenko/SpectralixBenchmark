# Benchmark Levels

This directory contains the public benchmark ladder artifacts used for paper
evaluation and runtime execution.

## Public Files

- `README.md`
- `level_a_eval.jsonl`
- `level_b_eval.jsonl`
- `level_c_eval.jsonl`
- `benchmark_v3_eval.jsonl`
- `levels_manifest.yaml`
- `paper_eval_manifest.yaml`

## Unified Record Schema

Each JSONL row follows the same top-level shape:

```json
{
  "record_id": "string",
  "level": "A | B | C",
  "source_id": "string",
  "source_split": "string",
  "source_license": "string",
  "task_family": "string",
  "task_subtype": "string",
  "difficulty": "easy | medium | hard",
  "coverage_tags": ["string", "..."],
  "input_text": "string",
  "input": {},
  "gold": {},
  "metadata": {}
}
```

## Level A

`Level A` is the reaction-understanding layer of the benchmark ladder.

Included sources:

- `PMechDB` manually curated challenging test
- `PMechDB` combinatorial test
- `USPTO-50K` test
- `ChEMU 2020` NER train/dev
- `CHORISO` public set
- `WEAVE2` public annotations

Primary task subtypes:

- `reaction_center_identification`
- `mechanistic_classification`
- `transformation_classification`

Auxiliary grounding subtypes associated with this source layer:

- `reagent_role_identification`
- `condition_role_identification`

In taxonomy-aware reporting these are exposed under auxiliary suite `G`
("Procedure Grounding"), even though the legacy source rows still originate
from the broader `Level A` construction layer.

## Level B

`Level B` is the single-step retrosynthesis layer.

Included sources:

- `ORDerly` retrosynthesis test split
- `PaRoutes selected_reactions_all`

Primary task subtypes:

- `immediate_precursor_prediction`
- `immediate_precursor_with_disconnection`

## Level C

`Level C` is the route-level synthesis planning layer.

Included source:

- `PaRoutes` reference routes (`n1` and `n5`)
- `benchmark_v1_0.jsonl`

Selection policy:

- include `PaRoutes` route trees as target-to-reference-route planning tasks
- include only tasks that explicitly ask for a synthesis plan or route proposal
- exclude forward multi-step execution and product-sequence recovery problems

## Usage Notes

- `benchmark_v3_eval.jsonl` is the primary end-to-end evaluation entrypoint.
- The per-level `*_eval.jsonl` files are the public paper-facing subsets.
- `benchmark_v1_0.jsonl` is retained separately as a source/compatibility file
  for selected legacy `Level C` route-design records, but it is not the primary
  benchmark entrypoint.
