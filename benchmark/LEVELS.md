# Benchmark Levels

This directory contains the benchmark ladder files used for the paper-oriented setup.

## Files

- `level_a.jsonl`
- `level_b.jsonl`
- `level_c.jsonl`
- `levels_manifest.yaml`

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

`level_a.jsonl` is a reaction-understanding file.

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
- `reagent_role_identification`
- `condition_role_identification`

## Level B

`level_b.jsonl` is a single-step retrosynthesis file.

Included sources:

- `ORDerly` retrosynthesis test split
- `PaRoutes selected_reactions_all`

Primary task subtypes:

- `immediate_precursor_prediction`
- `immediate_precursor_with_disconnection`

`level_b.jsonl` is now an extended agent pool. `ORDerly` remains the cleaner benchmark-like source, while `PaRoutes selected_reactions_all` expands coverage for large-scale agent solving.

## Level C

`level_c.jsonl` is a conservative multi-step synthesis planning subset taken from the internal pilot benchmark.

Included source:

- `PaRoutes` reference routes (`n1` and `n5`)
- `benchmark_v1_0.jsonl`

Selection policy:

- include `PaRoutes` route trees as target-to-reference-route planning tasks
- include only tasks that explicitly ask for a synthesis plan or route proposal
- exclude forward multi-step execution and product-sequence recovery problems

## Usage Notes

- The current `level_a.jsonl`, `level_b.jsonl`, and `level_c.jsonl` files are large agent pools.
- They are suitable for assigning tasks to agents at scale.
- They are not yet equivalent to final paper-ready eval subsets.
- Final paper eval sets should be smaller, expert-checked, and balanced across subcategories and difficulty.
