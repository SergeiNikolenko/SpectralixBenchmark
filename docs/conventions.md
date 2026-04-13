# Conventions

## General Engineering

- Keep changes minimal and directly tied to the task.
- Prefer explicit behavior over hidden side effects.
- Do not refactor unrelated code while implementing benchmark updates.
- Keep generated data and source code concerns separated.

## Benchmark Data Conventions

- Treat `benchmark/benchmark_v3_eval.jsonl` and the per-level `*_eval.jsonl`
  files as materialized benchmark artifacts.
- Do not hand-edit generated benchmark JSONL files.
- Use builder scripts for reproducible regeneration:
  - `spectralix_benchmark/build/level_benchmark_files.py`
  - `spectralix_benchmark/build/paper_eval_subsets.py`
- Keep deterministic subset selection behavior stable across runs.

## Schema Conventions

Unified benchmark rows should preserve these top-level fields:

- `record_id`
- `level`
- `source_id`
- `task_family`
- `task_subtype`
- `difficulty`
- `coverage_tags`
- `input_text`
- `input`
- `gold`
- `metadata`

If schema-related assumptions change, update:

- `benchmark/LEVELS.md`
- `benchmark/levels_manifest.yaml`
- `benchmark/paper_eval_manifest.yaml`

## Runtime and Evaluation

- Use `uv run` for repository scripts.
- Prefer smoke runs on `*_eval.jsonl` subsets before full runs.
- Keep sandbox assumptions explicit (`openshell` for production integrity, `local` for development checks).

## Documentation Hygiene

- Keep the top-level documentation entrypoints concise and navigable.
- Put detailed guidance in `docs/*` pages.
- Update docs in the same change when behavior or benchmark semantics change.
