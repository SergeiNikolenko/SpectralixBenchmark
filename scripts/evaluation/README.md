# Evaluation Pipeline

This directory contains the benchmark evaluation pipeline with hybrid scoring
actions (deterministic + LLM judge) and reliability-aware metrics.

## Scripts

- `student_validation.py`: runs the student model and writes `student_output.jsonl`.
- `llm_judge.py`: scores student answers against canonical answers.
- `run_full_matrix.py`: orchestrates full matrix runs across multiple student models.

## Key Principles

- Fixed judge model: the judge is controlled via `--judge-model` and is identical for all candidates in a run.
- No self-judge: the student model and judge model are independent.
- Hybrid scoring:
  - Deterministic (no LLM calls): `single_choice`, `multiple_choice`, `ordering`, `numeric`, `msms_structure_prediction`.
  - LLM judge only: `text`, `reaction_description`, `full_synthesis`, `structure`, and other non-deterministic types.
- Technical failures are separated from knowledge quality:
  - Student output includes `student_status`, `student_error`, `student_elapsed_ms`.
  - Rows with `student_status != ok` are marked as technical skips by judge.
  - These rows are excluded from quality aggregates and counted in reliability metrics.

## Output Fields

`llm_judge_output.jsonl` keeps backward-compatible fields and adds:

- `score_method`: `deterministic` or `llm_judge`
- `row_status`: `ok` or a technical/judge status
- `judge_model` (for `llm_judge` rows)
- `judge_request_id` (optional)
- `judge_latency_ms` (optional)

## Full Matrix Outputs

`run_full_matrix.py` writes:

- `runs/<run_id>/summary.csv`
- `runs/<run_id>/summary.json`
- `runs/<run_id>/<model>/metrics.json`
- `runs/<run_id>/<model>/breakdown_by_answer_type.json`
- `runs/<run_id>/<model>/errors_sample.json`

Metrics are separated into:

- `quality_normalized_score`: quality over rows with `row_status == ok`
- `reliability_ok_rate`: fraction of rows where both student and judge are `ok`
- `infra_error_rate`: per-category technical/judge error rates

## Recommended Profiles

Smoke profile:

```bash
uv run python scripts/evaluation/run_full_matrix.py \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --limit 5 \
  --timeout 60 \
  --workers 1 \
  --model-url http://localhost:11434/api/chat \
  --student-models llama3.1:8b \
  --judge-model gpt-4o-mini
```

Larger run profile:

```bash
uv run python scripts/evaluation/run_full_matrix.py \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --timeout 120 \
  --workers 1 \
  --model-url http://localhost:11434/api/chat \
  --student-models model_a model_b \
  --judge-model gpt-4o-mini
```

## Notes

- `workers` is currently reserved for compatibility and tuning; this pipeline version runs sequentially.
