# Tools Index

This page lists operational commands used by agents and maintainers.

## Environment

Repository root:

```bash
cd /Users/nikolenko/.codex/worktrees/a55a/SpectralixBenchmark
```

Dependencies:

```bash
uv sync
```

## Benchmark Build

Rebuild large benchmark pools:

```bash
uv run python scripts/build_level_benchmark_files.py
```

Rebuild paper eval subsets:

```bash
uv run python scripts/build_paper_eval_subsets.py
```

## Smoke Runs

Smoke student stage on legacy benchmark:

```bash
uv run python -m scripts.evaluation.student_validation \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --output-path scripts/evaluation/student_output_smoke.jsonl \
  --api-base-url "$API_BASE_URL" \
  --model-name "gpt-5-codex-mini" \
  --api-key "$CLIPROXY_API_KEY" \
  --limit 5
```

Existing ladder smoke outputs:

- `benchmark/smoke_reports/2026-03-13_gpt-5-codex-mini_local/level_a_smoke.jsonl`
- `benchmark/smoke_reports/2026-03-13_gpt-5-codex-mini_local/level_b_smoke.jsonl`
- `benchmark/smoke_reports/2026-03-13_gpt-5-codex-mini_local/level_c_smoke.jsonl`
- `benchmark/smoke_reports/2026-03-13_gpt-5-codex-mini_local/summary.json`

## Judge Modes

Run judge with rubric-based `g_eval`:

```bash
uv run python -m scripts.evaluation.llm_judge \
  --input-path scripts/evaluation/student_output.jsonl \
  --gold-path benchmark/benchmark_v1_0.jsonl \
  --judge-model "gpt-5-codex-mini" \
  --judge-model-url "$API_BASE_URL" \
  --judge-api-key "$CLIPROXY_API_KEY" \
  --judge-method g_eval \
  --judge-g-eval-fallback-structured true \
  --judge-structured-retries 2 \
  --output-path scripts/evaluation/llm_judge_output.jsonl
```

For implementation details:

- `docs/g_eval.md`

## Validation Helpers

Show benchmark manifests:

```bash
cat benchmark/levels_manifest.yaml
cat benchmark/paper_eval_manifest.yaml
```

Quick JSONL row count:

```bash
wc -l benchmark/level_a_eval.jsonl benchmark/level_b_eval.jsonl benchmark/level_c_eval.jsonl
```

## Security and Runtime

Before OpenShell-backed production-style runs:

```bash
docker info >/dev/null
openshell gateway start --name spectralix --port 18080 --plaintext --recreate
```

For security policy details:

- `docs/security_runbook.md`
