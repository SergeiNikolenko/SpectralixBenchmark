# Tools Index

This page lists operational commands used by agents and maintainers.

## Environment

Repository root:

```bash
cd /path/to/SpectralixBenchmark
```

Dependencies:

```bash
uv sync
```

## Benchmark Build

Rebuild large benchmark pools:

```bash
uv run spectralix-build-levels
```

Rebuild paper eval subsets:

```bash
uv run spectralix-build-paper-eval
```

Materialize the runtime-facing benchmark file:

```bash
uv run spectralix-materialize \
  --output benchmark/benchmark_v3_eval.jsonl
```

Builder status:

- `level_benchmark_files` and `paper_eval_subsets` expose dedicated argparse CLIs.
- They are dataset-dependent and fail fast if expected source files or benchmark pools are missing.
- `spectralix-build` provides a single umbrella entry point with `levels` and `paper-eval` subcommands.

## CLI Sanity Checks

Verify supported CLIs:

```bash
uv run spectralix-student --help
uv run spectralix-judge --help
uv run spectralix-matrix --help
uv run spectralix-materialize --help
uv run spectralix-build --help
uv run python -m spectralix_benchmark.evaluation.student_validation --help
uv run python -m spectralix_benchmark.evaluation.llm_judge --help
uv run python -m spectralix_benchmark.evaluation.run_full_matrix --help
uv run python -m spectralix_benchmark.evaluation.materialize_benchmark_v3_eval --help
```

## Smoke Runs

Smoke student stage on the materialized `benchmark_v3` eval set:

```bash
uv run spectralix-student \
  --benchmark-path benchmark/benchmark_v3_eval.jsonl \
  --output-path runs/smoke/student_output.jsonl \
  --api-base-url "$API_BASE_URL" \
  --model-name "gpt-5.4-mini" \
  --api-key "$CLIPROXY_API_KEY" \
  --agent-sandbox openshell \
  --agent-backend openshell_worker \
  --agent-tools-profile tools \
  --trace-log-enabled true \
  --limit 5
```

## Judge Modes

Run judge with rubric-based `g_eval`:

```bash
uv run spectralix-judge \
  --input-path runs/smoke/student_output.jsonl \
  --gold-path benchmark/benchmark_v3_eval.jsonl \
  --judge-model "gpt-5.4-mini" \
  --judge-model-url "$API_BASE_URL" \
  --judge-api-key "$CLIPROXY_API_KEY" \
  --judge-method g_eval \
  --judge-g-eval-fallback-structured true \
  --judge-structured-retries 2 \
  --output-path runs/smoke/llm_judge_output.jsonl
```

## Current Tool Profiles

Configured in `spectralix_benchmark/agents/agent_config.yaml`:

- `minimal`: no local tools
- `tools`: `chem_python_tool`, `workspace_list_tool`, `workspace_read_tool`, `shell_exec_tool`, `uv_run_tool`
- `tools_internet`: same as `tools` plus `safe_http_get_tool` (only when network tools are enabled)
- `full`: includes `workspace_write_tool` in addition to `tools` set

Notes:

- `tools` is the default practical profile for OpenShell runs.
- `workspace_write_tool` is intentionally excluded from `tools`.
- `safe_http_get_tool` is gated by `security.allow_network_tools` and host allowlist.

## Full Matrix (OpenShell + Tools)

```bash
uv run spectralix-matrix \
  --benchmark-path benchmark/benchmark_v3_eval.jsonl \
  --api-base-url "$API_BASE_URL" \
  --api-key "$CLIPROXY_API_KEY" \
  --models gpt-5.4-mini \
  --judge-model gpt-5.4-mini \
  --judge-model-url "$API_BASE_URL" \
  --judge-api-key "$CLIPROXY_API_KEY" \
  --agent-sandbox openshell \
  --agent-backend openshell_worker \
  --agent-tools-profile tools \
  --trace-log-enabled true \
  --verbose-output-enabled true
```

For implementation details:

- `docs/g_eval.md`
- `docs/architecture.md`

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
