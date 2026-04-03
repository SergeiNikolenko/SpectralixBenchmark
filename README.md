# SpectralixBenchmark

Benchmark and evaluation tooling for chemistry-focused AI systems:

- Organic chemistry tasks
- Tandem mass spectrometry (MS2) tasks
- Structured answer evaluation with deterministic + LLM judging

## Runtime Overview

Production evaluation runtime:

1. `OpenShell` as the sandboxed execution/runtime layer
2. OpenShell-managed inference routed through `https://inference.local/v1`
3. `PydanticAI` guard layer for structured validation/repair/retry

Primary runtime/config files:

- `scripts/agents/runtime.py`
- `scripts/agents/openshell_manager.py`
- `scripts/agents/openshell_worker.py`
- `scripts/agents/tool_registry.py`
- `scripts/agents/agent_config.yaml`
- `scripts/pydantic_guard/*`

## Benchmark Schema

Primary evaluation entrypoint: materialized `benchmark_v3` eval file
(`benchmark/benchmark_v3_eval.jsonl`), built from:

- `benchmark/level_a_eval.jsonl`
- `benchmark/level_b_eval.jsonl`
- `benchmark/level_c_eval.jsonl`

Legacy compatibility dataset (still supported): `benchmark/benchmark_v1_0.jsonl`

```json
{
  "exam_id": "string",
  "page_id": "integer | string",
  "question_id": "integer | string",
  "question_type": "string",
  "question_text": "string",
  "answer_type": "single_choice | multiple_choice | numeric | ordering | structure | full_synthesis | reaction_description | property_determination | msms_structure_prediction | text",
  "canonical_answer": "string",
  "max_score": "integer"
}
```

## Prerequisites

- Python 3.12+
- `uv`
- Local proxy (OpenAI-compatible) or direct OpenAI-compatible endpoint
- Docker daemon for local OpenShell gateway deployment

Repository root:

```bash
cd /path/to/SpectralixBenchmark
```

Install dependencies:

```bash
uv sync
```

## Environment Setup

Default local proxy values in this repository setup:

- API base URL: `http://127.0.0.1:8317/v1`
- API key: `ccs-internal-managed`

```bash
export API_BASE_URL="http://127.0.0.1:8317/v1"
export CLIPROXY_API_KEY="ccs-internal-managed"
```

Health check:

```bash
curl -sS \
  -H "Authorization: Bearer $CLIPROXY_API_KEY" \
  "$API_BASE_URL/models"
```

If this fails, start your local proxy first (example command from your setup):

```bash
/Users/nikolenko/.ccs/cliproxy/bin/plus/cli-proxy-api-plus \
  -standalone \
  -config /Users/nikolenko/.ccs/cliproxy/config.yaml
```

## `benchmark_v3` Eval Run

The `benchmark_v3` ladder uses:

- `benchmark/level_a_eval.jsonl`
- `benchmark/level_b_eval.jsonl`
- `benchmark/level_c_eval.jsonl`

The current student/judge runtime still expects the legacy evaluation contract
(`question_text`, `answer_type`, `canonical_answer`, `max_score`), so first
materialize the `benchmark_v3` eval subsets into one evaluation file:

```bash
uv run python -m scripts.evaluation.materialize_benchmark_v3_eval \
  --output benchmark/benchmark_v3_eval.jsonl
```

Then run the matrix pipeline against that materialized benchmark.

```bash
uv run python -m scripts.evaluation.run_full_matrix \
  --benchmark-path benchmark/benchmark_v3_eval.jsonl \
  --api-base-url "$API_BASE_URL" \
  --api-key "$CLIPROXY_API_KEY" \
  --models gpt-5.4-mini \
  --judge-model gpt-5.4 \
  --judge-model-url "$API_BASE_URL" \
  --judge-api-key "$CLIPROXY_API_KEY" \
  --judge-method g_eval \
  --judge-g-eval-fallback-structured true \
  --judge-reasoning-effort medium \
  --agent-sandbox openshell \
  --agent-tools-profile minimal \
  --trace-log-enabled true
```

## Evaluation Quick Start (`benchmark_v3`)

Materialize eval input first:

```bash
uv run python -m scripts.evaluation.materialize_benchmark_v3_eval \
  --output benchmark/benchmark_v3_eval.jsonl
```

### 1) Student stage smoke run (5 rows)

```bash
uv run python -m scripts.evaluation.student_validation \
  --benchmark-path benchmark/benchmark_v3_eval.jsonl \
  --output-path scripts/evaluation/student_output.jsonl \
  --api-base-url "$API_BASE_URL" \
  --model-name "gpt-5-codex-mini" \
  --api-key "$CLIPROXY_API_KEY" \
  --agent-sandbox openshell \
  --agent-tools-profile minimal \
  --student-guard-enabled true \
  --student-guard-mode on_failure \
  --student-guard-retries 2 \
  --resume-existing false \
  --limit 5
```

### 2) Judge run

```bash
uv run python -m scripts.evaluation.llm_judge \
  --input-path scripts/evaluation/student_output.jsonl \
  --gold-path benchmark/benchmark_v3_eval.jsonl \
  --judge-model "gpt-5-codex-mini" \
  --judge-model-url "$API_BASE_URL" \
  --judge-api-key "$CLIPROXY_API_KEY" \
  --judge-method g_eval \
  --judge-g-eval-fallback-structured true \
  --resume-existing false \
  --judge-structured-retries 2 \
  --output-path scripts/evaluation/llm_judge_output.jsonl
```

### 3) Full matrix run

```bash
uv run python -m scripts.evaluation.run_full_matrix \
  --benchmark-path benchmark/benchmark_v3_eval.jsonl \
  --api-base-url "$API_BASE_URL" \
  --api-key "$CLIPROXY_API_KEY" \
  --models gpt-5-codex-mini \
  --judge-model gpt-5-codex-mini \
  --agent-sandbox openshell \
  --agent-tools-profile minimal \
  --student-guard-enabled true \
  --student-guard-mode on_failure \
  --student-guard-retries 2 \
  --judge-method g_eval \
  --judge-g-eval-fallback-structured true \
  --resume-existing false \
  --judge-structured-retries 2
```

Inspect latest run:

```bash
latest_run=$(ls -td runs/* | head -n 1)
echo "$latest_run"
cat "$latest_run/summary.csv"
```

## Parsing Quick Start

Parser runtime uses `./exam_data` relative to current directory.
Run parsing commands from `scripts/parsing`.

Run parser:

```bash
cd /Users/nikolenko/.codex/worktrees/e20d/SpectralixBenchmark/scripts/parsing

uv run python exam-parser-pipeline.py \
  --agent-enabled true \
  --agent-max-steps 6 \
  --agent-config ../agents/agent_config.yaml \
  --model-marker gpt-5-codex-mini \
  --openai-base-url "$API_BASE_URL" \
  --api-key "$CLIPROXY_API_KEY" \
  --parser-structured-repair-enabled true \
  --parser-structured-retries 2
```

Flatten parsed output:

```bash
uv run python benchmark_collection.py
```

## Security and Tool/Internet Access

Default policy in `scripts/agents/agent_config.yaml`:

- OpenShell sandbox is the default executor (`sandbox.executor_type: openshell`)
- Local helper tools are selected by profile (`no_tools`, `minimal`, `tools`, `tools_internet`)
- Internet tools are disabled (`security.allow_network_tools: false`)
- `safe_http_get_tool` is not available unless network tools are explicitly enabled
- Tool network access is host-allowlisted only (`security.allowed_tool_hosts`)
- OpenShell policy allows only the sandbox workdir plus explicit temp paths
- Sandbox traffic uses `https://inference.local/v1`; upstream credentials stay in the host-side OpenShell provider config
- Student prompt excludes benchmark identifiers (`exam_id/page_id/question_id`)

This means internet access is controlled, not unrestricted.

## Sandbox Modes

- `--agent-sandbox openshell`:
  - Recommended runtime
  - Uses OpenShell gateway + sandbox policy enforcement
  - Supports tool/network controls through policy + tool profiles
- `--agent-sandbox local`:
  - Development fallback
  - No sandbox isolation and weak benchmark integrity guarantees
  - Do not use for production benchmarking

## Troubleshooting

`OpenShell preflight failed`

- Start Docker Desktop / daemon
- Validate with `docker info`
- Start or recreate the local OpenShell gateway:

```bash
openshell gateway start --name spectralix --port 18080 --plaintext --recreate
```
- Retry with `--agent-sandbox openshell`

`401 Unauthorized` from proxy/model endpoint

- Check API key value
- Verify `Authorization` header behavior in proxy
- Run `curl "$API_BASE_URL/models"` with the same key

`safe_http_get_tool` not available

- Expected with default config
- Enable both:
  - `security.allow_network_tools: true`
  - non-empty `security.allowed_tool_hosts`

## Additional Docs

- Evaluation details: `scripts/evaluation/README.md`
- Parsing details: `scripts/parsing/README.md`
- Architecture: `docs/architecture.md`
- Security controls: `docs/security_runbook.md`

## Judge Output Notes

- `score_method` in `llm_judge_output.jsonl` explicitly indicates score source:
  - `deterministic`
  - `g_eval`
  - `structured_fallback`
  - `llm_judge`
- With `--judge-g-eval-fallback-structured true`, open-ended rows can fall back
  from `g_eval` to structured judge when needed.

## Contacts

Maintainer (Innopolis University — AI Lab in Chemistry):

- Ivan Golov
- Email: `i.golov@innopolis.university`
- Telegram: [https://t.me/Ione_Golov](https://t.me/Ione_Golov)
