# SpectralixBenchmark

Benchmark and evaluation tooling for chemistry-focused AI systems:

- Organic chemistry tasks
- Tandem mass spectrometry (MS2) tasks
- Structured answer evaluation with deterministic + LLM judging

## Runtime Overview

Public evaluation runtime:

1. `OpenShell` as the sandboxed execution/runtime layer
2. OpenAI-compatible model access through a local or remote proxy endpoint
3. `PydanticAI` guard layer for structured validation/repair/retry

Primary runtime/config files:

- `scripts/agents/runtime.py`
- `scripts/agents/openshell_manager.py`
- `scripts/agents/openshell_worker.py`
- `scripts/agents/sgr_schemas.py`
- `scripts/agents/tool_registry.py`
- `scripts/agents/agent_config.yaml`
- `scripts/pydantic_guard/*`

Student inference uses a hidden two-phase flow:

1. Build and validate an internal SGR schema object (`A/B/C` + subtype-specific variants).
2. Generate the final benchmark-aligned answer using compact validated SGR context.

This keeps the raw student/judge pipeline stable while enriching emitted rows with explicit taxonomy and contract metadata.

## Repository Layout

- `benchmark/`: materialized evaluation datasets and manifests used by the paper/runtime.
- `scripts/agents/`: OpenShell runtime, tool registry, and SGR orchestration.
- `scripts/evaluation/`: student, judge, matrix runner, reporting, and taxonomy helpers.
- `docs/`: architecture, benchmark semantics, judging, and operational guidance.
- `external_sources/`: source inventories and manifests used to reconstruct benchmark provenance.

The clean repository intentionally excludes generated runs, rescue analyses, and raw parsing assets.

## Benchmark Schema

Primary evaluation entrypoint: materialized `benchmark_v3` eval file
(`benchmark/benchmark_v3_eval.jsonl`), built from:

- `benchmark/level_a_eval.jsonl`
- `benchmark/level_b_eval.jsonl`
- `benchmark/level_c_eval.jsonl`

Retained source/compatibility dataset for selected build steps:
`benchmark/benchmark_v1_0.jsonl`

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

Example local proxy values:

- API base URL: `http://127.0.0.1:8317/v1`
- API key: `<your-api-key>`

```bash
export API_BASE_URL="http://127.0.0.1:8317/v1"
export CLIPROXY_API_KEY="<your-api-key>"
```

Health check:

```bash
curl -sS \
  -H "Authorization: Bearer $CLIPROXY_API_KEY" \
  "$API_BASE_URL/models"
```

If this fails, start your local OpenAI-compatible proxy first. The exact
startup command depends on your environment and is intentionally not hardcoded in
this public README.

## Evaluation Quick Start (`benchmark_v3`)

Materialize the per-level subsets into the runtime-facing benchmark file:

```bash
uv run python -m scripts.evaluation.materialize_benchmark_v3_eval \
  --output benchmark/benchmark_v3_eval.jsonl
```

### 1) Student smoke run (5 rows)

```bash
uv run python -m scripts.evaluation.student_validation \
  --benchmark-path benchmark/benchmark_v3_eval.jsonl \
  --output-path scripts/evaluation/student_output.jsonl \
  --api-base-url "$API_BASE_URL" \
  --model-name "gpt-5.4-mini" \
  --api-key "$CLIPROXY_API_KEY" \
  --agent-sandbox openshell \
  --agent-backend openshell_worker \
  --agent-tools-profile tools \
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
  --judge-model "gpt-5.4-mini" \
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
  --models gpt-5.4-mini \
  --judge-model gpt-5.4-mini \
  --agent-sandbox openshell \
  --agent-backend openshell_worker \
  --agent-tools-profile tools \
  --student-guard-enabled true \
  --student-guard-mode on_failure \
  --student-guard-retries 2 \
  --judge-method g_eval \
  --judge-g-eval-fallback-structured true \
  --resume-existing false \
  --judge-structured-retries 2
```

## Optional Parsing Utilities

`scripts/parsing/` contains optional reconstruction helpers for working from
external raw exam PDFs. Raw parser inputs, parser iterations, and intermediate
outputs are intentionally not versioned in this repository.

## Runtime Notes

- `--agent-sandbox openshell` is the recommended runtime for benchmark-integrity
  runs.
- `--agent-sandbox local` is a development fallback only.
- Tool access is controlled through `scripts/agents/agent_config.yaml`.
- Detailed timeout, sandbox, and security behavior is documented in
  `docs/security_runbook.md`.

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
