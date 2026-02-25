# Evaluation Pipeline

## Components

- `student_validation.py`: student-answer generation (agent runtime by default)
- `llm_judge.py`: deterministic + LLM hybrid judge
- `run_full_matrix.py`: orchestrates multi-model runs and aggregates metrics

## Runtime Mode

- Uses `scripts/agents/runtime.py`
- Executes tool-capable agent with Docker sandbox
- Controlled by `scripts/agents/agent_config.yaml`
- Uses `scripts/pydantic_guard/*` for structured validation/repair on top of `smolagents`
- `--agent-enabled false` is accepted only for CLI compatibility and falls back to agent runtime
- Docker preflight runs once before row loop (fail-fast on sandbox issues)
- Agent session is reused across questions within one run

## Dependencies

```bash
uv sync
```

## Environment Variables

Supported API key variables:

- `OPENAI_API_KEY`
- `AITUNNEL_API_KEY`
- `OPENROUTER_API_KEY`
- `TOGETHER_API_KEY`

Optional auth customization:

- `OPENAI_API_KEY_HEADER` (default: `Authorization`)
- `OPENAI_API_KEY_PREFIX` (default: `Bearer`)

Network-tool note:

- `safe_http_get_tool` is enabled only when both are set:
  - `security.allow_network_tools: true`
  - non-empty `security.allowed_tool_hosts`

## `student_validation.py` CLI

Required:

- `--benchmark-path`
- `--output-path`
- `--model-name`
- one of:
  - `--model-url` (chat endpoint or base URL)
  - `--api-base-url`

Agent flags:

- `--agent-enabled` (default: `true`)
- `--agent-max-steps` (default: `6`)
- `--agent-sandbox` (default: `docker`)
- `--agent-tools-profile` (default: `full`)
- `--agent-config` (default: `scripts/agents/agent_config.yaml`)

Tools profiles:

- `full`: all allowlisted runtime tools
- `minimal`: reduced helper set
- `code_only`: no custom tools

Note:

- Base smolagents tools are enabled by default via `runtime.add_base_tools: true`
- This includes built-in `web_search` and `visit_webpage`
- For strict python-only mode, set `runtime.add_base_tools: false` in agent config

Student guard flags:

- `--student-guard-enabled` (default: `true`)
- `--student-guard-mode` (`on_failure|always|off`, default: `on_failure`)
- `--student-guard-retries` (default: `2`)
- `--student-guard-reasoning-effort` (`low|medium|high`, default: `high`)

Trace flags:

- `--trace-log-enabled` (default: `true`)
- `--trace-log-dir` (default: `<output-dir>/traces`)

Per-question trace logs include:

- full agent stdout/stderr stream (tool calls, code execution steps, runtime errors)
- raw model answer
- normalized final answer
- row metadata (`exam_id/page_id/question_id`, status, elapsed time)

## Example: Local Proxy Run (5 rows)

```bash
uv run python scripts/evaluation/student_validation.py \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --output-path scripts/evaluation/student_output.jsonl \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --model-name "gpt-5.3-codex-spark" \
  --api-key "ccs-internal-managed" \
  --student-guard-enabled true \
  --student-guard-mode on_failure \
  --student-guard-retries 2 \
  --student-guard-reasoning-effort high \
  --trace-log-enabled true \
  --trace-log-dir runs/debug_traces \
  --limit 5
```

## Judge Run

Judge structured flags:

- `--judge-structured-enabled` (default: `true`)
- `--judge-structured-retries` (default: `2`)
- `--judge-structured-fallback-legacy` (default: `true`)
- `--reasoning-effort` (`low|medium|high`, default: `high`)

```bash
uv run python scripts/evaluation/llm_judge.py \
  --input-path scripts/evaluation/student_output.jsonl \
  --gold-path benchmark/benchmark_v1_0.jsonl \
  --judge-model "gpt-5.3-codex-spark" \
  --reasoning-effort high \
  --judge-model-url "http://127.0.0.1:8317/v1" \
  --judge-api-key "ccs-internal-managed" \
  --judge-structured-enabled true \
  --judge-structured-retries 2 \
  --judge-structured-fallback-legacy true \
  --output-path scripts/evaluation/llm_judge_output.jsonl
```

## Full Matrix Run

`run_full_matrix.py` accepts `--student-models` and alias `--models`.
It also accepts `--api-base-url` as an alternative to `--model-url`.

```bash
uv run python scripts/evaluation/run_full_matrix.py \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --api-key "ccs-internal-managed" \
  --models gpt-5.3-codex-spark \
  --judge-model gpt-5.3-codex-spark \
  --agent-enabled true \
  --student-guard-enabled true \
  --student-guard-mode on_failure \
  --student-guard-retries 2 \
  --judge-reasoning-effort high \
  --trace-log-enabled true \
  --trace-log-dir runs/debug_traces \
  --judge-structured-enabled true \
  --judge-structured-retries 2 \
  --judge-structured-fallback-legacy true
```

## Output Contracts

Student output (`student_output.jsonl`) schema is stable:

- `exam_id`
- `page_id`
- `question_id`
- `question_type`
- `question_text`
- `answer_type`
- `student_answer`
- `student_status`
- `student_error`
- `student_elapsed_ms`

Judge output (`llm_judge_output.jsonl`) schema remains unchanged.
