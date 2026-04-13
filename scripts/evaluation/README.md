# Evaluation Pipeline

## Components

- `student_validation.py`: student-answer generation (agent runtime by default)
- `llm_judge.py`: deterministic + LLM hybrid judge
- `run_full_matrix.py`: orchestrates multi-model runs and aggregates metrics

## Runtime Mode

- Uses `scripts/agents/runtime.py`
- Executes tool-capable agent with OpenShell sandbox
- Controlled by `scripts/agents/agent_config.yaml`
- Uses `scripts/pydantic_guard/*` for structured validation/repair above the runtime
- OpenShell preflight runs once before row loop (fail-fast on gateway/sandbox issues)
- Agent runtime reuses a named OpenShell sandbox and configures gateway-scoped managed inference before row execution
- Student prompt hides `exam_id/page_id/question_id`; only `question_text` + `answer_type` are passed to the model

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

- `--agent-max-steps` (default: `6`)
- `--agent-sandbox` (default: `openshell`)
- `--agent-tools-profile` (default: `minimal`)
- `--agent-config` (default: `scripts/agents/agent_config.yaml`)

Tools profiles:

- `no_tools`: no local helper tools
- `minimal`: no helper tools, pure model path
- `tools`: local chemistry helpers without internet
- `tools_internet`: local chemistry helpers plus allowlisted HTTP fetch

Chemistry tool note:

- `tools` and `tools_internet` include `chem_python_tool`, which runs short snippets inside the sandbox
- runtime bootstrap installs required packages into `/sandbox/.venv` on first use
- prefer this tool for SMILES validation, canonicalization, formula checks, and small chemistry calculations

Leakage protection defaults:

- Benchmark path is not passed to student agent tasks
- OpenShell policy restricts filesystem access to sandbox paths
- Student traffic inside the sandbox goes to `https://inference.local/v1`; upstream API credentials stay on the host-side OpenShell provider
- `--agent-sandbox local` is unsafe for benchmark integrity and should be used only for debugging

Student guard flags:

- `--student-guard-enabled` (default: `true`)
- `--student-guard-mode` (`on_failure|always|off`, default: `on_failure`)
- `--student-guard-retries` (default: `2`)
- `--student-guard-reasoning-effort` (`low|medium|high`, default: `high`)

Trace flags:

- `--trace-log-enabled` (default: `true`)
- `--trace-log-dir` (default: `<output-dir>/traces`)
- `--verbose-output-enabled` (default: `false`)
- `--verbose-output-path` (default: `<output-dir>/student_output_verbose.jsonl`)
- `--resume-existing` (default: `false`, append only missing rows instead of overwrite)

Per-question trace logs include:

- best-effort reasoning summary:
  - `thought` values per step
  - provider reasoning summaries (when API returns them)
  - total reasoning tokens (if exposed by provider)
- human-readable step summary (`thought`, code block, tool calls, observations, errors)
- compact `RunResult` payload (`state`, output preview, step summaries)
- full agent stdout/stderr stream (tool calls, code execution steps, runtime errors)
- raw model answer
- normalized final answer
- row metadata (`exam_id/page_id/question_id`, status, elapsed time)

Note on reasoning visibility:

- Hidden provider-side chain-of-thought is generally not exposed by model APIs.
- Traces include observable reasoning artifacts only (steps, code, tool calls, observations, model-visible outputs).

Verbose student output:

- `student_output.jsonl` remains contract-stable for judge/matrix compatibility.
- If you need "everything in JSONL", enable `--verbose-output-enabled true`.
- This writes `student_output_verbose.jsonl` with additional fields:
  - `raw_answer`
  - `reasoning_summary`
  - `agent_run_details` (compact)
  - `trace_log_path`

Fail-fast on model limits:

- If provider returns quota/credits exhaustion (for example `insufficient_quota`, `billing_limit_reached`, `exceeded your current quota`), the run is aborted immediately.
- This is enforced both in student stage and judge stage to avoid silently writing rows with repeated technical errors.

## Example: Local Proxy Run (5 rows)

```bash
uv run python -m scripts.evaluation.student_validation \
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

- `--judge-structured-retries` (default: `2`)
- `--judge-method` (`structured|g_eval`, default: `g_eval`)
- `--judge-g-eval-fallback-structured` (`true|false`, default: `true`)
- `--reasoning-effort` (`low|medium|high`, default: `high`)
- `--resume-existing` (default: `false`, append only missing judged rows)

`g_eval` mode applies only to open-ended answer types and keeps deterministic scoring for exact-match types.
It uses rubric-guided structured judging and can fall back to the standard structured judge on failure.
`llm_judge_output.jsonl` exposes this via `score_method` (`g_eval`, `structured_fallback`, etc.).

```bash
uv run python -m scripts.evaluation.llm_judge \
  --input-path scripts/evaluation/student_output.jsonl \
  --gold-path benchmark/benchmark_v1_0.jsonl \
  --judge-model "gpt-5.4-mini" \
  --reasoning-effort high \
  --judge-model-url "http://127.0.0.1:8317/v1" \
  --judge-api-key "ccs-internal-managed" \
  --judge-g-eval-fallback-structured true \
  --judge-structured-retries 2 \
  --output-path scripts/evaluation/llm_judge_output.jsonl
```

## Full Matrix Run

`run_full_matrix.py` accepts `--student-models` and alias `--models`.
It also accepts `--api-base-url` as an alternative to `--model-url`.
It supports `--resume-existing true|false` and forwards it to student + judge stages.

```bash
uv run python -m scripts.evaluation.run_full_matrix \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --api-key "ccs-internal-managed" \
  --models gpt-5.3-codex-spark \
  --judge-model gpt-5.4-mini \
  --agent-sandbox openshell \
  --agent-tools-profile minimal \
  --student-guard-enabled true \
  --student-guard-mode on_failure \
  --student-guard-retries 2 \
  --judge-g-eval-fallback-structured true \
  --judge-reasoning-effort high \
  --trace-log-enabled true \
  --trace-log-dir runs/debug_traces \
  --judge-structured-retries 2
```

## `benchmark_v3` Eval Workflow (Primary)

The `benchmark_v3` eval ladder lives in:

- `benchmark/level_a_eval.jsonl`
- `benchmark/level_b_eval.jsonl`
- `benchmark/level_c_eval.jsonl`

The current evaluation runtime still uses the legacy benchmark contract, so
materialize the eval subsets first:

```bash
uv run python -m scripts.evaluation.materialize_benchmark_v3_eval \
  --output benchmark/benchmark_v3_eval.jsonl
```

Then run the matrix evaluation against the materialized file:

```bash
uv run python -m scripts.evaluation.run_full_matrix \
  --benchmark-path benchmark/benchmark_v3_eval.jsonl \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --api-key "ccs-internal-managed" \
  --models gpt-5.4-mini \
  --judge-model gpt-5.4 \
  --judge-model-url "http://127.0.0.1:8317/v1" \
  --judge-api-key "ccs-internal-managed" \
  --judge-method g_eval \
  --judge-g-eval-fallback-structured true \
  --judge-reasoning-effort medium \
  --agent-sandbox openshell \
  --agent-tools-profile minimal \
  --trace-log-enabled true
```

Notes:

- `--agent-sandbox openshell` is the default and preferred runtime.
- `--agent-sandbox local` is the practical fallback when OpenShell is unavailable.
- Current `/v1/models` availability should be checked before long runs. In this environment
  the local `ccs` endpoint exposes `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex-spark`,
  `gpt-5.2-codex`, `gpt-5-codex-mini`, and related `gpt-5.x` variants.
- `benchmark/benchmark_v1_0.jsonl` remains supported as a legacy input, but
  new benchmark reporting should use `benchmark/benchmark_v3_eval.jsonl`.

## Output Contracts

Student output (`student_output.jsonl`) schema is stable:

- `exam_id`
- `page_id`
- `question_id`
- `level`
- `task_subtype`
- `difficulty`
- `question_type`
- `question_text`
- `answer_type`
- `student_answer`
- `student_status`
- `student_error`
- `student_elapsed_ms`
- `student_input_tokens`
- `student_output_tokens`
- `student_total_tokens`
- `student_reasoning_tokens`

Judge output (`llm_judge_output.jsonl`) now includes benchmark taxonomy metadata fields alongside the legacy scoring fields.
