# Runtime Architecture

## Goals

- Keep benchmark contracts stable while enabling agentic execution.
- Enforce sandboxed execution for tool-capable model runs.
- Preserve deterministic scoring and reproducible evaluation outputs.
- Keep a single production runtime for student-stage inference (no legacy backend).
- Keep a single OpenShell-backed runtime while adding strict structured guards via `PydanticAI`.

## High-Level Flow

1. `student_validation.py`
- Reads benchmark rows
- Produces `student_answer` per row via `AgentRuntime`
- Applies optional `PydanticAI` student guard (`on_failure|always|off`)
- Writes `student_output.jsonl`

2. `llm_judge.py`
- Joins student rows with canonical benchmark rows
- Applies deterministic scoring where possible
- Uses structured `PydanticAI` judge for non-deterministic types
- Writes `llm_judge_output.jsonl`

3. `run_full_matrix.py`
- Runs student inference + judge across multiple models
- Produces per-model metrics and run summary artifacts

## Agent Layer (`scripts/agents/`)

- `config.py`
  - Default config
  - YAML loading and merge
  - OpenShell executor settings assembly

- `models.py`
  - OpenAI-compatible URL normalization
  - API key resolution
  - OpenShell managed inference base selection (`https://inference.local/v1`)
  - Upstream base URL rewriting for host-side provider configuration

- `tool_registry.py`
  - Explicit tool set
  - Tool profile resolution
  - Security-aware tool enablement (allowlist, network-tool gating)

- `prompts.py`
  - Student and parser task prompt builders

- `runtime.py`
  - `AgentRuntime` orchestration
  - One-time OpenShell sandbox preflight before execution
  - Local or OpenShell worker execution
  - Runtime error normalization

- `openshell_manager.py`
  - Gateway health checks
  - Provider + `inference.local` route configuration
  - Sandbox lifecycle management
  - Worker execution inside the sandbox

- `openshell_worker.py`
  - OpenAI-compatible chat loop inside the sandbox against `https://inference.local/v1`
  - Tool invocation and step capture

## Guard Layer (`scripts/pydantic_guard/`)

- `models.py`
  - OpenAI-compatible model builder for PydanticAI
- `schemas.py`
  - Strict schemas: `JudgeResult`, `StudentGuardOutput`, `ParsedQuestionSchema`
- `judge_structured.py`
  - Structured judge execution + retry
- `judge_geval.py`
  - rubric-guided G-Eval judge execution + retry
- `student_guard.py`
  - Answer-format validation + guard repair execution
- `parser_repair.py`
  - Structured parser output repair into schema-valid question lists
- `retry.py`
  - Shared retry helper for guard calls

## Security Model

- Sandbox: OpenShell (`executor_type=openshell`)
- Workspace: sandbox workdir only (`/sandbox`)
- Inference:
  - host-side OpenShell provider points at the upstream OpenAI-compatible endpoint
  - sandbox-visible client traffic uses `https://inference.local/v1`
- Process policy:
  - non-root sandbox user inside the runtime image
  - local execution fallback only for debugging
- Tool policy:
  - local tools selected by profile
  - outbound HTTP restricted by host allowlist
  - network tools disabled by default
  - MCP disabled by default

See `docs/security_runbook.md` for operational controls.

## Error Contract

Student stage status values include:

- `ok`
- `timeout`
- `connection_error`
- `http_error`
- `auth_error`
- `parse_error`
- `agent_step_error`
- `sandbox_error`

These values are persisted to `student_status` in output JSONL.

## Compatibility Guarantees

- `student_output.jsonl` schema unchanged
- `llm_judge_output.jsonl` schema unchanged
- Existing `--model-url` supported
- Added aliases for operational compatibility:
  - `--api-base-url`
  - `--models` (alias for `--student-models` in matrix runner)
  - `--resume-existing` (student/judge/matrix) for append-and-continue runs

## Judge Mode Semantics

- Deterministic answer types are scored without LLM.
- Open-ended answer types can use:
  - `g_eval` (rubric-guided),
  - fallback to structured judge when `--judge-g-eval-fallback-structured=true`.
- `llm_judge_output.jsonl` exposes score source in `score_method` to disambiguate:
  - `deterministic`
  - `g_eval`
  - `structured_fallback`
  - `llm_judge`

## Related Docs

- `docs/g_eval.md`
- `docs/security_runbook.md`
