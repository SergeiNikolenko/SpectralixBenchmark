# Runtime Architecture

## Goals

- Keep benchmark contracts stable while enabling agentic execution.
- Enforce sandboxed execution for tool-capable model runs.
- Preserve deterministic scoring and reproducible evaluation outputs.
- Keep a single production runtime for student-stage inference (no legacy backend).
- Keep `smolagents` as the base runtime while adding strict structured guards via `PydanticAI`.

## High-Level Flow

1. `student_validation.py`
- Reads benchmark rows
- Produces `student_answer` per row via `smolagents`
- Applies optional `PydanticAI` student guard (`on_failure|always|off`)
- Writes `student_output.jsonl`

2. `llm_judge.py`
- Joins student rows with canonical benchmark rows
- Applies deterministic scoring where possible
- Uses structured `PydanticAI` judge for non-deterministic types
- Optionally falls back to legacy JSON parse for judge on structured failure
- Writes `llm_judge_output.jsonl`

3. `run_full_matrix.py`
- Runs student inference + judge across multiple models
- Produces per-model metrics and run summary artifacts

## Agent Layer (`scripts/agents/`)

- `config.py`
  - Default config
  - YAML loading and merge
  - Docker executor kwargs assembly

- `models.py`
  - OpenAI-compatible URL normalization
  - API key resolution
  - `OpenAIModel` factory for `smolagents`

- `tool_registry.py`
  - Explicit tool set
  - Tool profile resolution
  - Security-aware tool enablement (allowlist)

- `prompts.py`
  - Student and parser task prompt builders

- `runtime.py`
  - `AgentRuntime` orchestration
  - Sandbox execution via `CodeAgent`
  - One-time Docker preflight before execution
  - Reused agent session across questions/pages
  - MCP tool loading (optional)
  - Runtime error normalization

## Guard Layer (`scripts/pydantic_guard/`)

- `models.py`
  - OpenAI-compatible model builder for PydanticAI
- `schemas.py`
  - Strict schemas: `JudgeResult`, `StudentGuardOutput`, `ParsedQuestionSchema`
- `judge_structured.py`
  - Structured judge execution + retry
- `student_guard.py`
  - Answer-format validation + guard repair execution
- `parser_repair.py`
  - Structured parser output repair into schema-valid question lists
- `retry.py`
  - Shared retry helper for guard calls

## Security Model

- Sandbox: Docker (`executor_type=docker`)
- Workspace mount: read-only
- Privilege drop: non-root, `cap_drop=ALL`, `no-new-privileges`
- Tool policy:
  - shell/file-write tools disabled by default
  - container network disabled by default
  - outbound HTTP restricted by host allowlist
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
- `smolagents` remains the only orchestration/runtime layer (not replaced by `PydanticAI`)
- Added aliases for operational compatibility:
  - `--api-base-url`
  - `--models` (alias for `--student-models` in matrix runner)
