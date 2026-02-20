# Runtime Architecture

## Goals

- Keep benchmark contracts stable while enabling agentic execution.
- Enforce sandboxed execution for tool-capable model runs.
- Preserve deterministic scoring and reproducible evaluation outputs.

## High-Level Flow

1. `student_validation.py`
- Reads benchmark rows
- Produces `student_answer` per row
- Writes `student_output.jsonl`

2. `llm_judge.py`
- Joins student rows with canonical benchmark rows
- Applies deterministic scoring where possible
- Uses LLM judge for non-deterministic types
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
  - MCP tool loading (optional)
  - Runtime error normalization

## Security Model

- Sandbox: Docker (`executor_type=docker`)
- Workspace mount: read-only
- Privilege drop: non-root, `cap_drop=ALL`, `no-new-privileges`
- Tool policy:
  - shell/file-write tools disabled by default
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
- Added aliases for operational compatibility:
  - `--api-base-url`
  - `--models` (alias for `--student-models` in matrix runner)
