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

## Agent Layer (`spectralix_benchmark/agents/`)

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
  - Student task prompt builders
  - Hidden SGR prompt builder for structured reasoning phase

- `runtime.py`
  - `AgentRuntime` orchestration
  - One-time OpenShell sandbox preflight before execution
  - Local or OpenShell worker execution
  - Runtime error normalization
  - Effective timeout scaling by benchmark level/subtype

- `openshell_manager.py`
  - Gateway health checks
  - Provider + `inference.local` route configuration
  - Sandbox lifecycle management
  - Worker execution inside the sandbox
  - Long client timeout configuration for extended benchmark calls

- `openshell_worker.py`
  - Two-phase student flow:
    - Hidden SGR schema generation and validation
    - Final benchmark answer generation from compact validated SGR context
  - OpenAI-compatible chat loop inside the sandbox against `https://inference.local/v1`
  - Tool invocation and step capture

- `sgr_schemas.py`
  - Level A/B/C generic schemas and subtype-specific schema variants
  - `level/task_subtype -> schema` selector
  - Schema validation and compact SGR snapshot helpers

## Hidden SGR Student Flow

Student mode uses a real schema-level reasoning stage, not prompt-only guidance:

1. Select schema from benchmark `level` + `task_subtype`.
2. Generate hidden SGR JSON.
3. Validate against the selected Pydantic schema.
4. Attempt one repair pass if validation fails.
5. Fall back to direct final-answer path only when repair also fails.
6. Generate final benchmark-aligned answer using compact validated SGR context.

Public benchmark outputs stay unchanged:

- `student_output.jsonl` enriched with taxonomy and contract metadata
- Judge interfaces unchanged
- SGR metadata is exposed only in debug/verbose artifacts

## Guard Layer (`spectralix_benchmark/guards/`)

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
- `http_error`
- `auth_error`
- `parse_error`
- `agent_step_error`
- `sandbox_error`

`connection_error` may still appear in legacy artifacts, but current runtime normalization maps most connection/network failures to `http_error`.

These status values are persisted to `student_status` in output JSONL.

## Effective Timeout Scaling

CLI `--timeout` is treated as a base timeout. Student calls are elevated to minimum floors:

- Level A/default student tasks: `>= 360s`
- Level B text/precursor/disconnection tasks: `>= 600s`
- Level C or `full_synthesis`: `>= 900s`

OpenShell SDK client timeout is also raised to at least `1200s`.

## Compatibility Guarantees

- `student_output.jsonl` enriched with taxonomy and contract metadata
- `llm_judge_output.jsonl` enriched with taxonomy metadata and used as the primary taxonomy-aware result row format
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
