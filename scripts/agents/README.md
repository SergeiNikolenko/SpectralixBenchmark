# Agent Runtime Package

This package contains the production runtime used by evaluation and parsing pipelines.

## Modules

- `runtime.py`: high-level orchestration and error mapping
- `config.py`: default config + YAML loading + Docker executor kwargs
- `models.py`: OpenAI-compatible model URL and auth handling
- `tool_registry.py`: custom tool definitions and tool profile assembly
- `prompts.py`: task prompt builders for student and parser stages
- `agent_config.yaml`: default runtime/sandbox/tool policy
- `pydantic_guard` block in config: defaults for student/judge/parser structured guards

## Design Rules

- Keep output contracts unchanged at pipeline boundaries.
- Fail fast on invalid runtime configuration.
- Default to Docker sandbox with restricted privileges.
- Allow tools only through explicit profile selection and host allowlists.
- Reuse one initialized `CodeAgent` session per runtime instance to reduce overhead.
- Keep container egress disabled by default (`security.allow_network_tools: false`).
- Do not pass benchmark identifiers/paths into student prompts.
- Keep benchmark lookup tooling disabled in default student tool profiles.
- Do not mount workspace into Docker sandbox unless explicitly required.
- Keep `smolagents` as orchestration runtime; use `PydanticAI` only as a typed guard layer.
