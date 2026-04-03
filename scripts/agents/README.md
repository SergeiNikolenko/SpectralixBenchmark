# Agent Runtime Package

This package contains the production runtime used by evaluation and parsing pipelines.

## Modules

- `runtime.py`: high-level orchestration and error mapping
- `config.py`: default config + YAML loading + OpenShell executor settings
- `models.py`: OpenAI-compatible model URL handling and managed-inference settings
- `tool_registry.py`: custom tool definitions and tool profile assembly
- `openshell_manager.py`: gateway, provider, reusable sandbox, and managed inference lifecycle
- `openshell_worker.py`: sandbox-local chat loop and tool execution against `https://inference.local/v1`
- `prompts.py`: task prompt builders for student and parser stages
- `agent_config.yaml`: default runtime/sandbox/tool policy

## Design Rules

- Keep output contracts unchanged at pipeline boundaries.
- Fail fast on invalid runtime configuration.
- Default to OpenShell sandbox execution.
- Allow tools only through explicit profile selection and host allowlists.
- Keep container egress disabled by default (`security.allow_network_tools: false`).
- Do not pass benchmark identifiers/paths into student prompts.
- Keep the local executor as a debugging fallback only.
- Keep `PydanticAI` as a typed guard layer above the runtime.
