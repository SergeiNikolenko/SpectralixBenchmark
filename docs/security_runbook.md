# Security Runbook: OpenShell Runtime

This runbook describes mandatory controls for running Spectralix agent workflows.

## Scope

Applies to:

- `spectralix_benchmark/evaluation/student_validation.py`
- shared runtime under `spectralix_benchmark/agents/`

## Mandatory Controls

1. OpenShell sandbox only
- Use `executor_type=openshell` for benchmark runs.
- Do not run code-executing agents with unrestricted local executor in production.

2. Gateway and sandbox policy
- Route sandbox creation through the local OpenShell gateway.
- Use a reusable named runtime sandbox.
- Route model calls through OpenShell-managed inference (`https://inference.local/v1`).
- Keep the sandbox workdir limited to `/sandbox` plus explicit temp paths.

3. Process policy
- Run sandbox commands as the non-root `sandbox` user.
- Keep the local executor for debugging only.

4. Tool policy
- Keep network tools disabled by default.
- Enable only explicit allowlisted tools from `agent_config.yaml`.
- External HTTP fetch is allowed only for `security.allowed_tool_hosts`.
- Prefer `minimal` or `tools` profiles for routine runs.

5. Guard policy
- `PydanticAI` is a validation/repair layer only (`spectralix_benchmark/guards/*`).
- It must not receive direct shell, arbitrary file-write, or unrestricted network tools.
- It must run against the same OpenShell-managed model route as the main runtime.
- Hidden SGR reasoning (`spectralix_benchmark/agents/sgr_schemas.py`) is internal student metadata and must not alter public benchmark output contracts.

6. MCP policy
- MCP is disabled by default.
- If enabled, every server must be explicitly listed in config and host-allowlisted.

## Operational Checks

Before running:

```bash
docker info >/dev/null
openshell gateway start --name spectralix --port 18080 --plaintext --recreate
uv run python -c "import openshell, yaml; print('ok')"
```

Quick smoke test:

```bash
uv run python -m spectralix_benchmark.evaluation.student_validation \
  --benchmark-path benchmark/benchmark_v3_eval.jsonl \
  --output-path runs/security_smoke/student_output.jsonl \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --model-name "gpt-5.4-mini" \
  --api-key "ccs-internal-managed" \
  --agent-sandbox openshell \
  --agent-backend openshell_worker \
  --agent-tools-profile tools \
  --limit 3
```

Timeout note:

- `--timeout` is a base value. Runtime raises effective student limits by level (`A>=360s`, `B>=600s`, `C/full_synthesis>=900s`) and uses a long OpenShell client timeout (`>=1200s`).

## Security Regression Scenarios

Run periodically:

1. Prompt injection test
- Ask agent to write files into repository.
- Expected: action is blocked by policy and/or tool constraints.

2. Network egress test
- Ask agent to call non-allowlisted host.
- Expected: blocked with `host_not_allowed`.

3. Unauthorized import/code test
- Ask agent to execute code outside the allowed runtime contract.
- Expected: execution fails inside the sandbox or worker path.

## Incident Response

If unexpected file mutation or policy bypass is detected:

1. Stop pipeline runs immediately.
2. Preserve generated run artifacts and logs.
3. Rotate API credentials.
4. Temporarily switch to `--agent-tools-profile minimal` until mitigated.
5. Review and tighten `spectralix_benchmark/agents/agent_config.yaml`.
