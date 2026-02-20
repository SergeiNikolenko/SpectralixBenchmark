# Security Runbook: smolagents Docker Sandbox

This runbook describes mandatory controls for running Spectralix agent workflows.

## Scope

Applies to:

- `scripts/evaluation/student_validation.py`
- `scripts/parsing/exam-parser-pipeline.py`
- shared runtime under `scripts/agents/`

## Mandatory Controls

1. Docker sandbox only
- Use `executor_type=docker`.
- Do not run code-executing agents with unrestricted local executor in production.

2. Container hardening
- `read_only: true`
- non-root user (`65534:65534`)
- `cap_drop: [ALL]`
- `security_opt: ["no-new-privileges"]`
- memory / CPU / pids limits
- writable `tmpfs` only for `/tmp`

3. Workspace policy
- Mount repository into container as read-only.
- Do not expose write mounts for source code paths.

4. Tool policy
- Keep shell/file-write tools disabled.
- Enable only explicit allowlisted tools from `agent_config.yaml`.
- External HTTP fetch is allowed only for `security.allowed_tool_hosts`.

5. MCP policy
- MCP is disabled by default.
- If enabled, every server must be explicitly listed in config and host-allowlisted.

## Operational Checks

Before running:

```bash
docker info >/dev/null
python3 -c "import smolagents, yaml; print('ok')"
```

Quick smoke test:

```bash
python3 scripts/evaluation/student_validation.py \
  --benchmark-path benchmark/benchmark_v1_0.jsonl \
  --output-path scripts/evaluation/student_output_smoke.jsonl \
  --api-base-url "http://127.0.0.1:8317/v1" \
  --model-name "gpt-4o-mini" \
  --api-key "ccs-internal-managed" \
  --limit 3
```

## Security Regression Scenarios

Run periodically:

1. Prompt injection test
- Ask agent to write files into repository.
- Expected: action is blocked by policy and/or sandbox constraints.

2. Network egress test
- Ask agent to call non-allowlisted host.
- Expected: blocked with `host_not_allowed`.

3. Unauthorized import/code test
- Ask agent to import privileged modules not in authorized imports.
- Expected: execution blocked by runtime restrictions.

## Incident Response

If unexpected file mutation or policy bypass is detected:

1. Stop pipeline runs immediately.
2. Preserve generated run artifacts and logs.
3. Rotate API credentials.
4. Set `--agent-enabled false` and use `student_validation_legacy.py` until mitigated.
5. Review and tighten `scripts/agents/agent_config.yaml`.
