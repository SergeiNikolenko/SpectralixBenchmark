from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional
import os

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency for runtime init
    yaml = None


DEFAULT_AGENT_CONFIG: Dict[str, Any] = {
    "model": {
        "temperature": 0.2,
        "max_tokens": 768,
        "requests_per_minute": 0,
        "reasoning_effort": "high",
    },
    "runtime": {
        "backend": "openshell_worker",
        "add_base_tools": True,
        "use_structured_outputs_internally": True,
        "code_block_tags": "markdown",
        "additional_authorized_imports": [
            "math",
            "statistics",
            "json",
            "re",
            "collections",
            "itertools",
            "pathlib",
            "typing",
            "rdkit",
        ],
    },
    "security": {
        "allowed_tool_hosts": [],
        "allow_network_tools": False,
        "enforce_container_network_isolation": True,
    },
    "sandbox": {
        "executor_type": "openshell",
        "openshell": {
            "gateway_name": "spectralix",
            "gateway_port": 18080,
            "gateway_plaintext": True,
            "auto_start_gateway": True,
            "sandbox_name": "spectralix-runtime",
            "sandbox_from": "base",
            "ready_timeout_seconds": 180,
            "delete_on_close": False,
            "native_codex": {
                "sandbox_name": "spectralix-codex-runtime",
                "sandbox_from": "codex",
                "codex_bin": "codex",
                "codex_home_dir": "/sandbox/.codex",
                "upload_auth_from": "~/.codex/auth.json",
            },
        },
    },
    "tools": {
        "profiles": {
            "no_tools": [],
            "minimal": [],
            "tools": [
                "chem_format_tool",
                "smiles_sanity_tool",
                "unit_convert_tool",
                "rubric_hint_tool",
                "json_array_validate_tool",
                "chem_python_tool",
                "workspace_list_tool",
                "workspace_read_tool",
                "shell_exec_tool",
                "uv_run_tool",
            ],
            "tools_internet": [
                "chem_format_tool",
                "smiles_sanity_tool",
                "unit_convert_tool",
                "rubric_hint_tool",
                "json_array_validate_tool",
                "chem_python_tool",
                "workspace_list_tool",
                "workspace_read_tool",
                "shell_exec_tool",
                "uv_run_tool",
                "safe_http_get_tool",
            ],
            "full": [
                "chem_format_tool",
                "smiles_sanity_tool",
                "unit_convert_tool",
                "rubric_hint_tool",
                "json_array_validate_tool",
                "chem_python_tool",
                "workspace_list_tool",
                "workspace_read_tool",
                "workspace_write_tool",
                "shell_exec_tool",
                "uv_run_tool",
            ],
        },
        "mcp": {
            "enabled": False,
            "servers": [],
        },
    },
}

REQUIRED_CONFIG_SECTIONS = ("model", "runtime", "security", "sandbox", "tools")
SUPPORTED_EXECUTORS = {"local", "openshell"}
SUPPORTED_BACKENDS = {"local_worker", "openshell_worker", "codex_native"}
SUPPORTED_REASONING_EFFORTS = {"low", "medium", "high"}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = deepcopy(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _expand_env_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_value(v) for v in value]
    return value


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required to load agent config. Install with: pip install PyYAML")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Agent config must be a mapping: {path}")
    return raw


def _require_mapping(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Agent config section '{key}' must be a mapping")
    return value


def _validate_profiles(tools_section: Dict[str, Any]) -> None:
    profiles = tools_section.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Agent config tools.profiles must be a non-empty mapping")

    for profile_name, tool_names in profiles.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError("Agent config tool profile names must be non-empty strings")
        if not isinstance(tool_names, list):
            raise ValueError(f"Tool profile '{profile_name}' must be a list")
        for tool_name in tool_names:
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(f"Tool profile '{profile_name}' contains invalid tool name")


def validate_agent_config(config: Dict[str, Any]) -> Dict[str, Any]:
    for section in REQUIRED_CONFIG_SECTIONS:
        _require_mapping(config, section)

    model_cfg = _require_mapping(config, "model")
    runtime_cfg = _require_mapping(config, "runtime")
    reasoning_effort = str(model_cfg.get("reasoning_effort") or "").strip().lower()
    if reasoning_effort and reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            "Agent config model.reasoning_effort must be one of "
            f"{sorted(SUPPORTED_REASONING_EFFORTS)}"
        )
    backend = str(runtime_cfg.get("backend") or "").strip().lower()
    if backend and backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported runtime backend '{backend}'. Supported: {sorted(SUPPORTED_BACKENDS)}"
        )

    sandbox = _require_mapping(config, "sandbox")
    executor_type = str(sandbox.get("executor_type") or "").strip().lower()
    if not executor_type:
        raise ValueError("Agent config sandbox.executor_type is required")
    if executor_type not in SUPPORTED_EXECUTORS:
        raise ValueError(
            f"Unsupported sandbox executor_type '{executor_type}'. Supported: {sorted(SUPPORTED_EXECUTORS)}"
        )

    if executor_type == "openshell":
        openshell_cfg = sandbox.get("openshell")
        if not isinstance(openshell_cfg, dict):
            raise ValueError("Agent config sandbox.openshell must be a mapping for openshell executor")
        native_codex_cfg = openshell_cfg.get("native_codex", {})
        if native_codex_cfg and not isinstance(native_codex_cfg, dict):
            raise ValueError("Agent config sandbox.openshell.native_codex must be a mapping")

    effective_backend = resolve_runtime_backend(config, executor_type=executor_type)
    if executor_type == "local" and effective_backend != "local_worker":
        raise ValueError("Local executor supports only runtime.backend=local_worker")
    if executor_type == "openshell" and effective_backend not in {"openshell_worker", "codex_native"}:
        raise ValueError(
            "OpenShell executor supports only runtime.backend in {'openshell_worker', 'codex_native'}"
        )

    security = _require_mapping(config, "security")
    allowed_hosts = security.get("allowed_tool_hosts", [])
    if not isinstance(allowed_hosts, list):
        raise ValueError("Agent config security.allowed_tool_hosts must be a list")
    if any(not isinstance(host, str) for host in allowed_hosts):
        raise ValueError("Agent config security.allowed_tool_hosts must contain only strings")
    allow_network_tools = security.get("allow_network_tools", False)
    if not isinstance(allow_network_tools, bool):
        raise ValueError("Agent config security.allow_network_tools must be a boolean")
    enforce_network_isolation = security.get("enforce_container_network_isolation", True)
    if not isinstance(enforce_network_isolation, bool):
        raise ValueError("Agent config security.enforce_container_network_isolation must be a boolean")
    if allow_network_tools and not allowed_hosts:
        raise ValueError(
            "security.allow_network_tools=true requires non-empty security.allowed_tool_hosts"
        )

    tools = _require_mapping(config, "tools")
    _validate_profiles(tools)

    mcp_cfg = tools.get("mcp", {})
    if not isinstance(mcp_cfg, dict):
        raise ValueError("Agent config tools.mcp must be a mapping")
    servers = mcp_cfg.get("servers", [])
    if not isinstance(servers, list):
        raise ValueError("Agent config tools.mcp.servers must be a list")

    return config


def load_agent_config(config_path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_AGENT_CONFIG)

    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Agent config file not found: {path}")
        config = _deep_merge(config, _load_yaml_config(path))

    if overrides:
        config = _deep_merge(config, overrides)

    expanded = _expand_env_value(config)
    return validate_agent_config(expanded)


def build_executor_kwargs(config: Dict[str, Any], workspace_dir: Path) -> Dict[str, Any]:
    _ = workspace_dir
    sandbox = config.get("sandbox", {})
    executor_type = str(sandbox.get("executor_type") or "").strip().lower()
    if executor_type != "openshell":
        return {}

    openshell_cfg = sandbox.get("openshell", {})
    return {
        "gateway_name": str(openshell_cfg.get("gateway_name") or "spectralix"),
        "gateway_port": int(openshell_cfg.get("gateway_port") or 18080),
        "gateway_plaintext": bool(openshell_cfg.get("gateway_plaintext", True)),
        "auto_start_gateway": bool(openshell_cfg.get("auto_start_gateway", True)),
        "sandbox_name": str(openshell_cfg.get("sandbox_name") or "spectralix-runtime"),
        "sandbox_from": str(openshell_cfg.get("sandbox_from") or "base"),
        "ready_timeout_seconds": float(openshell_cfg.get("ready_timeout_seconds") or 180),
        "delete_on_close": bool(openshell_cfg.get("delete_on_close", False)),
        "native_codex": deepcopy(openshell_cfg.get("native_codex") or {}),
    }


def resolve_runtime_backend(
    config: Dict[str, Any],
    *,
    executor_type: str,
    requested_backend: Optional[str] = None,
) -> str:
    normalized_executor = str(executor_type or "").strip().lower()
    normalized_requested = str(requested_backend or "").strip().lower()
    if normalized_executor == "local" and not normalized_requested:
        return "local_worker"
    if normalized_requested:
        return normalized_requested
    configured = str(((config.get("runtime") or {}).get("backend") or "")).strip().lower()
    if normalized_executor == "local":
        return "local_worker"
    if configured:
        return configured
    return "openshell_worker"
