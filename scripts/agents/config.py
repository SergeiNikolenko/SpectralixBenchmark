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
        ],
    },
    "security": {
        "allowed_tool_hosts": [],
        "allow_network_tools": False,
        "enforce_container_network_isolation": True,
    },
    "sandbox": {
        "executor_type": "docker",
        "docker": {
            "image_name": "spectralix-smolagent-sandbox",
            "build_new_image": False,
            "mount_workspace_readonly": False,
            "host": "127.0.0.1",
            "port": 8888,
            "container_run_kwargs": {
                "read_only": True,
                "user": "65534:65534",
                "mem_limit": "768m",
                "nano_cpus": 1_000_000_000,
                "pids_limit": 128,
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges"],
                "tmpfs": {
                    "/tmp": "rw,noexec,nosuid,size=128m",
                },
            },
        },
    },
    "tools": {
        "profiles": {
            "minimal": [
                "chem_format_tool",
                "smiles_sanity_tool",
                "unit_convert_tool",
                "rubric_hint_tool",
                "json_array_validate_tool",
            ],
            "full": [
                "chem_format_tool",
                "smiles_sanity_tool",
                "unit_convert_tool",
                "rubric_hint_tool",
                "json_array_validate_tool",
                "safe_http_get_tool",
            ],
        },
        "mcp": {
            "enabled": False,
            "servers": [],
        },
    },
}

REQUIRED_CONFIG_SECTIONS = ("model", "runtime", "security", "sandbox", "tools")
SUPPORTED_EXECUTORS = {"local", "docker"}
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
    reasoning_effort = str(model_cfg.get("reasoning_effort") or "").strip().lower()
    if reasoning_effort and reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            "Agent config model.reasoning_effort must be one of "
            f"{sorted(SUPPORTED_REASONING_EFFORTS)}"
        )

    sandbox = _require_mapping(config, "sandbox")
    executor_type = str(sandbox.get("executor_type") or "").strip().lower()
    if not executor_type:
        raise ValueError("Agent config sandbox.executor_type is required")
    if executor_type not in SUPPORTED_EXECUTORS:
        raise ValueError(
            f"Unsupported sandbox executor_type '{executor_type}'. Supported: {sorted(SUPPORTED_EXECUTORS)}"
        )

    if executor_type == "docker":
        docker_cfg = sandbox.get("docker")
        if not isinstance(docker_cfg, dict):
            raise ValueError("Agent config sandbox.docker must be a mapping for docker executor")

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
    sandbox = config.get("sandbox", {})
    docker_cfg = sandbox.get("docker", {})

    container_run_kwargs = deepcopy(docker_cfg.get("container_run_kwargs", {}))
    security_cfg = config.get("security", {})
    allow_network_tools = bool(security_cfg.get("allow_network_tools", False))
    enforce_network_isolation = bool(security_cfg.get("enforce_container_network_isolation", True))
    for key, default_value in (
        ("read_only", True),
        ("user", "65534:65534"),
        ("cap_drop", ["ALL"]),
        ("security_opt", ["no-new-privileges"]),
    ):
        container_run_kwargs.setdefault(key, default_value)

    volumes = dict(container_run_kwargs.get("volumes") or {})
    if bool(docker_cfg.get("mount_workspace_readonly", False)):
        volumes[str(workspace_dir.resolve())] = {"bind": "/workspace", "mode": "ro"}
    if volumes:
        container_run_kwargs["volumes"] = volumes
    else:
        container_run_kwargs.pop("volumes", None)
    if enforce_network_isolation:
        container_run_kwargs["network_disabled"] = not allow_network_tools
        if not allow_network_tools:
            container_run_kwargs.pop("network_mode", None)

    executor_kwargs: Dict[str, Any] = {
        "host": docker_cfg.get("host") or "127.0.0.1",
        "port": int(docker_cfg.get("port") or 8888),
        "image_name": docker_cfg.get("image_name", "spectralix-smolagent-sandbox"),
        "build_new_image": bool(docker_cfg.get("build_new_image", False)),
        "container_run_kwargs": container_run_kwargs,
    }

    dockerfile_content = docker_cfg.get("dockerfile_content")
    if dockerfile_content:
        executor_kwargs["dockerfile_content"] = dockerfile_content

    return executor_kwargs
