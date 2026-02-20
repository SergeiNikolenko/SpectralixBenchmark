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
    },
    "runtime": {
        "add_base_tools": True,
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
        "allow_shell_tools": False,
        "allow_file_write_tools": False,
    },
    "sandbox": {
        "executor_type": "docker",
        "docker": {
            "image_name": "spectralix-smolagent-sandbox",
            "build_new_image": True,
            "docker_host": None,
            "docker_port": None,
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
                "benchmark_lookup_tool",
            ],
            "full": [
                "chem_format_tool",
                "smiles_sanity_tool",
                "unit_convert_tool",
                "rubric_hint_tool",
                "json_array_validate_tool",
                "benchmark_lookup_tool",
                "safe_http_get_tool",
            ],
        },
        "mcp": {
            "enabled": False,
            "servers": [],
        },
    },
}


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


def load_agent_config(config_path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_AGENT_CONFIG)

    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Agent config file not found: {path}")
        config = _deep_merge(config, _load_yaml_config(path))

    if overrides:
        config = _deep_merge(config, overrides)

    return _expand_env_value(config)


def build_executor_kwargs(config: Dict[str, Any], workspace_dir: Path) -> Dict[str, Any]:
    sandbox = config.get("sandbox", {})
    docker_cfg = sandbox.get("docker", {})

    container_run_kwargs = deepcopy(docker_cfg.get("container_run_kwargs", {}))
    for key, default_value in (
        ("read_only", True),
        ("user", "65534:65534"),
        ("cap_drop", ["ALL"]),
        ("security_opt", ["no-new-privileges"]),
    ):
        container_run_kwargs.setdefault(key, default_value)

    volumes = dict(container_run_kwargs.get("volumes") or {})
    volumes[str(workspace_dir.resolve())] = {"bind": "/workspace", "mode": "ro"}
    container_run_kwargs["volumes"] = volumes

    executor_kwargs: Dict[str, Any] = {
        "docker_host": docker_cfg.get("docker_host"),
        "docker_port": docker_cfg.get("docker_port"),
        "image_name": docker_cfg.get("image_name", "spectralix-smolagent-sandbox"),
        "build_new_image": bool(docker_cfg.get("build_new_image", True)),
        "container_run_kwargs": container_run_kwargs,
    }

    dockerfile_content = docker_cfg.get("dockerfile_content")
    if dockerfile_content:
        executor_kwargs["dockerfile_content"] = dockerfile_content

    return executor_kwargs
