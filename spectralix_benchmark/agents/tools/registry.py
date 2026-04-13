from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .chem import chem_python_tool
from .network import safe_http_get_tool
from .workspace import shell_exec_tool, uv_run_tool, workspace_list_tool, workspace_read_tool, workspace_write_tool


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    func: Callable[..., str]
    description: str
    schema: Dict[str, Any]
    is_network_tool: bool = False


def _object_schema(*, properties: Dict[str, Dict[str, Any]], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOL_DEFINITIONS: Dict[str, ToolDefinition] = {
    "chem_python_tool": ToolDefinition(
        name="chem_python_tool",
        func=chem_python_tool,
        description="Run a short Python chemistry validation snippet and capture stdout and stderr.",
        schema=_object_schema(
            properties={
                "code": {"type": "string"},
                "timeout_sec": {"type": "integer", "minimum": 1},
            },
            required=["code"],
        ),
    ),
    "workspace_list_tool": ToolDefinition(
        name="workspace_list_tool",
        func=workspace_list_tool,
        description="List files and directories in the uploaded workspace snapshot.",
        schema=_object_schema(
            properties={
                "path": {"type": "string"},
                "max_entries": {"type": "integer", "minimum": 1},
            },
            required=[],
        ),
    ),
    "workspace_read_tool": ToolDefinition(
        name="workspace_read_tool",
        func=workspace_read_tool,
        description="Read a UTF-8 text file from the uploaded workspace snapshot.",
        schema=_object_schema(
            properties={
                "path": {"type": "string"},
                "max_bytes": {"type": "integer", "minimum": 1},
            },
            required=["path"],
        ),
    ),
    "workspace_write_tool": ToolDefinition(
        name="workspace_write_tool",
        func=workspace_write_tool,
        description="Write or append a UTF-8 text file inside the uploaded workspace snapshot.",
        schema=_object_schema(
            properties={
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["overwrite", "append"]},
            },
            required=["path", "content"],
        ),
    ),
    "shell_exec_tool": ToolDefinition(
        name="shell_exec_tool",
        func=shell_exec_tool,
        description="Run a short shell command inside the sandbox workspace and capture stdout and stderr.",
        schema=_object_schema(
            properties={
                "command": {"type": "string"},
                "timeout_sec": {"type": "integer", "minimum": 1},
                "workdir": {"type": "string"},
            },
            required=["command"],
        ),
    ),
    "uv_run_tool": ToolDefinition(
        name="uv_run_tool",
        func=uv_run_tool,
        description="Run uv inside the sandbox workspace using the runtime virtual environment.",
        schema=_object_schema(
            properties={
                "args": {"type": "string"},
                "timeout_sec": {"type": "integer", "minimum": 1},
                "workdir": {"type": "string"},
            },
            required=["args"],
        ),
    ),
    "safe_http_get_tool": ToolDefinition(
        name="safe_http_get_tool",
        func=safe_http_get_tool,
        description="Fetch a trusted HTTP(S) URL when the hostname is allowlisted.",
        schema=_object_schema(
            properties={
                "url": {"type": "string"},
                "timeout_sec": {"type": "integer", "minimum": 1},
            },
            required=["url"],
        ),
        is_network_tool=True,
    ),
}


TOOL_REGISTRY = {name: item.func for name, item in TOOL_DEFINITIONS.items()}


def build_tools(profile: str, config: Dict[str, Any]) -> List[Any]:
    profiles = ((config.get("tools") or {}).get("profiles") or {})
    if profile not in profiles:
        known_profiles = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"Unknown tools profile '{profile}'. Known profiles: {known_profiles}")

    selected = profiles[profile]
    security = config.get("security") or {}
    has_allowed_hosts = bool(security.get("allowed_tool_hosts"))
    allow_network_tools = bool(security.get("allow_network_tools", False))
    tools: List[Any] = []

    for name in selected:
        tool_definition = TOOL_DEFINITIONS.get(name)
        if tool_definition is None:
            raise ValueError(f"Unknown tool in profile '{profile}': {name}")
        if tool_definition.is_network_tool and (not has_allowed_hosts or not allow_network_tools):
            continue
        tools.append(tool_definition.func)
    return tools


def build_tool_definitions(profile: str, config: Dict[str, Any]) -> List[ToolDefinition]:
    profiles = ((config.get("tools") or {}).get("profiles") or {})
    if profile not in profiles:
        known_profiles = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"Unknown tools profile '{profile}'. Known profiles: {known_profiles}")

    selected = profiles[profile]
    security = config.get("security") or {}
    has_allowed_hosts = bool(security.get("allowed_tool_hosts"))
    allow_network_tools = bool(security.get("allow_network_tools", False))
    tool_definitions: List[ToolDefinition] = []

    for name in selected:
        tool_definition = TOOL_DEFINITIONS.get(name)
        if tool_definition is None:
            raise ValueError(f"Unknown tool in profile '{profile}': {name}")
        if tool_definition.is_network_tool and (not has_allowed_hosts or not allow_network_tools):
            continue
        tool_definitions.append(tool_definition)
    return tool_definitions

