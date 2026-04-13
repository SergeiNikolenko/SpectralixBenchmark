from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List
import json
import subprocess
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import os
import shlex


def _normalize_answer_type(answer_type: str) -> str:
    return (answer_type or "").strip().lower()


def _extract_answer_payload(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""

    text = text.replace("```", "")
    text = re.sub(r"`([^`]*)`", r"\1", text)
    answer_line = re.search(r"(?im)^\s*answer\s*:\s*(.+)$", text)
    if answer_line:
        return answer_line.group(1).strip()
    return text


def _normalize_sequence(text: str) -> str:
    sanitized = (text or "").replace("\n", ";").replace(",", ";").replace("|", ";")
    tokens: List[str] = []
    for raw_token in sanitized.split(";"):
        token = raw_token.strip()
        if not token:
            continue
        token = re.sub(r"^\d+[\.)]\s*", "", token)
        if token:
            tokens.append(token)
    return "; ".join(tokens)


def chem_format_tool(answer_type: str, raw_text: str) -> str:
    """
    Normalizes raw model output into benchmark-friendly format by answer_type.

    Args:
        answer_type: Benchmark answer type.
        raw_text: Raw model answer text.
    """
    payload = _extract_answer_payload(raw_text)
    if not payload:
        return ""

    normalized_type = _normalize_answer_type(answer_type)

    if normalized_type == "single_choice":
        tokens = re.findall(r"[A-Za-z0-9]+", payload)
        return tokens[0] if tokens else payload.strip()

    if normalized_type in {"multiple_choice", "ordering"}:
        seq = _normalize_sequence(payload)
        return seq if seq else payload.strip()

    if normalized_type in {"msms_structure_prediction", "structure"}:
        for line in payload.splitlines():
            candidate = line.strip()
            if candidate:
                return candidate
        return payload.strip()

    return re.sub(r"\s+", " ", payload).strip()


def smiles_sanity_tool(smiles: str) -> str:
    """
    Performs lightweight sanity checks for a SMILES candidate string.

    Args:
        smiles: Candidate SMILES string.
    """
    candidate = (smiles or "").strip().replace(" ", "")
    if not candidate:
        return json.dumps({"valid": False, "reason": "empty"})

    allowed_pattern = r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+$"
    if not re.match(allowed_pattern, candidate):
        return json.dumps({"valid": False, "reason": "illegal_characters", "normalized": candidate})

    if candidate.count("(") != candidate.count(")"):
        return json.dumps({"valid": False, "reason": "unbalanced_parentheses", "normalized": candidate})

    if candidate.count("[") != candidate.count("]"):
        return json.dumps({"valid": False, "reason": "unbalanced_brackets", "normalized": candidate})

    return json.dumps({"valid": True, "normalized": candidate})


def unit_convert_tool(value: float, from_unit: str, to_unit: str) -> str:
    """
    Converts between selected chemistry units.

    Args:
        value: Numeric value to convert.
        from_unit: Source unit (g, mg, kg, L, mL, uL, mol, mmol, umol).
        to_unit: Target unit.
    """
    src = (from_unit or "").strip().lower()
    dst = (to_unit or "").strip().lower()

    scales = {
        "g": 1.0,
        "mg": 1e-3,
        "kg": 1e3,
        "l": 1.0,
        "ml": 1e-3,
        "ul": 1e-6,
        "mol": 1.0,
        "mmol": 1e-3,
        "umol": 1e-6,
    }

    groups = {
        "mass": {"g", "mg", "kg"},
        "volume": {"l", "ml", "ul"},
        "amount": {"mol", "mmol", "umol"},
    }

    def group_of(u: str) -> str:
        for g_name, members in groups.items():
            if u in members:
                return g_name
        return ""

    src_group = group_of(src)
    dst_group = group_of(dst)

    if not src_group or not dst_group:
        return json.dumps({"status": "error", "reason": "unsupported_unit"})
    if src_group != dst_group:
        return json.dumps({"status": "error", "reason": "incompatible_unit_groups"})

    base_value = float(value) * scales[src]
    converted = base_value / scales[dst]

    return json.dumps(
        {
            "status": "ok",
            "value": converted,
            "from_unit": src,
            "to_unit": dst,
        }
    )


def rubric_hint_tool(answer_type: str) -> str:
    """
    Returns concise benchmark scoring hints by answer type.

    Args:
        answer_type: Benchmark answer type.
    """
    a = _normalize_answer_type(answer_type)
    hints = {
        "single_choice": "One label only (A/B/C/...).",
        "multiple_choice": "List labels separated by '; ' with no extra prose.",
        "ordering": "Return full order as '; '-separated tokens.",
        "numeric": "Return only the numeric value with unit only if required.",
        "structure": "Return one structure representation (prefer SMILES) without explanation.",
        "msms_structure_prediction": "Return one canonical SMILES candidate.",
        "full_synthesis": "Prioritize correct reagents, sequence, and key intermediates.",
        "reaction_description": "State transformation, reagents, and selectivity succinctly.",
        "property_determination": "Report derived property and minimal supporting rationale.",
        "text": "Lead with machine-readable answer, then short explanation if necessary.",
    }
    return hints.get(a, "Produce a precise, machine-readable answer with minimal extra text.")


def json_array_validate_tool(raw_json: str) -> str:
    """
    Validates that input is a JSON array and returns normalized compact JSON.

    Args:
        raw_json: Candidate JSON array string.
    """
    payload = (raw_json or "").strip()
    if not payload:
        return json.dumps({"valid": False, "reason": "empty"})

    # Allow markdown fences.
    payload = payload.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", payload)
        if not match:
            return json.dumps({"valid": False, "reason": "json_parse_error"})
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, list):
        return json.dumps({"valid": False, "reason": "not_array"})

    return json.dumps({"valid": True, "normalized": json.dumps(parsed, ensure_ascii=False)})


def chem_python_tool(code: str, timeout_sec: int = 20) -> str:
    """
    Runs a short Python snippet inside the project uv environment.
    Useful for chemistry validation, RDKit canonicalization, and small calculations.

    Args:
        code: Python code to execute. Print the final result to stdout.
        timeout_sec: Hard timeout in seconds.
    """
    snippet = (code or "").strip()
    if not snippet:
        return json.dumps({"status": "error", "reason": "empty_code"})

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.run(
            [os.environ.get("PYTHON_BIN", "python"), "-c", snippet],
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "error", "reason": "timeout"})
    except Exception as exc:
        return json.dumps({"status": "error", "reason": f"python_tool_error:{exc}"})

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    return json.dumps(
        {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "stdout": stdout[:12000],
            "stderr": stderr[:12000],
        },
        ensure_ascii=False,
    )


def _workspace_root() -> Path:
    return Path(os.getenv("AGENT_WORKSPACE_ROOT") or "/sandbox/workspace").resolve()


def _runtime_bin_dirs() -> List[str]:
    entries = [os.getenv("AGENT_UV_BIN") or "", os.getenv("PYTHON_BIN") or ""]
    dirs: List[str] = []
    for entry in entries:
        if not entry:
            continue
        parent = str(Path(entry).resolve().parent)
        if parent not in dirs:
            dirs.append(parent)
    return dirs


def _resolve_workspace_path(raw_path: str) -> Path:
    root = _workspace_root()
    candidate = (raw_path or "").strip()
    if not candidate:
        return root
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path_outside_workspace:{candidate}") from exc
    return resolved


def workspace_list_tool(path: str = ".", max_entries: int = 200) -> str:
    """
    List files and directories under the uploaded workspace snapshot.

    Args:
        path: Relative path under the workspace root.
        max_entries: Maximum number of entries to return.
    """
    try:
        target = _resolve_workspace_path(path)
    except Exception as exc:
        return json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False)

    if not target.exists():
        return json.dumps({"status": "error", "reason": "not_found"}, ensure_ascii=False)
    if not target.is_dir():
        return json.dumps({"status": "error", "reason": "not_directory"}, ensure_ascii=False)

    entries: List[Dict[str, Any]] = []
    limit = max(1, min(int(max_entries), 500))
    root = _workspace_root()
    for item in sorted(target.iterdir(), key=lambda value: (not value.is_dir(), value.name.lower()))[:limit]:
        relative = str(item.relative_to(root))
        try:
            size = item.stat().st_size
        except OSError:
            size = None
        entries.append(
            {
                "path": relative,
                "type": "dir" if item.is_dir() else "file",
                "size": size,
            }
        )

    return json.dumps({"status": "ok", "entries": entries}, ensure_ascii=False)


def workspace_read_tool(path: str, max_bytes: int = 12000) -> str:
    """
    Read a UTF-8 text file from the uploaded workspace snapshot.

    Args:
        path: Relative path under the workspace root.
        max_bytes: Maximum number of bytes to read from the file.
    """
    try:
        target = _resolve_workspace_path(path)
    except Exception as exc:
        return json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False)

    if not target.exists():
        return json.dumps({"status": "error", "reason": "not_found"}, ensure_ascii=False)
    if not target.is_file():
        return json.dumps({"status": "error", "reason": "not_file"}, ensure_ascii=False)

    payload = target.read_text(encoding="utf-8", errors="replace")
    clipped = payload[: max(1, int(max_bytes))]
    return json.dumps(
        {
            "status": "ok",
            "path": str(target.relative_to(_workspace_root())),
            "content": clipped,
            "truncated": len(clipped) < len(payload),
        },
        ensure_ascii=False,
    )


def workspace_write_tool(path: str, content: str, mode: str = "overwrite") -> str:
    """
    Write a UTF-8 text file under the uploaded workspace snapshot.

    Args:
        path: Relative path under the workspace root.
        content: File content to write.
        mode: Either 'overwrite' or 'append'.
    """
    try:
        target = _resolve_workspace_path(path)
    except Exception as exc:
        return json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False)

    write_mode = (mode or "overwrite").strip().lower()
    if write_mode not in {"overwrite", "append"}:
        return json.dumps({"status": "error", "reason": "invalid_mode"}, ensure_ascii=False)

    target.parent.mkdir(parents=True, exist_ok=True)
    if write_mode == "append":
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content or "")
    else:
        target.write_text(content or "", encoding="utf-8")

    return json.dumps(
        {
            "status": "ok",
            "path": str(target.relative_to(_workspace_root())),
            "bytes_written": len((content or "").encode("utf-8")),
            "mode": write_mode,
        },
        ensure_ascii=False,
    )


def shell_exec_tool(command: str, timeout_sec: int = 30, workdir: str = ".") -> str:
    """
    Run a short shell command inside the sandbox workspace.

    Args:
        command: Shell command to execute.
        timeout_sec: Hard timeout in seconds.
        workdir: Relative working directory under the workspace root.
    """
    snippet = (command or "").strip()
    if not snippet:
        return json.dumps({"status": "error", "reason": "empty_command"}, ensure_ascii=False)

    try:
        cwd = _resolve_workspace_path(workdir)
    except Exception as exc:
        return json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False)
    if not cwd.exists() or not cwd.is_dir():
        return json.dumps({"status": "error", "reason": "invalid_workdir"}, ensure_ascii=False)

    try:
        argv = shlex.split(snippet)
    except ValueError as exc:
        return json.dumps({"status": "error", "reason": f"invalid_command:{exc}"}, ensure_ascii=False)
    if not argv:
        return json.dumps({"status": "error", "reason": "empty_command"}, ensure_ascii=False)

    allowed_commands = {
        "python",
        "python3",
        "/sandbox/.venv/bin/python",
        "uv",
        "/sandbox/.venv/bin/uv",
        "ls",
        "cat",
        "find",
        "rg",
        "grep",
        "pwd",
        "echo",
        "head",
        "sed",
    }
    for runtime_path in (os.getenv("AGENT_UV_BIN") or "", os.getenv("PYTHON_BIN") or ""):
        if runtime_path:
            allowed_commands.add(runtime_path)
            allowed_commands.add(str(Path(runtime_path).resolve()))
    executable = argv[0]
    if executable not in allowed_commands:
        return json.dumps(
            {"status": "error", "reason": f"command_not_allowed:{executable}"},
            ensure_ascii=False,
        )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    runtime_dirs = _runtime_bin_dirs()
    if runtime_dirs:
        env["PATH"] = f"{':'.join(runtime_dirs)}:{env.get('PATH', '')}"

    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "error", "reason": "timeout"}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "error", "reason": f"shell_tool_error:{exc}"}, ensure_ascii=False)

    return json.dumps(
        {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip()[:12000],
            "stderr": (proc.stderr or "").strip()[:12000],
            "cwd": str(cwd.relative_to(_workspace_root())),
            "argv": argv,
        },
        ensure_ascii=False,
    )


def uv_run_tool(args: str, timeout_sec: int = 60, workdir: str = ".") -> str:
    """
    Run uv inside the sandbox workspace using the runtime virtual environment.

    Args:
        args: Arguments passed to uv, for example 'run python script.py'.
        timeout_sec: Hard timeout in seconds.
        workdir: Relative working directory under the workspace root.
    """
    raw_args = (args or "").strip()
    if not raw_args:
        return json.dumps({"status": "error", "reason": "empty_args"}, ensure_ascii=False)

    try:
        cwd = _resolve_workspace_path(workdir)
    except Exception as exc:
        return json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False)
    if not cwd.exists() or not cwd.is_dir():
        return json.dumps({"status": "error", "reason": "invalid_workdir"}, ensure_ascii=False)

    uv_bin = os.getenv("AGENT_UV_BIN") or "/sandbox/.venv/bin/uv"
    if not Path(uv_bin).exists():
        return json.dumps({"status": "error", "reason": "uv_not_available"}, ensure_ascii=False)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    runtime_dirs = _runtime_bin_dirs()
    if runtime_dirs:
        env["PATH"] = f"{':'.join(runtime_dirs)}:{env.get('PATH', '')}"

    try:
        proc = subprocess.run(
            [uv_bin, *shlex.split(raw_args)],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "error", "reason": "timeout"}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"status": "error", "reason": f"uv_tool_error:{exc}"}, ensure_ascii=False)

    return json.dumps(
        {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip()[:12000],
            "stderr": (proc.stderr or "").strip()[:12000],
            "cwd": str(cwd.relative_to(_workspace_root())),
            "argv": [uv_bin, *shlex.split(raw_args)],
        },
        ensure_ascii=False,
    )


def safe_http_get_tool(url: str, timeout_sec: int = 10) -> str:
    """
    Fetches a trusted URL only if host is allowlisted in AGENT_ALLOWED_HOSTS.

    Args:
        url: HTTPS/HTTP URL.
        timeout_sec: Request timeout in seconds.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return json.dumps({"status": "error", "reason": "unsupported_scheme"})

    host = (parsed.hostname or "").lower()
    allowed_hosts = {
        h.strip().lower()
        for h in (os.getenv("AGENT_ALLOWED_HOSTS") or "").split(",")
        if h.strip()
    }
    allow_all_hosts = "*" in allowed_hosts

    if not allow_all_hosts and allowed_hosts and host not in allowed_hosts:
        return json.dumps({"status": "error", "reason": "host_not_allowed", "host": host})

    req = Request(url, headers={"User-Agent": "SpectralixAgent/1.0"}, method="GET")
    try:
        with urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
            body = resp.read(4_096).decode("utf-8", errors="replace")
            return json.dumps(
                {
                    "status": "ok",
                    "code": getattr(resp, "status", 200),
                    "content_preview": body,
                },
                ensure_ascii=False,
            )
    except Exception as exc:
        return json.dumps({"status": "error", "reason": f"http_error:{exc}"})


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    func: Callable[..., str]
    description: str
    schema: Dict[str, Any]
    is_network_tool: bool = False


def _object_schema(
    *,
    properties: Dict[str, Dict[str, Any]],
    required: List[str],
) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOL_DEFINITIONS: Dict[str, ToolDefinition] = {
    "chem_format_tool": ToolDefinition(
        name="chem_format_tool",
        func=chem_format_tool,
        description="Normalize a raw chemistry answer into benchmark-friendly machine-readable format.",
        schema=_object_schema(
            properties={
                "answer_type": {"type": "string"},
                "raw_text": {"type": "string"},
            },
            required=["answer_type", "raw_text"],
        ),
    ),
    "smiles_sanity_tool": ToolDefinition(
        name="smiles_sanity_tool",
        func=smiles_sanity_tool,
        description="Perform lightweight syntax checks on a candidate SMILES string.",
        schema=_object_schema(
            properties={
                "smiles": {"type": "string"},
            },
            required=["smiles"],
        ),
    ),
    "unit_convert_tool": ToolDefinition(
        name="unit_convert_tool",
        func=unit_convert_tool,
        description="Convert common chemistry units for mass, volume, and amount.",
        schema=_object_schema(
            properties={
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
            required=["value", "from_unit", "to_unit"],
        ),
    ),
    "rubric_hint_tool": ToolDefinition(
        name="rubric_hint_tool",
        func=rubric_hint_tool,
        description="Return a short answer-format hint for a benchmark answer type.",
        schema=_object_schema(
            properties={
                "answer_type": {"type": "string"},
            },
            required=["answer_type"],
        ),
    ),
    "json_array_validate_tool": ToolDefinition(
        name="json_array_validate_tool",
        func=json_array_validate_tool,
        description="Validate that a string contains a JSON array and return a compact normalized version.",
        schema=_object_schema(
            properties={
                "raw_json": {"type": "string"},
            },
            required=["raw_json"],
        ),
    ),
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
            # Keep tool disabled when allowlist is empty.
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
