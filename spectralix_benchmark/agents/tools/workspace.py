from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List


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
    """List files and directories under the uploaded workspace snapshot."""
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
        entries.append({"path": relative, "type": "dir" if item.is_dir() else "file", "size": size})

    return json.dumps({"status": "ok", "entries": entries}, ensure_ascii=False)


def workspace_read_tool(path: str, max_bytes: int = 12000) -> str:
    """Read a UTF-8 text file from the uploaded workspace snapshot."""
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
    """Write a UTF-8 text file under the uploaded workspace snapshot."""
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
    """Run a short allowlisted command inside the sandbox workspace."""
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
        return json.dumps({"status": "error", "reason": f"command_not_allowed:{executable}"}, ensure_ascii=False)

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
    """Run uv inside the sandbox workspace using the runtime virtual environment."""
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
        argv = [uv_bin, *shlex.split(raw_args)]
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
        return json.dumps({"status": "error", "reason": f"uv_tool_error:{exc}"}, ensure_ascii=False)

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

