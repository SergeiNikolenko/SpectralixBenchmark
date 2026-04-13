from __future__ import annotations

import json
import os
import subprocess


def chem_python_tool(code: str, timeout_sec: int = 20) -> str:
    """
    Run a short Python snippet inside the runtime environment.

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

