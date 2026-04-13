from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import os
import subprocess
import sys
import tempfile

from .prompts import build_student_task


def _codex_home_dir(payload: Dict[str, Any]) -> str:
    env_override = str(os.getenv("CODEX_HOME") or "").strip()
    if env_override:
        return env_override
    openshell_cfg = (((payload.get("config") or {}).get("sandbox") or {}).get("openshell") or {})
    native_cfg = openshell_cfg.get("native_codex") or {}
    return str(native_cfg.get("codex_home_dir") or "/sandbox/.codex")


def _build_prompt(payload: Dict[str, Any]) -> str:
    mode = str(payload.get("mode") or "").strip().lower()
    if mode == "student":
        return build_student_task(payload["question"])
    raise RuntimeError(f"Unsupported worker mode: {mode}")


def _codex_command(payload: Dict[str, Any], output_path: Path) -> List[str]:
    model = payload.get("model") or {}
    tools_profile = str(payload.get("tools_profile") or "minimal").strip().lower()
    prompt = _build_prompt(payload)
    command = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
        str(output_path),
        "--sandbox",
        "workspace-write" if tools_profile in {"tools", "tools_internet", "full"} else "read-only",
    ]
    if tools_profile == "tools_internet":
        command.append("--search")
    model_name = str(model.get("model_name") or "").strip()
    if model_name:
        command.extend(["--model", model_name])
    command.append(prompt)
    return command


def _usage_to_openai_shape(event_usage: Dict[str, Any]) -> Dict[str, Any]:
    input_tokens = int(event_usage.get("input_tokens") or 0)
    output_tokens = int(event_usage.get("output_tokens") or 0)
    total_tokens = input_tokens + output_tokens
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_details": {
            "reasoning_tokens": int(event_usage.get("reasoning_tokens") or 0),
        },
        "codex_usage": event_usage,
    }


def _parse_event_stream(stdout_text: str) -> Dict[str, Any]:
    usage: Dict[str, Any] = {}
    assistant_messages: List[str] = []
    for line in stdout_text.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type") or "").strip()
        if event_type == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = dict(event["usage"])
        if event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and str(item.get("type") or "") == "agent_message":
                text = str(item.get("text") or "").strip()
                if text:
                    assistant_messages.append(text)
    return {
        "usage": usage,
        "assistant_messages": assistant_messages,
    }


def _main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    codex_home = _codex_home_dir(payload)
    auth_path = Path(codex_home) / "auth.json"
    if not auth_path.exists():
        raise RuntimeError(f"Native Codex auth file is missing: {auth_path}")

    with tempfile.TemporaryDirectory(prefix="codex-native-", dir="/tmp") as tmp_dir:
        output_path = Path(tmp_dir) / "last_message.txt"
        workdir = "/sandbox" if Path("/sandbox").exists() else os.getcwd()
        process = subprocess.run(
            _codex_command(payload, output_path),
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=int(payload.get("timeout_sec") or 0) or None,
            env={
                **os.environ,
                "CODEX_HOME": codex_home,
                "HOME": "/sandbox",
                "PYTHONUNBUFFERED": "1",
            },
            check=False,
        )

        parsed_stream = _parse_event_stream(process.stdout or "")
        final_answer = ""
        if output_path.exists():
            final_answer = output_path.read_text(encoding="utf-8", errors="replace").strip()
        if not final_answer:
            assistant_messages = parsed_stream.get("assistant_messages") or []
            if assistant_messages:
                final_answer = str(assistant_messages[-1]).strip()

        if process.returncode != 0:
            stderr_text = (process.stderr or "").strip()
            error_text = stderr_text or (process.stdout or "").strip() or "codex exec failed"
            result = {
                "state": "error",
                "output": final_answer,
                "error": error_text[:500],
                "steps": [],
            }
        else:
            usage = _usage_to_openai_shape(parsed_stream.get("usage") or {})
            result = {
                "state": "success" if final_answer else "error",
                "output": final_answer,
                "error": "" if final_answer else "codex returned empty final answer",
                "steps": [
                    {
                        "step_number": 1,
                        "model_output_message": {
                            "content": final_answer,
                            "raw": {"usage": usage},
                        },
                        "tool_calls": [],
                    }
                ],
            }

    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
