from __future__ import annotations

from typing import Any, Dict, List
import json
import os
import sys
import time
import base64
from urllib.parse import urlparse

from openai import OpenAI

from .prompts import build_parse_page_task, build_student_task
from .tool_registry import build_tool_definitions


def _build_client(model: Dict[str, Any]) -> OpenAI:
    api_base = str(model["api_base"]).rstrip("/")
    header_name = (os.getenv("OPENAI_API_KEY_HEADER") or "Authorization").strip() or "Authorization"
    prefix = (os.getenv("OPENAI_API_KEY_PREFIX") or "Bearer").strip()
    default_headers: Dict[str, str] = {}
    api_key = str(model.get("api_key") or "")
    parsed = urlparse(api_base)
    if (parsed.hostname or "").lower() == "inference.local":
        api_key = "openshell-managed"
    if header_name.lower() != "authorization" or prefix.lower() != "bearer":
        token_value = f"{prefix} {api_key}".strip() if prefix else api_key
        default_headers[header_name] = token_value
        api_key = "unused_api_key"
    return OpenAI(
        base_url=api_base,
        api_key=api_key,
        default_headers=default_headers or None,
        timeout=60.0,
    )


def _extract_assistant_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if hasattr(item, "text") and getattr(item, "text"):
                parts.append(str(getattr(item, "text")).strip())
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]).strip())
        return "\n".join(part for part in parts if part).strip()
    return ""


def _tool_schema(tool_definition: Any) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_definition.name,
            "description": tool_definition.description,
            "parameters": tool_definition.schema,
        },
    }


def _serialize_tool_call(tool_call: Any, output_text: str) -> Dict[str, Any]:
    function = getattr(tool_call, "function", None)
    return {
        "name": str(getattr(function, "name", "") or ""),
        "arguments": str(getattr(function, "arguments", "") or ""),
        "output_preview": output_text[:500],
    }


def _chat_completion(
    *,
    client: OpenAI,
    model: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tool_definitions: List[Any],
) -> Any:
    kwargs: Dict[str, Any] = {
        "model": str(model["model_name"]),
        "messages": messages,
        "temperature": float(model.get("temperature", 0.2)),
        "max_tokens": int(model.get("max_tokens", 768)),
    }
    reasoning_effort = str(model.get("reasoning_effort") or "").strip()
    if reasoning_effort:
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
    if tool_definitions:
        kwargs["tools"] = [_tool_schema(item) for item in tool_definitions]
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ["429", "rate limit", "timeout", "temporarily unavailable", "503", "502", "500"])


def _chat_completion_with_retry(
    *,
    client: OpenAI,
    model: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tool_definitions: List[Any],
    max_attempts: int = 3,
) -> Any:
    attempt = 0
    while True:
        attempt += 1
        try:
            return _chat_completion(
                client=client,
                model=model,
                messages=messages,
                tool_definitions=tool_definitions,
            )
        except Exception:
            if attempt >= max_attempts:
                raise
            time.sleep(float(attempt))


def _run_tool_loop(
    *,
    payload: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    model = payload["model"]
    config = payload["config"]
    client = _build_client(model)
    tool_definitions = build_tool_definitions(str(payload["tools_profile"]), config)
    tool_lookup = {item.name: item for item in tool_definitions}

    os.environ["AGENT_ALLOWED_HOSTS"] = ",".join(config.get("security", {}).get("allowed_tool_hosts") or [])
    os.environ["PYTHON_BIN"] = sys.executable
    os.environ["AGENT_WORKSPACE_ROOT"] = "/sandbox/workspace"
    os.environ["AGENT_UV_BIN"] = "/sandbox/.venv/bin/uv"

    steps: List[Dict[str, Any]] = []
    final_answer = ""
    state = "error"
    error_message = ""
    requests_per_minute = max(0, int(model.get("requests_per_minute") or 0))
    min_interval_seconds = (60.0 / requests_per_minute) if requests_per_minute > 0 else 0.0
    next_request_not_before = 0.0

    for step_number in range(1, int(payload.get("max_steps") or 1) + 1):
        if min_interval_seconds > 0:
            sleep_for = next_request_not_before - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
        started_at = time.perf_counter()
        response = _chat_completion_with_retry(
            client=client,
            model=model,
            messages=messages,
            tool_definitions=tool_definitions,
        )
        if min_interval_seconds > 0:
            next_request_not_before = time.monotonic() + min_interval_seconds
        duration = time.perf_counter() - started_at
        response_raw = response.model_dump()
        choice = response.choices[0]
        message = choice.message
        assistant_text = _extract_assistant_text(message)
        tool_calls = list(getattr(message, "tool_calls", None) or [])

        step_payload: Dict[str, Any] = {
            "step_number": step_number,
            "timing": {"duration": duration},
            "model_output_message": {
                "content": assistant_text,
                "raw": response_raw,
            },
            "tool_calls": [],
        }

        if not tool_calls:
            final_answer = assistant_text
            state = "success" if final_answer else "error"
            if not final_answer:
                error_message = "model returned empty final answer"
            steps.append(step_payload)
            break

        messages.append(
            {
                "role": "assistant",
                "content": assistant_text or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            tool_name = str(tool_call.function.name)
            tool_definition = tool_lookup.get(tool_name)
            if tool_definition is None:
                tool_output = json.dumps({"status": "error", "reason": f"unknown_tool:{tool_name}"})
            else:
                try:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                    if not isinstance(tool_args, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    tool_output = str(tool_definition.func(**tool_args))
                except Exception as exc:
                    tool_output = json.dumps({"status": "error", "reason": f"tool_call_error:{exc}"})
            step_payload["tool_calls"].append(_serialize_tool_call(tool_call, tool_output))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )

        steps.append(step_payload)

    if state != "success" and not error_message:
        error_message = "agent reached max steps without producing a final answer"

    return {
        "state": state,
        "output": final_answer,
        "error": error_message,
        "steps": steps,
    }


def _student_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    question = payload["question"]
    config = payload["config"]
    tool_definitions = build_tool_definitions(str(payload["tools_profile"]), config)
    runtime_context = {
        "tools_profile": str(payload["tools_profile"]),
        "workspace_root": "/sandbox/workspace",
        "available_tools": [item.name for item in tool_definitions],
    }
    return [
        {"role": "system", "content": "You are a chemistry benchmark agent operating inside an OpenShell sandbox."},
        {"role": "user", "content": build_student_task(question, runtime_context=runtime_context)},
    ]


def _parser_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_bytes = base64.b64decode(payload["image_base64"])
    image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    return [
        {"role": "system", "content": "You parse chemistry exam pages accurately and return strict JSON arrays only."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": build_parse_page_task(
                        exam_id=str(payload["exam_id"]),
                        page_id=int(payload["page_id"]),
                        marker_prompt=str(payload["marker_prompt"]),
                        image_path=str(payload.get("image_path") or ""),
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                },
            ],
        },
    ]


def _main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    mode = str(payload.get("mode") or "").strip().lower()

    if mode == "student":
        result = _run_tool_loop(payload=payload, messages=_student_messages(payload))
    elif mode == "parser":
        result = _run_tool_loop(payload=payload, messages=_parser_messages(payload))
    else:
        raise ValueError(f"Unsupported worker mode: {mode}")

    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
