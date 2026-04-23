from __future__ import annotations

from typing import Any, Dict, List
import json
import os
import sys
import time
from urllib.parse import urlparse

import httpx
from openai import OpenAI

from .prompts import build_student_sgr_task, build_student_task
from .sgr_schemas import (
    extract_contract_check,
    extract_final_answer_value,
    get_sgr_schema_spec,
    schema_template_lines,
    validate_sgr_payload,
)
from .tool_registry import build_tool_definitions


def _build_client(model: Dict[str, Any], *, request_timeout_sec: float | None = None) -> OpenAI:
    api_base = str(model["api_base"]).rstrip("/")
    header_name = (os.getenv("OPENAI_API_KEY_HEADER") or "Authorization").strip() or "Authorization"
    prefix = (os.getenv("OPENAI_API_KEY_PREFIX") or "Bearer").strip()
    default_headers: Dict[str, str] = {}
    api_key = str(model.get("api_key") or "")
    parsed = urlparse(api_base)
    hostname = (parsed.hostname or "").lower()
    use_local_transport = hostname in {"127.0.0.1", "localhost", "inference.local"}
    if hostname == "inference.local":
        api_key = "openshell-managed"
    if header_name.lower() != "authorization" or prefix.lower() != "bearer":
        token_value = f"{prefix} {api_key}".strip() if prefix else api_key
        default_headers[header_name] = token_value
        api_key = "unused_api_key"
    default_headers.setdefault("Connection", "close")
    client_timeout = max(60.0, float(request_timeout_sec or model.get("request_timeout_sec") or 60.0))
    client_kwargs: Dict[str, Any] = {
        "base_url": api_base,
        "api_key": api_key,
        "default_headers": default_headers or None,
        "timeout": client_timeout,
    }
    if use_local_transport:
        client_kwargs["http_client"] = httpx.Client(
            timeout=client_timeout,
            trust_env=False,
            headers={"Connection": "close"},
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=0),
        )
    return OpenAI(**client_kwargs)


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


def _tool_support_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "does not support tools" in text or "tool" in text and "invalid_request_error" in text


def _env_truthy(name: str) -> bool:
    value = (os.getenv(name) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
        except Exception as exc:
            if _tool_support_error(exc) or not _is_transient_error(exc):
                raise
            if attempt >= max_attempts:
                raise
            time.sleep(float(attempt))


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    payload = (raw_text or "").strip()
    if not payload:
        raise ValueError("empty_json")
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    import re

    match = re.search(r"\{[\s\S]*\}", payload)
    if not match:
        raise ValueError("json_object_not_found")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("json_is_not_object")
    return parsed


def _manual_tool_protocol_message(tool_definitions: List[Any]) -> Dict[str, str]:
    tool_lines = []
    for item in tool_definitions:
        schema_json = json.dumps(item.schema, ensure_ascii=False, separators=(",", ":"))
        tool_lines.append(f"- {item.name}: {item.description} | json_schema={schema_json}")
    return {
        "role": "system",
        "content": (
            "Native function-calling is unavailable for this model. "
            "If you need a tool, respond with exactly one JSON object and nothing else in this format: "
            '{"tool_call":{"name":"<tool_name>","arguments":{...}}}. '
            "Use only one tool call per message. "
            "If no tool is needed, return the final benchmark answer as plain text instead.\n"
            "Available tools:\n"
            + ("\n".join(tool_lines) if tool_lines else "- none")
        ),
    }


def _extract_manual_tool_call(raw_text: str, tool_lookup: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        parsed = _extract_json_object(raw_text)
    except Exception:
        return None
    if "tool_call" in parsed and isinstance(parsed["tool_call"], dict):
        parsed = parsed["tool_call"]
    name = str(parsed.get("name") or "").strip()
    arguments = parsed.get("arguments")
    if not name or name not in tool_lookup or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def _step_budget_message(*, step_number: int, max_steps: int, phase: str) -> Dict[str, str]:
    remaining_after_this = max(0, max_steps - step_number)
    phase_name = phase or "main"
    if remaining_after_this == 0:
        instruction = (
            "This is your last allowed step. "
            "Return the final benchmark answer now. Do not call more tools unless a final tool call is strictly required."
        )
    elif remaining_after_this == 1:
        instruction = (
            "You have one step left after this one. "
            "If the answer is nearly ready, finalize now instead of exploring."
        )
    else:
        instruction = (
            "Conserve steps. "
            "If the answer is already clear, finalize instead of using another tool."
        )
    return {
        "role": "system",
        "content": (
            f"Step budget status: phase={phase_name}; current_step={step_number}; "
            f"max_steps={max_steps}; remaining_after_this={remaining_after_this}. "
            f"{instruction}"
        ),
    }


def _build_sgr_repair_task(schema_name: str, schema_lines: str, invalid_payload: str, error_message: str) -> str:
    return (
        "Repair the invalid SGR JSON into a valid JSON object.\n"
        "Return ONLY one valid JSON object, with no markdown fences and no extra text.\n"
        "Do not omit required fields.\n"
        f"Schema name: {schema_name}\n"
        f"Validation error: {error_message}\n"
        "Invalid payload:\n"
        f"{invalid_payload}\n\n"
        "Schema template:\n"
        f"{schema_lines}"
    )


def _build_runtime_context(payload: Dict[str, Any], tool_definitions: List[Any]) -> Dict[str, Any]:
    return {
        "tools_profile": str(payload["tools_profile"]),
        "workspace_root": str(payload.get("workspace_root") or "/sandbox/workspace"),
        "available_tools": [item.name for item in tool_definitions],
    }


def _sgr_tool_definitions(question: Dict[str, Any], tool_definitions: List[Any]) -> List[Any]:
    level = str(question.get("level") or "").strip().upper()
    if level == "A":
        allowed = set()
    elif level == "B":
        allowed = {"chem_python_tool"}
    elif level == "C":
        allowed = {"chem_python_tool", "workspace_list_tool", "workspace_read_tool", "uv_run_tool"}
    else:
        allowed = set()
    return [item for item in tool_definitions if item.name in allowed]


def _student_tool_definitions(question: Dict[str, Any], tool_definitions: List[Any]) -> List[Any]:
    return _sgr_tool_definitions(question, tool_definitions)


def _generate_sgr_payload(
    *,
    payload: Dict[str, Any],
    tool_definitions: List[Any],
) -> Dict[str, Any]:
    question = payload.get("question") or {}
    spec = get_sgr_schema_spec(str(question.get("level") or ""), str(question.get("task_subtype") or ""))
    sgr_tools = _sgr_tool_definitions(question, tool_definitions)
    schema_lines = schema_template_lines(spec.template)
    runtime_context = _build_runtime_context(payload, sgr_tools)
    runtime_context["sgr_schema_name"] = spec.schema_name
    runtime_context["sgr_schema_lines"] = schema_lines
    steps_so_far: List[Dict[str, Any]] = []
    try:
        initial_result = _run_tool_loop(
            payload=payload,
            messages=[
                {"role": "system", "content": "You produce hidden structured chemistry reasoning JSON only."},
                {"role": "user", "content": build_student_sgr_task(question, runtime_context=runtime_context, schema_spec=spec)},
            ],
            tool_definitions=sgr_tools,
            step_phase="sgr",
            step_number_offset=0,
        )
        initial_steps = list(initial_result.get("steps") or [])
        steps_so_far.extend(initial_steps)
        try:
            validated = validate_sgr_payload(
                str(question.get("level") or ""),
                str(question.get("task_subtype") or ""),
                _extract_json_object(str(initial_result.get("output") or "")),
            )
            return {
                "sgr_schema_name": spec.schema_name,
                "sgr_payload": validated.model_dump(),
                "sgr_validation_status": "validated",
                "sgr_repair_attempted": False,
                "sgr_fallback_used": False,
                "steps": initial_steps,
                "sgr_error": "",
            }
        except Exception as first_error:
            repair_result = _run_tool_loop(
                payload=payload,
                messages=[
                    {"role": "system", "content": "You repair hidden structured chemistry reasoning JSON only."},
                    {
                        "role": "user",
                        "content": _build_sgr_repair_task(
                            schema_name=spec.schema_name,
                            schema_lines=schema_lines,
                            invalid_payload=str(initial_result.get("output") or ""),
                            error_message=str(first_error),
                        ),
                    },
                ],
                tool_definitions=[],
                step_phase="sgr_repair",
                step_number_offset=len(initial_steps),
            )
            combined_steps = initial_steps + list(repair_result.get("steps") or [])
            steps_so_far = combined_steps
            try:
                validated = validate_sgr_payload(
                    str(question.get("level") or ""),
                    str(question.get("task_subtype") or ""),
                    _extract_json_object(str(repair_result.get("output") or "")),
                )
                return {
                    "sgr_schema_name": spec.schema_name,
                    "sgr_payload": validated.model_dump(),
                    "sgr_validation_status": "validated_after_repair",
                    "sgr_repair_attempted": True,
                    "sgr_fallback_used": False,
                    "steps": combined_steps,
                    "sgr_error": "",
                }
            except Exception:
                return {
                    "sgr_schema_name": spec.schema_name,
                    "sgr_payload": None,
                    "sgr_validation_status": "fallback_after_repair_failure",
                    "sgr_repair_attempted": True,
                    "sgr_fallback_used": True,
                    "steps": combined_steps,
                    "sgr_error": "",
                }
    except Exception as exc:
        return {
            "sgr_schema_name": spec.schema_name,
            "sgr_payload": None,
            "sgr_validation_status": "fallback_on_generation_error",
            "sgr_repair_attempted": False,
            "sgr_fallback_used": True,
            "steps": steps_so_far,
            "sgr_error": str(exc),
        }


def _run_tool_loop(
    *,
    payload: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tool_definitions: List[Any] | None = None,
    step_phase: str = "",
    step_number_offset: int = 0,
) -> Dict[str, Any]:
    model = payload["model"]
    config = payload["config"]
    client = _build_client(model, request_timeout_sec=float(payload.get("timeout_sec") or 60.0))
    tool_definitions = list(tool_definitions) if tool_definitions is not None else build_tool_definitions(str(payload["tools_profile"]), config)
    tool_lookup = {item.name: item for item in tool_definitions}

    os.environ["AGENT_ALLOWED_HOSTS"] = ",".join(config.get("security", {}).get("allowed_tool_hosts") or [])
    os.environ["PYTHON_BIN"] = sys.executable
    os.environ["AGENT_WORKSPACE_ROOT"] = str(payload.get("workspace_root") or "/sandbox/workspace")
    os.environ["AGENT_UV_BIN"] = str(payload.get("uv_bin") or "/sandbox/.venv/bin/uv")

    steps: List[Dict[str, Any]] = []
    final_answer = ""
    state = "error"
    error_message = ""
    requests_per_minute = max(0, int(model.get("requests_per_minute") or 0))
    min_interval_seconds = (60.0 / requests_per_minute) if requests_per_minute > 0 else 0.0
    next_request_not_before = 0.0
    manual_tool_mode = bool(tool_definitions) and _env_truthy("SPECTRALIX_FORCE_MANUAL_TOOL_PROTOCOL")

    max_steps = max(1, int(payload.get("max_steps") or 1))
    for step_number in range(1, max_steps + 1):
        if min_interval_seconds > 0:
            sleep_for = next_request_not_before - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
        started_at = time.perf_counter()
        request_messages = list(messages)
        request_messages.append(
            _step_budget_message(
                step_number=step_number,
                max_steps=max_steps,
                phase=step_phase,
            )
        )
        if manual_tool_mode and tool_definitions:
            request_messages.append(_manual_tool_protocol_message(tool_definitions))
        try:
            response = _chat_completion_with_retry(
                client=client,
                model=model,
                messages=request_messages,
                tool_definitions=[] if manual_tool_mode else tool_definitions,
            )
        except Exception as exc:
            if tool_definitions and not manual_tool_mode and _tool_support_error(exc):
                manual_tool_mode = True
                request_messages = list(messages)
                request_messages.append(
                    _step_budget_message(
                        step_number=step_number,
                        max_steps=max_steps,
                        phase=step_phase,
                    )
                )
                request_messages.append(_manual_tool_protocol_message(tool_definitions))
                response = _chat_completion_with_retry(
                    client=client,
                    model=model,
                    messages=request_messages,
                    tool_definitions=[],
                )
            else:
                raise
        if min_interval_seconds > 0:
            next_request_not_before = time.monotonic() + min_interval_seconds
        duration = time.perf_counter() - started_at
        response_raw = response.model_dump()
        choice = response.choices[0]
        message = choice.message
        assistant_text = _extract_assistant_text(message)
        tool_calls = list(getattr(message, "tool_calls", None) or [])

        step_payload: Dict[str, Any] = {
            "step_number": step_number + step_number_offset,
            "phase": step_phase or "main",
            "tool_protocol": "manual_json" if manual_tool_mode else "native",
            "timing": {"duration": duration},
            "model_output_message": {
                "content": assistant_text,
                "raw": response_raw,
            },
            "tool_calls": [],
        }

        manual_tool_call = _extract_manual_tool_call(assistant_text, tool_lookup) if manual_tool_mode and tool_definitions else None
        if manual_tool_call is not None:
            tool_name = str(manual_tool_call["name"])
            tool_definition = tool_lookup[tool_name]
            try:
                tool_output = str(tool_definition.func(**manual_tool_call["arguments"]))
            except Exception as exc:
                tool_output = json.dumps({"status": "error", "reason": f"tool_call_error:{exc}"})
            step_payload["tool_calls"].append(
                {
                    "name": tool_name,
                    "arguments": json.dumps(manual_tool_call["arguments"], ensure_ascii=False),
                    "output_preview": tool_output[:500],
                }
            )
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool result for {tool_name}:\n{tool_output}\n\n"
                        "Continue. If another tool is needed, respond with one JSON tool_call object only. "
                        "Otherwise provide the final benchmark answer as plain text."
                    ),
                }
            )
            steps.append(step_payload)
            continue

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


def _student_messages(payload: Dict[str, Any], tool_definitions: List[Any]) -> List[Dict[str, Any]]:
    question = payload["question"]
    runtime_context = _build_runtime_context(payload, tool_definitions)
    sgr_context = payload.get("sgr_context")
    return [
        {"role": "system", "content": "You are a chemistry benchmark agent operating inside an OpenShell sandbox."},
        {"role": "user", "content": build_student_task(question, runtime_context=runtime_context, sgr_context=sgr_context if isinstance(sgr_context, dict) else None)},
    ]


def _run_student_with_sgr(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = payload.get("question") or {}
    tool_definitions = build_tool_definitions(str(payload["tools_profile"]), payload["config"])
    student_tool_definitions = _student_tool_definitions(question, tool_definitions)
    sgr_meta = _generate_sgr_payload(payload=payload, tool_definitions=tool_definitions)
    student_payload = dict(payload)
    sgr_candidate_final_answer = ""
    if isinstance(sgr_meta.get("sgr_payload"), dict):
        sgr_payload = sgr_meta["sgr_payload"]
        sgr_candidate_final_answer = extract_final_answer_value(sgr_payload)
        student_payload["sgr_context"] = {
            "schema_name": str(sgr_meta.get("sgr_schema_name") or ""),
            "contract_check": extract_contract_check(sgr_payload),
            "candidate_final_answer": sgr_candidate_final_answer,
        }
    sgr_steps = list(sgr_meta.get("steps") or [])
    try:
        result = _run_tool_loop(
            payload=student_payload,
            messages=_student_messages(student_payload, student_tool_definitions),
            tool_definitions=student_tool_definitions,
            step_phase="final_answer",
            step_number_offset=len(sgr_steps),
        )
    except Exception:
        if not sgr_candidate_final_answer:
            raise
        result = {
            "state": "success",
            "output": sgr_candidate_final_answer,
            "error": "",
            "steps": sgr_steps,
        }
    result["steps"] = sgr_steps + list(result.get("steps") or [])
    result["sgr_schema_name"] = sgr_meta.get("sgr_schema_name")
    result["sgr_payload"] = sgr_meta.get("sgr_payload")
    result["sgr_validation_status"] = sgr_meta.get("sgr_validation_status")
    result["sgr_repair_attempted"] = bool(sgr_meta.get("sgr_repair_attempted"))
    result["sgr_fallback_used"] = bool(sgr_meta.get("sgr_fallback_used"))
    result["sgr_error"] = str(sgr_meta.get("sgr_error") or "")
    return result


def _run_student_without_sgr(payload: Dict[str, Any]) -> Dict[str, Any]:
    tool_definitions = build_tool_definitions(str(payload["tools_profile"]), payload["config"])
    question = payload.get("question") or {}
    student_tool_definitions = _student_tool_definitions(question, tool_definitions)
    result = _run_tool_loop(
        payload=payload,
        messages=_student_messages(payload, student_tool_definitions),
        tool_definitions=student_tool_definitions,
        step_phase="final_answer",
    )
    result["sgr_schema_name"] = ""
    result["sgr_payload"] = None
    result["sgr_validation_status"] = "disabled"
    result["sgr_repair_attempted"] = False
    result["sgr_fallback_used"] = False
    result["sgr_error"] = ""
    return result
def _main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    mode = str(payload.get("mode") or "").strip().lower()

    if mode == "student":
        if bool(payload.get("sgr_enabled", True)):
            result = _run_student_with_sgr(payload)
        else:
            result = _run_student_without_sgr(payload)
    else:
        raise ValueError(f"Unsupported worker mode: {mode}")

    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
