from __future__ import annotations

from typing import Any, Dict, Optional


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _extract_reasoning_tokens(details: Dict[str, Any]) -> int:
    for key in ("reasoning_tokens", "output_tokens_reasoning", "completion_tokens_reasoning"):
        value = details.get(key)
        if isinstance(value, int):
            return value
    return 0


def extract_run_usage(run_result: Any) -> Optional[Dict[str, Any]]:
    usage_getter = getattr(run_result, "usage", None)
    if usage_getter is None:
        return None

    usage_obj = usage_getter() if callable(usage_getter) else usage_getter
    if usage_obj is None:
        return None

    details_raw = getattr(usage_obj, "details", {}) or {}
    details = details_raw if isinstance(details_raw, dict) else {}
    input_tokens = _coerce_int(getattr(usage_obj, "input_tokens", 0))
    output_tokens = _coerce_int(getattr(usage_obj, "output_tokens", 0))

    return {
        "judge_requests": _coerce_int(getattr(usage_obj, "requests", 0)),
        "judge_tool_calls": _coerce_int(getattr(usage_obj, "tool_calls", 0)),
        "judge_input_tokens": input_tokens,
        "judge_output_tokens": output_tokens,
        "judge_total_tokens": input_tokens + output_tokens,
        "judge_cache_write_tokens": _coerce_int(getattr(usage_obj, "cache_write_tokens", 0)),
        "judge_cache_read_tokens": _coerce_int(getattr(usage_obj, "cache_read_tokens", 0)),
        "judge_input_audio_tokens": _coerce_int(getattr(usage_obj, "input_audio_tokens", 0)),
        "judge_cache_audio_read_tokens": _coerce_int(getattr(usage_obj, "cache_audio_read_tokens", 0)),
        "judge_reasoning_tokens": _extract_reasoning_tokens(details),
        "judge_usage_details": details,
    }
