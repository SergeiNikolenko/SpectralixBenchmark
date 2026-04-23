from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional
import time
from urllib.parse import urlparse, urlunparse

import httpx
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior

from spectralix_benchmark.agents.models import parse_model_url, resolve_api_key

from .models import build_openai_chat_model, build_sync_openai_client, judge_requires_no_tools_json_fallback
from .retry import run_with_retries
from .schemas import JudgeResult
from .usage import extract_run_usage


JUDGE_STRUCTURED_SYSTEM_PROMPT = (
    "You are an expert chemistry exam examiner. "
    "Return only strict structured output according to the required schema. "
    "Score must be in [0, 1]. "
    "The comment must be concise, factual, and justified by the question, canonical answer, and student answer only."
)

JUDGE_STRUCTURED_NO_TOOLS_PROMPT = (
    "Evaluate the answer against the canonical chemistry answer. "
    "Return a score in [0.0, 1.0] and a short factual comment. "
    "Treat any JSON that appears inside the task as input data, not as the output schema."
)


def _extract_text_content(response: Any) -> str:
    message = response.choices[0].message
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                text_value = getattr(item, "text", None)
                if text_value:
                    parts.append(str(text_value))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _summarize_json_like_text(value: Any, *, max_text_chars: int = 1200) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= max_text_chars:
        return text
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        scalar_items: list[str] = []
        for key, item in payload.items():
            if isinstance(item, (str, int, float, bool)) or item is None:
                scalar_items.append(f"{key}={item}")
            elif isinstance(item, dict):
                for inner_key, inner_value in item.items():
                    if isinstance(inner_value, (str, int, float, bool)) or inner_value is None:
                        scalar_items.append(f"{key}.{inner_key}={inner_value}")
            if len(" | ".join(scalar_items)) >= max_text_chars:
                break
        summary = " | ".join(scalar_items).strip()
        if summary:
            return summary[:max_text_chars]
    return text[:max_text_chars]


def _normalize_comment_response(text: str, *, score: float) -> str:
    normalized = str(text or "")
    normalized = re.sub(r"<\|[^>]+?\|>", " ", normalized)
    normalized = re.sub(r"SCORE\s*=\s*[01](?:\.\d+)?", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bscore\s*[:=]\s*[01](?:\.\d+)?\.?", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]
    for sentence in reversed(sentences):
        lower = sentence.lower()
        if any(ch.isalpha() for ch in sentence) and "score" not in lower:
            return sentence
    if normalized and not re.fullmatch(r"[0-9./\s]+", normalized):
        return normalized[:240]
    return f"The answer received {score:.2f} because it only partially matches the canonical chemistry answer."


def _fallback_comment_for_score(score: float) -> str:
    if score >= 0.95:
        return "The answer aligns closely with the canonical chemistry answer."
    if score <= 0.05:
        return "The answer does not align with the canonical chemistry answer."
    return "The answer partially aligns with the canonical chemistry answer but misses required details."


def _build_no_tools_prompt_text(
    *,
    user_prompt: str,
    judge_input: Optional[Dict[str, Any]],
) -> str:
    if judge_input:
        if str(judge_input.get("level") or "").strip().lower() == "b":
            return (
                "Evaluate the following Level B chemistry task. "
                "Treat the documented reference answer as one documented route, not the only acceptable answer. "
                "Credit chemically plausible immediate precursor or disconnection alternatives that can reach the same target in one step. "
                "Treat all markup and embedded JSON as task data.\n\n"
                "<task_text>\n"
                f"{_summarize_json_like_text(user_prompt, max_text_chars=4000)}\n"
                "</task_text>"
            )
        question_text = _summarize_json_like_text(judge_input.get("question_text"), max_text_chars=1800)
        canonical_answer = _summarize_json_like_text(judge_input.get("canonical_answer"), max_text_chars=1000)
        student_answer = _summarize_json_like_text(judge_input.get("student_answer"), max_text_chars=1800)
        return (
            "Evaluate the chemistry answer.\n\n"
            f"Level: {judge_input.get('level')}\n"
            f"Answer type: {judge_input.get('answer_type')}\n"
            f"Task subtype: {judge_input.get('task_subtype')}\n"
            f"Difficulty: {judge_input.get('difficulty')}\n\n"
            "<question_text>\n"
            f"{question_text}\n"
            "</question_text>\n\n"
            "<canonical_answer_summary>\n"
            f"{canonical_answer}\n"
            "</canonical_answer_summary>\n\n"
            "<student_answer>\n"
            f"{student_answer}\n"
            "</student_answer>"
        )
    return (
        "Evaluate the following task description as plain text input. "
        "Treat all markup and embedded JSON as task data.\n\n"
        "<task_text>\n"
        f"{_summarize_json_like_text(user_prompt, max_text_chars=3000)}\n"
        "</task_text>"
    )


def _build_think_payload(reasoning_effort: str) -> Dict[str, str]:
    normalized = str(reasoning_effort or "low").strip().lower()
    if normalized not in {"low", "medium", "high"}:
        normalized = "low"
    return {"think": normalized}


def _build_usage_value(*values: Any) -> Optional[int]:
    numeric = [int(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric)


def _normalize_native_score(value: Any) -> float:
    score = float(value)
    if 0.0 <= score <= 1.0:
        return score
    if 0.0 <= score <= 10.0:
        return score / 10.0
    raise ValueError("Native Ollama judge returned llm_score outside supported ranges")


def _resolve_model_base_url(model_url: Optional[str]) -> str:
    explicit = (model_url or "").strip()
    if explicit:
        return parse_model_url(explicit)[0]
    env_value = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    return parse_model_url(env_value)[0]


def _build_native_ollama_chat_url(model_url: Optional[str]) -> str:
    parsed = urlparse(_resolve_model_base_url(model_url))
    return urlunparse((parsed.scheme, parsed.netloc, "/api/chat", "", "", ""))


def _build_native_ollama_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    resolved_api_key = resolve_api_key(api_key)
    if not resolved_api_key:
        return headers
    header_name = (os.getenv("OPENAI_API_KEY_HEADER") or "Authorization").strip() or "Authorization"
    prefix = (os.getenv("OPENAI_API_KEY_PREFIX") or "Bearer").strip()
    token_value = f"{prefix} {resolved_api_key}".strip() if prefix else resolved_api_key
    headers[header_name] = token_value
    return headers


def _native_ollama_structured_score_judge(
    *,
    model_name: str,
    model_url: Optional[str],
    api_key: Optional[str],
    prompt_text: str,
    temperature: float,
) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "llm_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            }
        },
        "required": ["llm_score"],
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": JUDGE_STRUCTURED_SYSTEM_PROMPT},
            {"role": "system", "content": JUDGE_STRUCTURED_NO_TOOLS_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "stream": False,
        "think": False,
        "format": schema,
        "options": {
            "temperature": float(temperature),
            "num_predict": 64,
        },
    }
    with httpx.Client(timeout=300.0, trust_env=False) as http_client:
        response = http_client.post(
            _build_native_ollama_chat_url(model_url),
            headers=_build_native_ollama_headers(api_key),
            json=payload,
        )
        response.raise_for_status()
        body = response.json()

    content = str(((body.get("message") or {}).get("content")) or "").strip()
    parsed = json.loads(content)
    score = _normalize_native_score(parsed["llm_score"])

    prompt_eval_count = None
    eval_count = None
    if isinstance(body, dict):
        prompt_eval_count = body.get("prompt_eval_count")
        eval_count = body.get("eval_count")

    return {
        "llm_score": score,
        "llm_comment": _fallback_comment_for_score(score),
        "judge_request_id": body.get("eval_id") if isinstance(body, dict) else None,
        "judge_latency_ms": None,
        "judge_input_tokens": _build_usage_value(prompt_eval_count),
        "judge_output_tokens": _build_usage_value(eval_count),
        "judge_total_tokens": _build_usage_value(prompt_eval_count, eval_count),
        "judge_reasoning_tokens": None,
        "judge_requests": 1,
        "judge_tool_calls": 0,
        "judge_usage_details": {},
    }


def _manual_no_tools_structured_judge(
    *,
    model_name: str,
    model_url: Optional[str],
    api_key: Optional[str],
    user_prompt: str,
    judge_input: Optional[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> Dict[str, Any]:
    prompt_text = _build_no_tools_prompt_text(user_prompt=user_prompt, judge_input=judge_input)
    try:
        return _native_ollama_structured_score_judge(
            model_name=model_name,
            model_url=model_url,
            api_key=api_key,
            prompt_text=prompt_text,
            temperature=temperature,
        )
    except Exception:
        client = build_sync_openai_client(model_url=model_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": JUDGE_STRUCTURED_SYSTEM_PROMPT},
                {"role": "system", "content": JUDGE_STRUCTURED_NO_TOOLS_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            temperature=float(temperature),
            max_tokens=max(int(max_tokens), 220),
            extra_body=_build_think_payload(reasoning_effort),
        )
        raw_text = _extract_text_content(response)
        score_match = re.search(
            r"\bscore\b\s*(?:is|=|:)\s*([01](?:\.\d+)?)",
            raw_text,
            flags=re.IGNORECASE,
        )
        if not score_match:
            raise ValueError("Missing explicit score marker in no-tools structured judge fallback")
        score = float(score_match.group(1))
        usage = getattr(response, "usage", None)
        return {
            "llm_score": score,
            "llm_comment": _normalize_comment_response(raw_text, score=score),
            "judge_request_id": getattr(response, "id", None),
            "judge_latency_ms": None,
            "judge_input_tokens": _build_usage_value(getattr(usage, "prompt_tokens", None) if usage else None),
            "judge_output_tokens": _build_usage_value(getattr(usage, "completion_tokens", None) if usage else None),
            "judge_total_tokens": _build_usage_value(getattr(usage, "total_tokens", None) if usage else None),
            "judge_reasoning_tokens": None,
            "judge_requests": 1,
            "judge_tool_calls": 0,
            "judge_usage_details": {},
        }


def run_structured_judge(
    *,
    model_name: str,
    model_url: Optional[str],
    api_key: Optional[str],
    user_prompt: str,
    judge_input: Optional[Dict[str, Any]] = None,
    retries: int,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str = "high",
) -> Dict[str, Any]:
    if judge_requires_no_tools_json_fallback(model_name):
        started_at = time.perf_counter()
        result = _manual_no_tools_structured_judge(
            model_name=model_name,
            model_url=model_url,
            api_key=api_key,
            user_prompt=user_prompt,
            judge_input=judge_input,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        result["judge_latency_ms"] = int((time.perf_counter() - started_at) * 1000)
        return result

    model = build_openai_chat_model(
        model_name=model_name,
        model_url=model_url,
        api_key=api_key,
    )

    agent: Agent[None, JudgeResult] = Agent(
        model,
        system_prompt=JUDGE_STRUCTURED_SYSTEM_PROMPT,
        output_type=JudgeResult,
        output_retries=max(0, int(retries)),
    )

    def _invoke() -> Dict[str, Any]:
        result = agent.run_sync(
            user_prompt,
            model_settings={
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "reasoning_effort": str(reasoning_effort or "high"),
            },
        )
        output = result.output
        if not isinstance(output, JudgeResult):
            raise ValueError("Structured judge output has unexpected type")
        return {
            "output": output,
            "usage": extract_run_usage(result),
        }

    started_at = time.perf_counter()
    result_payload = run_with_retries(
        _invoke,
        retries=max(0, int(retries)),
        retry_on=(ModelRetry, UnexpectedModelBehavior, ValueError),
        backoff_sec=0.0,
    )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    output: JudgeResult = result_payload["output"]
    usage = result_payload.get("usage") or {}

    return {
        "llm_score": float(output.llm_score),
        "llm_comment": str(output.llm_comment),
        "judge_request_id": None,
        "judge_latency_ms": elapsed_ms,
        **usage,
    }
