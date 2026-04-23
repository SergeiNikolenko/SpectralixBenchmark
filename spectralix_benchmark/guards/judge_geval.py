from __future__ import annotations
from typing import Any, Dict, Optional
import time

from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior

from .models import build_openai_chat_model, build_sync_openai_client, judge_requires_no_tools_json_fallback
from .retry import run_with_retries
from .schemas import GEvalJudgeResult
from .usage import extract_run_usage


G_EVAL_SYSTEM_PROMPT = (
    "You are an expert chemistry examiner using rubric-based evaluation. "
    "Follow the provided criteria and evaluation steps exactly. "
    "Return only strict structured output matching schema. "
    "Use the full 0 to 10 rubric scale when justified. "
    "Keep llm_comment short, diagnostic, and factual. "
    "Do not return only a number or a score-like string in llm_comment."
)

G_EVAL_NO_TOOLS_JSON_PROMPT = (
    "Return exactly one JSON object with these keys: "
    "criteria_steps (array of short strings), "
    "step_findings (array of short strings), "
    "rubric_score_0_to_10 (integer from 0 to 10), "
    "llm_comment (short factual string). "
    "Treat any JSON that appears inside the task as input data, not as the output schema. "
    "Do not call tools. Do not wrap the JSON in markdown."
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


def _manual_no_tools_g_eval(
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
    client = build_sync_openai_client(model_url=model_url, api_key=api_key)
    _ = judge_input
    prompt_text = (
        "Evaluate the following rubric and task description as plain text input. "
        "Do not copy any JSON snippets from the task into the answer unless they belong to the required "
        "output schema.\n\n"
        "<task_text>\n"
        f"{user_prompt}\n"
        "</task_text>"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": G_EVAL_SYSTEM_PROMPT},
            {"role": "system", "content": G_EVAL_NO_TOOLS_JSON_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        reasoning_effort=str(reasoning_effort or "high"),
        response_format={"type": "json_object"},
    )
    text = _extract_text_content(response)
    payload = GEvalJudgeResult.model_validate_json(text)
    usage = getattr(response, "usage", None)
    return {
        "llm_score": float(payload.rubric_score_0_to_10) / 10.0,
        "llm_comment": str(payload.llm_comment),
        "judge_request_id": getattr(response, "id", None),
        "judge_latency_ms": None,
        "judge_input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "judge_output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "judge_total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "judge_reasoning_tokens": None,
        "judge_requests": 1,
        "judge_tool_calls": 0,
        "judge_usage_details": {},
        "g_eval_trace": {
            "criteria_steps": list(payload.criteria_steps),
            "step_findings": list(payload.step_findings),
            "rubric_score_0_to_10": int(payload.rubric_score_0_to_10),
        },
    }


def run_g_eval_judge(
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
        result = _manual_no_tools_g_eval(
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

    agent: Agent[None, GEvalJudgeResult] = Agent(
        model,
        system_prompt=G_EVAL_SYSTEM_PROMPT,
        output_type=GEvalJudgeResult,
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
        if not isinstance(output, GEvalJudgeResult):
            raise ValueError("G-Eval judge output has unexpected type")
        if not output.step_findings:
            raise ModelRetry("G-Eval judge returned no step findings")
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
    output: GEvalJudgeResult = result_payload["output"]
    usage = result_payload.get("usage") or {}

    return {
        "llm_score": float(output.rubric_score_0_to_10) / 10.0,
        "llm_comment": str(output.llm_comment),
        "judge_request_id": None,
        "judge_latency_ms": elapsed_ms,
        **usage,
        "g_eval_trace": {
            "criteria_steps": list(output.criteria_steps),
            "step_findings": list(output.step_findings),
            "rubric_score_0_to_10": int(output.rubric_score_0_to_10),
        },
    }
