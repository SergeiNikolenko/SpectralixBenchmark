from __future__ import annotations

from typing import Any, Dict, Optional
import time

from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior

from .models import build_openai_chat_model
from .retry import run_with_retries
from .schemas import GEvalJudgeResult
from .usage import extract_run_usage


G_EVAL_SYSTEM_PROMPT = (
    "You are an expert chemistry exam examiner using rubric-based evaluation. "
    "Follow the provided criteria and evaluation steps. "
    "Return only strict structured output matching schema. "
    "Use the full 0 to 10 rubric scale when justified."
)


def run_g_eval_judge(
    *,
    model_name: str,
    model_url: Optional[str],
    api_key: Optional[str],
    user_prompt: str,
    retries: int,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str = "high",
) -> Dict[str, Any]:
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
