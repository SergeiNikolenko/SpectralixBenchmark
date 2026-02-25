from __future__ import annotations

from typing import Any, Dict, Optional
import time

from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior

from .models import build_openai_chat_model
from .retry import run_with_retries
from .schemas import JudgeResult


JUDGE_STRUCTURED_SYSTEM_PROMPT = (
    "You are an expert chemistry exam examiner. "
    "Return only strict structured output according to the required schema. "
    "Score must be in [0, 1], and comment must be concise and factual."
)


def run_structured_judge(
    *,
    model_name: str,
    model_url: Optional[str],
    api_key: Optional[str],
    user_prompt: str,
    retries: int,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
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

    def _invoke() -> JudgeResult:
        result = agent.run_sync(
            user_prompt,
            model_settings={
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            },
        )
        output = result.output
        if not isinstance(output, JudgeResult):
            raise ValueError("Structured judge output has unexpected type")
        return output

    started_at = time.perf_counter()
    output = run_with_retries(
        _invoke,
        retries=max(0, int(retries)),
        retry_on=(ModelRetry, UnexpectedModelBehavior, ValueError),
        backoff_sec=0.0,
    )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    return {
        "llm_score": float(output.llm_score),
        "llm_comment": str(output.llm_comment),
        "judge_request_id": None,
        "judge_latency_ms": elapsed_ms,
    }
