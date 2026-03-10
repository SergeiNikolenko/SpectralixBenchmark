from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior

from .models import build_openai_chat_model
from .retry import run_with_retries
from .schemas import ParsedQuestionSchema, parsed_questions_to_dicts


PARSER_REPAIR_SYSTEM_PROMPT = (
    "You repair parser outputs for chemistry exam pages. "
    "Return a strict list of question objects matching schema. "
    "Preserve source meaning, do not invent missing content, and return an empty list if the page has no questions."
)


def repair_parsed_questions(
    *,
    raw_response: str,
    marker_prompt: str,
    exam_id: str,
    page_id: int,
    model_name: str,
    model_url: Optional[str],
    api_key: Optional[str],
    retries: int,
    reasoning_effort: str = "high",
) -> List[Dict[str, Any]]:
    model = build_openai_chat_model(
        model_name=model_name,
        model_url=model_url,
        api_key=api_key,
    )

    agent: Agent[None, list[ParsedQuestionSchema]] = Agent(
        model,
        output_type=list[ParsedQuestionSchema],
        system_prompt=PARSER_REPAIR_SYSTEM_PROMPT,
        output_retries=max(0, int(retries)),
    )

    prompt = (
        "<task>\n"
        "Repair the parser output so it matches schema without inventing information.\n"
        "</task>\n\n"
        "<page_context>\n"
        f"Exam ID: {exam_id}\n"
        f"Page ID: {page_id}\n"
        "</page_context>\n\n"
        "<rules>\n"
        "- Preserve source wording when possible.\n"
        "- Repair structure and field normalization only.\n"
        "- If a field is missing and cannot be recovered from the raw response, leave it empty or use error status rather than guessing.\n"
        "</rules>\n\n"
        "<original_parsing_instructions>\n"
        f"{marker_prompt}\n"
        "</original_parsing_instructions>\n\n"
        "<raw_model_response>\n"
        f"{raw_response}\n"
        "</raw_model_response>\n"
    )

    def _invoke() -> List[Dict[str, Any]]:
        result = agent.run_sync(
            prompt,
            model_settings={
                "temperature": 0.0,
                "max_tokens": 1200,
                "reasoning_effort": str(reasoning_effort or "high"),
            },
        )
        output = result.output
        if not isinstance(output, list):
            raise ValueError("Parser repair output must be a list")
        for item in output:
            if not isinstance(item, ParsedQuestionSchema):
                raise ModelRetry("Output list contains invalid item type")
        return parsed_questions_to_dicts(output)

    return run_with_retries(
        _invoke,
        retries=max(0, int(retries)),
        retry_on=(ModelRetry, UnexpectedModelBehavior, ValueError),
        backoff_sec=0.0,
    )
