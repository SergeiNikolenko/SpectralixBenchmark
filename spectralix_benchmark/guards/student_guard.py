from __future__ import annotations

from typing import Any, Dict, Optional
import re

from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior

from .models import build_openai_chat_model
from .retry import run_with_retries
from .schemas import StudentGuardOutput


FORMAT_RULES = {
    "single_choice": "Return one option label only (example: A).",
    "multiple_choice": "Return option labels separated by '; ' (example: A; D).",
    "ordering": "Return ordered labels or indices separated by '; ' (example: 4; 2; 3; 1).",
    "msms_structure_prediction": "Return one SMILES string only.",
    "structure": "Return one SMILES string only.",
    "numeric": "Return numeric value only unless unit is explicitly required.",
}


def is_answer_invalid(answer_type: str, normalized_answer: str) -> bool:
    text = (normalized_answer or "").strip()
    if not text:
        return True

    t = (answer_type or "").strip().lower()
    if t == "single_choice":
        return re.fullmatch(r"[A-Za-z0-9]+", text) is None
    if t in {"multiple_choice", "ordering"}:
        tokens = [part.strip() for part in text.replace(",", ";").split(";") if part.strip()]
        return len(tokens) == 0
    if t in {"msms_structure_prediction", "structure"}:
        return len(text.split()) != 1
    if t == "numeric":
        return re.search(r"-?\d+(?:\.\d+)?", text) is None

    return False


def _build_prompt(question: Dict[str, Any], raw_answer: str, normalized_answer: str) -> str:
    answer_type = str(question.get("answer_type") or "").strip().lower()
    format_rule = FORMAT_RULES.get(
        answer_type,
        "Return a concise machine-readable final answer with no markdown fences.",
    )
    return (
        "<task>\n"
        "Repair the answer format without changing meaning unless a minimal rewrite is required.\n"
        "</task>\n\n"
        "<rules>\n"
        "- Preserve the intended chemistry answer whenever possible.\n"
        "- Repair formatting only.\n"
        "- Do not add explanations, markdown fences, or benchmark commentary.\n"
        "- If the answer cannot be repaired without guessing, return format_ok=false.\n"
        "</rules>\n\n"
        "<question>\n"
        f"{question.get('question_text', '')}\n"
        "</question>\n\n"
        "<answer_format>\n"
        f"Answer type: {answer_type}\n"
        f"Required format: {format_rule}\n"
        "</answer_format>\n\n"
        "<raw_answer>\n"
        f"{raw_answer or ''}\n"
        "</raw_answer>\n\n"
        "<current_normalized_answer>\n"
        f"{normalized_answer or ''}\n"
        "</current_normalized_answer>"
    )


def run_student_guard(
    *,
    question: Dict[str, Any],
    raw_answer: str,
    normalized_answer: str,
    model_name: str,
    model_url: Optional[str],
    api_key: Optional[str],
    retries: int,
    reasoning_effort: str = "high",
) -> StudentGuardOutput:
    model = build_openai_chat_model(
        model_name=model_name,
        model_url=model_url,
        api_key=api_key,
    )
    agent: Agent[None, StudentGuardOutput] = Agent(
        model,
        output_type=StudentGuardOutput,
        system_prompt=(
            "You repair chemistry answers into strict machine-readable format. "
            "Return only structured output matching schema. "
            "Do not solve a different problem, and do not add explanations beyond the schema fields."
        ),
        output_retries=max(0, int(retries)),
    )

    prompt = _build_prompt(question, raw_answer, normalized_answer)

    def _invoke() -> StudentGuardOutput:
        result = agent.run_sync(
            prompt,
            model_settings={
                "temperature": 0.0,
                "max_tokens": 256,
                "reasoning_effort": str(reasoning_effort or "high"),
            },
        )
        output = result.output
        if not isinstance(output, StudentGuardOutput):
            raise ValueError("Student guard output has unexpected type")
        if not output.format_ok:
            raise ModelRetry("Returned answer format is not acceptable")
        if not output.final_answer.strip():
            raise ModelRetry("Returned final_answer is empty")
        return output

    return run_with_retries(
        _invoke,
        retries=max(0, int(retries)),
        retry_on=(ModelRetry, UnexpectedModelBehavior, ValueError),
        backoff_sec=0.0,
    )
