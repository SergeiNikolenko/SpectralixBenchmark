from __future__ import annotations

from typing import Any, Dict


STUDENT_AGENT_INSTRUCTION = (
    "You are a chemistry benchmark agent. "
    "You must produce concise, machine-readable answers that strictly follow the requested format."
)

PARSER_AGENT_INSTRUCTION = (
    "You are a document parsing agent for chemistry exam pages. "
    "Return only a JSON array of parsed questions and never include markdown fences."
)

DEFAULT_FORMAT_INSTRUCTION = (
    "Start the response with 'Answer: <machine-readable answer>'. "
    "If needed, add a very short explanation after that line."
)

FORMAT_INSTRUCTIONS = {
    "single_choice": "Return only one option label (example: A).",
    "multiple_choice": "Return only option labels separated by '; ' (example: A; D).",
    "ordering": "Return only the ordered labels/numbers separated by '; ' (example: 4; 2; 3; 1).",
    "msms_structure_prediction": "Return only one SMILES string on a single line.",
    "structure": "Return only one SMILES string on a single line.",
}


def _format_instruction(answer_type: str) -> str:
    normalized = (answer_type or "").strip().lower()
    return FORMAT_INSTRUCTIONS.get(normalized, DEFAULT_FORMAT_INSTRUCTION)


def build_student_task(question: Dict[str, Any]) -> str:
    answer_type = str(question.get("answer_type", "") or "")
    question_text = question.get("question_text", "")
    return (
        f"{STUDENT_AGENT_INSTRUCTION}\n\n"
        f"Output format rule: {_format_instruction(answer_type)}\n\n"
        "Question metadata:\n"
        f"- exam_id: {question.get('exam_id')}\n"
        f"- page_id: {question.get('page_id')}\n"
        f"- question_id: {question.get('question_id')}\n"
        f"- answer_type: {answer_type}\n\n"
        "Question:\n"
        f"{question_text}\n\n"
        "If helpful, call available tools for formatting and sanity checks before final answer."
    )


def build_parse_page_task(exam_id: str, page_id: int, marker_prompt: str, image_path: str) -> str:
    return (
        f"{PARSER_AGENT_INSTRUCTION}\n\n"
        f"Exam ID: {exam_id}\n"
        f"Page ID: {page_id}\n"
        f"Image path: {image_path}\n\n"
        "Follow this extraction specification exactly:\n"
        f"{marker_prompt}\n\n"
        "Before finalizing, validate the JSON with available validation tool. "
        "Output only a valid JSON array and nothing else."
    )
