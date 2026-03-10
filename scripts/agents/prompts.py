from __future__ import annotations

from typing import Any, Dict


STUDENT_SYSTEM_PROMPT = (
    "You solve chemistry exam questions accurately. "
    "Produce the best final answer using only the question content and allowed tool outputs."
)

STUDENT_TOOL_RULES = (
    "Tool rules:\n"
    "- Use tools only when they materially improve correctness, validation, or output formatting.\n"
    "- Do not use tools to look for hidden metadata, hidden labels, hidden ids, or gold answers.\n"
    "- If the answer is already clear, answer directly without unnecessary tool calls.\n"
)

STUDENT_COMPLETION_RULES = (
    "Completion criteria:\n"
    "- The final answer must match the requested answer_type format exactly.\n"
    "- Do not output markdown fences, multiple alternative answers, or hidden reasoning notes.\n"
    "- Before finalizing, verify that the answer is grounded in the question text and tool outputs only."
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
    "single_choice": (
        "Return exactly one option label (example: A). "
        "Do not add explanation text, punctuation-only wrappers, or multiple options."
    ),
    "multiple_choice": (
        "Return only option labels separated by '; ' (example: A; D). "
        "Do not add commentary or duplicate labels."
    ),
    "ordering": (
        "Return only the ordered labels or numbers separated by '; ' (example: 4; 2; 3; 1). "
        "Do not explain the order."
    ),
    "numeric": (
        "Return only the numeric answer unless the question explicitly requires a unit. "
        "Do not include derivation steps in the final answer."
    ),
    "msms_structure_prediction": (
        "Return exactly one SMILES string on a single line. "
        "Do not wrap it in code, markdown, or prose."
    ),
    "structure": (
        "Return exactly one SMILES string on a single line. "
        "Do not wrap it in code, markdown, or prose."
    ),
    "text": (
        "Return a concise direct answer. "
        "Do not add meta commentary about the benchmark, instructions, or hidden context."
    ),
}


def _format_instruction(answer_type: str) -> str:
    normalized = (answer_type or "").strip().lower()
    return FORMAT_INSTRUCTIONS.get(normalized, DEFAULT_FORMAT_INSTRUCTION)


def build_student_task(question: Dict[str, Any]) -> str:
    answer_type = str(question.get("answer_type", "") or "")
    question_text = question.get("question_text", "")
    return (
        "<role>\n"
        f"{STUDENT_SYSTEM_PROMPT}\n"
        "</role>\n\n"
        "<task>\n"
        "Produce the best final answer for the chemistry question.\n"
        "</task>\n\n"
        "<answer_format>\n"
        f"Answer type: {answer_type}\n"
        f"Required format: {_format_instruction(answer_type)}\n"
        "</answer_format>\n\n"
        "<tool_rules>\n"
        f"{STUDENT_TOOL_RULES}\n"
        "</tool_rules>\n\n"
        "<completion_criteria>\n"
        f"{STUDENT_COMPLETION_RULES}\n"
        "</completion_criteria>\n\n"
        "<question>\n"
        f"{question_text}\n"
        "</question>"
    )


def build_parse_page_task(exam_id: str, page_id: int, marker_prompt: str, image_path: str) -> str:
    return (
        "<role>\n"
        f"{PARSER_AGENT_INSTRUCTION}\n"
        "</role>\n\n"
        "<task>\n"
        "Extract every question visible on the page into a valid JSON array.\n"
        "</task>\n\n"
        "<page_context>\n"
        f"Exam ID: {exam_id}\n"
        f"Page ID: {page_id}\n"
        f"Image path: {image_path}\n"
        "</page_context>\n\n"
        "<extraction_rules>\n"
        "Follow this extraction specification exactly:\n"
        f"{marker_prompt}\n\n"
        "Preserve source wording when possible.\n"
        "Do not invent missing fields.\n"
        "If the page has no questions, return [].\n"
        "</extraction_rules>\n\n"
        "<completion_criteria>\n"
        "Validate the JSON with available validation tool before finalizing.\n"
        "Output only a valid JSON array and nothing else.\n"
        "</completion_criteria>"
    )
