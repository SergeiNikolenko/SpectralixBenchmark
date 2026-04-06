from __future__ import annotations

from typing import Any, Dict, Iterable, List


STUDENT_SYSTEM_PROMPT = (
    "You solve chemistry benchmark questions accurately and compactly. "
    "Use only the question content and allowed tool outputs. "
    "Prefer the shortest final answer that fully satisfies the contract."
)

STUDENT_TOOL_RULES = (
    "Tool rules:\n"
    "- Use tools only when they materially improve correctness, validation, or compact formatting.\n"
    "- Prefer zero-tool answers when the chemistry answer is already clear.\n"
    "- Use chem_python_tool only for chemistry validation, RDKit canonicalization, molecular formula checks, or quick reaction-related calculations.\n"
    "- Use workspace_list_tool and workspace_read_tool to inspect uploaded code or benchmark files when local repository context materially helps.\n"
    "- Use shell_exec_tool or uv_run_tool only for short sandbox-local inspection, validation, or package setup commands.\n"
    "- Use safe_http_get_tool only when the question explicitly requires current external information. Benchmark questions almost never need internet access.\n"
    "- Do not use tools to look for hidden metadata, hidden labels, hidden ids, or gold answers.\n"
    "- Never browse just to confirm chemistry that can be answered from the prompt.\n"
)

TOOL_DECISION_RULES = (
    "Tool decision order:\n"
    "- If the answer is already clear from the question, answer directly with no tool calls.\n"
    "- If you need answer-format guidance or compact normalization, use rubric_hint_tool or chem_format_tool first.\n"
    "- If you need chemistry validation, prefer chem_python_tool before shell_exec_tool or uv_run_tool.\n"
    "- If you need local repository context, use workspace_list_tool and workspace_read_tool before shell_exec_tool.\n"
    "- Use shell_exec_tool or uv_run_tool only for short, targeted local inspection or validation.\n"
    "- Use workspace_write_tool only when the task explicitly requires creating or editing a file.\n"
    "- Use safe_http_get_tool only when current external information is explicitly necessary.\n"
)

STUDENT_COMPLETION_RULES = (
    "Completion criteria:\n"
    "- The final answer must match the requested answer_type format exactly.\n"
    "- Do not output markdown fences, multiple alternative answers, or hidden reasoning notes.\n"
    "- Keep the final answer as short as possible while staying complete.\n"
    "- Use one line when possible; use multiple lines only when the contract explicitly needs them.\n"
    "- Before finalizing, verify that the answer is grounded in the question text and tool outputs only.\n"
    "- Before finalizing, check that the answer is at the correct planning depth for the task level."
)

STUDENT_CODE_AGENT_RULES = (
    "Code-agent execution rules:\n"
    "- If no tool is needed, call final_answer(...) immediately instead of writing exploratory code.\n"
    "- Never place raw chemistry answers directly in executable code.\n"
    "- SMILES strings, precursor lists, reaction labels, and JSON answers must be passed as quoted strings to final_answer(...).\n"
    "- Do not emit bare chemical formulas, SMILES, or natural-language answers as Python code.\n"
    "- Bad: CCO.CN or precursor_a + precursor_b\n"
    '- Good: final_answer("CCO.CN")\n'
    '- Good: final_answer("precursor_a + precursor_b")\n'
    '- Good: final_answer(\'Answer: {"steps":[...]}\')'
)

PARSER_AGENT_INSTRUCTION = (
    "You are a document parsing agent for chemistry exam pages. "
    "Return only a JSON array of parsed questions and never include markdown fences."
)

DEFAULT_FORMAT_INSTRUCTION = (
    "Start the response with 'Answer: <machine-readable answer>'. "
    "If needed, add a very short explanation after that line."
)

LEVEL_CONTRACTS = {
    "a": (
        "Level A contract:\n"
        "- Solve only the local reaction-understanding task.\n"
        "- If reaction center is requested, include the full local event, not a partial subset.\n"
        "- If mechanistic class is requested, give one best-fitting class label only.\n"
        "- Do not replace the requested local transformation with a broader reaction narrative."
    ),
    "b": (
        "Level B contract:\n"
        "- Propose immediate precursors for one retrosynthetic step only.\n"
        "- Do not jump to earlier building blocks or a longer route.\n"
        "- State the main disconnection explicitly.\n"
        "- Prefer the most direct target-forming precursor set over alternative strategic routes."
    ),
    "c": (
        "Level C contract:\n"
        "- Provide a complete connected multistep route.\n"
        "- Include key intermediates or products for each step.\n"
        "- Do not give only a retrosynthetic idea or disconnected route fragments.\n"
        "- Ensure the final step reaches the exact target."
    ),
}

LEVEL_SELF_CHECKS = {
    "a": (
        "Level A self-check:\n"
        "- Did I identify the exact local transformation rather than a broader story?\n"
        "- If reaction center is requested, did I include all relevant atoms or bond changes?\n"
        "- If a class label is requested, did I choose one best label?"
    ),
    "b": (
        "Level B self-check:\n"
        "- Are these immediate precursors rather than earlier retrosynthetic building blocks?\n"
        "- Did I state the intended bond disconnection?\n"
        "- Did I avoid giving a full route or multiple unrelated alternatives?"
    ),
    "c": (
        "Level C self-check:\n"
        "- Do the steps connect into a coherent route?\n"
        "- Does each step have plausible inputs and an intermediate or product?\n"
        "- Does the final step reach the exact target?"
    ),
}

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
        "Return a compact machine-readable answer. "
        "For single-step retrosynthesis tasks, use exactly these lines when applicable: "
        "'Answer: <immediate precursor 1>; <immediate precursor 2>' and "
        "'Disconnection: <very short bond-change description>'. "
        "Do not give a longer route, earlier building blocks, or meta commentary."
    ),
    "reaction_description": (
        "Return a compact machine-readable chemistry answer. "
        "Use only the exact local transformation information requested. "
        "Do not replace a reaction-center or mechanistic-class answer with a broad reaction summary."
    ),
    "full_synthesis": (
        "Return a compact structured route with connected steps. "
        "Preferred format: 'Answer: Step 1: ... | Step 2: ... | Step 3: ...'. "
        "Include key intermediates and the final target-reaching step. "
        "Do not answer with only a high-level retrosynthetic idea."
    ),
}


def _format_instruction(answer_type: str) -> str:
    normalized = (answer_type or "").strip().lower()
    return FORMAT_INSTRUCTIONS.get(normalized, DEFAULT_FORMAT_INSTRUCTION)


def _level_instruction(level: str) -> str:
    normalized = (level or "").strip().lower()
    return LEVEL_CONTRACTS.get(normalized, "Level contract: match the requested planning depth exactly.")


def _level_self_check(level: str) -> str:
    normalized = (level or "").strip().lower()
    return LEVEL_SELF_CHECKS.get(normalized, "Self-check: verify planning depth, format, and exact target match before finalizing.")


def _format_tool_map(available_tools: Iterable[str]) -> str:
    names = {str(name).strip() for name in available_tools if str(name).strip()}
    if not names:
        return "Active tools: none."

    groups: List[str] = []
    chemistry = [name for name in ["chem_python_tool", "smiles_sanity_tool", "unit_convert_tool"] if name in names]
    formatting = [name for name in ["chem_format_tool", "rubric_hint_tool", "json_array_validate_tool"] if name in names]
    workspace = [name for name in ["workspace_list_tool", "workspace_read_tool", "workspace_write_tool"] if name in names]
    execution = [name for name in ["shell_exec_tool", "uv_run_tool"] if name in names]
    network = [name for name in ["safe_http_get_tool"] if name in names]

    if chemistry:
        groups.append(f"Chemistry validation: {', '.join(chemistry)}")
    if formatting:
        groups.append(f"Formatting helpers: {', '.join(formatting)}")
    if workspace:
        groups.append(f"Workspace inspection/editing: {', '.join(workspace)}")
    if execution:
        groups.append(f"Local execution: {', '.join(execution)}")
    if network:
        groups.append(f"Network access: {', '.join(network)}")

    unused = sorted(names.difference({
        *chemistry,
        *formatting,
        *workspace,
        *execution,
        *network,
    }))
    if unused:
        groups.append(f"Other tools: {', '.join(unused)}")

    return "\n".join(f"- {group}" for group in groups)


def build_student_task(question: Dict[str, Any], runtime_context: Dict[str, Any] | None = None) -> str:
    level = str(question.get("level", "") or "")
    answer_type = str(question.get("answer_type", "") or "")
    task_subtype = str(question.get("task_subtype", "") or "")
    question_text = question.get("question_text", "")
    runtime_context = runtime_context or {}
    tools_profile = str(runtime_context.get("tools_profile") or "minimal")
    workspace_root = str(runtime_context.get("workspace_root") or "/sandbox/workspace")
    available_tools = list(runtime_context.get("available_tools") or [])
    return (
        "<role>\n"
        f"{STUDENT_SYSTEM_PROMPT}\n"
        "</role>\n\n"
        "<task>\n"
        "Produce the best final answer for the chemistry question.\n"
        "</task>\n\n"
        "<answer_format>\n"
        f"Benchmark level: {level}\n"
        f"Answer type: {answer_type}\n"
        f"Task subtype: {task_subtype}\n"
        f"Required format: {_format_instruction(answer_type)}\n"
        "</answer_format>\n\n"
        "<runtime_context>\n"
        f"Tools profile: {tools_profile}\n"
        f"Workspace root: {workspace_root}\n"
        f"{_format_tool_map(available_tools)}\n"
        "</runtime_context>\n\n"
        "<task_contract>\n"
        f"{_level_instruction(level)}\n"
        "</task_contract>\n\n"
        "<tool_rules>\n"
        f"{STUDENT_TOOL_RULES}\n"
        "</tool_rules>\n\n"
        "<tool_decision_order>\n"
        f"{TOOL_DECISION_RULES}\n"
        "</tool_decision_order>\n\n"
        "<completion_criteria>\n"
        f"{STUDENT_COMPLETION_RULES}\n"
        "</completion_criteria>\n\n"
        "<pre_final_self_check>\n"
        f"{_level_self_check(level)}\n"
        "</pre_final_self_check>\n\n"
        "<code_agent_rules>\n"
        f"{STUDENT_CODE_AGENT_RULES}\n"
        "</code_agent_rules>\n\n"
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
