from __future__ import annotations

from typing import Any, Dict, List


GENERIC_G_EVAL_SPEC = {
    "criteria": [
        "Chemical correctness relative to the canonical answer",
        "Match to the requested task contract",
        "Presence of key required elements without material contradictions",
    ],
    "evaluation_steps": [
        "Identify the core chemistry claim in the canonical answer.",
        "Check whether the student answer matches the same task and planning depth.",
        "Check whether key required elements are present and chemically plausible.",
        "Assign a rubric score from 0 to 10.",
    ],
    "rubric": [
        "0 = entirely incorrect, incompatible, or missing",
        "2 = mostly incorrect with only minor relevant overlap",
        "4 = partially correct but missing or contradicting important chemistry details",
        "6 = materially correct but incomplete or imprecise on important details",
        "8 = mostly correct with only minor omissions or imprecision",
        "10 = fully correct and complete for the requested answer type",
    ],
}


G_EVAL_SPECS: Dict[str, Dict[str, List[str]]] = {
    "text": GENERIC_G_EVAL_SPEC,
    "reaction_description": {
        "criteria": [
            "Correct identification of the requested local transformation",
            "Chemical plausibility of any mechanism, selectivity, or reagent claims",
            "Coverage of the essential local reaction outcome without contradictions",
        ],
        "evaluation_steps": [
            "Identify the local transformation or label required by the canonical answer.",
            "Check whether the student answered that exact local question rather than a broader summary.",
            "Check whether any chemistry claims are plausible and whether critical local details are missing.",
            "Assign a rubric score from 0 to 10.",
        ],
        "rubric": GENERIC_G_EVAL_SPEC["rubric"],
    },
    "property_determination": {
        "criteria": [
            "Correctness of the determined property or value",
            "Consistency with chemistry context in the question",
            "Completeness of qualifiers such as sign, trend, or unit when relevant",
            "Absence of contradictions",
        ],
        "evaluation_steps": [
            "Identify the target property the question asks for.",
            "Compare the student answer with the canonical answer for correctness.",
            "Check whether required qualifiers such as sign, trend, or units are present when relevant.",
            "Check for contradictions or unsupported chemistry claims.",
            "Assign a rubric score from 0 to 10.",
        ],
        "rubric": GENERIC_G_EVAL_SPEC["rubric"],
    },
    "full_synthesis": {
        "criteria": [
            "Route plausibility toward the target",
            "Connected multistep structure with chemically plausible key intermediates or transformations",
            "Coverage of critical target-reaching steps without fatal contradictions",
        ],
        "evaluation_steps": [
            "Identify the essential target-reaching route requirement from the canonical answer.",
            "Check whether the student route is connected and can plausibly reach the same target.",
            "Check whether key intermediates or transformations are chemically valid and whether critical steps are missing.",
            "Assign a rubric score from 0 to 10.",
        ],
        "rubric": GENERIC_G_EVAL_SPEC["rubric"],
    },
}


LEVEL_SPEC_OVERRIDES: Dict[str, Dict[str, List[str]]] = {
    "a": G_EVAL_SPECS["reaction_description"],
    "b": {
        "criteria": [
            "Correct immediate precursor set for one retrosynthetic step",
            "Correct main disconnection at the requested planning depth",
            "Credit chemically plausible immediate alternatives, but penalize earlier retrosynthetic jumps or unrelated routes",
        ],
        "evaluation_steps": [
            "Identify the expected immediate precursor set and main disconnection from the canonical answer.",
            "Check whether the student proposed immediate precursors rather than earlier building blocks or a multistep plan.",
            "Check whether the student disconnection is chemically plausible and aligned with the same target-forming step.",
            "Assign a rubric score from 0 to 10.",
        ],
        "rubric": GENERIC_G_EVAL_SPEC["rubric"],
    },
    "c": G_EVAL_SPECS["full_synthesis"],
}


def get_g_eval_spec(
    answer_type: Any,
    *,
    level: Any = None,
    task_subtype: Any = None,
) -> Dict[str, List[str]]:
    normalized_answer_type = str(answer_type or "").strip().lower()
    normalized_level = str(level or "").strip().lower()
    normalized_task_subtype = str(task_subtype or "").strip().lower()

    spec = G_EVAL_SPECS.get(normalized_answer_type, GENERIC_G_EVAL_SPEC)

    if normalized_level in LEVEL_SPEC_OVERRIDES:
        spec = LEVEL_SPEC_OVERRIDES[normalized_level]

    if normalized_level == "b" and normalized_task_subtype == "immediate_precursor_with_disconnection":
        spec = {
            "criteria": [
                "Correct immediate precursor set",
                "Explicit and correct main disconnection",
                "No jump to earlier retrosynthetic stages or unrelated alternatives",
            ],
            "evaluation_steps": [
                "Identify the canonical immediate precursor set and disconnection.",
                "Check whether the student answer includes both immediate precursors and a matching disconnection.",
                "Penalize answers that skip backward to earlier building blocks or propose a different route family.",
                "Assign a rubric score from 0 to 10.",
            ],
            "rubric": GENERIC_G_EVAL_SPEC["rubric"],
        }

    return {
        "criteria": list(spec["criteria"]),
        "evaluation_steps": list(spec["evaluation_steps"]),
        "rubric": list(spec["rubric"]),
    }
