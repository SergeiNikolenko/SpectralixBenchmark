from __future__ import annotations

from typing import Any, Dict, List


GENERIC_G_EVAL_SPEC = {
    "criteria": [
        "Chemical correctness relative to the canonical answer",
        "Relevance to the question being asked",
        "Completeness of the final answer for the requested answer type",
        "Absence of contradictions or materially incorrect claims",
    ],
    "evaluation_steps": [
        "Identify the core chemistry claim in the canonical answer.",
        "Compare the student answer against that claim for correctness.",
        "Check whether key required elements are present or missing.",
        "Check whether the answer contains chemistry contradictions or incompatible statements.",
        "Assign a rubric score from 0 to 10 based on overall correctness and completeness.",
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
            "Correct identification of the main transformation",
            "Chemical plausibility of mechanism, selectivity, or reagents if mentioned",
            "Coverage of the essential reaction outcome",
            "Absence of incorrect reaction claims",
        ],
        "evaluation_steps": [
            "Identify the expected reaction outcome from the canonical answer.",
            "Check whether the student described the same transformation.",
            "Check whether any stated reagent, mechanism, or selectivity claim is chemically plausible.",
            "Check whether critical reaction details are missing or contradicted.",
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
            "Validity of the proposed synthetic route relative to the canonical answer",
            "Chemical plausibility of key intermediates, reagents, and transformations",
            "Coverage of critical steps required to reach the target",
            "Absence of fatal route contradictions",
        ],
        "evaluation_steps": [
            "Identify the essential route or target transformation in the canonical answer.",
            "Check whether the student route can plausibly reach the same target.",
            "Check whether key intermediates, reagents, or transformations are chemically valid.",
            "Check whether critical steps are missing or contradicted.",
            "Assign a rubric score from 0 to 10.",
        ],
        "rubric": GENERIC_G_EVAL_SPEC["rubric"],
    },
}


def get_g_eval_spec(answer_type: Any) -> Dict[str, List[str]]:
    normalized = str(answer_type or "").strip().lower()
    spec = G_EVAL_SPECS.get(normalized, GENERIC_G_EVAL_SPEC)
    return {
        "criteria": list(spec["criteria"]),
        "evaluation_steps": list(spec["evaluation_steps"]),
        "rubric": list(spec["rubric"]),
    }
