from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional

BENCHMARK_TAXONOMY_ID = "benchmark_taxonomy"


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _default_legacy_answer_type(level: str) -> str:
    normalized = _normalize(level)
    if normalized == "a":
        return "reaction_description"
    if normalized == "c":
        return "full_synthesis"
    return "text"


SUBTRACK_SPECS: Dict[str, Dict[str, Any]] = {
    "reaction_center_identification": {
        "benchmark_suite": "A",
        "suite_label": "Local Reaction Reasoning",
        "planning_horizon": "local",
        "task_mode": "recognition",
        "benchmark_subtrack": "A1",
        "subtrack_label": "Bond-Change Localization",
        "core_depth_score": True,
        "expected_output_schema": "reaction_edit_summary",
        "judge_rubric_id": "g_eval.reaction_description.v1",
        "difficulty_proxies": [
            "atom_mapping_availability",
            "bond_edit_count",
            "local_reaction_center_completeness",
        ],
    },
    "mechanistic_classification": {
        "benchmark_suite": "A",
        "suite_label": "Local Reaction Reasoning",
        "planning_horizon": "local",
        "task_mode": "recognition",
        "benchmark_subtrack": "A2",
        "subtrack_label": "Mechanistic Inference",
        "core_depth_score": True,
        "expected_output_schema": "mechanistic_label",
        "judge_rubric_id": "g_eval.reaction_description.v1",
        "difficulty_proxies": [
            "mechanistic_ambiguity",
            "charge_and_polarity_pattern",
            "local_explanation_burden",
        ],
    },
    "transformation_classification": {
        "benchmark_suite": "A",
        "suite_label": "Local Reaction Reasoning",
        "planning_horizon": "local",
        "task_mode": "recognition",
        "benchmark_subtrack": "A1",
        "subtrack_label": "Bond-Change Localization",
        "core_depth_score": True,
        "expected_output_schema": "transformation_label",
        "judge_rubric_id": "g_eval.reaction_description.v1",
        "difficulty_proxies": [
            "functional_group_change_pattern",
            "local_transformation_specificity",
        ],
    },
    "reagent_role_identification": {
        "benchmark_suite": "G",
        "suite_label": "Procedure Grounding",
        "planning_horizon": "auxiliary",
        "task_mode": "grounding_extraction",
        "benchmark_subtrack": "G1",
        "subtrack_label": "Reagent Role Identification",
        "core_depth_score": False,
        "expected_output_schema": "role_mapping",
        "judge_rubric_id": "g_eval.reaction_description.v1",
        "difficulty_proxies": [
            "procedure_text_grounding",
            "role_slot_assignment",
            "reagent_role_ambiguity",
        ],
    },
    "condition_role_identification": {
        "benchmark_suite": "G",
        "suite_label": "Procedure Grounding",
        "planning_horizon": "auxiliary",
        "task_mode": "grounding_extraction",
        "benchmark_subtrack": "G2",
        "subtrack_label": "Condition Role Identification",
        "core_depth_score": False,
        "expected_output_schema": "condition_role_mapping",
        "judge_rubric_id": "g_eval.reaction_description.v1",
        "difficulty_proxies": [
            "procedure_text_grounding",
            "condition_role_disambiguation",
            "solvent_catalyst_assignment",
        ],
    },
    "immediate_precursor_prediction": {
        "benchmark_suite": "B",
        "suite_label": "Single-Step Disconnection Reasoning",
        "planning_horizon": "single_step",
        "task_mode": "constrained_generation",
        "benchmark_subtrack": "B1",
        "subtrack_label": "Precursor Proposal",
        "core_depth_score": True,
        "expected_output_schema": "precursor_set",
        "judge_rubric_id": "g_eval.text.v1",
        "difficulty_proxies": [
            "precursor_count",
            "disconnection_obviousness",
            "procedure_context_constraints",
        ],
    },
    "immediate_precursor_with_disconnection": {
        "benchmark_suite": "B",
        "suite_label": "Single-Step Disconnection Reasoning",
        "planning_horizon": "single_step",
        "task_mode": "constrained_generation",
        "benchmark_subtrack": "B2",
        "subtrack_label": "Disconnection Justification",
        "core_depth_score": True,
        "expected_output_schema": "precursor_set_with_disconnection",
        "judge_rubric_id": "g_eval.text.immediate_precursor_with_disconnection.v1",
        "difficulty_proxies": [
            "plausible_disconnection_count",
            "route_family_drift_risk",
            "structural_constraint_burden",
        ],
    },
    "reference_route_planning": {
        "benchmark_suite": "C",
        "suite_label": "Route-Level Synthesis Planning",
        "planning_horizon": "route_level",
        "task_mode": "open_planning",
        "benchmark_subtrack": "C2",
        "subtrack_label": "Reference-Route Planning",
        "core_depth_score": True,
        "expected_output_schema": "reference_route_plan",
        "judge_rubric_id": "g_eval.full_synthesis.v1",
        "difficulty_proxies": [
            "route_depth",
            "branching_and_convergence",
            "reference_route_alignment",
        ],
    },
    "route_design": {
        "benchmark_suite": "C",
        "suite_label": "Route-Level Synthesis Planning",
        "planning_horizon": "route_level",
        "task_mode": "open_planning",
        "benchmark_subtrack": "C3",
        "subtrack_label": "Open Route Design",
        "core_depth_score": True,
        "expected_output_schema": "route_design_plan",
        "judge_rubric_id": "g_eval.full_synthesis.v1",
        "difficulty_proxies": [
            "route_depth",
            "constraint_satisfaction",
            "key_intermediate_selection",
        ],
    },
}

GENERIC_LEVEL_SPECS: Dict[str, Dict[str, Any]] = {
    "a": {
        "benchmark_suite": "A",
        "suite_label": "Local Reaction Reasoning",
        "planning_horizon": "local",
        "task_mode": "recognition",
        "benchmark_subtrack": "A0",
        "subtrack_label": "Generic Local Reaction Reasoning",
        "core_depth_score": True,
        "expected_output_schema": "local_reasoning_answer",
        "judge_rubric_id": "g_eval.reaction_description.v1",
        "difficulty_proxies": ["legacy_level_a_difficulty"],
    },
    "b": {
        "benchmark_suite": "B",
        "suite_label": "Single-Step Disconnection Reasoning",
        "planning_horizon": "single_step",
        "task_mode": "constrained_generation",
        "benchmark_subtrack": "B0",
        "subtrack_label": "Generic Single-Step Disconnection Reasoning",
        "core_depth_score": True,
        "expected_output_schema": "single_step_retrosynthesis_answer",
        "judge_rubric_id": "g_eval.text.v1",
        "difficulty_proxies": ["legacy_level_b_difficulty"],
    },
    "c": {
        "benchmark_suite": "C",
        "suite_label": "Route-Level Synthesis Planning",
        "planning_horizon": "route_level",
        "task_mode": "open_planning",
        "benchmark_subtrack": "C0",
        "subtrack_label": "Generic Route-Level Synthesis Planning",
        "core_depth_score": True,
        "expected_output_schema": "route_planning_answer",
        "judge_rubric_id": "g_eval.full_synthesis.v1",
        "difficulty_proxies": ["legacy_level_c_difficulty"],
    },
}

CORE_SUITES = ("A", "B", "C")


def get_benchmark_taxonomy_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    task_subtype = _normalize(row.get("task_subtype"))
    level = _normalize(row.get("level"))
    spec = SUBTRACK_SPECS.get(task_subtype) or GENERIC_LEVEL_SPECS.get(level, {})
    benchmark_suite = spec.get("benchmark_suite")
    legacy_answer_type = row.get("answer_type") or _default_legacy_answer_type(level)
    metadata = {
        "taxonomy_version": BENCHMARK_TAXONOMY_ID,
        "planning_horizon": spec.get("planning_horizon"),
        "task_mode": spec.get("task_mode"),
        "benchmark_suite": benchmark_suite,
        "benchmark_subtrack": spec.get("benchmark_subtrack"),
        "suite_label": spec.get("suite_label"),
        "subtrack_label": spec.get("subtrack_label"),
        "core_depth_score": bool(spec.get("core_depth_score", benchmark_suite in CORE_SUITES)),
        "difficulty_proxies": list(spec.get("difficulty_proxies", [])),
        "eval_contract_id": f"legacy_eval_contract.{_normalize(legacy_answer_type) or 'unknown'}.v1",
        "expected_output_schema": spec.get("expected_output_schema"),
        "judge_rubric_id": spec.get("judge_rubric_id"),
        "legacy_level": row.get("level"),
        "legacy_answer_type": legacy_answer_type,
        "legacy_difficulty": row.get("difficulty"),
        "difficulty_taxonomy_status": (
            "legacy_proxy" if row.get("difficulty") is not None else "unavailable"
        ),
    }
    return metadata


def overlay_benchmark_taxonomy_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(row)
    payload.pop("difficulty_v2_status", None)
    payload.update(get_benchmark_taxonomy_metadata(row))
    return payload


def _status(row: Mapping[str, Any], key: str) -> str:
    return str(row.get(key, "ok") or "ok")


def _status_pair(row: Mapping[str, Any]) -> tuple[str, str]:
    return _status(row, "student_status"), _status(row, "row_status")


def _is_ok_pair(student_status: str, row_status: str) -> bool:
    return student_status == "ok" and row_status == "ok"


def _bucket_template() -> Dict[str, Any]:
    return {
        "count": 0,
        "quality_count": 0,
        "quality_total_score": 0.0,
        "quality_max_score": 0.0,
        "end_to_end_max_score": 0.0,
        "ok_count": 0,
    }


def _bucketize(rows: Iterable[Mapping[str, Any]], key_name: str) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = defaultdict(_bucket_template)
    for row in rows:
        bucket_key = str(row.get(key_name) or "unknown")
        bucket = buckets[bucket_key]
        bucket["count"] += 1
        bucket["end_to_end_max_score"] += float(row.get("max_score", 0) or 0)

        student_status, row_status = _status_pair(row)
        if _is_ok_pair(student_status, row_status):
            bucket["ok_count"] += 1

        if row_status == "ok" and row.get("final_score") is not None:
            bucket["quality_count"] += 1
            bucket["quality_total_score"] += float(row.get("final_score", 0) or 0)
            bucket["quality_max_score"] += float(row.get("max_score", 0) or 0)

    result: Dict[str, Dict[str, Any]] = {}
    for bucket_key, values in sorted(buckets.items()):
        quality_max = float(values["quality_max_score"])
        end_to_end_max = float(values["end_to_end_max_score"])
        count = int(values["count"])
        quality_norm = (
            float(values["quality_total_score"]) / quality_max if quality_max > 0 else None
        )
        end_to_end = (
            float(values["quality_total_score"]) / end_to_end_max if end_to_end_max > 0 else None
        )
        result[bucket_key] = {
            "count": count,
            "quality_count": int(values["quality_count"]),
            "quality_total_score": float(values["quality_total_score"]),
            "quality_max_score": quality_max,
            "end_to_end_max_score": end_to_end_max,
            "quality_score": quality_norm,
            "end_to_end_score": end_to_end,
            "ok_rate": (int(values["ok_count"]) / count) if count else None,
        }
    return result


def compute_benchmark_taxonomy_metrics(judge_rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    overlay_rows = [overlay_benchmark_taxonomy_fields(row) for row in judge_rows]

    breakdown_by_suite = _bucketize(overlay_rows, "benchmark_suite")
    breakdown_by_subtrack = _bucketize(overlay_rows, "benchmark_subtrack")
    breakdown_by_task_subtype = _bucketize(overlay_rows, "task_subtype")
    breakdown_by_task_mode = _bucketize(overlay_rows, "task_mode")
    breakdown_by_planning_horizon = _bucketize(overlay_rows, "planning_horizon")

    core_quality_scores = [
        breakdown_by_suite.get(suite, {}).get("quality_score") for suite in CORE_SUITES
    ]
    core_e2e_scores = [
        breakdown_by_suite.get(suite, {}).get("end_to_end_score") for suite in CORE_SUITES
    ]

    return {
        "taxonomy_version": BENCHMARK_TAXONOMY_ID,
        "core_suites": list(CORE_SUITES),
        "auxiliary_suites": ["G"],
        "macro_depth_quality_score": (
            mean(core_quality_scores) if all(score is not None for score in core_quality_scores) else None
        ),
        "macro_depth_end_to_end_score": (
            mean(core_e2e_scores) if all(score is not None for score in core_e2e_scores) else None
        ),
        "auxiliary_grounding_quality_score": breakdown_by_suite.get("G", {}).get(
            "quality_score"
        ),
        "auxiliary_grounding_end_to_end_score": breakdown_by_suite.get("G", {}).get(
            "end_to_end_score"
        ),
        "breakdown_by_suite": breakdown_by_suite,
        "breakdown_by_subtrack": breakdown_by_subtrack,
        "breakdown_by_task_subtype": breakdown_by_task_subtype,
        "breakdown_by_task_mode": breakdown_by_task_mode,
        "breakdown_by_planning_horizon": breakdown_by_planning_horizon,
    }
