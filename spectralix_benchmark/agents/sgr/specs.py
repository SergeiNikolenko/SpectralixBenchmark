from __future__ import annotations

from typing import Any, Dict, Mapping

from pydantic import BaseModel, ValidationError

from .models import (
    GenericASchema,
    GenericBSchema,
    GenericCSchema,
    ImmediatePrecursorPredictionSchema,
    ImmediatePrecursorWithDisconnectionSchema,
    MechanisticClassificationSchema,
    ReactionCenterIdentificationSchema,
    ReferenceRoutePlanningSchema,
    RoleIdentificationSchema,
    RouteDesignSchema,
    SGRBaseSchema,
    SGRSchemaSpec,
    TransformationClassificationSchema,
)


def _generic_a_template() -> Dict[str, Any]:
    return {
        "level": "A",
        "task_subtype": "<from benchmark>",
        "input_parse": {
            "reaction_entities": [],
            "mapped_atoms_or_species": [],
            "explicit_reagents_or_conditions": [],
        },
        "local_structure": {
            "core_event_region": "",
            "relevant_atoms_or_species": [],
            "relevant_bonds_or_roles": [],
        },
        "reasoning_focus": {
            "requested_local_task": "",
            "irrelevant_information_to_ignore": [],
        },
        "derived_local_result": {
            "extracted_local_change_or_assignment": "",
        },
        "contract_check": {
            "exact_local_task_answered": False,
            "broader_summary_leak": False,
            "missing_required_local_detail": False,
            "answer_matches_requested_task": False,
            "answer_matches_requested_depth": False,
            "answer_matches_exact_benchmark_contract": False,
            "broader_or_alternative_answer_leak": False,
        },
        "final_answer": {"value": ""},
    }


_REACTION_CENTER_TEMPLATE = {
    "level": "A",
    "task_subtype": "reaction_center_identification",
    "input_parse": {"mapped_atoms": [], "reactants": "", "products": ""},
    "reaction_edit_schema": {
        "atoms_involved": [],
        "bonds_broken": [],
        "bonds_formed": [],
        "bond_order_changes": [],
        "charge_changes": [],
        "metal_or_leaving_group_transfer": [],
    },
    "derived_local_event": {"complete_local_event": ""},
    "contract_check": {
        "all_local_edits_included": False,
        "partial_event_only": False,
        "broader_reaction_story_instead_of_local_event": False,
        "answer_matches_requested_task": False,
        "answer_matches_requested_depth": False,
        "answer_matches_exact_benchmark_contract": False,
        "broader_or_alternative_answer_leak": False,
    },
    "final_answer": {"value": ""},
}

_MECHANISTIC_TEMPLATE = {
    "level": "A",
    "task_subtype": "mechanistic_classification",
    "input_parse": {"mapped_atoms": [], "reactants": "", "products": ""},
    "reaction_edit_schema": {
        "atoms_involved": [],
        "key_bond_changes": [],
        "pi_system_changes": [],
        "charge_or_polarity_pattern": [],
    },
    "candidate_labels": {"possible_mechanistic_labels": []},
    "label_selection": {"best_label": "", "why_other_labels_rejected": []},
    "contract_check": {
        "exactly_one_label_selected": False,
        "generic_textbook_label_used_instead_of_benchmark_label": False,
        "classification_grounded_in_local_pattern": False,
        "answer_matches_requested_task": False,
        "answer_matches_requested_depth": False,
        "answer_matches_exact_benchmark_contract": False,
        "broader_or_alternative_answer_leak": False,
    },
    "final_answer": {"value": ""},
}

_ROLE_TEMPLATE = {
    "level": "A",
    "task_subtype": "reagent_role_identification",
    "input_parse": {"mentioned_species": [], "mentioned_conditions": []},
    "role_schema": {
        "reactant_candidates": [],
        "reagent_candidates": [],
        "solvent_candidates": [],
        "catalyst_candidates": [],
        "product_candidates": [],
    },
    "role_assignment": {
        "requested_roles_only": {},
        "ambiguous_assignments": [],
        "final_role_mapping": {},
    },
    "contract_check": {
        "role_swap_present": False,
        "extra_unrequested_species_added": False,
        "benchmark_role_convention_respected": False,
        "answer_matches_requested_task": False,
        "answer_matches_requested_depth": False,
        "answer_matches_exact_benchmark_contract": False,
        "broader_or_alternative_answer_leak": False,
    },
    "final_answer": {"value": ""},
}

_TRANSFORMATION_TEMPLATE = {
    "level": "A",
    "task_subtype": "transformation_classification",
    "input_parse": {"reactants": "", "products": ""},
    "transformation_schema": {
        "key_functional_group_change": "",
        "key_bond_change": "",
        "local_transformation_pattern": "",
    },
    "label_selection": {"best_transformation_label": ""},
    "contract_check": {
        "one_label_only": False,
        "label_matches_local_change": False,
        "answer_matches_requested_task": False,
        "answer_matches_requested_depth": False,
        "answer_matches_exact_benchmark_contract": False,
        "broader_or_alternative_answer_leak": False,
    },
    "final_answer": {"value": ""},
}

_GENERIC_B_TEMPLATE = {
    "level": "B",
    "task_subtype": "immediate_precursor_prediction",
    "target_parse": {"target_smiles": "", "key_bonds_or_handles": []},
    "candidate_disconnections": {
        "disconnection_1": "",
        "disconnection_2": "",
        "disconnection_3": "",
    },
    "candidate_precursor_sets": {"set_1": [], "set_2": [], "set_3": []},
    "single_step_check": {
        "set_1_is_immediate": False,
        "set_2_is_immediate": False,
        "set_3_is_immediate": False,
    },
    "forward_regeneration_check": {
        "set_1_regenerates_exact_target": False,
        "set_2_regenerates_exact_target": False,
        "set_3_regenerates_exact_target": False,
    },
    "selection": {
        "chosen_disconnection": "",
        "chosen_precursor_set": [],
        "rejected_as_earlier_stage": [],
        "rejected_as_wrong_family": [],
    },
    "contract_check": {
        "immediate_step_only": False,
        "exact_target_preserved": False,
        "alternative_non_immediate_route_leak": False,
        "answer_matches_requested_task": False,
        "answer_matches_requested_depth": False,
        "answer_matches_exact_benchmark_contract": False,
        "broader_or_alternative_answer_leak": False,
    },
    "final_answer": {"value": ""},
}

_DISCONNECTION_TEMPLATE = {
    "level": "B",
    "task_subtype": "immediate_precursor_with_disconnection",
    "target_parse": {"target_smiles": "", "key_bonds_or_handles": []},
    "candidate_disconnections": {
        "candidate_1": {"broken_bond": "", "precursor_set": [], "forward_reaction_type": ""},
        "candidate_2": {"broken_bond": "", "precursor_set": [], "forward_reaction_type": ""},
    },
    "selection": {
        "chosen_broken_bond": "",
        "chosen_precursor_set": [],
        "chosen_disconnection_rationale": "",
    },
    "contract_check": {
        "disconnection_explicitly_stated": False,
        "immediate_step_only": False,
        "exact_target_preserved": False,
        "answer_matches_requested_task": False,
        "answer_matches_requested_depth": False,
        "answer_matches_exact_benchmark_contract": False,
        "broader_or_alternative_answer_leak": False,
    },
    "final_answer": {"value": ""},
}

_GENERIC_C_TEMPLATE = {
    "level": "C",
    "task_subtype": "route_design",
    "target_parse": {"target_smiles": "", "target_defining_constraints": []},
    "route_graph": {
        "target_node": "",
        "late_intermediates": [],
        "branch_fragments": [],
        "convergence_points": [],
    },
    "branch_plan": {
        "branch_1_goal": "",
        "branch_1_steps": [],
        "branch_2_goal": "",
        "branch_2_steps": [],
    },
    "convergence_plan": {"merge_step": "", "merge_inputs": [], "merge_output": ""},
    "late_stage_validation": {
        "scaffold_preserved": False,
        "heterocycle_identity_preserved": False,
        "regiochemistry_preserved": False,
        "linker_length_preserved": False,
        "substitution_pattern_preserved": False,
    },
    "target_reachability_check": {
        "exact_target_reached": False,
        "plausible_but_non_target_convergent": False,
    },
    "contract_check": {
        "complete_connected_route": False,
        "exact_target_final_step": False,
        "disconnected_fragments_only": False,
        "answer_matches_requested_task": False,
        "answer_matches_requested_depth": False,
        "answer_matches_exact_benchmark_contract": False,
        "broader_or_alternative_answer_leak": False,
    },
    "final_answer": {"value": ""},
}

_REFERENCE_ROUTE_TEMPLATE = {
    "level": "C",
    "task_subtype": "reference_route_planning",
    "target_parse": {"target_smiles": "", "target_defining_constraints": []},
    "reference_alignment": {
        "expected_route_depth": "",
        "likely_key_convergence_pattern": "",
    },
    "route_graph": {
        "target_node": "",
        "key_reference_like_intermediates": [],
        "branch_fragments": [],
        "convergence_points": [],
    },
    "target_reachability_check": {
        "exact_target_reached": False,
        "route_depth_sufficient": False,
        "convergence_logic_complete": False,
    },
    "contract_check": {
        "route_not_just_retrosynthetic_idea": False,
        "final_step_reaches_exact_target": False,
        "answer_matches_requested_task": False,
        "answer_matches_requested_depth": False,
        "answer_matches_exact_benchmark_contract": False,
        "broader_or_alternative_answer_leak": False,
    },
    "final_answer": {"value": ""},
}

_ROUTE_DESIGN_TEMPLATE = {
    "level": "C",
    "task_subtype": "route_design",
    "target_parse": {"target_smiles": "", "target_defining_constraints": []},
    "strategic_route_options": {"option_1": "", "option_2": ""},
    "selected_strategy": {"chosen_strategy": "", "why_selected": ""},
    "route_graph": {
        "target_node": "",
        "intermediates": [],
        "branch_fragments": [],
        "convergence_points": [],
    },
    "target_reachability_check": {
        "exact_target_reached": False,
        "structural_constraints_preserved": False,
    },
    "contract_check": {
        "connected_route_present": False,
        "exact_target_final_step": False,
        "answer_matches_requested_task": False,
        "answer_matches_requested_depth": False,
        "answer_matches_exact_benchmark_contract": False,
        "broader_or_alternative_answer_leak": False,
    },
    "final_answer": {"value": ""},
}


_GENERIC_LEVEL_SPECS: Dict[str, SGRSchemaSpec] = {
    "A": SGRSchemaSpec("sgr_a_generic", GenericASchema, _generic_a_template()),
    "B": SGRSchemaSpec("sgr_b_generic", GenericBSchema, _GENERIC_B_TEMPLATE),
    "C": SGRSchemaSpec("sgr_c_generic", GenericCSchema, _GENERIC_C_TEMPLATE),
}

_SUBTYPE_SPECS: Dict[tuple[str, str], SGRSchemaSpec] = {
    ("A", "reaction_center_identification"): SGRSchemaSpec(
        "sgr_a_reaction_center_identification",
        ReactionCenterIdentificationSchema,
        _REACTION_CENTER_TEMPLATE,
    ),
    ("A", "mechanistic_classification"): SGRSchemaSpec(
        "sgr_a_mechanistic_classification",
        MechanisticClassificationSchema,
        _MECHANISTIC_TEMPLATE,
    ),
    ("A", "reagent_role_identification"): SGRSchemaSpec(
        "sgr_a_reagent_role_identification",
        RoleIdentificationSchema,
        _ROLE_TEMPLATE,
    ),
    ("A", "condition_role_identification"): SGRSchemaSpec(
        "sgr_a_condition_role_identification",
        RoleIdentificationSchema,
        {**_ROLE_TEMPLATE, "task_subtype": "condition_role_identification"},
    ),
    ("A", "transformation_classification"): SGRSchemaSpec(
        "sgr_a_transformation_classification",
        TransformationClassificationSchema,
        _TRANSFORMATION_TEMPLATE,
    ),
    ("B", "immediate_precursor_prediction"): SGRSchemaSpec(
        "sgr_b_immediate_precursor_prediction",
        ImmediatePrecursorPredictionSchema,
        _GENERIC_B_TEMPLATE,
    ),
    ("B", "immediate_precursor_with_disconnection"): SGRSchemaSpec(
        "sgr_b_immediate_precursor_with_disconnection",
        ImmediatePrecursorWithDisconnectionSchema,
        _DISCONNECTION_TEMPLATE,
    ),
    ("C", "reference_route_planning"): SGRSchemaSpec(
        "sgr_c_reference_route_planning",
        ReferenceRoutePlanningSchema,
        _REFERENCE_ROUTE_TEMPLATE,
    ),
    ("C", "route_design"): SGRSchemaSpec(
        "sgr_c_route_design",
        RouteDesignSchema,
        _ROUTE_DESIGN_TEMPLATE,
    ),
}


def _normalize_level(level: str) -> str:
    return str(level or "").strip().upper()


def _normalize_task_subtype(task_subtype: str) -> str:
    return str(task_subtype or "").strip().lower()


def get_sgr_schema_spec(level: str, task_subtype: str) -> SGRSchemaSpec:
    normalized_level = _normalize_level(level)
    generic_spec = _GENERIC_LEVEL_SPECS.get(normalized_level)
    if generic_spec is None:
        raise ValueError(f"Unsupported SGR level: {level!r}")
    normalized_subtype = _normalize_task_subtype(task_subtype)
    return _SUBTYPE_SPECS.get((normalized_level, normalized_subtype), generic_spec)


def select_sgr_schema(level: str, task_subtype: str) -> tuple[type[SGRBaseSchema], str]:
    spec = get_sgr_schema_spec(level, task_subtype)
    return spec.model, spec.schema_name


def validate_sgr_payload(level: str, task_subtype: str, payload: Mapping[str, Any]) -> SGRBaseSchema:
    schema_model, _ = select_sgr_schema(level, task_subtype)
    return schema_model.model_validate(dict(payload))


_EMPTY_PRUNED_VALUES = ("", [], {}, None)


def _prune_payload(value: Any, *, max_list_items: int = 4) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            pruned = _prune_payload(item, max_list_items=max_list_items)
            if pruned in _EMPTY_PRUNED_VALUES:
                continue
            result[str(key)] = pruned
        return result
    if isinstance(value, list):
        items = [_prune_payload(item, max_list_items=max_list_items) for item in value]
        items = [item for item in items if item not in _EMPTY_PRUNED_VALUES]
        if len(items) > max_list_items:
            return items[:max_list_items] + [f"... ({len(items) - max_list_items} more)"]
        return items
    return value


def compact_sgr_payload(payload: Mapping[str, Any] | BaseModel) -> Dict[str, Any]:
    pruned = _prune_payload(payload)
    return pruned if isinstance(pruned, dict) else {}


def extract_contract_check(payload: Mapping[str, Any] | BaseModel) -> Dict[str, Any]:
    compact = compact_sgr_payload(payload)
    contract = compact.get("contract_check")
    return contract if isinstance(contract, dict) else {}


def extract_final_answer_value(payload: Mapping[str, Any] | BaseModel) -> str:
    compact = compact_sgr_payload(payload)
    final_answer = compact.get("final_answer")
    if isinstance(final_answer, dict):
        return str(final_answer.get("value") or "").strip()
    return ""


def schema_template_lines(template: Mapping[str, Any]) -> str:
    lines: list[str] = []

    def _walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(item, Mapping):
                    lines.append(f"{prefix}{key}:")
                    _walk(item, prefix + "  ")
                else:
                    lines.append(f"{prefix}{key}: {item}")
        else:
            lines.append(f"{prefix}{value}")

    _walk(template)
    return "\n".join(lines)


__all__ = [
    "compact_sgr_payload",
    "extract_contract_check",
    "extract_final_answer_value",
    "get_sgr_schema_spec",
    "schema_template_lines",
    "select_sgr_schema",
    "validate_sgr_payload",
    "ValidationError",
]

