from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Mapping

from pydantic import BaseModel, Field, ValidationError, model_validator


class FinalAnswerField(BaseModel):
    value: str = Field(min_length=1)


class SGRBaseSchema(BaseModel):
    EXPECTED_LEVEL: ClassVar[str] = ""
    EXPECTED_TASK_SUBTYPES: ClassVar[tuple[str, ...]] = ()

    level: str
    task_subtype: str

    @model_validator(mode="after")
    def _validate_scope(self) -> "SGRBaseSchema":
        expected_level = str(self.EXPECTED_LEVEL or "").strip().upper()
        if expected_level and str(self.level or "").strip().upper() != expected_level:
            raise ValueError(f"level must be {expected_level}")
        expected_subtypes = tuple(self.EXPECTED_TASK_SUBTYPES or ())
        if expected_subtypes and str(self.task_subtype or "").strip().lower() not in expected_subtypes:
            raise ValueError(f"task_subtype must be one of {', '.join(expected_subtypes)}")
        return self


class UniversalReconciliation(BaseModel):
    answer_matches_requested_task: bool = False
    answer_matches_requested_depth: bool = False
    answer_matches_exact_benchmark_contract: bool = False
    broader_or_alternative_answer_leak: bool = False


class GenericAInputParse(BaseModel):
    reaction_entities: list[str] = Field(default_factory=list)
    mapped_atoms_or_species: list[str] = Field(default_factory=list)
    explicit_reagents_or_conditions: list[str] = Field(default_factory=list)


class GenericALocalStructure(BaseModel):
    core_event_region: str = ""
    relevant_atoms_or_species: list[str] = Field(default_factory=list)
    relevant_bonds_or_roles: list[str] = Field(default_factory=list)


class GenericAReasoningFocus(BaseModel):
    requested_local_task: str = ""
    irrelevant_information_to_ignore: list[str] = Field(default_factory=list)


class GenericADerivedLocalResult(BaseModel):
    extracted_local_change_or_assignment: str = ""


class GenericAContractCheck(UniversalReconciliation):
    exact_local_task_answered: bool = False
    broader_summary_leak: bool = False
    missing_required_local_detail: bool = False


class GenericASchema(SGRBaseSchema):
    EXPECTED_LEVEL = "A"

    input_parse: GenericAInputParse
    local_structure: GenericALocalStructure
    reasoning_focus: GenericAReasoningFocus
    derived_local_result: GenericADerivedLocalResult
    contract_check: GenericAContractCheck
    final_answer: FinalAnswerField


class ReactionCenterInputParse(BaseModel):
    mapped_atoms: list[str] = Field(default_factory=list)
    reactants: str = ""
    products: str = ""


class ReactionEditSchema(BaseModel):
    atoms_involved: list[str] = Field(default_factory=list)
    bonds_broken: list[str] = Field(default_factory=list)
    bonds_formed: list[str] = Field(default_factory=list)
    bond_order_changes: list[str] = Field(default_factory=list)
    charge_changes: list[str] = Field(default_factory=list)
    metal_or_leaving_group_transfer: list[str] = Field(default_factory=list)


class DerivedLocalEvent(BaseModel):
    complete_local_event: str = ""


class ReactionCenterContractCheck(UniversalReconciliation):
    all_local_edits_included: bool = False
    partial_event_only: bool = False
    broader_reaction_story_instead_of_local_event: bool = False


class ReactionCenterIdentificationSchema(SGRBaseSchema):
    EXPECTED_LEVEL = "A"
    EXPECTED_TASK_SUBTYPES = ("reaction_center_identification",)

    input_parse: ReactionCenterInputParse
    reaction_edit_schema: ReactionEditSchema
    derived_local_event: DerivedLocalEvent
    contract_check: ReactionCenterContractCheck
    final_answer: FinalAnswerField


class MechanisticInputParse(BaseModel):
    mapped_atoms: list[str] = Field(default_factory=list)
    reactants: str = ""
    products: str = ""


class MechanisticReactionEditSchema(BaseModel):
    atoms_involved: list[str] = Field(default_factory=list)
    key_bond_changes: list[str] = Field(default_factory=list)
    pi_system_changes: list[str] = Field(default_factory=list)
    charge_or_polarity_pattern: list[str] = Field(default_factory=list)


class CandidateLabels(BaseModel):
    possible_mechanistic_labels: list[str] = Field(default_factory=list)


class LabelSelection(BaseModel):
    best_label: str = ""
    why_other_labels_rejected: list[str] = Field(default_factory=list)


class MechanisticContractCheck(UniversalReconciliation):
    exactly_one_label_selected: bool = False
    generic_textbook_label_used_instead_of_benchmark_label: bool = False
    classification_grounded_in_local_pattern: bool = False


class MechanisticClassificationSchema(SGRBaseSchema):
    EXPECTED_LEVEL = "A"
    EXPECTED_TASK_SUBTYPES = ("mechanistic_classification",)

    input_parse: MechanisticInputParse
    reaction_edit_schema: MechanisticReactionEditSchema
    candidate_labels: CandidateLabels
    label_selection: LabelSelection
    contract_check: MechanisticContractCheck
    final_answer: FinalAnswerField


class RoleInputParse(BaseModel):
    mentioned_species: list[str] = Field(default_factory=list)
    mentioned_conditions: list[str] = Field(default_factory=list)


class RoleSchema(BaseModel):
    reactant_candidates: list[str] = Field(default_factory=list)
    reagent_candidates: list[str] = Field(default_factory=list)
    solvent_candidates: list[str] = Field(default_factory=list)
    catalyst_candidates: list[str] = Field(default_factory=list)
    product_candidates: list[str] = Field(default_factory=list)


class RoleAssignment(BaseModel):
    requested_roles_only: dict[str, str] = Field(default_factory=dict)
    ambiguous_assignments: list[str] = Field(default_factory=list)
    final_role_mapping: dict[str, str] = Field(default_factory=dict)


class RoleContractCheck(UniversalReconciliation):
    role_swap_present: bool = False
    extra_unrequested_species_added: bool = False
    benchmark_role_convention_respected: bool = False


class RoleIdentificationSchema(SGRBaseSchema):
    EXPECTED_LEVEL = "A"
    EXPECTED_TASK_SUBTYPES = ("reagent_role_identification", "condition_role_identification")

    input_parse: RoleInputParse
    role_schema: RoleSchema
    role_assignment: RoleAssignment
    contract_check: RoleContractCheck
    final_answer: FinalAnswerField


class TransformationInputParse(BaseModel):
    reactants: str = ""
    products: str = ""


class TransformationSchema(BaseModel):
    key_functional_group_change: str = ""
    key_bond_change: str = ""
    local_transformation_pattern: str = ""


class TransformationLabelSelection(BaseModel):
    best_transformation_label: str = ""


class TransformationContractCheck(UniversalReconciliation):
    one_label_only: bool = False
    label_matches_local_change: bool = False


class TransformationClassificationSchema(SGRBaseSchema):
    EXPECTED_LEVEL = "A"
    EXPECTED_TASK_SUBTYPES = ("transformation_classification",)

    input_parse: TransformationInputParse
    transformation_schema: TransformationSchema
    label_selection: TransformationLabelSelection
    contract_check: TransformationContractCheck
    final_answer: FinalAnswerField


class BTargetParse(BaseModel):
    target_smiles: str = ""
    key_bonds_or_handles: list[str] = Field(default_factory=list)


class BCandidateDisconnections(BaseModel):
    disconnection_1: str = ""
    disconnection_2: str = ""
    disconnection_3: str = ""


class BCandidatePrecursorSets(BaseModel):
    set_1: list[str] = Field(default_factory=list)
    set_2: list[str] = Field(default_factory=list)
    set_3: list[str] = Field(default_factory=list)


class SingleStepCheck(BaseModel):
    set_1_is_immediate: bool = False
    set_2_is_immediate: bool = False
    set_3_is_immediate: bool = False


class ForwardRegenerationCheck(BaseModel):
    set_1_regenerates_exact_target: bool = False
    set_2_regenerates_exact_target: bool = False
    set_3_regenerates_exact_target: bool = False


class BSelection(BaseModel):
    chosen_disconnection: str = ""
    chosen_precursor_set: list[str] = Field(default_factory=list)
    rejected_as_earlier_stage: list[str] = Field(default_factory=list)
    rejected_as_wrong_family: list[str] = Field(default_factory=list)


class BContractCheck(UniversalReconciliation):
    immediate_step_only: bool = False
    exact_target_preserved: bool = False
    alternative_non_immediate_route_leak: bool = False


class GenericBSchema(SGRBaseSchema):
    EXPECTED_LEVEL = "B"

    target_parse: BTargetParse
    candidate_disconnections: BCandidateDisconnections
    candidate_precursor_sets: BCandidatePrecursorSets
    single_step_check: SingleStepCheck
    forward_regeneration_check: ForwardRegenerationCheck
    selection: BSelection
    contract_check: BContractCheck
    final_answer: FinalAnswerField


class ImmediatePrecursorPredictionSchema(GenericBSchema):
    EXPECTED_LEVEL = "B"
    EXPECTED_TASK_SUBTYPES = ("immediate_precursor_prediction",)


class DisconnectionCandidate(BaseModel):
    broken_bond: str = ""
    precursor_set: list[str] = Field(default_factory=list)
    forward_reaction_type: str = ""


class DisconnectionCandidates(BaseModel):
    candidate_1: DisconnectionCandidate
    candidate_2: DisconnectionCandidate


class DisconnectionSelection(BaseModel):
    chosen_broken_bond: str = ""
    chosen_precursor_set: list[str] = Field(default_factory=list)
    chosen_disconnection_rationale: str = ""


class DisconnectionContractCheck(UniversalReconciliation):
    disconnection_explicitly_stated: bool = False
    immediate_step_only: bool = False
    exact_target_preserved: bool = False


class ImmediatePrecursorWithDisconnectionSchema(SGRBaseSchema):
    EXPECTED_LEVEL = "B"
    EXPECTED_TASK_SUBTYPES = ("immediate_precursor_with_disconnection",)

    target_parse: BTargetParse
    candidate_disconnections: DisconnectionCandidates
    selection: DisconnectionSelection
    contract_check: DisconnectionContractCheck
    final_answer: FinalAnswerField


class CTargetParse(BaseModel):
    target_smiles: str = ""
    target_defining_constraints: list[str] = Field(default_factory=list)


class RouteGraph(BaseModel):
    target_node: str = ""
    late_intermediates: list[str] = Field(default_factory=list)
    branch_fragments: list[str] = Field(default_factory=list)
    convergence_points: list[str] = Field(default_factory=list)


class BranchPlan(BaseModel):
    branch_1_goal: str = ""
    branch_1_steps: list[str] = Field(default_factory=list)
    branch_2_goal: str = ""
    branch_2_steps: list[str] = Field(default_factory=list)


class ConvergencePlan(BaseModel):
    merge_step: str = ""
    merge_inputs: list[str] = Field(default_factory=list)
    merge_output: str = ""


class LateStageValidation(BaseModel):
    scaffold_preserved: bool = False
    heterocycle_identity_preserved: bool = False
    regiochemistry_preserved: bool = False
    linker_length_preserved: bool = False
    substitution_pattern_preserved: bool = False


class TargetReachabilityCheck(BaseModel):
    exact_target_reached: bool = False
    plausible_but_non_target_convergent: bool = False


class CContractCheck(UniversalReconciliation):
    complete_connected_route: bool = False
    exact_target_final_step: bool = False
    disconnected_fragments_only: bool = False


class GenericCSchema(SGRBaseSchema):
    EXPECTED_LEVEL = "C"

    target_parse: CTargetParse
    route_graph: RouteGraph
    branch_plan: BranchPlan
    convergence_plan: ConvergencePlan
    late_stage_validation: LateStageValidation
    target_reachability_check: TargetReachabilityCheck
    contract_check: CContractCheck
    final_answer: FinalAnswerField


class ReferenceAlignment(BaseModel):
    expected_route_depth: str = ""
    likely_key_convergence_pattern: str = ""


class ReferenceRouteGraph(BaseModel):
    target_node: str = ""
    key_reference_like_intermediates: list[str] = Field(default_factory=list)
    branch_fragments: list[str] = Field(default_factory=list)
    convergence_points: list[str] = Field(default_factory=list)


class ReferenceTargetReachabilityCheck(BaseModel):
    exact_target_reached: bool = False
    route_depth_sufficient: bool = False
    convergence_logic_complete: bool = False


class ReferenceContractCheck(UniversalReconciliation):
    route_not_just_retrosynthetic_idea: bool = False
    final_step_reaches_exact_target: bool = False


class ReferenceRoutePlanningSchema(SGRBaseSchema):
    EXPECTED_LEVEL = "C"
    EXPECTED_TASK_SUBTYPES = ("reference_route_planning",)

    target_parse: CTargetParse
    reference_alignment: ReferenceAlignment
    route_graph: ReferenceRouteGraph
    target_reachability_check: ReferenceTargetReachabilityCheck
    contract_check: ReferenceContractCheck
    final_answer: FinalAnswerField


class StrategicRouteOptions(BaseModel):
    option_1: str = ""
    option_2: str = ""


class SelectedStrategy(BaseModel):
    chosen_strategy: str = ""
    why_selected: str = ""


class RouteDesignGraph(BaseModel):
    target_node: str = ""
    intermediates: list[str] = Field(default_factory=list)
    branch_fragments: list[str] = Field(default_factory=list)
    convergence_points: list[str] = Field(default_factory=list)


class RouteDesignReachabilityCheck(BaseModel):
    exact_target_reached: bool = False
    structural_constraints_preserved: bool = False


class RouteDesignContractCheck(UniversalReconciliation):
    connected_route_present: bool = False
    exact_target_final_step: bool = False


class RouteDesignSchema(SGRBaseSchema):
    EXPECTED_LEVEL = "C"
    EXPECTED_TASK_SUBTYPES = ("route_design",)

    target_parse: CTargetParse
    strategic_route_options: StrategicRouteOptions
    selected_strategy: SelectedStrategy
    route_graph: RouteDesignGraph
    target_reachability_check: RouteDesignReachabilityCheck
    contract_check: RouteDesignContractCheck
    final_answer: FinalAnswerField


@dataclass(frozen=True)
class SGRSchemaSpec:
    schema_name: str
    model: type[SGRBaseSchema]
    template: Mapping[str, Any]


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
    "SGRBaseSchema",
    "SGRSchemaSpec",
    "compact_sgr_payload",
    "extract_contract_check",
    "extract_final_answer_value",
    "get_sgr_schema_spec",
    "schema_template_lines",
    "select_sgr_schema",
    "validate_sgr_payload",
    "ValidationError",
]
