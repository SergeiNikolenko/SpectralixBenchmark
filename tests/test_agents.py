import unittest
import os
import tempfile
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from scripts.agents import openshell_worker
from scripts.agents.config import build_executor_kwargs, load_agent_config, resolve_runtime_backend
from scripts.agents.models import ensure_chat_completions_url, parse_model_url, sandbox_visible_api_base
from scripts.agents.prompts import build_parse_page_task, build_student_sgr_task, build_student_task
from scripts.agents.runtime import AgentRuntime
from scripts.agents.sgr_schemas import (
    GenericASchema,
    GenericBSchema,
    MechanisticClassificationSchema,
    ReferenceRoutePlanningSchema,
    select_sgr_schema,
)
from scripts.agents.tool_registry import (
    build_tools,
    safe_http_get_tool,
    workspace_list_tool,
    workspace_read_tool,
    workspace_write_tool,
    shell_exec_tool,
    uv_run_tool,
)


class ModelAdapterTests(unittest.TestCase):
    def test_parse_model_url_chat_endpoint(self):
        api_base, chat = parse_model_url("https://api.openai.com/v1/chat/completions")
        self.assertEqual(api_base, "https://api.openai.com/v1")
        self.assertEqual(chat, "https://api.openai.com/v1/chat/completions")

    def test_parse_model_url_v1_base(self):
        api_base, chat = parse_model_url("http://127.0.0.1:8317/v1")
        self.assertEqual(api_base, "http://127.0.0.1:8317/v1")
        self.assertEqual(chat, "http://127.0.0.1:8317/v1/chat/completions")

    def test_ensure_chat_completions_url(self):
        self.assertEqual(
            ensure_chat_completions_url("http://127.0.0.1:8317/v1"),
            "http://127.0.0.1:8317/v1/chat/completions",
        )

    def test_sandbox_visible_api_base_rewrites_localhost(self):
        self.assertEqual(
            sandbox_visible_api_base("http://127.0.0.1:8317/v1"),
            "http://host.openshell.internal:8317/v1",
        )


class ToolPolicyTests(unittest.TestCase):
    def test_safe_http_tool_disabled_without_allowlist(self):
        config = load_agent_config(config_path=None)
        config["security"]["allowed_tool_hosts"] = []
        tools = build_tools("full", config)
        tool_names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools}
        self.assertNotIn("safe_http_get_tool", tool_names)

    def test_safe_http_tool_enabled_with_allowlist(self):
        config = load_agent_config(config_path=None)
        config["security"]["allowed_tool_hosts"] = ["api.openai.com"]
        config["security"]["allow_network_tools"] = True
        tools = build_tools("tools_internet", config)
        tool_names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools}
        self.assertIn("safe_http_get_tool", tool_names)

    def test_full_profile_tools_are_expected(self):
        config = load_agent_config(config_path=None)
        tools = build_tools("full", config)
        tool_names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools}
        self.assertIn("chem_python_tool", tool_names)
        self.assertIn("workspace_read_tool", tool_names)
        self.assertIn("workspace_write_tool", tool_names)
        self.assertIn("shell_exec_tool", tool_names)
        self.assertIn("uv_run_tool", tool_names)
        self.assertNotIn("rubric_hint_tool", tool_names)
        self.assertNotIn("chem_format_tool", tool_names)
        self.assertNotIn("smiles_sanity_tool", tool_names)

    def test_tools_profile_excludes_workspace_write_by_default(self):
        config = load_agent_config(config_path=None)
        tools = build_tools("tools", config)
        tool_names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools}
        self.assertNotIn("workspace_write_tool", tool_names)

    def test_safe_http_tool_disabled_when_network_tools_off(self):
        config = load_agent_config(config_path=None)
        config["security"]["allowed_tool_hosts"] = ["api.openai.com"]
        config["security"]["allow_network_tools"] = False
        tools = build_tools("full", config)
        tool_names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools}
        self.assertNotIn("safe_http_get_tool", tool_names)

    def test_unknown_tool_fails_fast(self):
        config = load_agent_config(config_path=None)
        config["tools"]["profiles"]["full"] = ["unknown_tool_name"]
        with self.assertRaises(ValueError):
            build_tools("full", config)

    def test_unknown_profile_fails_fast(self):
        config = load_agent_config(config_path=None)
        with self.assertRaises(ValueError):
            build_tools("missing_profile", config)

    def test_safe_http_tool_wildcard_allows_any_host_check(self):
        previous = os.environ.get("AGENT_ALLOWED_HOSTS")
        os.environ["AGENT_ALLOWED_HOSTS"] = "*"
        try:
            result = safe_http_get_tool("http://nonexistent.invalid", timeout_sec=1)
        finally:
            if previous is None:
                os.environ.pop("AGENT_ALLOWED_HOSTS", None)
            else:
                os.environ["AGENT_ALLOWED_HOSTS"] = previous
        self.assertNotIn("host_not_allowed", result)


class ConfigTests(unittest.TestCase):
    def test_missing_config_file_raises(self):
        missing = Path("scripts/agents/does_not_exist.yaml")
        with self.assertRaises(FileNotFoundError):
            load_agent_config(config_path=missing)

    def test_default_enables_base_tools_flag(self):
        config = load_agent_config(config_path=None)
        self.assertTrue(config["runtime"]["add_base_tools"])

    def test_invalid_overrides_fail_validation(self):
        with self.assertRaises(ValueError):
            load_agent_config(config_path=None, overrides={"tools": None})

    def test_executor_kwargs_openshell_fields(self):
        config = load_agent_config(config_path=None)
        kwargs = build_executor_kwargs(config, workspace_dir=Path("."))
        self.assertEqual(kwargs["gateway_name"], "spectralix")
        self.assertEqual(kwargs["gateway_port"], 18080)
        self.assertEqual(kwargs["sandbox_name"], "spectralix-runtime")
        self.assertEqual(kwargs["sandbox_from"], "base")
        self.assertFalse(kwargs["delete_on_close"])
        self.assertEqual(kwargs["native_codex"]["sandbox_from"], "codex")

    def test_runtime_backend_defaults_follow_executor(self):
        config = load_agent_config(config_path=None)
        self.assertEqual(resolve_runtime_backend(config, executor_type="openshell"), "openshell_worker")
        self.assertEqual(resolve_runtime_backend(config, executor_type="local"), "local_worker")

    def test_network_tools_require_allowlist(self):
        with self.assertRaises(ValueError):
            load_agent_config(
                config_path=None,
                overrides={
                    "security": {
                        "allow_network_tools": True,
                        "allowed_tool_hosts": [],
                    }
                },
            )

    def test_network_can_be_enabled_when_allowlist_is_present(self):
        config = load_agent_config(
            config_path=None,
            overrides={
                "security": {
                    "allow_network_tools": True,
                    "allowed_tool_hosts": ["api.openai.com"],
                }
            },
        )
        kwargs = build_executor_kwargs(config, workspace_dir=Path("."))
        self.assertEqual(kwargs["gateway_name"], "spectralix")


class StatusMappingTests(unittest.TestCase):
    def test_classify_sandbox_error(self):
        status = AgentRuntime._classify_error(RuntimeError("openshell sandbox create failed"))
        self.assertEqual(status, "sandbox_error")

    def test_classify_auth_error(self):
        status = AgentRuntime._classify_error(RuntimeError("401 unauthorized"))
        self.assertEqual(status, "auth_error")

    def test_classify_exit_code_124_as_timeout(self):
        status = AgentRuntime._classify_error(RuntimeError("OpenShell worker failed with exit_code=124: unknown error"))
        self.assertEqual(status, "timeout")

    def test_classify_deadline_exceeded_as_timeout(self):
        status = AgentRuntime._classify_error(RuntimeError("grpc deadline exceeded while executing worker"))
        self.assertEqual(status, "timeout")


class RuntimeInitTests(unittest.TestCase):
    def test_local_executor_uses_empty_executor_kwargs(self):
        runtime = AgentRuntime(
            model_url="http://127.0.0.1:8317/v1",
            model_name="gpt-4o-mini",
            api_key="test-key",
            sandbox="local",
            tools_profile="minimal",
        )
        self.assertEqual(runtime.executor_type, "local")
        self.assertEqual(runtime.executor_kwargs, {})
        self.assertEqual(runtime.get_runtime_metadata()["sandbox_runtime"], "local_worker")
        runtime.close()


class WorkspaceToolTests(unittest.TestCase):
    def test_workspace_read_write_and_list_tools(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_root = os.environ.get("AGENT_WORKSPACE_ROOT")
            os.environ["AGENT_WORKSPACE_ROOT"] = tmp_dir
            try:
                write_result = workspace_write_tool("notes/example.txt", "hello", mode="overwrite")
                self.assertIn('"status": "ok"', write_result)
                read_result = workspace_read_tool("notes/example.txt")
                self.assertIn('"content": "hello"', read_result)
                list_result = workspace_list_tool("notes")
                self.assertIn("example.txt", list_result)
            finally:
                if previous_root is None:
                    os.environ.pop("AGENT_WORKSPACE_ROOT", None)
                else:
                    os.environ["AGENT_WORKSPACE_ROOT"] = previous_root

    def test_shell_exec_tool_runs_in_workspace(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_root = os.environ.get("AGENT_WORKSPACE_ROOT")
            os.environ["AGENT_WORKSPACE_ROOT"] = tmp_dir
            try:
                result = shell_exec_tool("pwd", workdir=".", timeout_sec=5)
                self.assertIn('"status": "ok"', result)
                self.assertIn(tmp_dir, result)
            finally:
                if previous_root is None:
                    os.environ.pop("AGENT_WORKSPACE_ROOT", None)
                else:
                    os.environ["AGENT_WORKSPACE_ROOT"] = previous_root

    def test_shell_exec_tool_blocks_non_allowlisted_command(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_root = os.environ.get("AGENT_WORKSPACE_ROOT")
            os.environ["AGENT_WORKSPACE_ROOT"] = tmp_dir
            try:
                result = shell_exec_tool("bash", workdir=".", timeout_sec=5)
                self.assertIn("command_not_allowed:bash", result)
            finally:
                if previous_root is None:
                    os.environ.pop("AGENT_WORKSPACE_ROOT", None)
                else:
                    os.environ["AGENT_WORKSPACE_ROOT"] = previous_root

    def test_uv_run_tool_reports_missing_uv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_root = os.environ.get("AGENT_WORKSPACE_ROOT")
            previous_uv = os.environ.get("AGENT_UV_BIN")
            os.environ["AGENT_WORKSPACE_ROOT"] = tmp_dir
            os.environ["AGENT_UV_BIN"] = str(Path(tmp_dir) / "missing-uv")
            try:
                result = uv_run_tool("run python -V", workdir=".", timeout_sec=5)
                self.assertIn("uv_not_available", result)
            finally:
                if previous_root is None:
                    os.environ.pop("AGENT_WORKSPACE_ROOT", None)
                else:
                    os.environ["AGENT_WORKSPACE_ROOT"] = previous_root
                if previous_uv is None:
                    os.environ.pop("AGENT_UV_BIN", None)
                else:
                    os.environ["AGENT_UV_BIN"] = previous_uv

    def test_runtime_metadata_for_openshell_config(self):
        runtime = AgentRuntime(
            model_url="http://127.0.0.1:8317/v1",
            model_name="gpt-5.4-mini",
            api_key="test-key",
            sandbox="openshell",
            tools_profile="tools",
        )
        metadata = runtime.get_runtime_metadata()
        self.assertEqual(metadata["sandbox_runtime"], "openshell")
        self.assertEqual(metadata["executor_type"], "openshell")
        self.assertEqual(metadata["runtime_backend"], "openshell_worker")
        self.assertIn("chem_python_tool", metadata["configured_local_tools"])
        runtime.close()

    def test_runtime_metadata_for_codex_native_backend(self):
        runtime = AgentRuntime(
            model_url="http://127.0.0.1:8317/v1",
            model_name="gpt-5.4-mini",
            api_key="test-key",
            sandbox="openshell",
            backend="codex_native",
            tools_profile="minimal",
        )
        metadata = runtime.get_runtime_metadata()
        self.assertEqual(metadata["executor_type"], "openshell")
        self.assertEqual(metadata["runtime_backend"], "codex_native")
        self.assertEqual(metadata["sandbox_runtime"], "openshell_codex_native")
        self.assertFalse(metadata["managed_inference"])
        runtime.close()

    def test_runtime_metadata_exposes_network_tools_when_enabled(self):
        config_path = Path("scripts/agents/_test_network_true.yaml")
        config_path.write_text(
            "\n".join(
                [
                    "security:",
                    "  allow_network_tools: true",
                    "  allowed_tool_hosts:",
                    "    - api.openai.com",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            runtime = AgentRuntime(
                model_url="http://127.0.0.1:8317/v1",
                model_name="gpt-5.4-mini",
                api_key="test-key",
                sandbox="local",
                tools_profile="tools_internet",
                config_path=config_path,
            )
            metadata = runtime.get_runtime_metadata()
            self.assertTrue(metadata["allow_network_tools"])
            self.assertIn("api.openai.com", metadata["allowed_tool_hosts"])
            runtime.close()
        finally:
            config_path.unlink(missing_ok=True)

    def test_payload_timeout_scales_for_heavier_levels(self):
        runtime = AgentRuntime(
            model_url="http://127.0.0.1:8317/v1",
            model_name="gpt-5.4-mini",
            api_key="test-key",
            sandbox="openshell",
            tools_profile="minimal",
            timeout_sec=120,
        )
        self.assertEqual(
            runtime._payload_timeout_seconds(
                {
                    "mode": "student",
                    "question": {"level": "A", "answer_type": "reaction_description"},
                }
            ),
            240,
        )
        self.assertEqual(
            runtime._payload_timeout_seconds(
                {
                    "mode": "student",
                    "question": {
                        "level": "B",
                        "answer_type": "text",
                        "task_subtype": "immediate_precursor_with_disconnection",
                    },
                }
            ),
            360,
        )
        self.assertEqual(
            runtime._payload_timeout_seconds(
                {
                    "mode": "student",
                    "question": {"level": "C", "answer_type": "full_synthesis"},
                }
            ),
            540,
        )
        runtime.close()


class PromptSecurityTests(unittest.TestCase):
    def test_student_prompt_hides_question_ids(self):
        question = {
            "exam_id": "exam_1",
            "page_id": "9",
            "question_id": "13",
            "answer_type": "multiple_choice",
            "question_text": "Select answers",
        }
        prompt = build_student_task(question)
        self.assertNotIn("exam_id", prompt)
        self.assertNotIn("page_id", prompt)
        self.assertNotIn("question_id", prompt)
        self.assertNotIn("Question metadata", prompt)

    def test_student_prompt_includes_contract_sections(self):
        prompt = build_student_task(
            {
                "answer_type": "structure",
                "question_text": "Propose the product structure.",
            },
            runtime_context={
                "tools_profile": "tools",
                "workspace_root": "/sandbox/workspace",
                "available_tools": ["chem_python_tool", "workspace_read_tool", "uv_run_tool"],
            },
        )
        self.assertIn("<role>", prompt)
        self.assertIn("<answer_format>", prompt)
        self.assertIn("<tool_rules>", prompt)
        self.assertIn("<runtime_context>", prompt)
        self.assertIn("<tool_decision_order>", prompt)
        self.assertIn("<completion_criteria>", prompt)
        self.assertIn("Return exactly one SMILES string", prompt)
        self.assertIn("Do not use tools to look for hidden metadata", prompt)
        self.assertIn("Chemistry validation: chem_python_tool", prompt)
        self.assertIn("Workspace inspection/editing: workspace_read_tool", prompt)

    def test_parser_prompt_includes_extraction_contract(self):
        prompt = build_parse_page_task(
            exam_id="exam_7",
            page_id=4,
            marker_prompt="Extract all questions.",
            image_path="/tmp/page.png",
        )
        self.assertIn("<task>", prompt)
        self.assertIn("<extraction_rules>", prompt)
        self.assertIn("Do not invent missing fields.", prompt)
        self.assertIn("return []", prompt)


class SGRSchemaTests(unittest.TestCase):
    def test_selector_prefers_subtype_specific_schema(self):
        schema_model, schema_name = select_sgr_schema("A", "mechanistic_classification")
        self.assertIs(schema_model, MechanisticClassificationSchema)
        self.assertEqual(schema_name, "sgr_a_mechanistic_classification")

    def test_selector_falls_back_to_generic_schema_for_unknown_subtype(self):
        schema_model, schema_name = select_sgr_schema("B", "non_existing_subtype")
        self.assertIs(schema_model, GenericBSchema)
        self.assertEqual(schema_name, "sgr_b_generic")

    def test_selector_rejects_unknown_level(self):
        with self.assertRaises(ValueError):
            select_sgr_schema("Z", "anything")

    def test_subtype_specific_schema_rejects_task_subtype_mismatch(self):
        payload = {
            "level": "A",
            "task_subtype": "reaction_center_identification",
            "input_parse": {
                "mapped_atoms": [],
                "reactants": "A",
                "products": "B",
            },
            "reaction_edit_schema": {
                "atoms_involved": [],
                "key_bond_changes": [],
                "pi_system_changes": [],
                "charge_or_polarity_pattern": [],
            },
            "candidate_labels": {"possible_mechanistic_labels": ["SN2"]},
            "label_selection": {
                "best_label": "SN2",
                "why_other_labels_rejected": [],
            },
            "contract_check": {
                "answer_matches_requested_task": True,
                "answer_matches_requested_depth": True,
                "answer_matches_exact_benchmark_contract": True,
                "broader_or_alternative_answer_leak": False,
                "exactly_one_label_selected": True,
                "generic_textbook_label_used_instead_of_benchmark_label": False,
                "classification_grounded_in_local_pattern": True,
            },
            "final_answer": {"value": "SN2"},
        }
        with self.assertRaises(ValidationError):
            MechanisticClassificationSchema.model_validate(payload)

    def test_generic_schema_requires_contract_check_and_final_answer(self):
        payload = {
            "level": "A",
            "task_subtype": "free_form_local_task",
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
        }
        with self.assertRaises(ValidationError):
            GenericASchema.model_validate(payload)

    def test_golden_samples_validate_for_a_b_c(self):
        level_a_payload = {
            "level": "A",
            "task_subtype": "local_task",
            "input_parse": {
                "reaction_entities": ["A", "B"],
                "mapped_atoms_or_species": ["1", "2"],
                "explicit_reagents_or_conditions": ["base"],
            },
            "local_structure": {
                "core_event_region": "C1-C2",
                "relevant_atoms_or_species": ["1", "2"],
                "relevant_bonds_or_roles": ["formed"],
            },
            "reasoning_focus": {
                "requested_local_task": "reaction_center_identification",
                "irrelevant_information_to_ignore": ["yield"],
            },
            "derived_local_result": {
                "extracted_local_change_or_assignment": "C1-C2 formed",
            },
            "contract_check": {
                "answer_matches_requested_task": True,
                "answer_matches_requested_depth": True,
                "answer_matches_exact_benchmark_contract": True,
                "broader_or_alternative_answer_leak": False,
                "exact_local_task_answered": True,
                "broader_summary_leak": False,
                "missing_required_local_detail": False,
            },
            "final_answer": {"value": "C1-C2 formed"},
        }
        parsed_a = GenericASchema.model_validate(level_a_payload)
        self.assertEqual(parsed_a.final_answer.value, "C1-C2 formed")

        level_b_schema, _ = select_sgr_schema("B", "unknown_subtype")
        level_b_payload = {
            "level": "B",
            "task_subtype": "unknown_subtype",
            "target_parse": {
                "target_smiles": "CCO",
                "key_bonds_or_handles": ["C-O"],
            },
            "candidate_disconnections": {
                "disconnection_1": "C-O",
                "disconnection_2": "",
                "disconnection_3": "",
            },
            "candidate_precursor_sets": {
                "set_1": ["CCBr", "OH-"],
                "set_2": [],
                "set_3": [],
            },
            "single_step_check": {
                "set_1_is_immediate": True,
                "set_2_is_immediate": False,
                "set_3_is_immediate": False,
            },
            "forward_regeneration_check": {
                "set_1_regenerates_exact_target": True,
                "set_2_regenerates_exact_target": False,
                "set_3_regenerates_exact_target": False,
            },
            "selection": {
                "chosen_disconnection": "C-O",
                "chosen_precursor_set": ["CCBr", "OH-"],
                "rejected_as_earlier_stage": [],
                "rejected_as_wrong_family": [],
            },
            "contract_check": {
                "answer_matches_requested_task": True,
                "answer_matches_requested_depth": True,
                "answer_matches_exact_benchmark_contract": True,
                "broader_or_alternative_answer_leak": False,
                "immediate_step_only": True,
                "exact_target_preserved": True,
                "alternative_non_immediate_route_leak": False,
            },
            "final_answer": {"value": "CCBr; [OH-]"},
        }
        parsed_b = level_b_schema.model_validate(level_b_payload)
        self.assertEqual(parsed_b.final_answer.value, "CCBr; [OH-]")

        level_c_payload = {
            "level": "C",
            "task_subtype": "reference_route_planning",
            "target_parse": {
                "target_smiles": "CCN",
                "target_defining_constraints": ["ethyl-amine"],
            },
            "reference_alignment": {
                "expected_route_depth": "2-3",
                "likely_key_convergence_pattern": "late amination",
            },
            "route_graph": {
                "target_node": "CCN",
                "key_reference_like_intermediates": ["CCBr"],
                "branch_fragments": ["CC"],
                "convergence_points": ["amination"],
            },
            "target_reachability_check": {
                "exact_target_reached": True,
                "route_depth_sufficient": True,
                "convergence_logic_complete": True,
            },
            "contract_check": {
                "answer_matches_requested_task": True,
                "answer_matches_requested_depth": True,
                "answer_matches_exact_benchmark_contract": True,
                "broader_or_alternative_answer_leak": False,
                "route_not_just_retrosynthetic_idea": True,
                "final_step_reaches_exact_target": True,
            },
            "final_answer": {"value": "Step 1: CC -> CCBr | Step 2: CCBr -> CCN"},
        }
        parsed_c = ReferenceRoutePlanningSchema.model_validate(level_c_payload)
        self.assertIn("Step 2", parsed_c.final_answer.value)


class _ValidatedPayload:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


class OpenShellWorkerSGRTests(unittest.TestCase):
    def _student_payload(self) -> dict:
        return {
            "question": {
                "level": "A",
                "task_subtype": "reaction_center_identification",
                "question_text": "Identify local edits.",
            },
            "model": {
                "model_name": "gpt-5.4-mini",
                "temperature": 0.1,
                "max_tokens": 64,
            },
            "config": load_agent_config(config_path=None),
            "tools_profile": "minimal",
            "max_steps": 2,
        }

    def test_worker_returns_hidden_sgr_metadata_validated(self):
        payload = self._student_payload()
        spec = mock.Mock(schema_name="sgr_a_reaction_center_identification", template={"level": "A"})
        sgr_payload = {
            "level": "A",
            "task_subtype": "reaction_center_identification",
            "contract_check": {"answer_matches_requested_task": True},
            "final_answer": {"value": "demo"},
        }
        with (
            mock.patch.object(openshell_worker, "get_sgr_schema_spec", return_value=spec),
            mock.patch.object(openshell_worker, "validate_sgr_payload", return_value=_ValidatedPayload(sgr_payload)),
            mock.patch.object(
                openshell_worker,
                "_run_tool_loop",
                side_effect=[
                    {"state": "success", "output": '{"level":"A"}', "error": "", "steps": [{"step_number": 1}]},
                    {"state": "success", "output": "Answer: ok", "error": "", "steps": [{"step_number": 2}]},
                ],
            ) as mocked_run_loop,
        ):
            result = openshell_worker._run_student_with_sgr(payload)

        self.assertEqual(result["state"], "success")
        self.assertEqual(result["sgr_schema_name"], "sgr_a_reaction_center_identification")
        self.assertEqual(result["sgr_validation_status"], "validated")
        self.assertFalse(result["sgr_repair_attempted"])
        self.assertFalse(result["sgr_fallback_used"])
        self.assertIsInstance(result["sgr_payload"], dict)
        self.assertEqual(len(result["steps"]), 2)

        called_payload = mocked_run_loop.call_args_list[1].kwargs["payload"]
        self.assertIn("sgr_context", called_payload)
        self.assertEqual(called_payload["sgr_context"]["schema_name"], "sgr_a_reaction_center_identification")

    def test_worker_repairs_sgr_once_before_success(self):
        payload = self._student_payload()
        spec = mock.Mock(schema_name="sgr_a_reaction_center_identification", template={"level": "A"})
        sgr_payload = {
            "level": "A",
            "task_subtype": "reaction_center_identification",
            "contract_check": {"answer_matches_requested_task": True},
            "final_answer": {"value": "demo"},
        }
        with (
            mock.patch.object(openshell_worker, "get_sgr_schema_spec", return_value=spec),
            mock.patch.object(
                openshell_worker,
                "validate_sgr_payload",
                return_value=_ValidatedPayload(sgr_payload),
            ),
            mock.patch.object(
                openshell_worker,
                "_run_tool_loop",
                side_effect=[
                    {"state": "success", "output": "not-json", "error": "", "steps": [{"step_number": 1}]},
                    {"state": "success", "output": '{"level":"A"}', "error": "", "steps": [{"step_number": 2}]},
                    {"state": "success", "output": "Answer: ok", "error": "", "steps": [{"step_number": 3}]},
                ],
            ),
        ):
            result = openshell_worker._run_student_with_sgr(payload)

        self.assertEqual(result["sgr_validation_status"], "validated_after_repair")
        self.assertTrue(result["sgr_repair_attempted"])
        self.assertFalse(result["sgr_fallback_used"])
        self.assertIsInstance(result["sgr_payload"], dict)
        self.assertEqual(len(result["steps"]), 3)

    def test_worker_falls_back_when_sgr_repair_fails(self):
        payload = self._student_payload()
        spec = mock.Mock(schema_name="sgr_a_reaction_center_identification", template={"level": "A"})
        with (
            mock.patch.object(openshell_worker, "get_sgr_schema_spec", return_value=spec),
            mock.patch.object(openshell_worker, "validate_sgr_payload", side_effect=ValueError("bad")),
            mock.patch.object(
                openshell_worker,
                "_run_tool_loop",
                side_effect=[
                    {"state": "success", "output": "bad", "error": "", "steps": [{"step_number": 1}]},
                    {"state": "success", "output": "still-bad", "error": "", "steps": [{"step_number": 2}]},
                    {"state": "success", "output": "Answer: ok", "error": "", "steps": [{"step_number": 3}]},
                ],
            ),
        ):
            result = openshell_worker._run_student_with_sgr(payload)

        self.assertEqual(result["sgr_validation_status"], "fallback_after_repair_failure")
        self.assertTrue(result["sgr_repair_attempted"])
        self.assertTrue(result["sgr_fallback_used"])
        self.assertIsNone(result["sgr_payload"])
        self.assertEqual(len(result["steps"]), 3)

    def test_student_sgr_prompt_includes_schema_block(self):
        spec = mock.Mock(schema_name="sgr_b_generic", template={"level": "B", "contract_check": {}, "final_answer": {"value": ""}})
        prompt = build_student_sgr_task(
            {
                "level": "B",
                "task_subtype": "immediate_precursor_prediction",
                "answer_type": "text",
                "question_text": "Predict immediate precursors.",
            },
            runtime_context={
                "tools_profile": "tools",
                "workspace_root": "/sandbox/workspace",
                "available_tools": ["chem_python_tool"],
            },
            schema_spec=spec,
        )
        self.assertIn("<sgr_schema>", prompt)
        self.assertIn("Schema name: sgr_b_generic", prompt)
        self.assertIn("Return exactly one JSON object", prompt)

    def test_student_prompt_includes_validated_sgr_context(self):
        prompt = build_student_task(
            {
                "level": "C",
                "answer_type": "full_synthesis",
                "task_subtype": "route_design",
                "question_text": "Design a route.",
            },
            runtime_context={
                "tools_profile": "tools",
                "workspace_root": "/sandbox/workspace",
                "available_tools": ["chem_python_tool", "workspace_read_tool"],
            },
            sgr_context={
                "schema_name": "sgr_c_route_design",
                "payload": {
                    "contract_check": {"answer_matches_requested_task": True},
                    "final_answer": {"value": "Step 1: ..."},
                },
            },
        )
        self.assertIn("<validated_sgr_context>", prompt)
        self.assertIn("sgr_c_route_design", prompt)
        self.assertIn("Do not print the schema or hidden fields", prompt)


if __name__ == "__main__":
    unittest.main()
