import unittest
import os
from pathlib import Path

from scripts.agents.config import build_executor_kwargs, load_agent_config
from scripts.agents.models import ensure_chat_completions_url, parse_model_url
from scripts.agents.prompts import build_parse_page_task, build_student_task
from scripts.agents.runtime import AgentRuntime
from scripts.agents.tool_registry import build_tools, safe_http_get_tool


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
        tools = build_tools("full", config)
        tool_names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools}
        self.assertIn("safe_http_get_tool", tool_names)

    def test_full_profile_tools_are_expected(self):
        config = load_agent_config(config_path=None)
        tools = build_tools("full", config)
        tool_names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools}
        self.assertIn("chem_format_tool", tool_names)
        self.assertIn("smiles_sanity_tool", tool_names)

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

    def test_default_enables_base_tools(self):
        config = load_agent_config(config_path=None)
        self.assertTrue(config["runtime"]["add_base_tools"])

    def test_invalid_overrides_fail_validation(self):
        with self.assertRaises(ValueError):
            load_agent_config(config_path=None, overrides={"tools": None})

    def test_executor_kwargs_host_port_fields(self):
        config = load_agent_config(config_path=None)
        kwargs = build_executor_kwargs(config, workspace_dir=Path("."))
        self.assertIn("host", kwargs)
        self.assertIn("port", kwargs)
        self.assertNotIn("docker_host", kwargs)
        self.assertNotIn("docker_port", kwargs)
        self.assertFalse(kwargs["build_new_image"])
        self.assertTrue(kwargs["container_run_kwargs"]["network_disabled"])
        self.assertNotIn("volumes", kwargs["container_run_kwargs"])

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
        self.assertFalse(kwargs["container_run_kwargs"]["network_disabled"])


class StatusMappingTests(unittest.TestCase):
    def test_classify_sandbox_error(self):
        status = AgentRuntime._classify_error(RuntimeError("docker executor failed"))
        self.assertEqual(status, "sandbox_error")

    def test_classify_auth_error(self):
        status = AgentRuntime._classify_error(RuntimeError("401 unauthorized"))
        self.assertEqual(status, "auth_error")


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

    def test_base_tools_disabled_when_network_tools_off(self):
        runtime = AgentRuntime(
            model_url="http://127.0.0.1:8317/v1",
            model_name="gpt-4o-mini",
            api_key="test-key",
            sandbox="local",
            tools_profile="minimal",
        )
        self.assertFalse(runtime.allow_network_tools)
        self.assertFalse(runtime.add_base_tools)
        runtime.close()

    def test_base_tools_enabled_only_when_network_tools_allowed(self):
        config_path = Path("scripts/agents/_test_base_tools_true.yaml")
        config_path.write_text(
            "\n".join(
                [
                    "runtime:",
                    "  add_base_tools: true",
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
                model_name="gpt-4o-mini",
                api_key="test-key",
                sandbox="local",
                tools_profile="minimal",
                config_path=config_path,
            )
            self.assertTrue(runtime.allow_network_tools)
            self.assertTrue(runtime.add_base_tools)
            runtime.close()
        finally:
            config_path.unlink(missing_ok=True)


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
        self.assertNotIn("benchmark", prompt.lower())

    def test_student_prompt_includes_contract_sections(self):
        prompt = build_student_task(
            {
                "answer_type": "structure",
                "question_text": "Propose the product structure.",
            }
        )
        self.assertIn("<role>", prompt)
        self.assertIn("<answer_format>", prompt)
        self.assertIn("<tool_rules>", prompt)
        self.assertIn("<completion_criteria>", prompt)
        self.assertIn("Return exactly one SMILES string", prompt)
        self.assertIn("Do not use tools to look for hidden metadata", prompt)

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


if __name__ == "__main__":
    unittest.main()
