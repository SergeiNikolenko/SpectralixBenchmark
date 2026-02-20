import unittest
from pathlib import Path

from scripts.agents.config import load_agent_config
from scripts.agents.models import ensure_chat_completions_url, parse_model_url
from scripts.agents.runtime import AgentRuntime
from scripts.agents.tool_registry import build_tools


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
        tools = build_tools("full", config)
        tool_names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools}
        self.assertIn("safe_http_get_tool", tool_names)

    def test_unknown_tool_fails_fast(self):
        config = load_agent_config(config_path=None)
        config["tools"]["profiles"]["full"] = ["unknown_tool_name"]
        with self.assertRaises(ValueError):
            build_tools("full", config)


class ConfigTests(unittest.TestCase):
    def test_missing_config_file_raises(self):
        missing = Path("scripts/agents/does_not_exist.yaml")
        with self.assertRaises(FileNotFoundError):
            load_agent_config(config_path=missing)


class StatusMappingTests(unittest.TestCase):
    def test_classify_sandbox_error(self):
        status = AgentRuntime._classify_error(RuntimeError("docker executor failed"))
        self.assertEqual(status, "sandbox_error")

    def test_classify_auth_error(self):
        status = AgentRuntime._classify_error(RuntimeError("401 unauthorized"))
        self.assertEqual(status, "auth_error")


if __name__ == "__main__":
    unittest.main()
