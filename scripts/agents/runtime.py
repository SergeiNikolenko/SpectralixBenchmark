from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import os
import re
from urllib.parse import urlparse

from .config import build_executor_kwargs, load_agent_config
from .models import build_openai_model
from .prompts import build_parse_page_task, build_student_task
from .tool_registry import build_tools


@dataclass
class AgentRuntimeError(Exception):
    status: str
    message: str

    def __str__(self) -> str:
        return self.message


class AgentRuntime:
    def __init__(
        self,
        model_url: str,
        model_name: str,
        *,
        benchmark_path: Optional[Path] = None,
        api_key: Optional[str] = None,
        config_path: Optional[Path] = None,
        max_steps: int = 6,
        sandbox: str = "docker",
        tools_profile: str = "full",
        timeout_sec: int = 120,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_dir = Path(__file__).resolve().parents[2]
        self.benchmark_path = Path(benchmark_path).resolve() if benchmark_path else None
        self.max_steps = max(1, int(max_steps))
        self.timeout_sec = max(1, int(timeout_sec))
        self.tools_profile = tools_profile

        self.config = load_agent_config(config_path)
        self.executor_type = sandbox or (self.config.get("sandbox") or {}).get("executor_type", "docker")
        self.executor_kwargs = (
            build_executor_kwargs(self.config, self.workspace_dir)
            if self.executor_type == "docker"
            else {}
        )

        model_cfg = self.config.get("model") or {}
        model_kwargs: Dict[str, Any] = {
            "temperature": model_cfg.get("temperature", 0.2),
            "max_tokens": model_cfg.get("max_tokens", 768),
            "reasoning_effort": str(model_cfg.get("reasoning_effort") or "high"),
        }
        rpm = int(model_cfg.get("requests_per_minute") or 0)
        if rpm > 0:
            model_kwargs["requests_per_minute"] = rpm

        self.model = build_openai_model(
            model_name=model_name,
            model_url=model_url,
            api_key=api_key,
            model_kwargs=model_kwargs,
        )

        runtime_cfg = self.config.get("runtime") or {}
        self.add_base_tools = bool(runtime_cfg.get("add_base_tools", False))
        self.use_structured_outputs_internally = bool(
            runtime_cfg.get("use_structured_outputs_internally", True)
        )
        self.code_block_tags = runtime_cfg.get("code_block_tags", "markdown")
        self.additional_authorized_imports = list(
            runtime_cfg.get("additional_authorized_imports") or []
        )
        if self.add_base_tools:
            try:
                import ddgs  # noqa: F401
                import markdownify  # noqa: F401
            except Exception as exc:
                raise ValueError(
                    "runtime.add_base_tools=true requires toolkit dependencies. "
                    "Install with: pip install 'smolagents[toolkit]' (or ddgs + markdownify) "
                    "or set add_base_tools=false."
                ) from exc

        security_cfg = self.config.get("security") or {}
        self.allowed_hosts = [h.strip() for h in security_cfg.get("allowed_tool_hosts") or [] if h.strip()]
        self._agent: Optional[Any] = None
        self._agent_stack: Optional[ExitStack] = None
        self._preflight_done = False

    def solve_question(self, question: Dict[str, Any]) -> str:
        task = build_student_task(question)
        if self.benchmark_path:
            task += f"\n\nBenchmark path for optional lookup tool: {self.benchmark_path}"
        return self._run_agent_task(task=task)

    def parse_page(self, image_path: str, exam_id: str, page_id: int, marker_prompt: str) -> str:
        task = build_parse_page_task(
            exam_id=exam_id,
            page_id=page_id,
            marker_prompt=marker_prompt,
            image_path=image_path,
        )

        image = self._load_image(image_path)
        raw = self._run_agent_task(task=task, images=[image])
        parsed_array = self._extract_json_array(raw)
        return json.dumps(parsed_array, ensure_ascii=False)

    def _load_image(self, image_path: str):
        try:
            from PIL import Image
        except ImportError as exc:
            raise AgentRuntimeError(
                status="parse_error",
                message="Pillow is required for parser agent image handling. Install with: pip install pillow",
            ) from exc

        try:
            with Image.open(image_path) as image:
                return image.copy()
        except Exception as exc:
            raise AgentRuntimeError(
                status="parse_error",
                message=f"Failed to open image: {exc}",
            ) from exc

    def _extract_json_array(self, raw: str) -> List[Dict[str, Any]]:
        payload = (raw or "").strip()
        if not payload:
            raise AgentRuntimeError(status="parse_error", message="Parser agent returned empty output")

        payload = payload.replace("```json", "").replace("```", "").strip()

        parsed: Any
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", payload)
            if not match:
                raise AgentRuntimeError(
                    status="parse_error",
                    message="Parser agent did not return a JSON array",
                )
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise AgentRuntimeError(
                    status="parse_error",
                    message=f"Parser JSON extraction failed: {exc}",
                ) from exc

        if not isinstance(parsed, list):
            raise AgentRuntimeError(status="parse_error", message="Parser output JSON is not an array")

        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise AgentRuntimeError(
                    status="parse_error",
                    message=f"Parser output item {idx} is not an object",
                )
        return parsed

    def _run_agent_task(self, task: str, images: Optional[List[Any]] = None) -> str:
        try:
            from smolagents import CodeAgent, MCPClient
        except ImportError as exc:
            raise AgentRuntimeError(
                status="parse_error",
                message="smolagents is not installed. Install with: pip install 'smolagents[docker]'",
            ) from exc

        previous_hosts = os.getenv("AGENT_ALLOWED_HOSTS")
        if self.allowed_hosts:
            os.environ["AGENT_ALLOWED_HOSTS"] = ",".join(self.allowed_hosts)

        try:
            self._ensure_agent(CodeAgent, MCPClient)
            if self._agent is None:
                raise AgentRuntimeError(status="parse_error", message="Agent runtime initialization failed")
            result = self._run_code_agent(agent=self._agent, task=task, images=images)
            return "" if result is None else str(result).strip()

        except AgentRuntimeError:
            raise
        except Exception as exc:
            status = self._classify_error(exc)
            self.logger.exception("Agent runtime task failed with status=%s", status)
            raise AgentRuntimeError(status=status, message=str(exc)[:240]) from exc
        finally:
            if previous_hosts is None:
                os.environ.pop("AGENT_ALLOWED_HOSTS", None)
            else:
                os.environ["AGENT_ALLOWED_HOSTS"] = previous_hosts

    def _run_code_agent(self, agent: Any, task: str, images: Optional[List[Any]]) -> Any:
        run_kwargs = {"task": task, "max_steps": self.max_steps, "images": images}
        return agent.run(**run_kwargs)

    def _ensure_agent(self, code_agent_cls: Any, mcp_client_cls: Any) -> None:
        if self._agent is not None:
            return
        self.preflight()
        stack = ExitStack()
        try:
            local_tools = build_tools(self.tools_profile, self.config)
            mcp_tools = self._load_mcp_tools(stack, mcp_client_cls)
            all_tools = [*local_tools, *mcp_tools]

            agent = code_agent_cls(
                model=self.model,
                tools=all_tools,
                add_base_tools=self.add_base_tools,
                max_steps=self.max_steps,
                executor_type=self.executor_type,
                executor_kwargs=self.executor_kwargs,
                additional_authorized_imports=self.additional_authorized_imports,
                use_structured_outputs_internally=self.use_structured_outputs_internally,
                code_block_tags=self.code_block_tags,
            )
            if hasattr(agent, "__enter__") and hasattr(agent, "__exit__"):
                agent = stack.enter_context(agent)
            self._agent = agent
            self._agent_stack = stack
        except Exception:
            stack.close()
            raise

    def preflight(self) -> None:
        if self._preflight_done:
            return
        if self.executor_type == "docker":
            self._preflight_docker()
        self._preflight_done = True

    def _preflight_docker(self) -> None:
        try:
            import docker
            client = docker.from_env()
            client.ping()
            client.close()
        except Exception as exc:
            raise AgentRuntimeError(
                status="sandbox_error",
                message="Docker preflight failed: could not connect to Docker daemon",
            ) from exc

    def close(self) -> None:
        stack = self._agent_stack
        self._agent = None
        self._agent_stack = None
        if stack is not None:
            stack.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return

    def _load_mcp_tools(self, stack: ExitStack, mcp_client_cls: Any) -> List[Any]:
        tools_cfg = self.config.get("tools") or {}
        mcp_cfg = tools_cfg.get("mcp") or {}
        servers = mcp_cfg.get("servers") or []
        if not mcp_cfg.get("enabled", False) or not servers:
            return []

        loaded_tools: List[Any] = []
        allowed_hosts = set(self.allowed_hosts)

        for server in servers:
            if not server.get("enabled", False):
                continue

            transport = (server.get("transport") or "streamable-http").strip().lower()
            server_name = server.get("name", "unnamed")

            try:
                if transport == "streamable-http":
                    url = str(server.get("url") or "").strip()
                    if not url:
                        continue

                    host = self._extract_host(url)
                    if allowed_hosts and host not in allowed_hosts:
                        self.logger.warning(
                            "Skipping MCP server %s: host not allowlisted (%s)",
                            server_name,
                            host,
                        )
                        continue

                    params: Any = {"url": url, "transport": "streamable-http"}
                elif transport == "stdio":
                    try:
                        from mcp import StdioServerParameters
                    except Exception as exc:
                        self.logger.warning(
                            "Stdio MCP unavailable for server %s: %s",
                            server_name,
                            exc,
                        )
                        continue

                    command = str(server.get("command") or "").strip()
                    if not command:
                        self.logger.warning(
                            "Skipping MCP server %s: empty stdio command",
                            server_name,
                        )
                        continue

                    params = StdioServerParameters(
                        command=command,
                        args=list(server.get("args") or []),
                        env={**os.environ, **(server.get("env") or {})},
                    )
                else:
                    self.logger.warning(
                        "Unsupported MCP transport for server %s: %s",
                        server_name,
                        transport,
                    )
                    continue

                server_tools = stack.enter_context(mcp_client_cls(params))
                loaded_tools.extend(server_tools)

            except Exception as exc:
                self.logger.warning("Failed to load MCP server %s: %s", server_name, exc)

        return loaded_tools

    @staticmethod
    def _extract_host(url: str) -> str:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower()

    @staticmethod
    def _classify_error(error: Exception) -> str:
        text = str(error).lower()

        if "timeout" in text or "timed out" in text:
            return "timeout"
        if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
            return "auth_error"
        if any(keyword in text for keyword in ["docker", "sandbox", "executor", "container"]):
            return "sandbox_error"
        if any(keyword in text for keyword in ["tool", "step", "code execution", "interpreter"]):
            return "agent_step_error"
        if "connection" in text or "network" in text:
            return "http_error"

        return "parse_error"
