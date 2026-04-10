from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import base64
import json
import logging
import shutil
import subprocess
import sys

from .config import build_executor_kwargs, load_agent_config, resolve_runtime_backend
from .models import ModelSettings, build_model_settings
from .openshell_manager import OpenShellManager, OpenShellManagerError
from .prompts import build_parse_page_task, build_student_task
from .tool_registry import build_tool_definitions


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
        api_key: Optional[str] = None,
        config_path: Optional[Path] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        max_steps: int = 6,
        sandbox: str = "openshell",
        backend: Optional[str] = None,
        tools_profile: str = "minimal",
        timeout_sec: int = 120,
        sgr_enabled: bool = True,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_dir = Path(__file__).resolve().parents[2]
        self.max_steps = max(1, int(max_steps))
        self.timeout_sec = max(1, int(timeout_sec))
        self.tools_profile = tools_profile
        self.config = load_agent_config(config_path, overrides=config_overrides)
        self.executor_type = sandbox or (self.config.get("sandbox") or {}).get("executor_type", "openshell")
        self.runtime_backend = resolve_runtime_backend(
            self.config,
            executor_type=self.executor_type,
            requested_backend=backend,
        )
        self.sgr_enabled = bool(sgr_enabled)
        self.executor_kwargs = (
            build_executor_kwargs(self.config, self.workspace_dir)
            if self.executor_type == "openshell"
            else {}
        )

        model_cfg = self.config.get("model") or {}
        self.model_settings = build_model_settings(
            model_name=model_name,
            model_url=model_url,
            api_key=api_key,
            model_kwargs={
                "temperature": model_cfg.get("temperature", 0.2),
                "max_tokens": model_cfg.get("max_tokens", 768),
                "reasoning_effort": str(model_cfg.get("reasoning_effort") or "medium"),
                "requests_per_minute": int(model_cfg.get("requests_per_minute") or 0),
            },
            sandbox_visible=self.executor_type == "openshell",
        )

        self._last_run_details: Optional[Dict[str, Any]] = None
        self._preflight_done = False
        self._openshell_manager: Optional[OpenShellManager] = None

    def solve_question(self, question: Dict[str, Any]) -> str:
        payload = {
            "mode": "student",
            "question": question,
            "model": self._payload_model_settings(),
            "config": self.config,
            "tools_profile": self.tools_profile,
            "max_steps": self.max_steps,
            "sgr_enabled": self.sgr_enabled,
            "workspace_root": self._payload_workspace_root(),
            "uv_bin": self._payload_uv_bin(),
        }
        result = self._run_payload(payload)
        return str(result.get("output") or "").strip()

    def parse_page(self, image_path: str, exam_id: str, page_id: int, marker_prompt: str) -> str:
        image_bytes = Path(image_path).read_bytes()
        payload = {
            "mode": "parser",
            "exam_id": exam_id,
            "page_id": page_id,
            "marker_prompt": marker_prompt,
            "image_path": image_path,
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "model": self._payload_model_settings(),
            "config": self.config,
            "tools_profile": "minimal",
            "max_steps": self.max_steps,
            "workspace_root": self._payload_workspace_root(),
            "uv_bin": self._payload_uv_bin(),
        }
        result = self._run_payload(payload)
        raw = str(result.get("output") or "").strip()
        parsed_array = self._extract_json_array(raw)
        return json.dumps(parsed_array, ensure_ascii=False)

    def preflight(self) -> None:
        if self._preflight_done:
            return
        if self.executor_type == "openshell":
            self._openshell_manager = OpenShellManager(
                workspace_dir=self.workspace_dir,
                executor_kwargs=self.executor_kwargs,
            )
            self._openshell_manager.ensure_sandbox(
                model_settings=self.model_settings,
                tools_profile=self.tools_profile,
                backend=self.runtime_backend,
            )
        self._preflight_done = True

    def close(self) -> None:
        if self._openshell_manager is not None:
            self._openshell_manager.close()
        self._openshell_manager = None

    def get_last_run_details(self) -> Optional[Dict[str, Any]]:
        if self._last_run_details is None:
            return None
        return json.loads(json.dumps(self._last_run_details, ensure_ascii=False))

    def get_runtime_metadata(self) -> Dict[str, Any]:
        tool_definitions = build_tool_definitions(self.tools_profile, self.config)
        requested_tools = list((((self.config.get("tools") or {}).get("profiles") or {}).get(self.tools_profile) or []))
        configured_local_tools = (
            []
            if self.runtime_backend == "codex_native"
            else [item.name for item in tool_definitions]
        )
        return {
            "executor_type": self.executor_type,
            "runtime_backend": self.runtime_backend,
            "tools_profile": self.tools_profile,
            "requested_local_tools": requested_tools,
            "configured_local_tools": configured_local_tools,
            "allow_network_tools": bool((self.config.get("security") or {}).get("allow_network_tools", False)),
            "allowed_tool_hosts": list((self.config.get("security") or {}).get("allowed_tool_hosts") or []),
            "model_api_base": self.model_settings.api_base,
            "managed_inference": self.executor_type == "openshell" and self.runtime_backend == "openshell_worker",
            "sandbox_runtime": (
                "openshell_codex_native"
                if self.runtime_backend == "codex_native"
                else ("openshell" if self.executor_type == "openshell" else "local_worker")
            ),
        }

    def _payload_model_settings(self) -> Dict[str, Any]:
        payload = dict(self.model_settings.__dict__)
        if self.executor_type == "openshell":
            payload["api_key"] = ""
        return payload

    def _payload_workspace_root(self) -> str:
        if self.executor_type == "local":
            return str(self.workspace_dir)
        return "/sandbox/workspace"

    def _payload_uv_bin(self) -> str:
        if self.executor_type == "local":
            return str(shutil.which("uv") or "")
        return "/sandbox/.venv/bin/uv"

    def _run_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._last_run_details = None
            self.preflight()
            worker_timeout = self._payload_timeout_seconds(payload)
            payload = dict(payload)
            payload["timeout_sec"] = worker_timeout
            if self.executor_type == "local":
                result = self._run_local_worker(payload, timeout_seconds=worker_timeout)
            else:
                if self._openshell_manager is None:
                    raise AgentRuntimeError(status="sandbox_error", message="OpenShell manager not initialized")
                if self.runtime_backend == "codex_native":
                    result = self._openshell_manager.exec_codex_worker(
                        payload=payload,
                        timeout_seconds=worker_timeout,
                    )
                else:
                    result = self._openshell_manager.exec_worker(payload=payload, timeout_seconds=worker_timeout)
            self._last_run_details = result
            if str(result.get("state") or "").strip().lower() != "success":
                message = str(result.get("error") or result.get("output") or "agent worker returned error state").strip()
                raise AgentRuntimeError(status="agent_step_error", message=message[:240] or "agent worker failed")
            return result
        except AgentRuntimeError:
            raise
        except Exception as exc:
            status = self._classify_error(exc)
            self.logger.exception("Agent runtime task failed with status=%s", status)
            raise AgentRuntimeError(status=status, message=str(exc)[:240]) from exc

    def _run_local_worker(self, payload: Dict[str, Any], *, timeout_seconds: int) -> Dict[str, Any]:
        process = subprocess.run(
            [sys.executable, "-m", "scripts.agents.openshell_worker"],
            cwd=str(self.workspace_dir),
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if process.returncode != 0:
            raise AgentRuntimeError(
                status="agent_step_error",
                message=(process.stderr or process.stdout or "local worker failed").strip()[:240],
            )
        return json.loads((process.stdout or "").strip() or "{}")

    def _extract_json_array(self, raw: str) -> List[Dict[str, Any]]:
        payload = (raw or "").strip()
        if not payload:
            raise AgentRuntimeError(status="parse_error", message="Parser agent returned empty output")

        payload = payload.replace("```json", "").replace("```", "").strip()

        try:
            parsed: Any = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AgentRuntimeError(status="parse_error", message=f"Parser JSON extraction failed: {exc}") from exc

        if not isinstance(parsed, list):
            raise AgentRuntimeError(status="parse_error", message="Parser output JSON is not an array")
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise AgentRuntimeError(status="parse_error", message=f"Parser output item {idx} is not an object")
        return parsed

    @staticmethod
    def _classify_error(error: Exception) -> str:
        text = str(error).lower()
        if "timeout" in text or "timed out" in text or "exit_code=124" in text or "deadline exceeded" in text:
            return "timeout"
        if (
            "429" in text
            or "rate limit" in text
            or "insufficient_quota" in text
            or "model_cooldown" in text
            or "cooling down" in text
        ):
            return "http_error"
        if "401" in text or "403" in text or "unauthorized" in text or "forbidden" in text:
            return "auth_error"
        if any(keyword in text for keyword in ["openshell", "sandbox", "gateway", "docker"]):
            return "sandbox_error"
        if any(keyword in text for keyword in ["codex", "auth required", "invalid refresh token"]):
            return "auth_error"
        if any(keyword in text for keyword in ["tool", "step", "worker failed"]):
            return "agent_step_error"
        if "connection" in text or "network" in text:
            return "http_error"
        return "parse_error"

    def _payload_timeout_seconds(self, payload: Dict[str, Any]) -> int:
        timeout = self.timeout_sec
        mode = str(payload.get("mode") or "").strip().lower()
        if mode != "student":
            return timeout

        question = payload.get("question") or {}
        level = str(question.get("level") or "").strip().upper()
        answer_type = str(question.get("answer_type") or "").strip().lower()
        task_subtype = str(question.get("task_subtype") or "").strip().lower()

        if level == "C" or answer_type == "full_synthesis":
            return max(timeout, 900)
        if level == "B" and (
            answer_type == "text" or "disconnection" in task_subtype or "precursor" in task_subtype
        ):
            return max(timeout, 600)
        return max(timeout, 360)
