from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import shutil
import subprocess
import os
import re
import uuid

from .models import ModelSettings


class OpenShellManagerError(RuntimeError):
    pass


@dataclass
class OpenShellSandboxHandle:
    gateway_name: str
    sandbox_name: str
    sandbox_id: str
    created_by_manager: bool = False


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value or "").strip()).strip("-._")
    return cleaned or "runtime"


class OpenShellManager:
    _WORKSPACE_UPLOAD_ENTRIES = (
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "benchmark",
        "docs",
        "scripts",
        "tests",
    )

    def __init__(
        self,
        *,
        workspace_dir: Path,
        executor_kwargs: Dict[str, Any],
    ) -> None:
        self.workspace_dir = workspace_dir
        self.executor_kwargs = dict(executor_kwargs or {})
        self.openshell_bin = shutil.which("openshell") or os.path.expanduser("~/.local/bin/openshell")
        self.gateway_name = str(self.executor_kwargs.get("gateway_name") or "spectralix")
        self.gateway_port = int(self.executor_kwargs.get("gateway_port") or 18080)
        self.gateway_plaintext = bool(self.executor_kwargs.get("gateway_plaintext", True))
        self.auto_start_gateway = bool(self.executor_kwargs.get("auto_start_gateway", True))
        self.sandbox_name = str(self.executor_kwargs.get("sandbox_name") or "spectralix-runtime")
        self.sandbox_from = str(self.executor_kwargs.get("sandbox_from") or "base")
        self.ready_timeout_seconds = float(self.executor_kwargs.get("ready_timeout_seconds") or 180)
        self.delete_on_close = bool(self.executor_kwargs.get("delete_on_close", False))
        native_codex_cfg = dict(self.executor_kwargs.get("native_codex") or {})
        self.native_codex_sandbox_name = str(native_codex_cfg.get("sandbox_name") or "spectralix-codex-runtime")
        self.native_codex_sandbox_from = str(native_codex_cfg.get("sandbox_from") or "codex")
        self.native_codex_bin = str(native_codex_cfg.get("codex_bin") or "codex")
        self.native_codex_home_dir = str(native_codex_cfg.get("codex_home_dir") or "/sandbox/.codex")
        self.native_codex_upload_auth_from = str(
            native_codex_cfg.get("upload_auth_from") or "~/.codex/auth.json"
        )
        self._handle: Optional[OpenShellSandboxHandle] = None
        self._client = None

    def ensure_gateway(self) -> None:
        if not self.auto_start_gateway:
            return
        if self._gateway_is_healthy():
            return
        command = [
            self.openshell_bin,
            "gateway",
            "start",
            "--name",
            self.gateway_name,
            "--port",
            str(self.gateway_port),
            "--recreate",
        ]
        if self.gateway_plaintext:
            command.append("--plaintext")
        self._run_cli(command, timeout_seconds=900)

    def _ensure_managed_inference(self, *, model_settings: ModelSettings) -> None:
        upstream_api_base = str(model_settings.upstream_api_base or "").strip()
        if not upstream_api_base:
            return

        provider_name = "spectralix-openai-local"
        get_result = subprocess.run(
            [self.openshell_bin, "provider", "get", "-g", self.gateway_name, provider_name],
            cwd=str(self.workspace_dir),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        provider_command = [
            self.openshell_bin,
            "provider",
            "update" if get_result.returncode == 0 else "create",
            "-g",
            self.gateway_name,
        ]
        if get_result.returncode != 0:
            provider_command.extend(
                [
                    "--name",
                    provider_name,
                    "--type",
                    "openai",
                ]
            )
        provider_command.extend(
            [
                "--credential",
                f"OPENAI_API_KEY={model_settings.api_key}",
                "--config",
                f"OPENAI_BASE_URL={upstream_api_base}",
            ]
        )
        if get_result.returncode == 0:
            provider_command.append(provider_name)
        self._run_cli(provider_command, timeout_seconds=120)
        self._run_cli(
            [
                self.openshell_bin,
                "inference",
                "set",
                "-g",
                self.gateway_name,
                "--provider",
                provider_name,
                "--model",
                model_settings.model_name,
            ],
            timeout_seconds=120,
        )

    def ensure_sandbox(
        self,
        *,
        model_settings: ModelSettings,
        tools_profile: str,
        backend: str = "openshell_worker",
    ) -> OpenShellSandboxHandle:
        if self._handle is not None:
            return self._handle

        self.ensure_gateway()
        if backend != "codex_native":
            self._ensure_managed_inference(model_settings=model_settings)
        sandbox_name = self._sandbox_name_for_backend(backend)
        sandbox_from = self._sandbox_from_for_backend(backend)
        try:
            client = self._client_for_gateway()
            created_by_manager = False
            try:
                sandbox_ref = client.get(sandbox_name)
            except Exception:
                sandbox_ref = None
            if sandbox_ref is None:
                self._run_cli(
                    [
                        self.openshell_bin,
                        "sandbox",
                        "create",
                        "-g",
                        self.gateway_name,
                        "--name",
                        sandbox_name,
                        "--from",
                        sandbox_from,
                        "--no-tty",
                        "--",
                        "true",
                    ],
                    timeout_seconds=1800,
                )
                created_by_manager = True
            sandbox_ref = client.wait_ready(
                sandbox_name,
                timeout_seconds=self.ready_timeout_seconds,
            )
            self._handle = OpenShellSandboxHandle(
                gateway_name=self.gateway_name,
                sandbox_name=sandbox_name,
                sandbox_id=sandbox_ref.id,
                created_by_manager=created_by_manager,
            )
            self._upload_runtime_sources(self._handle)
            if backend == "codex_native":
                self._ensure_runtime_venv(self._handle)
                self._bootstrap_codex_home(self._handle)
            else:
                self._bootstrap_runtime_dependencies(self._handle, tools_profile=tools_profile)
            return self._handle
        except Exception:
            raise

    def exec_worker(
        self,
        *,
        payload: Dict[str, Any],
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        if self._handle is None:
            raise OpenShellManagerError("OpenShell sandbox has not been initialized")

        result = self._client_for_gateway().exec(
            self._handle.sandbox_id,
            ["/sandbox/.venv/bin/python", "-m", "scripts.agents.openshell_worker"],
            workdir="/sandbox",
            stdin=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout_seconds=timeout_seconds,
            env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/sandbox"},
        )
        stdout = self._text_output(result.stdout).strip()
        stderr = self._text_output(result.stderr).strip()
        if result.exit_code != 0:
            if result.exit_code == 124:
                raise OpenShellManagerError(
                    f"OpenShell worker timed out after {timeout_seconds}s: stdout={stdout[:200]} stderr={stderr[:200]}"
                )
            raise OpenShellManagerError(
                f"OpenShell worker failed with exit_code={result.exit_code}: {stderr or 'unknown error'}"
            )
        if not stdout:
            raise OpenShellManagerError("OpenShell worker returned empty stdout")
        return json.loads(stdout)

    def exec_codex_worker(
        self,
        *,
        payload: Dict[str, Any],
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        if self._handle is None:
            raise OpenShellManagerError("OpenShell sandbox has not been initialized")

        result = self._client_for_gateway().exec(
            self._handle.sandbox_id,
            ["/sandbox/.venv/bin/python", "-m", "scripts.agents.codex_native_worker"],
            workdir="/sandbox",
            stdin=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout_seconds=timeout_seconds,
            env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/sandbox"},
        )
        stdout = self._text_output(result.stdout).strip()
        stderr = self._text_output(result.stderr).strip()
        if result.exit_code != 0:
            if result.exit_code == 124:
                raise OpenShellManagerError(
                    f"OpenShell native Codex worker timed out after {timeout_seconds}s: stdout={stdout[:200]} stderr={stderr[:200]}"
                )
            raise OpenShellManagerError(
                f"OpenShell native Codex worker failed with exit_code={result.exit_code}: {stderr or 'unknown error'}"
            )
        if not stdout:
            raise OpenShellManagerError("OpenShell native Codex worker returned empty stdout")
        return json.loads(stdout)

    def _gateway_is_healthy(self) -> bool:
        completed = subprocess.run(
            [self.openshell_bin, "status", "-g", self.gateway_name],
            cwd=str(self.workspace_dir),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return completed.returncode == 0

    def close(self) -> None:
        if self._handle and self.delete_on_close and self._handle.created_by_manager:
            try:
                self._client_for_gateway().delete(self._handle.sandbox_name)
            except Exception:
                pass
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._handle = None
        self._client = None

    def _client_for_gateway(self):
        if self._client is not None:
            return self._client
        try:
            from openshell import SandboxClient
        except ImportError as exc:
            raise OpenShellManagerError(
                "OpenShell Python SDK is not installed. Install project dependencies with uv sync."
            ) from exc
        self._client = SandboxClient.from_active_cluster(cluster=self.gateway_name, timeout=30.0)
        return self._client

    def _run_cli(self, command: list[str], *, timeout_seconds: int) -> None:
        completed = subprocess.run(
            command,
            cwd=str(self.workspace_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode == 0:
            return
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        raise OpenShellManagerError(
            f"OpenShell CLI failed: {' '.join(command)} | stdout={stdout[:500]} | stderr={stderr[:500]}"
        )

    def _sandbox_name_for_backend(self, backend: str) -> str:
        base_name = self.native_codex_sandbox_name if backend == "codex_native" else self.sandbox_name
        return base_name or f"spectralix-runtime-{uuid.uuid4().hex[:8]}"

    def _sandbox_from_for_backend(self, backend: str) -> str:
        return self.native_codex_sandbox_from if backend == "codex_native" else self.sandbox_from

    def _upload_runtime_sources(self, handle: OpenShellSandboxHandle) -> None:
        self._run_cli(
            [
                self.openshell_bin,
                "sandbox",
                "upload",
                "-g",
                self.gateway_name,
                handle.sandbox_name,
                "scripts",
                "/sandbox/scripts",
            ],
            timeout_seconds=300,
        )
        prepare = self._client_for_gateway().exec(
            handle.sandbox_id,
            ["python3", "-c", "from pathlib import Path; Path('/sandbox/workspace').mkdir(parents=True, exist_ok=True)"],
            workdir="/sandbox",
            timeout_seconds=30,
            env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/sandbox"},
        )
        if prepare.exit_code != 0:
            stderr = self._text_output(prepare.stderr).strip()
            raise OpenShellManagerError(
                f"OpenShell workspace bootstrap failed with exit_code={prepare.exit_code}: {stderr or 'unknown error'}"
            )
        for entry in self._WORKSPACE_UPLOAD_ENTRIES:
            source = self.workspace_dir / entry
            if not source.exists():
                continue
            destination = f"/sandbox/workspace/{entry}"
            self._run_cli(
                [
                    self.openshell_bin,
                    "sandbox",
                    "upload",
                    "-g",
                    self.gateway_name,
                    handle.sandbox_name,
                    str(source),
                    destination,
                ],
                timeout_seconds=300,
            )

    def _bootstrap_runtime_dependencies(self, handle: OpenShellSandboxHandle, *, tools_profile: str) -> None:
        self._ensure_runtime_venv(handle)
        packages = ["openai>=1.3.0", "uv>=0.8.0"]
        if tools_profile in {"tools", "tools_internet", "full"}:
            packages.append("rdkit>=2025.9.6")

        check_script = [
            "/sandbox/.venv/bin/python",
            "-c",
            (
                "import importlib.util; "
                f"mods={['openai', 'uv'] + (['rdkit'] if tools_profile in {'tools', 'tools_internet', 'full'} else [])!r}; "
                "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
                "print('\\n'.join(missing))"
            ),
        ]
        check = self._client_for_gateway().exec(
            handle.sandbox_id,
            check_script,
            workdir="/sandbox",
            timeout_seconds=60,
            env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/sandbox"},
        )
        missing = [
            line.strip()
            for line in self._text_output(check.stdout).splitlines()
            if line.strip()
        ]
        if check.exit_code == 0 and not missing:
            return

        command = [
            "/sandbox/.venv/bin/pip",
            "install",
            "--disable-pip-version-check",
            *packages,
        ]
        result = self._client_for_gateway().exec(
            handle.sandbox_id,
            command,
            workdir="/sandbox",
            timeout_seconds=900,
            env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/sandbox"},
        )
        if result.exit_code != 0:
            stderr = self._text_output(result.stderr).strip()
            raise OpenShellManagerError(
                f"OpenShell bootstrap failed with exit_code={result.exit_code}: {stderr or 'unknown error'}"
            )

    def _bootstrap_codex_home(self, handle: OpenShellSandboxHandle) -> None:
        auth_source = Path(os.path.expanduser(self.native_codex_upload_auth_from))
        if not auth_source.exists():
            raise OpenShellManagerError(
                f"Native Codex auth file not found: {auth_source}"
            )
        prepare = self._client_for_gateway().exec(
            handle.sandbox_id,
            [
                "python3",
                "-c",
                (
                    "from pathlib import Path; "
                    f"home=Path({self.native_codex_home_dir!r}); "
                    "home.mkdir(parents=True, exist_ok=True)"
                ),
            ],
            workdir="/sandbox",
            timeout_seconds=30,
            env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/sandbox"},
        )
        if prepare.exit_code != 0:
            stderr = self._text_output(prepare.stderr).strip()
            raise OpenShellManagerError(
                f"OpenShell native Codex home bootstrap failed with exit_code={prepare.exit_code}: {stderr or 'unknown error'}"
            )
        self._run_cli(
            [
                self.openshell_bin,
                "sandbox",
                "upload",
                "-g",
                self.gateway_name,
                handle.sandbox_name,
                str(auth_source),
                f"{self.native_codex_home_dir.rstrip('/')}/auth.json",
            ],
            timeout_seconds=300,
        )

    def _ensure_runtime_venv(self, handle: OpenShellSandboxHandle) -> None:
        check = self._client_for_gateway().exec(
            handle.sandbox_id,
            ["python3", "-c", "from pathlib import Path; raise SystemExit(0 if Path('/sandbox/.venv/bin/python').exists() else 1)"],
            workdir="/sandbox",
            timeout_seconds=30,
            env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/sandbox"},
        )
        if check.exit_code == 0:
            return
        create = self._client_for_gateway().exec(
            handle.sandbox_id,
            ["python3", "-m", "venv", "/sandbox/.venv"],
            workdir="/sandbox",
            timeout_seconds=180,
            env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": "/sandbox"},
        )
        if create.exit_code != 0:
            stderr = self._text_output(create.stderr).strip()
            raise OpenShellManagerError(
                f"OpenShell venv bootstrap failed with exit_code={create.exit_code}: {stderr or 'unknown error'}"
            )

    @staticmethod
    def _text_output(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
