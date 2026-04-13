from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse
import os

CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
DEFAULT_API_PATH = "/v1"
OPEN_SHELL_MANAGED_INFERENCE_BASE = "https://inference.local/v1"


@dataclass(frozen=True)
class ModelSettings:
    model_name: str
    api_base: str
    api_key: str
    temperature: float
    max_tokens: int
    reasoning_effort: str
    requests_per_minute: int
    upstream_api_base: Optional[str] = None


def resolve_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("AITUNNEL_API_KEY")
        or os.getenv("TOGETHER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )


def _resolve_api_path(path: str) -> str:
    normalized_path = path.rstrip("/")
    if not normalized_path:
        return DEFAULT_API_PATH

    if normalized_path.endswith(CHAT_COMPLETIONS_SUFFIX):
        normalized_path = normalized_path[: -len(CHAT_COMPLETIONS_SUFFIX)]
    elif CHAT_COMPLETIONS_SUFFIX in normalized_path:
        normalized_path = normalized_path.split(CHAT_COMPLETIONS_SUFFIX, 1)[0]
    elif "/v1" in normalized_path:
        v1_index = normalized_path.find("/v1")
        normalized_path = normalized_path[: v1_index + len("/v1")]

    return normalized_path or DEFAULT_API_PATH


def parse_model_url(model_url: str) -> Tuple[str, str]:
    cleaned = (model_url or "").strip()
    if not cleaned:
        raise ValueError("model_url must not be empty")

    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid model URL: {model_url}")

    api_path = _resolve_api_path(parsed.path)

    api_base = urlunparse((parsed.scheme, parsed.netloc, api_path, "", "", ""))
    chat_url = api_base.rstrip("/") + CHAT_COMPLETIONS_SUFFIX
    return api_base, chat_url


def ensure_chat_completions_url(model_url: str) -> str:
    _, chat_url = parse_model_url(model_url)
    return chat_url


def _sandbox_host(hostname: str) -> str:
    lowered = (hostname or "").strip().lower()
    if lowered in {"127.0.0.1", "localhost"}:
        return "host.openshell.internal"
    return hostname


def sandbox_visible_api_base(model_url: str) -> str:
    parsed = urlparse(parse_model_url(model_url)[0])
    host = _sandbox_host(parsed.hostname or "")
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))


def build_model_settings(
    *,
    model_name: str,
    model_url: str,
    api_key: Optional[str],
    model_kwargs: Optional[Dict[str, Any]] = None,
    sandbox_visible: bool = False,
) -> ModelSettings:
    model_kwargs = dict(model_kwargs or {})
    resolved_api_key = resolve_api_key(api_key)
    if not resolved_api_key:
        raise ValueError(
            "API key not found. Set OPENAI_API_KEY (or AITUNNEL_API_KEY) or pass explicit api_key."
        )

    api_base = sandbox_visible_api_base(model_url) if sandbox_visible else parse_model_url(model_url)[0]
    upstream_api_base = sandbox_visible_api_base(model_url) if sandbox_visible else None
    if sandbox_visible:
        api_base = OPEN_SHELL_MANAGED_INFERENCE_BASE
    return ModelSettings(
        model_name=model_name,
        api_base=api_base,
        api_key=resolved_api_key,
        temperature=float(model_kwargs.get("temperature", 0.2)),
        max_tokens=int(model_kwargs.get("max_tokens", 768)),
        reasoning_effort=str(model_kwargs.get("reasoning_effort") or "medium"),
        requests_per_minute=int(model_kwargs.get("requests_per_minute") or 0),
        upstream_api_base=upstream_api_base,
    )
