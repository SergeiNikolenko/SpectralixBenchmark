from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse
import os

CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
DEFAULT_API_PATH = "/v1"


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
    """
    Convert full endpoint URL to OpenAI-compatible api_base.

    Returns:
        (api_base, chat_completions_url)
    """
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
    """Return a normalized /chat/completions endpoint URL."""
    _, chat_url = parse_model_url(model_url)
    return chat_url


def build_openai_model(
    model_name: str,
    model_url: str,
    api_key: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    model_kwargs = dict(model_kwargs or {})
    resolved_api_key = resolve_api_key(api_key)
    if not resolved_api_key:
        raise ValueError(
            "API key not found. Set OPENAI_API_KEY (or AITUNNEL_API_KEY) or pass explicit api_key."
        )

    api_base, _ = parse_model_url(model_url)

    header_name = (os.getenv("OPENAI_API_KEY_HEADER") or "Authorization").strip() or "Authorization"
    prefix = (os.getenv("OPENAI_API_KEY_PREFIX") or "Bearer").strip()
    header_is_authorization = header_name.lower() == "authorization"
    prefix_is_bearer = prefix.lower() == "bearer"

    client_kwargs: Dict[str, Any] = {}
    model_api_key = resolved_api_key

    # Custom auth headers for gateways that do not use standard Authorization: Bearer.
    if not (header_is_authorization and prefix_is_bearer):
        token_value = f"{prefix} {resolved_api_key}".strip() if prefix else resolved_api_key
        client_kwargs["default_headers"] = {header_name: token_value}
        if not header_is_authorization:
            model_api_key = "unused_api_key"

    try:
        from smolagents import OpenAIModel
    except ImportError as exc:
        raise ImportError(
            "smolagents is not installed. Install with: pip install 'smolagents[docker]'"
        ) from exc

    return OpenAIModel(
        model_id=model_name,
        api_base=api_base,
        api_key=model_api_key,
        client_kwargs=client_kwargs,
        **model_kwargs,
    )
