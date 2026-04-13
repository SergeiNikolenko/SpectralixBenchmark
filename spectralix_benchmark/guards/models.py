from __future__ import annotations

from typing import Optional
import os
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from spectralix_benchmark.agents.models import parse_model_url, resolve_api_key


def _resolve_model_url(model_url: Optional[str]) -> str:
    explicit = (model_url or "").strip()
    if explicit:
        return explicit
    env_value = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    if not env_value:
        return "https://api.openai.com/v1"
    return env_value


def _use_local_transport(api_base: str) -> bool:
    hostname = (urlparse(api_base).hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "inference.local"}


def build_openai_chat_model(
    *,
    model_name: str,
    model_url: Optional[str],
    api_key: Optional[str],
) -> OpenAIChatModel:
    resolved_api_key = resolve_api_key(api_key)
    if not resolved_api_key:
        raise ValueError("API key not found for PydanticAI guard runtime")

    api_base, _ = parse_model_url(_resolve_model_url(model_url))
    client_kwargs = {}
    if _use_local_transport(api_base):
        client_kwargs["http_client"] = httpx.AsyncClient(trust_env=False)

    header_name = (os.getenv("OPENAI_API_KEY_HEADER") or "Authorization").strip() or "Authorization"
    prefix = (os.getenv("OPENAI_API_KEY_PREFIX") or "Bearer").strip()
    header_is_authorization = header_name.lower() == "authorization"
    prefix_is_bearer = prefix.lower() == "bearer"

    if header_is_authorization and prefix_is_bearer:
        if client_kwargs:
            async_client = AsyncOpenAI(
                base_url=api_base,
                api_key=resolved_api_key,
                **client_kwargs,
            )
            provider = OpenAIProvider(openai_client=async_client)
        else:
            provider = OpenAIProvider(base_url=api_base, api_key=resolved_api_key)
    else:
        token_value = f"{prefix} {resolved_api_key}".strip() if prefix else resolved_api_key
        async_client = AsyncOpenAI(
            base_url=api_base,
            api_key="unused_api_key",
            default_headers={header_name: token_value},
            **client_kwargs,
        )
        provider = OpenAIProvider(openai_client=async_client)

    return OpenAIChatModel(model_name, provider=provider)
