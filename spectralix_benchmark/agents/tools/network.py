from __future__ import annotations

import json
import os
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def safe_http_get_tool(url: str, timeout_sec: int = 10) -> str:
    """
    Fetches a trusted URL only if host is allowlisted in AGENT_ALLOWED_HOSTS.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return json.dumps({"status": "error", "reason": "unsupported_scheme"})

    host = (parsed.hostname or "").lower()
    allowed_hosts = {h.strip().lower() for h in (os.getenv("AGENT_ALLOWED_HOSTS") or "").split(",") if h.strip()}
    allow_all_hosts = "*" in allowed_hosts
    if not allow_all_hosts and allowed_hosts and host not in allowed_hosts:
        return json.dumps({"status": "error", "reason": "host_not_allowed", "host": host})

    req = Request(url, headers={"User-Agent": "SpectralixAgent/1.0"}, method="GET")
    try:
        with urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
            return json.dumps(
                {
                    "status": "ok",
                    "code": getattr(resp, "status", 200),
                    "content_preview": body,
                },
                ensure_ascii=False,
            )
    except Exception as exc:
        return json.dumps({"status": "error", "reason": f"http_error:{exc}"})

