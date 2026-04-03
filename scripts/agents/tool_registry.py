from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List
import json
import subprocess
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import os


def _normalize_answer_type(answer_type: str) -> str:
    return (answer_type or "").strip().lower()


def _extract_answer_payload(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""

    text = text.replace("```", "")
    text = re.sub(r"`([^`]*)`", r"\1", text)
    answer_line = re.search(r"(?im)^\s*answer\s*:\s*(.+)$", text)
    if answer_line:
        return answer_line.group(1).strip()
    return text


def _normalize_sequence(text: str) -> str:
    sanitized = (text or "").replace("\n", ";").replace(",", ";").replace("|", ";")
    tokens: List[str] = []
    for raw_token in sanitized.split(";"):
        token = raw_token.strip()
        if not token:
            continue
        token = re.sub(r"^\d+[\.)]\s*", "", token)
        if token:
            tokens.append(token)
    return "; ".join(tokens)


def chem_format_tool(answer_type: str, raw_text: str) -> str:
    """
    Normalizes raw model output into benchmark-friendly format by answer_type.

    Args:
        answer_type: Benchmark answer type.
        raw_text: Raw model answer text.
    """
    payload = _extract_answer_payload(raw_text)
    if not payload:
        return ""

    normalized_type = _normalize_answer_type(answer_type)

    if normalized_type == "single_choice":
        tokens = re.findall(r"[A-Za-z0-9]+", payload)
        return tokens[0] if tokens else payload.strip()

    if normalized_type in {"multiple_choice", "ordering"}:
        seq = _normalize_sequence(payload)
        return seq if seq else payload.strip()

    if normalized_type in {"msms_structure_prediction", "structure"}:
        for line in payload.splitlines():
            candidate = line.strip()
            if candidate:
                return candidate
        return payload.strip()

    return re.sub(r"\s+", " ", payload).strip()


def smiles_sanity_tool(smiles: str) -> str:
    """
    Performs lightweight sanity checks for a SMILES candidate string.

    Args:
        smiles: Candidate SMILES string.
    """
    candidate = (smiles or "").strip().replace(" ", "")
    if not candidate:
        return json.dumps({"valid": False, "reason": "empty"})

    allowed_pattern = r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+$"
    if not re.match(allowed_pattern, candidate):
        return json.dumps({"valid": False, "reason": "illegal_characters", "normalized": candidate})

    if candidate.count("(") != candidate.count(")"):
        return json.dumps({"valid": False, "reason": "unbalanced_parentheses", "normalized": candidate})

    if candidate.count("[") != candidate.count("]"):
        return json.dumps({"valid": False, "reason": "unbalanced_brackets", "normalized": candidate})

    return json.dumps({"valid": True, "normalized": candidate})


def unit_convert_tool(value: float, from_unit: str, to_unit: str) -> str:
    """
    Converts between selected chemistry units.

    Args:
        value: Numeric value to convert.
        from_unit: Source unit (g, mg, kg, L, mL, uL, mol, mmol, umol).
        to_unit: Target unit.
    """
    src = (from_unit or "").strip().lower()
    dst = (to_unit or "").strip().lower()

    scales = {
        "g": 1.0,
        "mg": 1e-3,
        "kg": 1e3,
        "l": 1.0,
        "ml": 1e-3,
        "ul": 1e-6,
        "mol": 1.0,
        "mmol": 1e-3,
        "umol": 1e-6,
    }

    groups = {
        "mass": {"g", "mg", "kg"},
        "volume": {"l", "ml", "ul"},
        "amount": {"mol", "mmol", "umol"},
    }

    def group_of(u: str) -> str:
        for g_name, members in groups.items():
            if u in members:
                return g_name
        return ""

    src_group = group_of(src)
    dst_group = group_of(dst)

    if not src_group or not dst_group:
        return json.dumps({"status": "error", "reason": "unsupported_unit"})
    if src_group != dst_group:
        return json.dumps({"status": "error", "reason": "incompatible_unit_groups"})

    base_value = float(value) * scales[src]
    converted = base_value / scales[dst]

    return json.dumps(
        {
            "status": "ok",
            "value": converted,
            "from_unit": src,
            "to_unit": dst,
        }
    )


def rubric_hint_tool(answer_type: str) -> str:
    """
    Returns concise benchmark scoring hints by answer type.

    Args:
        answer_type: Benchmark answer type.
    """
    a = _normalize_answer_type(answer_type)
    hints = {
        "single_choice": "One label only (A/B/C/...).",
        "multiple_choice": "List labels separated by '; ' with no extra prose.",
        "ordering": "Return full order as '; '-separated tokens.",
        "numeric": "Return only the numeric value with unit only if required.",
        "structure": "Return one structure representation (prefer SMILES) without explanation.",
        "msms_structure_prediction": "Return one canonical SMILES candidate.",
        "full_synthesis": "Prioritize correct reagents, sequence, and key intermediates.",
        "reaction_description": "State transformation, reagents, and selectivity succinctly.",
        "property_determination": "Report derived property and minimal supporting rationale.",
        "text": "Lead with machine-readable answer, then short explanation if necessary.",
    }
    return hints.get(a, "Produce a precise, machine-readable answer with minimal extra text.")


def json_array_validate_tool(raw_json: str) -> str:
    """
    Validates that input is a JSON array and returns normalized compact JSON.

    Args:
        raw_json: Candidate JSON array string.
    """
    payload = (raw_json or "").strip()
    if not payload:
        return json.dumps({"valid": False, "reason": "empty"})

    # Allow markdown fences.
    payload = payload.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", payload)
        if not match:
            return json.dumps({"valid": False, "reason": "json_parse_error"})
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, list):
        return json.dumps({"valid": False, "reason": "not_array"})

    return json.dumps({"valid": True, "normalized": json.dumps(parsed, ensure_ascii=False)})


def chem_python_tool(code: str, timeout_sec: int = 20) -> str:
    """
    Runs a short Python snippet inside the project uv environment.
    Useful for chemistry validation, RDKit canonicalization, and small calculations.

    Args:
        code: Python code to execute. Print the final result to stdout.
        timeout_sec: Hard timeout in seconds.
    """
    snippet = (code or "").strip()
    if not snippet:
        return json.dumps({"status": "error", "reason": "empty_code"})

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.run(
            [os.environ.get("PYTHON_BIN", "python"), "-c", snippet],
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "error", "reason": "timeout"})
    except Exception as exc:
        return json.dumps({"status": "error", "reason": f"python_tool_error:{exc}"})

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    return json.dumps(
        {
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "stdout": stdout[:12000],
            "stderr": stderr[:12000],
        },
        ensure_ascii=False,
    )


def safe_http_get_tool(url: str, timeout_sec: int = 10) -> str:
    """
    Fetches a trusted URL only if host is allowlisted in AGENT_ALLOWED_HOSTS.

    Args:
        url: HTTPS/HTTP URL.
        timeout_sec: Request timeout in seconds.
    """
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return json.dumps({"status": "error", "reason": "unsupported_scheme"})

    host = (parsed.hostname or "").lower()
    allowed_hosts = {
        h.strip().lower()
        for h in (os.getenv("AGENT_ALLOWED_HOSTS") or "").split(",")
        if h.strip()
    }
    allow_all_hosts = "*" in allowed_hosts

    if not allow_all_hosts and allowed_hosts and host not in allowed_hosts:
        return json.dumps({"status": "error", "reason": "host_not_allowed", "host": host})

    req = Request(url, headers={"User-Agent": "SpectralixAgent/1.0"}, method="GET")
    try:
        with urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
            body = resp.read(4_096).decode("utf-8", errors="replace")
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


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    func: Callable[..., str]
    description: str
    schema: Dict[str, Any]
    is_network_tool: bool = False


def _object_schema(
    *,
    properties: Dict[str, Dict[str, Any]],
    required: List[str],
) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOL_DEFINITIONS: Dict[str, ToolDefinition] = {
    "chem_format_tool": ToolDefinition(
        name="chem_format_tool",
        func=chem_format_tool,
        description="Normalize a raw chemistry answer into benchmark-friendly machine-readable format.",
        schema=_object_schema(
            properties={
                "answer_type": {"type": "string"},
                "raw_text": {"type": "string"},
            },
            required=["answer_type", "raw_text"],
        ),
    ),
    "smiles_sanity_tool": ToolDefinition(
        name="smiles_sanity_tool",
        func=smiles_sanity_tool,
        description="Perform lightweight syntax checks on a candidate SMILES string.",
        schema=_object_schema(
            properties={
                "smiles": {"type": "string"},
            },
            required=["smiles"],
        ),
    ),
    "unit_convert_tool": ToolDefinition(
        name="unit_convert_tool",
        func=unit_convert_tool,
        description="Convert common chemistry units for mass, volume, and amount.",
        schema=_object_schema(
            properties={
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
            required=["value", "from_unit", "to_unit"],
        ),
    ),
    "rubric_hint_tool": ToolDefinition(
        name="rubric_hint_tool",
        func=rubric_hint_tool,
        description="Return a short answer-format hint for a benchmark answer type.",
        schema=_object_schema(
            properties={
                "answer_type": {"type": "string"},
            },
            required=["answer_type"],
        ),
    ),
    "json_array_validate_tool": ToolDefinition(
        name="json_array_validate_tool",
        func=json_array_validate_tool,
        description="Validate that a string contains a JSON array and return a compact normalized version.",
        schema=_object_schema(
            properties={
                "raw_json": {"type": "string"},
            },
            required=["raw_json"],
        ),
    ),
    "chem_python_tool": ToolDefinition(
        name="chem_python_tool",
        func=chem_python_tool,
        description="Run a short Python chemistry validation snippet and capture stdout and stderr.",
        schema=_object_schema(
            properties={
                "code": {"type": "string"},
                "timeout_sec": {"type": "integer", "minimum": 1},
            },
            required=["code"],
        ),
    ),
    "safe_http_get_tool": ToolDefinition(
        name="safe_http_get_tool",
        func=safe_http_get_tool,
        description="Fetch a trusted HTTP(S) URL when the hostname is allowlisted.",
        schema=_object_schema(
            properties={
                "url": {"type": "string"},
                "timeout_sec": {"type": "integer", "minimum": 1},
            },
            required=["url"],
        ),
        is_network_tool=True,
    ),
}


TOOL_REGISTRY = {name: item.func for name, item in TOOL_DEFINITIONS.items()}


def build_tools(profile: str, config: Dict[str, Any]) -> List[Any]:
    profiles = ((config.get("tools") or {}).get("profiles") or {})
    if profile not in profiles:
        known_profiles = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"Unknown tools profile '{profile}'. Known profiles: {known_profiles}")
    selected = profiles[profile]
    security = config.get("security") or {}
    has_allowed_hosts = bool(security.get("allowed_tool_hosts"))
    allow_network_tools = bool(security.get("allow_network_tools", False))
    tools: List[Any] = []

    for name in selected:
        tool_definition = TOOL_DEFINITIONS.get(name)
        if tool_definition is None:
            raise ValueError(f"Unknown tool in profile '{profile}': {name}")

        if tool_definition.is_network_tool and (not has_allowed_hosts or not allow_network_tools):
            # Keep tool disabled when allowlist is empty.
            continue

        tools.append(tool_definition.func)

    return tools


def build_tool_definitions(profile: str, config: Dict[str, Any]) -> List[ToolDefinition]:
    profiles = ((config.get("tools") or {}).get("profiles") or {})
    if profile not in profiles:
        known_profiles = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"Unknown tools profile '{profile}'. Known profiles: {known_profiles}")

    selected = profiles[profile]
    security = config.get("security") or {}
    has_allowed_hosts = bool(security.get("allowed_tool_hosts"))
    allow_network_tools = bool(security.get("allow_network_tools", False))

    tool_definitions: List[ToolDefinition] = []
    for name in selected:
        tool_definition = TOOL_DEFINITIONS.get(name)
        if tool_definition is None:
            raise ValueError(f"Unknown tool in profile '{profile}': {name}")
        if tool_definition.is_network_tool and (not has_allowed_hosts or not allow_network_tools):
            continue
        tool_definitions.append(tool_definition)
    return tool_definitions
