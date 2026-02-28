from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import os

try:
    from smolagents import tool
except Exception:  # pragma: no cover - graceful import fallback
    def tool(func):  # type: ignore
        return func


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


@tool
def benchmark_lookup_tool(benchmark_path: str, exam_id: str, page_id: str, question_id: str) -> str:
    """
    Looks up benchmark entries and returns a sanitized question row as JSON.
    Gold labels/answers and scoring metadata are always redacted.

    Args:
        benchmark_path: Path to benchmark JSONL file.
        exam_id: Exam identifier.
        page_id: Page identifier.
        question_id: Question identifier.
    """
    path = Path(benchmark_path)
    if not path.exists():
        return json.dumps({"status": "not_found", "reason": f"file_missing:{benchmark_path}"})

    target_exam = str(exam_id)
    target_page = str(page_id)
    target_q = str(question_id)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                str(row.get("exam_id")) == target_exam
                and str(row.get("page_id")) == target_page
                and str(row.get("question_id")) == target_q
            ):
                redacted_keys = {
                    "canonical_answer",
                    "answer",
                    "gold_answer",
                    "expected_answer",
                    "max_score",
                    "rubric",
                    "score_rubric",
                }
                sanitized = {k: v for k, v in row.items() if k not in redacted_keys}
                return json.dumps(
                    {
                        "status": "ok",
                        "row": sanitized,
                        "redacted_fields": sorted(k for k in row.keys() if k in redacted_keys),
                    },
                    ensure_ascii=False,
                )

    return json.dumps(
        {
            "status": "not_found",
            "reason": "row_not_found",
            "lookup": {
                "exam_id": target_exam,
                "page_id": target_page,
                "question_id": target_q,
            },
        },
        ensure_ascii=False,
    )


@tool
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


@tool
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


@tool
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


@tool
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


@tool
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


@tool
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


TOOL_REGISTRY = {
    "benchmark_lookup_tool": benchmark_lookup_tool,
    "chem_format_tool": chem_format_tool,
    "smiles_sanity_tool": smiles_sanity_tool,
    "unit_convert_tool": unit_convert_tool,
    "rubric_hint_tool": rubric_hint_tool,
    "json_array_validate_tool": json_array_validate_tool,
    "safe_http_get_tool": safe_http_get_tool,
}


def build_tools(profile: str, config: Dict[str, Any]) -> List[Any]:
    profiles = ((config.get("tools") or {}).get("profiles") or {})
    if profile not in profiles:
        known_profiles = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"Unknown tools profile '{profile}'. Known profiles: {known_profiles}")
    selected = profiles[profile]
    security = config.get("security") or {}
    has_allowed_hosts = bool(security.get("allowed_tool_hosts"))
    allow_shell_tools = bool(security.get("allow_shell_tools", False))
    allow_file_write_tools = bool(security.get("allow_file_write_tools", False))
    allow_network_tools = bool(security.get("allow_network_tools", False))
    tools: List[Any] = []

    for name in selected:
        tool_obj = TOOL_REGISTRY.get(name)
        if tool_obj is None:
            raise ValueError(f"Unknown tool in profile '{profile}': {name}")

        if name == "safe_http_get_tool" and (not has_allowed_hosts or not allow_network_tools):
            # Keep tool disabled when allowlist is empty.
            continue

        if name.startswith("shell_") and not allow_shell_tools:
            continue
        if name.startswith("file_write_") and not allow_file_write_tools:
            continue

        tools.append(tool_obj)

    return tools
