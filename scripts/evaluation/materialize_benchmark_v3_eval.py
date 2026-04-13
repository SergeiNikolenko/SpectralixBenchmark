from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from scripts.evaluation.benchmark_taxonomy import get_benchmark_taxonomy_metadata


DEFAULT_INPUTS = [
    Path("benchmark/level_a_eval.jsonl"),
    Path("benchmark/level_b_eval.jsonl"),
    Path("benchmark/level_c_eval.jsonl"),
]
DEFAULT_OUTPUT = Path("benchmark/benchmark_v3_eval.jsonl")


def _answer_type_for_level(level: str) -> str:
    normalized = str(level or "").strip().upper()
    if normalized == "A":
        return "reaction_description"
    if normalized == "C":
        return "full_synthesis"
    return "text"


def _max_score_for_level(level: str) -> int:
    normalized = str(level or "").strip().upper()
    if normalized == "C":
        return 4
    if normalized == "B":
        return 2
    return 2


def _question_text(row: Dict[str, Any]) -> str:
    taxonomy = get_benchmark_taxonomy_metadata(row)
    payload = {
        "level": row.get("level"),
        "task_family": row.get("task_family"),
        "task_subtype": row.get("task_subtype"),
        "difficulty": row.get("difficulty"),
        "coverage_tags": row.get("coverage_tags"),
        "benchmark_suite": taxonomy.get("benchmark_suite"),
        "benchmark_subtrack": taxonomy.get("benchmark_subtrack"),
        "planning_horizon": taxonomy.get("planning_horizon"),
        "task_mode": taxonomy.get("task_mode"),
        "difficulty_proxies": taxonomy.get("difficulty_proxies"),
        "eval_contract_id": taxonomy.get("eval_contract_id"),
        "expected_output_schema": taxonomy.get("expected_output_schema"),
        "judge_rubric_id": taxonomy.get("judge_rubric_id"),
        "input": row.get("input"),
    }
    return (
        f"{row.get('input_text', '')}\n\n"
        "Use the structured benchmark context below as the full task specification.\n"
        "Return the best chemistry answer grounded only in this context.\n\n"
        "<benchmark_context>\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "</benchmark_context>"
    ).strip()


def _canonical_answer(row: Dict[str, Any]) -> str:
    return json.dumps(row.get("gold", {}), ensure_ascii=False, sort_keys=True)


def _to_contract_row(row: Dict[str, Any]) -> Dict[str, Any]:
    level = str(row.get("level") or "").strip().upper()
    source_id = str(row.get("source_id") or "unknown")
    source_split = str(row.get("source_split") or "eval")
    record_id = str(row.get("record_id") or "row")
    taxonomy = get_benchmark_taxonomy_metadata(row)
    return {
        "exam_id": f"benchmark_v3_{level.lower()}",
        "page_id": f"{source_id}_{source_split}",
        "question_id": record_id,
        "question_type": "text",
        "question_text": _question_text(row),
        "answer_type": _answer_type_for_level(level),
        "max_score": _max_score_for_level(level),
        "canonical_answer": _canonical_answer(row),
        "status": "ok",
        "error_comment": None,
        "level": row.get("level"),
        "source_id": row.get("source_id"),
        "source_split": row.get("source_split"),
        "source_license": row.get("source_license"),
        "task_family": row.get("task_family"),
        "task_subtype": row.get("task_subtype"),
        "difficulty": row.get("difficulty"),
        "coverage_tags": row.get("coverage_tags"),
        **taxonomy,
        "benchmark_v3_record_id": row.get("record_id"),
        "benchmark_v3_input_text": row.get("input_text"),
        "benchmark_v3_input": row.get("input"),
        "benchmark_v3_gold": row.get("gold"),
        "benchmark_v3_metadata": row.get("metadata"),
    }


def _iter_rows(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)


def materialize(inputs: List[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as f:
        for row in _iter_rows(inputs):
            f.write(json.dumps(_to_contract_row(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize benchmark_v3 eval subsets into the legacy evaluation contract."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Input benchmark_v3 eval JSONL files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL in legacy evaluation contract format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = materialize(inputs=args.inputs, output=args.output)
    print(f"[INFO] Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
