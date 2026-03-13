from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmark"

LEVEL_A_POOL = BENCHMARK_DIR / "level_a.jsonl"
LEVEL_B_POOL = BENCHMARK_DIR / "level_b.jsonl"
LEVEL_C_POOL = BENCHMARK_DIR / "level_c.jsonl"

LEVEL_A_EVAL = BENCHMARK_DIR / "level_a_eval.jsonl"
LEVEL_B_EVAL = BENCHMARK_DIR / "level_b_eval.jsonl"
LEVEL_C_EVAL = BENCHMARK_DIR / "level_c_eval.jsonl"
PAPER_EVAL_MANIFEST = BENCHMARK_DIR / "paper_eval_manifest.yaml"
PAPER_EVAL_DOC = BENCHMARK_DIR / "PAPER_EVALS.md"


def stable_rank(record_id: str) -> int:
    return int(hashlib.sha1(record_id.encode("utf-8")).hexdigest(), 16)


@dataclass
class Rule:
    name: str
    quota: int
    predicate: Callable[[dict], bool]
    heap: list[tuple[int, str]] = field(default_factory=list)

    def consider(self, row: dict) -> None:
        if not self.predicate(row):
            return
        rank = stable_rank(row["record_id"])
        item = (-rank, row["record_id"])
        if len(self.heap) < self.quota:
            heapq.heappush(self.heap, item)
            return
        if item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    @property
    def selected_ids(self) -> list[str]:
        return [record_id for _, record_id in sorted(self.heap, reverse=True)]


def build_selection(pool_path: Path, rules: list[Rule]) -> set[str]:
    with pool_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for rule in rules:
                rule.consider(row)
    selected: set[str] = set()
    for rule in rules:
        selected.update(rule.selected_ids)
    return selected


def fill_to_target(pool_path: Path, selected_ids: set[str], target_count: int) -> set[str]:
    slots_needed = target_count - len(selected_ids)
    if slots_needed <= 0:
        return selected_ids

    heap: list[tuple[int, str]] = []
    with pool_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = row["record_id"]
            if record_id in selected_ids:
                continue
            rank = stable_rank(record_id)
            item = (-rank, record_id)
            if len(heap) < slots_needed:
                heapq.heappush(heap, item)
                continue
            if item > heap[0]:
                heapq.heapreplace(heap, item)

    for _, record_id in sorted(heap, reverse=True):
        selected_ids.add(record_id)
        if len(selected_ids) >= target_count:
            break
    return selected_ids


def write_subset(pool_path: Path, output_path: Path, selected_ids: set[str]) -> int:
    count = 0
    written_ids: set[str] = set()
    with pool_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as sink:
        for line in source:
            row = json.loads(line)
            record_id = row["record_id"]
            if record_id not in selected_ids:
                continue
            if record_id in written_ids:
                continue
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            written_ids.add(record_id)
            count += 1
    return count


def make_level_a_rules() -> list[Rule]:
    return [
        Rule(
            name="a_reaction_center",
            quota=76,
            predicate=lambda row: row["source_id"] == "pmechdb"
            and row["task_subtype"] == "reaction_center_identification",
        ),
        Rule(
            name="a_mechanistic_class",
            quota=76,
            predicate=lambda row: row["source_id"] == "pmechdb"
            and row["task_subtype"] == "mechanistic_classification",
        ),
        Rule(
            name="a_transformation_type",
            quota=76,
            predicate=lambda row: row["source_id"] == "uspto_50k",
        ),
        Rule(
            name="a_reagent_roles_chemu",
            quota=80,
            predicate=lambda row: row["source_id"] == "chemu_2020",
        ),
        Rule(
            name="a_reagent_roles_weave2",
            quota=32,
            predicate=lambda row: row["source_id"] == "weave2",
        ),
        Rule(
            name="a_condition_roles_choriso",
            quota=80,
            predicate=lambda row: row["source_id"] == "choriso",
        ),
    ]


def make_level_b_rules() -> list[Rule]:
    return [
        Rule(
            name="b_orderly_easy",
            quota=70,
            predicate=lambda row: row["source_id"] == "orderly" and row["difficulty"] == "easy",
        ),
        Rule(
            name="b_orderly_medium",
            quota=70,
            predicate=lambda row: row["source_id"] == "orderly"
            and row["difficulty"] == "medium",
        ),
        Rule(
            name="b_orderly_hard",
            quota=70,
            predicate=lambda row: row["source_id"] == "orderly" and row["difficulty"] == "hard",
        ),
        Rule(
            name="b_paroutes_easy",
            quota=70,
            predicate=lambda row: row["source_id"] == "paroutes"
            and row["difficulty"] == "easy",
        ),
        Rule(
            name="b_paroutes_medium",
            quota=70,
            predicate=lambda row: row["source_id"] == "paroutes"
            and row["difficulty"] == "medium",
        ),
        Rule(
            name="b_paroutes_hard",
            quota=70,
            predicate=lambda row: row["source_id"] == "paroutes"
            and row["difficulty"] == "hard",
        ),
    ]


def make_level_c_rules() -> list[Rule]:
    return [
        Rule(
            name="c_pilot_route_design",
            quota=13,
            predicate=lambda row: row["source_id"] == "benchmark_v1_0",
        ),
        Rule(
            name="c_n1_medium",
            quota=34,
            predicate=lambda row: row["source_id"] == "paroutes"
            and row["source_split"] == "n1"
            and row["difficulty"] == "medium",
        ),
        Rule(
            name="c_n1_hard",
            quota=34,
            predicate=lambda row: row["source_id"] == "paroutes"
            and row["source_split"] == "n1"
            and row["difficulty"] == "hard",
        ),
        Rule(
            name="c_n5_medium",
            quota=34,
            predicate=lambda row: row["source_id"] == "paroutes"
            and row["source_split"] == "n5"
            and row["difficulty"] == "medium",
        ),
        Rule(
            name="c_n5_hard",
            quota=35,
            predicate=lambda row: row["source_id"] == "paroutes"
            and row["source_split"] == "n5"
            and row["difficulty"] == "hard",
        ),
    ]


def write_manifest(counts: dict[str, int]) -> None:
    lines = [
        "schema_version: paper_eval_v1",
        "files:",
        f"  level_a_eval: {{path: benchmark/level_a_eval.jsonl, records: {counts['A']}}}",
        f"  level_b_eval: {{path: benchmark/level_b_eval.jsonl, records: {counts['B']}}}",
        f"  level_c_eval: {{path: benchmark/level_c_eval.jsonl, records: {counts['C']}}}",
        "selection_policy:",
        "  level_a: source- and subtype-balanced deterministic sample",
        "  level_b: source- and difficulty-balanced deterministic sample",
        "  level_c: route-source-balanced deterministic sample with pilot tasks kept intact",
        "notes:",
        "  - A target size: 420",
        "  - B target size: 420",
        "  - C target size: 150",
        "  - C has no meaningful easy bucket in the current pool, so balancing uses medium/hard only.",
    ]
    PAPER_EVAL_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc() -> None:
    text = """# Paper Eval Subsets

These files are compact paper-oriented evaluation subsets built on top of the large benchmark pools.

## Files

- `level_a_eval.jsonl`
- `level_b_eval.jsonl`
- `level_c_eval.jsonl`
- `paper_eval_manifest.yaml`

## Intended Sizes

- `Level A`: 420 records
- `Level B`: 420 records
- `Level C`: 150 records

## Balancing Policy

### Level A

Balanced by source and task subtype:

- `PMechDB` reaction center identification
- `PMechDB` mechanistic classification
- `USPTO-50K` transformation classification
- `ChEMU 2020` reagent-role extraction
- `WEAVE2` reagent-role extraction
- `CHORISO` condition-role identification

### Level B

Balanced by source and difficulty:

- `ORDerly`: easy / medium / hard
- `PaRoutes selected_reactions_all`: easy / medium / hard

### Level C

Balanced by route source and difficulty:

- all internal pilot route-design tasks are kept
- `PaRoutes n1`: medium / hard
- `PaRoutes n5`: medium / hard

## Determinism

Sampling is deterministic and based on a stable hash of `record_id`.
"""
    PAPER_EVAL_DOC.write_text(text, encoding="utf-8")


def build() -> dict[str, int]:
    level_a_rules = make_level_a_rules()
    level_b_rules = make_level_b_rules()
    level_c_rules = make_level_c_rules()

    selected_a = fill_to_target(LEVEL_A_POOL, build_selection(LEVEL_A_POOL, level_a_rules), 420)
    selected_b = fill_to_target(LEVEL_B_POOL, build_selection(LEVEL_B_POOL, level_b_rules), 420)
    selected_c = fill_to_target(LEVEL_C_POOL, build_selection(LEVEL_C_POOL, level_c_rules), 150)

    counts = {
        "A": write_subset(LEVEL_A_POOL, LEVEL_A_EVAL, selected_a),
        "B": write_subset(LEVEL_B_POOL, LEVEL_B_EVAL, selected_b),
        "C": write_subset(LEVEL_C_POOL, LEVEL_C_EVAL, selected_c),
    }
    write_manifest(counts)
    write_doc()
    return counts


def main() -> None:
    counts = build()
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
