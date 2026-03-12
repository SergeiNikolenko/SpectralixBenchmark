from __future__ import annotations

import csv
import json
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
EXTERNAL_SOURCES_DIR = REPO_ROOT / "external_sources"

LEVEL_A_OUTPUT = BENCHMARK_DIR / "level_a.jsonl"
LEVEL_B_OUTPUT = BENCHMARK_DIR / "level_b.jsonl"
LEVEL_C_OUTPUT = BENCHMARK_DIR / "level_c.jsonl"
LEVELS_MANIFEST = BENCHMARK_DIR / "levels_manifest.yaml"

PMECHDB_MANUAL_TEST = (
    EXTERNAL_SOURCES_DIR
    / "level_a/pmechdb/extracted/dataset/pmechdb_data/manually_curated_test_challenging.csv"
)
PMECHDB_COMBINATORIAL_TEST = (
    EXTERNAL_SOURCES_DIR
    / "level_a/pmechdb/extracted/dataset/pmechdb_data/combinatorial_test.csv"
)
USPTO_50K_TEST = EXTERNAL_SOURCES_DIR / "level_a/uspto_50k/raw/uspto50k_test.csv"
CHEMU_NER_TRAIN = EXTERNAL_SOURCES_DIR / "level_a/chemu_2020/raw/chemu.ner.train.zip"
CHEMU_NER_DEV = EXTERNAL_SOURCES_DIR / "level_a/chemu_2020/raw/chemu.ner.dev.zip"
CHORISO_PUBLIC = EXTERNAL_SOURCES_DIR / "level_a/choriso/extracted/choriso_public/choriso_public.tsv"
WEAVE2_DIR = EXTERNAL_SOURCES_DIR / "level_a/weave2/extracted/weave2/weave2"

ORDERLY_RETRO_TEST = EXTERNAL_SOURCES_DIR / "level_b/orderly/raw/orderly_retro_test.parquet"
PAROUTES_SELECTED_REACTIONS = (
    EXTERNAL_SOURCES_DIR / "level_b/paroutes/extracted/PaRoutes-main/data/selected_reactions_all.csv"
)
PAROUTES_N1_ROUTES = (
    EXTERNAL_SOURCES_DIR / "level_b/paroutes/extracted/PaRoutes-main/data/n1-routes.json"
)
PAROUTES_N5_ROUTES = (
    EXTERNAL_SOURCES_DIR / "level_b/paroutes/extracted/PaRoutes-main/data/n5-routes.json"
)

PILOT_BENCHMARK = BENCHMARK_DIR / "benchmark_v1_0.jsonl"

LEVEL_C_RECORD_IDS = {
    ("exam_3", "7"),
    ("exam_3", "8"),
    ("exam_4", "4"),
    ("exam_4", "8"),
    ("exam_4", "9"),
    ("exam_4", "10"),
    ("exam_7", "2"),
    ("exam_8", "2"),
    ("exam_8", "5"),
    ("exam_8", "6"),
    ("exam_10", "2"),
    ("exam_11", "8"),
    ("exam_11", "9"),
}

LICENSES = {
    "pmechdb": "CC-BY-NC-ND",
    "uspto_50k": "CC-BY-4.0",
    "chemu_2020": "CC-BY-NC-3.0",
    "choriso": "CC-BY-4.0",
    "weave2": "unknown",
    "orderly": "CC-BY-4.0",
    "paroutes": "Apache-2.0",
    "benchmark_v1_0": "internal_pilot",
}

CHEMU_ROLE_LABELS = {
    "STARTING_MATERIAL": "starting_materials",
    "REACTION_PRODUCT": "reaction_products",
    "SOLVENT": "solvents",
    "OTHER_COMPOUND": "other_compounds",
    "TEMPERATURE": "temperature",
    "TIME": "time",
}


@dataclass(frozen=True)
class BuildStats:
    level_a_records: int
    level_b_records: int
    level_c_records: int


def json_ready(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def parse_reaction_triplet(value: str) -> tuple[str, str, str]:
    parts = value.split(">")
    if len(parts) != 3:
        return value, "", ""
    return parts[0], parts[1], parts[2]


def molecule_list(value: str) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(".") if item]


def split_pmechdb_reaction_and_arrow_codes(value: str) -> tuple[str, list[str]]:
    stripped = value.strip().strip('"')
    if " " not in stripped:
        return stripped, []
    reaction, arrow_codes = stripped.rsplit(" ", 1)
    center = [item for item in arrow_codes.split(";") if item]
    return reaction, center


def safe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_bool(value: Any) -> bool | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def has_stereo(smiles: str) -> bool:
    return "@" in smiles


def dedupe_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = value.strip()
        if not text or text == "empty" or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_record(
    *,
    record_id: str,
    level: str,
    source_id: str,
    source_split: str,
    task_family: str,
    task_subtype: str,
    input_text: str,
    input_payload: dict[str, Any],
    gold_payload: dict[str, Any],
    metadata: dict[str, Any],
    difficulty: str,
    coverage_tags: list[str],
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "level": level,
        "source_id": source_id,
        "source_split": source_split,
        "source_license": LICENSES[source_id],
        "task_family": task_family,
        "task_subtype": task_subtype,
        "difficulty": difficulty,
        "coverage_tags": coverage_tags,
        "input_text": input_text,
        "input": input_payload,
        "gold": gold_payload,
        "metadata": metadata,
    }


def iter_level_a_records() -> Iterator[dict[str, Any]]:
    yield from iter_pmechdb_manual_challenging_records()
    yield from iter_pmechdb_combinatorial_test_records()
    yield from iter_uspto_50k_test_records()
    yield from iter_chemu_ner_records(CHEMU_NER_TRAIN, split="train")
    yield from iter_chemu_ner_records(CHEMU_NER_DEV, split="dev")
    yield from iter_choriso_public_records()
    yield from iter_weave2_records()


def iter_pmechdb_manual_challenging_records() -> Iterator[dict[str, Any]]:
    with PMECHDB_MANUAL_TEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader, start=1):
            if not row:
                continue
            reaction_smirks, reaction_center = split_pmechdb_reaction_and_arrow_codes(row[0])
            reactants, reagents, products = parse_reaction_triplet(reaction_smirks)
            yield build_record(
                record_id=f"pmechdb_manual_test_challenging_{index:05d}",
                level="A",
                source_id="pmechdb",
                source_split="test_challenging",
                task_family="Reaction Understanding",
                task_subtype="reaction_center_identification",
                difficulty="hard",
                coverage_tags=["reaction_center", "major_product"],
                input_text="Identify the reaction center for the mapped transformation.",
                input_payload={
                    "reaction_smirks": reaction_smirks,
                    "reactants": reactants,
                    "reagents": reagents,
                    "products": products,
                },
                gold_payload={
                    "reaction_center": reaction_center,
                    "mechanistic_class": None,
                    "transformation_type": None,
                    "major_product": products,
                },
                metadata={
                    "source_path": str(PMECHDB_MANUAL_TEST.relative_to(REPO_ROOT)),
                    "selection_note": "PMechDB manually curated challenging test row.",
                },
            )


def iter_pmechdb_combinatorial_test_records() -> Iterator[dict[str, Any]]:
    with PMECHDB_COMBINATORIAL_TEST.open(newline="", encoding="utf-8") as handle:
        next(handle)
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            reaction_smirks, reaction_center = split_pmechdb_reaction_and_arrow_codes(
                row["SMIRKS"]
            )
            reactants, reagents, products = parse_reaction_triplet(reaction_smirks)
            yield build_record(
                record_id=f"pmechdb_combinatorial_test_{index:05d}",
                level="A",
                source_id="pmechdb",
                source_split="test",
                task_family="Reaction Understanding",
                task_subtype="mechanistic_classification",
                difficulty="medium",
                coverage_tags=["mechanistic_class", "reaction_center", "major_product"],
                input_text="Classify the mechanistic class of the mapped reaction and identify the reaction center.",
                input_payload={
                    "reaction_smirks": reaction_smirks,
                    "reactants": reactants,
                    "reagents": reagents,
                    "products": products,
                    "conditions": {
                        "nucleophile_solvent": row["Nu Solvent"],
                        "temperature_k": row["Temp(K)"],
                        "sn_parameter": row["sN(N+E)"],
                    },
                },
                gold_payload={
                    "reaction_center": reaction_center,
                    "mechanistic_class": row["orbital pair classification"].strip() or None,
                    "transformation_type": row["orbital pair classification"].strip() or None,
                    "major_product": products,
                },
                metadata={
                    "source_path": str(PMECHDB_COMBINATORIAL_TEST.relative_to(REPO_ROOT)),
                    "selection_note": "PMechDB combinatorial test row.",
                },
            )


def iter_uspto_50k_test_records() -> Iterator[dict[str, Any]]:
    with USPTO_50K_TEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            reaction_smiles = row["reactants>reagents>production"]
            reactants, reagents, products = parse_reaction_triplet(reaction_smiles)
            coverage = ["transformation_type", "major_product"]
            if has_stereo(products):
                coverage.append("stereochemical_outcome")
            yield build_record(
                record_id=f"uspto_50k_test_{row['id']}",
                level="A",
                source_id="uspto_50k",
                source_split="test",
                task_family="Reaction Understanding",
                task_subtype="transformation_classification",
                difficulty="easy",
                coverage_tags=coverage,
                input_text="Identify the transformation class for the reaction and report the major product.",
                input_payload={
                    "reaction_smiles": reaction_smiles,
                    "reactants": reactants,
                    "reagents": reagents,
                    "products": products,
                },
                gold_payload={
                    "reaction_class": row["class"],
                    "transformation_type": f"class_{row['class']}",
                    "major_product": products,
                    "reaction_center": None,
                },
                metadata={
                    "source_row_id": row["id"],
                    "source_path": str(USPTO_50K_TEST.relative_to(REPO_ROOT)),
                    "selection_note": "USPTO-50K official test split row.",
                },
            )


def parse_brat_annotations(text: str) -> list[tuple[str, str]]:
    annotations: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        label = parts[1].split(" ", 1)[0]
        value = parts[2].strip()
        annotations.append((label, value))
    return annotations


def iter_chemu_ner_records(zip_path: Path, split: str) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as archive:
        txt_files = sorted(name for name in archive.namelist() if name.endswith(".txt"))
        for name in txt_files:
            stem = Path(name).stem
            ann_name = f"{Path(name).parent}/{stem}.ann"
            if ann_name not in archive.namelist():
                continue
            text = archive.read(name).decode("utf-8", errors="replace").strip()
            ann_text = archive.read(ann_name).decode("utf-8", errors="replace")
            grouped = {
                "starting_materials": [],
                "reaction_products": [],
                "solvents": [],
                "other_compounds": [],
                "temperature": [],
                "time": [],
            }
            for label, value in parse_brat_annotations(ann_text):
                group_key = CHEMU_ROLE_LABELS.get(label)
                if not group_key:
                    continue
                grouped[group_key].append(value)
            if not any(grouped.values()):
                continue
            yield build_record(
                record_id=f"chemu_ner_{split}_{stem}",
                level="A",
                source_id="chemu_2020",
                source_split=split,
                task_family="Reaction Understanding",
                task_subtype="reagent_role_identification",
                difficulty="medium",
                coverage_tags=["reagent_roles", "procedural_text"],
                input_text="Identify reagent roles and reaction conditions from the procedure text.",
                input_payload={
                    "procedure_text": text,
                },
                gold_payload={
                    "reagent_roles": {
                        "starting_materials": dedupe_preserve(grouped["starting_materials"]),
                        "reaction_products": dedupe_preserve(grouped["reaction_products"]),
                        "solvents": dedupe_preserve(grouped["solvents"]),
                        "other_compounds": dedupe_preserve(grouped["other_compounds"]),
                    },
                    "conditions": {
                        "temperature": dedupe_preserve(grouped["temperature"]),
                        "time": dedupe_preserve(grouped["time"]),
                    },
                },
                metadata={
                    "source_path": str(zip_path.relative_to(REPO_ROOT)),
                    "annotation_file": ann_name,
                    "selection_note": "ChEMU NER procedure record.",
                },
            )


def iter_choriso_public_records() -> Iterator[dict[str, Any]]:
    with CHORISO_PUBLIC.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, dialect="excel-tab")
        for index, row in enumerate(reader, start=1):
            reactants, reagents, products = parse_reaction_triplet(row["canonic_rxn"])
            catalyst = dedupe_preserve(row["catalyst"].split("|"))
            solvent = dedupe_preserve(row["solvent"].split("|"))
            reagent_roles = dedupe_preserve(row["reagent"].split("|"))
            coverage = ["reagent_roles", "reaction_center_possible", "transformation_type_possible"]
            if has_stereo(row["rxnmapper_aam"]):
                coverage.append("stereochemical_outcome_possible")
            difficulty = "hard" if catalyst and solvent else "medium"
            yield build_record(
                record_id=f"choriso_public_{index:06d}",
                level="A",
                source_id="choriso",
                source_split="public",
                task_family="Reaction Understanding",
                task_subtype="condition_role_identification",
                difficulty=difficulty,
                coverage_tags=coverage,
                input_text="Identify the roles of reagents, solvent, and catalyst for the reaction.",
                input_payload={
                    "reaction_smiles": row["canonic_rxn"],
                    "mapped_reaction_smiles": row["rxnmapper_aam"],
                    "reactants": reactants,
                    "reagents": reagents,
                    "products": products,
                },
                gold_payload={
                    "reagent_roles": {
                        "reagents": reagent_roles,
                        "solvents": solvent,
                        "catalysts": catalyst,
                    },
                    "yield_percent": safe_float(row["yield"]),
                },
                metadata={
                    "source_path": str(CHORISO_PUBLIC.relative_to(REPO_ROOT)),
                    "selection_note": "CHORISO public reaction record.",
                },
            )


def map_weave2_label(label: str) -> str | None:
    if "REACTANT" in label:
        return "reactants"
    if "REAGENT" in label:
        return "reagents"
    if "SOLVENT" in label:
        return "solvents"
    if "PRODUCT" in label:
        return "products"
    if label == "YIELD":
        return "yield"
    return None


def iter_weave2_records() -> Iterator[dict[str, Any]]:
    for ann_path in sorted(WEAVE2_DIR.glob("*.ann")):
        txt_path = ann_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        ann_text = ann_path.read_text(encoding="utf-8", errors="replace")
        grouped = {
            "reactants": [],
            "reagents": [],
            "solvents": [],
            "products": [],
            "yield": [],
        }
        for label, value in parse_brat_annotations(ann_text):
            key = map_weave2_label(label)
            if not key:
                continue
            grouped[key].append(value)
        if not any(grouped.values()):
            continue
        yield build_record(
            record_id=f"weave2_{ann_path.stem}",
            level="A",
            source_id="weave2",
            source_split="public",
            task_family="Reaction Understanding",
            task_subtype="reagent_role_identification",
            difficulty="medium",
            coverage_tags=["reagent_roles", "procedural_text"],
            input_text="Identify reactants, reagents, solvents, and products from the patent procedure text.",
            input_payload={
                "procedure_text": text,
            },
            gold_payload={
                "reagent_roles": {
                    "reactants": dedupe_preserve(grouped["reactants"]),
                    "reagents": dedupe_preserve(grouped["reagents"]),
                    "solvents": dedupe_preserve(grouped["solvents"]),
                    "products": dedupe_preserve(grouped["products"]),
                },
                "yield_values": dedupe_preserve(grouped["yield"]),
            },
            metadata={
                "source_path": str(ann_path.relative_to(REPO_ROOT)),
                "selection_note": "WEAVE2 procedure annotation record.",
            },
        )


def iter_level_b_records() -> Iterator[dict[str, Any]]:
    yield from iter_orderly_retro_records()
    yield from iter_paroutes_selected_reaction_records()


def infer_orderly_difficulty(precursor_count: int, agents: list[str], solvents: list[str]) -> str:
    if precursor_count <= 1:
        return "easy"
    if precursor_count == 2 and len(agents) <= 2 and len(solvents) <= 1:
        return "medium"
    return "hard"


def iter_orderly_retro_records() -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(ORDERLY_RETRO_TEST)
    column_names = parquet_file.schema.names
    reactant_columns = sorted(name for name in column_names if name.startswith("reactant_"))
    product_columns = sorted(name for name in column_names if name.startswith("product_"))
    agent_columns = sorted(name for name in column_names if name.startswith("agent_"))
    solvent_columns = sorted(name for name in column_names if name.startswith("solvent_"))

    for batch in parquet_file.iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            reactants = [row[name] for name in reactant_columns if row.get(name)]
            products = [row[name] for name in product_columns if row.get(name)]
            agents = [row[name] for name in agent_columns if row.get(name)]
            solvents = [row[name] for name in solvent_columns if row.get(name)]
            target = products[0] if products else None
            if not target or not reactants:
                continue
            difficulty = infer_orderly_difficulty(len(reactants), agents, solvents)
            yield build_record(
                record_id=f"orderly_retro_test_{int(row['index']):07d}",
                level="B",
                source_id="orderly",
                source_split="test",
                task_family="Single-Step Retrosynthesis",
                task_subtype="immediate_precursor_prediction",
                difficulty=difficulty,
                coverage_tags=["precursor_set", "constraints", "procedure_context"],
                input_text="Propose plausible immediate precursors for the target molecule.",
                input_payload={
                    "target": target,
                },
                gold_payload={
                    "precursor_set": reactants,
                    "proposed_disconnection": None,
                    "justification": None,
                    "key_transformation": None,
                    "constraints": {
                        "agents": agents,
                        "solvents": solvents,
                        "temperature": row.get("temperature"),
                        "rxn_time": row.get("rxn_time"),
                        "yield": row.get("yield_000"),
                    },
                },
                metadata={
                    "original_index": row.get("original_index"),
                    "procedure_details": row.get("procedure_details"),
                    "rxn_str": row.get("rxn_str"),
                    "extracted_from_file": row.get("extracted_from_file"),
                    "grant_date": row.get("grant_date"),
                    "selection_note": "ORDerly official retrosynthesis test split row.",
                    "source_path": str(ORDERLY_RETRO_TEST.relative_to(REPO_ROOT)),
                },
            )


def infer_paroutes_b_difficulty(row: dict[str, str]) -> str:
    n_reactants = safe_int(row.get("NReactants")) or 0
    n_ring_change = safe_int(row.get("NRingChange")) or 0
    ring_bond_made = safe_int(row.get("RingBondMade")) or 0
    unmapped_prod_atoms = safe_int(row.get("UnmappedProdAtoms")) or 0
    bad = bool(row.get("BadMolecules") or row.get("BadMolecules2"))
    unsanitizable = safe_bool(row.get("HasUnsanitizableReactants")) or False
    radical = safe_bool(row.get("HasUnmappedRadicalAtom")) or False
    ring_breaker = safe_bool(row.get("RingBreaker")) or False

    if (
        n_reactants <= 2
        and n_ring_change <= 1
        and ring_bond_made <= 0
        and not unsanitizable
        and unmapped_prod_atoms == 0
        and not bad
        and not radical
    ):
        return "easy"
    if n_reactants > 4 or unmapped_prod_atoms > 0 or unsanitizable or radical or bad:
        return "hard"
    if ring_breaker or ring_bond_made > 0:
        return "hard"
    return "medium"


def iter_paroutes_selected_reaction_records() -> Iterator[dict[str, Any]]:
    with PAROUTES_SELECTED_REACTIONS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, dialect="excel-tab")
        for index, row in enumerate(reader, start=1):
            raw_reaction = row.get("rsmi_processed") or row.get("rsmi") or ""
            reactants, reagents, products = parse_reaction_triplet(raw_reaction)
            precursor_set = molecule_list(reactants)
            reagent_list = molecule_list(reagents)
            if not precursor_set or not products:
                continue
            difficulty = infer_paroutes_b_difficulty(row)
            yield build_record(
                record_id=f"paroutes_selected_{index:07d}",
                level="B",
                source_id="paroutes",
                source_split="selected_reactions_all",
                task_family="Single-Step Retrosynthesis",
                task_subtype="immediate_precursor_with_disconnection",
                difficulty=difficulty,
                coverage_tags=["precursor_set", "disconnection_hint", "structural_constraints"],
                input_text="Given the product, propose immediate precursors and a plausible disconnection.",
                input_payload={
                    "target": products,
                },
                gold_payload={
                    "precursor_set": precursor_set,
                    "proposed_disconnection": {
                        "inferred_from_classification": row.get("classification"),
                        "ring_breaker": safe_bool(row.get("RingBreaker")),
                        "n_ring_change": safe_int(row.get("NRingChange")),
                    },
                    "justification": None,
                    "key_transformation": {
                        "classification_raw": row.get("classification"),
                    },
                    "constraints": {
                        "reagents": reagent_list,
                        "n_reactants": safe_int(row.get("NReactants")),
                        "n_products": safe_int(row.get("NProducts")),
                        "mapped_reactants": safe_int(row.get("NMappedReactants")),
                        "mapped_products": safe_int(row.get("NMappedProducts")),
                        "n_ring_change": safe_int(row.get("NRingChange")),
                        "ring_bond_made": safe_int(row.get("RingBondMade")),
                        "ring_made_size": safe_int(row.get("RingMadeSize")),
                        "ring_breaker": safe_bool(row.get("RingBreaker")),
                        "unmapped_prod_atoms": safe_int(row.get("UnmappedProdAtoms")),
                        "widow_atoms": safe_int(row.get("WidowAtoms")),
                        "has_unmapped_radical_atom": safe_bool(row.get("HasUnmappedRadicalAtom")),
                        "has_unsanitizable_reactants": safe_bool(
                            row.get("HasUnsanitizableReactants")
                        ),
                        "bad_molecules": row.get("BadMolecules") or None,
                        "bad_molecules2": row.get("BadMolecules2") or None,
                    },
                },
                metadata={
                    "source_row_id": row.get("id"),
                    "source": row.get("source"),
                    "date": row.get("date"),
                    "rsmi": row.get("rsmi"),
                    "rsmi_processed": row.get("rsmi_processed"),
                    "pseudo_hash": row.get("PseudoHash"),
                    "product_size": safe_int(row.get("ProductSize")),
                    "source_path": str(PAROUTES_SELECTED_REACTIONS.relative_to(REPO_ROOT)),
                    "selection_note": "PaRoutes selected one-step reaction row.",
                },
            )


def iter_level_c_records() -> Iterator[dict[str, Any]]:
    yield from iter_paroutes_route_records(PAROUTES_N1_ROUTES, route_set="n1")
    yield from iter_paroutes_route_records(PAROUTES_N5_ROUTES, route_set="n5")
    yield from iter_internal_pilot_level_c_records()


def infer_paroutes_c_difficulty(route: dict[str, Any]) -> str:
    depth = route_reaction_depth(route)
    branching = count_branching_reaction_nodes(route)
    if depth <= 3 and branching == 0:
        return "easy"
    if depth <= 6 and branching <= 1:
        return "medium"
    return "hard"


def iter_paroutes_route_records(path: Path, route_set: str) -> Iterator[dict[str, Any]]:
    routes = json.loads(path.read_text(encoding="utf-8"))
    for index, route in enumerate(routes, start=1):
        yield build_record(
            record_id=f"paroutes_{route_set}_{index:05d}",
            level="C",
            source_id="paroutes",
            source_split=route_set,
            task_family="Multi-Step Synthesis Planning",
            task_subtype="reference_route_planning",
            difficulty=infer_paroutes_c_difficulty(route),
            coverage_tags=["route_depth", "branch_handling", "convergence", "reference_route"],
            input_text="Propose a multi-step synthesis route for the target molecule.",
            input_payload={
                "target": route["smiles"],
            },
            gold_payload={
                "reference_route": route,
                "route_depth": route_reaction_depth(route),
                "reaction_steps": count_reaction_nodes(route),
                "branching_reaction_nodes": count_branching_reaction_nodes(route),
                "terminal_molecules": count_terminal_molecules(route),
                "terminal_in_stock": count_terminal_molecules(route, require_in_stock=True),
            },
            metadata={
                "route_set": route_set,
                "original_index": index,
                "selection_note": "PaRoutes reference route benchmark target.",
                "source_path": str(path.relative_to(REPO_ROOT)),
            },
        )


def iter_internal_pilot_level_c_records() -> Iterator[dict[str, Any]]:
    with PILOT_BENCHMARK.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["exam_id"], str(row["question_id"]))
            if key not in LEVEL_C_RECORD_IDS:
                continue
            yield build_record(
                record_id=f"{row['exam_id']}/{row['question_id']}",
                level="C",
                source_id="benchmark_v1_0",
                source_split="pilot_internal",
                task_family="Multi-Step Synthesis Planning",
                task_subtype="route_design",
                difficulty="hard",
                coverage_tags=[
                    "route_design",
                    "key_intermediates",
                    "reagent_constraints",
                ],
                input_text=row["question_text"],
                input_payload={
                    "question_text": row["question_text"],
                },
                gold_payload={
                    "reference_answer": row["canonical_answer"],
                },
                metadata={
                    "exam_id": row["exam_id"],
                    "page_id": row["page_id"],
                    "question_id": row["question_id"],
                    "question_type": row["question_type"],
                    "answer_type": row["answer_type"],
                    "max_score": row["max_score"],
                    "selection_rule": "strict_route_planning_subset",
                    "selection_note": (
                        "Selected from benchmark_v1_0 because the task explicitly asks "
                        "for a synthesis plan or route proposal, not forward product prediction."
                    ),
                    "source_path": str(PILOT_BENCHMARK.relative_to(REPO_ROOT)),
                },
            )


def route_reaction_depth(node: dict[str, Any]) -> int:
    children = node.get("children") or []
    if not children:
        return 0
    increment = 1 if node.get("type") == "reaction" else 0
    return increment + max(route_reaction_depth(child) for child in children)


def count_reaction_nodes(node: dict[str, Any]) -> int:
    count = 1 if node.get("type") == "reaction" else 0
    for child in node.get("children") or []:
        count += count_reaction_nodes(child)
    return count


def count_branching_reaction_nodes(node: dict[str, Any]) -> int:
    children = node.get("children") or []
    count = 1 if node.get("type") == "reaction" and len(children) > 1 else 0
    for child in children:
        count += count_branching_reaction_nodes(child)
    return count


def count_terminal_molecules(node: dict[str, Any], require_in_stock: bool = False) -> int:
    children = node.get("children") or []
    if not children:
        if node.get("type") != "mol":
            return 0
        if require_in_stock:
            return 1 if node.get("in_stock") else 0
        return 1
    return sum(
        count_terminal_molecules(child, require_in_stock=require_in_stock) for child in children
    )


def write_manifest(stats: BuildStats) -> None:
    lines = [
        "schema_version: v2",
        "files:",
        "  level_a:",
        f"    path: {LEVEL_A_OUTPUT.relative_to(REPO_ROOT)}",
        f"    records: {stats.level_a_records}",
        "    sources:",
        "      - pmechdb_manual_test_challenging",
        "      - pmechdb_combinatorial_test",
        "      - uspto_50k_test",
        "      - chemu_ner_train_dev",
        "      - choriso_public",
        "      - weave2_public",
        "    coverage:",
        "      - transformation_type",
        "      - reaction_center",
        "      - reagent_roles",
        "      - mechanistic_class",
        "      - stereochemical_outcome_partial",
        "  level_b:",
        f"    path: {LEVEL_B_OUTPUT.relative_to(REPO_ROOT)}",
        f"    records: {stats.level_b_records}",
        "    sources:",
        "      - orderly_retro_test",
        "      - paroutes_selected_reactions_all",
        "    coverage:",
        "      - precursor_set",
        "      - disconnection_hint",
        "      - structural_constraints",
        "  level_c:",
        f"    path: {LEVEL_C_OUTPUT.relative_to(REPO_ROOT)}",
        f"    records: {stats.level_c_records}",
        "    sources:",
        "      - paroutes_n1_reference_routes",
        "      - paroutes_n5_reference_routes",
        "      - benchmark_v1_0 strict route-planning subset",
        "    coverage:",
        "      - route_depth",
        "      - branch_handling",
        "      - convergence",
        "      - key_intermediates_partial",
        "notes:",
        "  - difficulty and coverage_tags are now first-class fields on every record.",
        "  - level_a and level_b are large agent pools, not paper-ready eval subsets.",
        "  - mechanistic_class/selectivity remain incomplete and need curated eval-layer annotation.",
    ]
    LEVELS_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build() -> BuildStats:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    level_a_records = write_jsonl(LEVEL_A_OUTPUT, iter_level_a_records())
    level_b_records = write_jsonl(LEVEL_B_OUTPUT, iter_level_b_records())
    level_c_records = write_jsonl(LEVEL_C_OUTPUT, iter_level_c_records())
    stats = BuildStats(
        level_a_records=level_a_records,
        level_b_records=level_b_records,
        level_c_records=level_c_records,
    )
    write_manifest(stats)
    return stats


def main() -> None:
    stats = build()
    print(
        json.dumps(
            {
                "level_a_records": stats.level_a_records,
                "level_b_records": stats.level_b_records,
                "level_c_records": stats.level_c_records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
