## External Benchmark Sources

This directory is the provenance layer for rebuilding the benchmark from raw
sources. The public repository tracks only metadata and instructions. Downloaded
archives and extracted datasets are intentionally ignored.

## Directory Structure

- `level_a/`: sources used for local reaction reasoning tasks
- `level_b/`: sources used for single-step retrosynthesis tasks
- `shared/`: sources reused across multiple benchmark levels
- `blocked/`: sources that were intentionally not downloaded because they are
  commercial or require non-automatable access

## Source Inventory

The complete machine-readable manifest is in `manifest.yaml`. The current source
catalog is:

### Level A

- `chemu_2020`: [Mendeley Data](https://data.mendeley.com/datasets/wy6745bjfj/2)
- `choriso`: [Figshare](https://figshare.com/articles/dataset/CHORISO_-_chemical_reaction_SMILES_from_academic_journals/22598230)
- `pmechdb`: [official download page](https://deeprxn.ics.uci.edu/pmechdb/download)
- `uspto_50k`: [Zenodo](https://zenodo.org/records/8114657)
- `uspto_llm`: [Zenodo](https://zenodo.org/records/14396156)
- `weave2`: [Zenodo](https://zenodo.org/records/8386296)

### Level B

- `lowe_uspto`: [Figshare](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873)
- `orderly`: [Figshare collection](https://figshare.com/articles/collection/ORDerly_supplementary_datasets/23502372)
- `paroutes`: [GitHub](https://github.com/MolecularAI/PaRoutes)

### Shared

- `ord`: [GitHub](https://github.com/Open-Reaction-Database/ord-data)

## Blocked or Restricted Sources

These sources are documented but not redistributed:

- `RMechDB`: request-form plus email-delivery workflow
- `Pistachio`: commercial
- `Reaxys`: commercial

See `blocked/*.txt` for the exact notes captured during collection.

## Expected Local Layout

For each source, populate this structure locally:

```text
external_sources/<group>/<source>/
  raw/
  extracted/
```

The exact files expected by the build scripts are listed in `manifest.yaml`.

## Rebuild Workflow

After downloading the required public sources into the expected layout:

1. build the normalized benchmark pools
2. derive the paper-eval subsets
3. materialize the runtime-facing `benchmark_v3_eval.jsonl`

Commands:

```bash
uv run python -m spectralix_benchmark.build.level_benchmark_files
uv run python -m spectralix_benchmark.build.paper_eval_subsets
uv run python -m spectralix_benchmark.evaluation.materialize_benchmark_v3_eval \
  --output benchmark/benchmark_v3_eval.jsonl
```

## Notes

- `benchmark/benchmark_v1_0.jsonl` remains part of the public repository because
  selected `Level C` route-design tasks still depend on it.
- The public repository intentionally excludes raw downloaded datasets to keep
  the benchmark package lightweight and redistributable.
