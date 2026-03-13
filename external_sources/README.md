## External Benchmark Sources

Collected on 2026-03-12.

This directory stores external sources gathered for the new benchmark ladder:

- `level_a/`: reaction understanding sources
- `level_b/`: single-step retrosynthesis sources
- `shared/`: sources useful for both levels
- `blocked/`: sources that could not be downloaded automatically because of access, licensing, or commercial restrictions

Downloaded sources in this directory are third-party data. Check each source license before reuse or redistribution.

## Normalized Layout

Each source directory follows the same structure:

- `raw/`: original downloaded files
- `extracted/`: locally unpacked or normalized contents

Example:

```text
external_sources/level_a/pmechdb/
  raw/
  extracted/
```

## Source Groups

### Level A

- `chemu_2020/`
- `choriso/`
- `pmechdb/`
- `rmechdb/`
- `uspto_50k/`
- `uspto_llm/`
- `weave2/`

### Level B

- `lowe_uspto/`
- `orderly/`
- `paroutes/`

### Shared

- `ord/`

## Key Artifacts

- [manifest.yaml](/Users/nikolenko/.codex/worktrees/a55a/SpectralixBenchmark/external_sources/manifest.yaml)
- [inventory.csv](/Users/nikolenko/.codex/worktrees/a55a/SpectralixBenchmark/external_sources/inventory.csv)
- [inventory.md](/Users/nikolenko/.codex/worktrees/a55a/SpectralixBenchmark/external_sources/inventory.md)

## Notes

- `PMechDB` bulk data was downloaded through the official form workflow and unpacked locally.
- `RMechDB` does not expose a public bulk URL; the official workflow is request form plus email delivery.
- `Pistachio` and `Reaxys` are commercial products and were not downloaded.
