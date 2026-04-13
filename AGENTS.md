# Agent Context TOC

This file is the entrypoint for AI agents working in this repository.
Keep it short. Load only the docs you need.

## Read First

1. [README.md](README.md) - public quick start and repository layout
2. [docs/architecture.md](docs/architecture.md) - runtime, package boundaries, and execution flow
3. [docs/conventions.md](docs/conventions.md) - repository conventions and regeneration rules
4. [docs/benchmark_ladder.md](docs/benchmark_ladder.md) - A/B/C ladder semantics
5. [docs/benchmark_construction.md](docs/benchmark_construction.md) - source-to-benchmark construction logic
6. [docs/benchmark_taxonomy.md](docs/benchmark_taxonomy.md) - taxonomy and reporting model
7. [docs/g_eval.md](docs/g_eval.md) - judge and rubric behavior
8. [docs/tools/README.md](docs/tools/README.md) - operational commands
9. [docs/security_runbook.md](docs/security_runbook.md) - runtime and sandbox controls

## Package Map

- `spectralix_benchmark/agents/`
  - OpenShell runtime, worker orchestration, tool registry, SGR schemas
- `spectralix_benchmark/evaluation/`
  - student stage, judge stage, matrix runner, taxonomy-aware metrics
- `spectralix_benchmark/guards/`
  - typed validation and retry layer for student/judge calls
- `spectralix_benchmark/build/`
  - reproducible benchmark construction utilities
- `benchmark/`
  - public benchmark artifacts and manifests
- `external_sources/`
  - provenance manifests, source links, and rebuild instructions

## Task Routing

- Benchmark semantics or source mapping:
  - [docs/benchmark_ladder.md](docs/benchmark_ladder.md)
  - [docs/benchmark_construction.md](docs/benchmark_construction.md)
- Runtime, worker, SGR, or judge behavior:
  - [docs/architecture.md](docs/architecture.md)
  - [docs/g_eval.md](docs/g_eval.md)
  - [docs/security_runbook.md](docs/security_runbook.md)
- Taxonomy, metrics, and result interpretation:
  - [docs/benchmark_taxonomy.md](docs/benchmark_taxonomy.md)
  - [docs/quality.md](docs/quality.md)
- Build/rebuild commands:
  - [docs/tools/README.md](docs/tools/README.md)
  - [external_sources/README.md](external_sources/README.md)

## Working Rules

- Treat `AGENTS.md` as a TOC, not a monolith. Put details in `docs/*`.
- Prefer changing the package under `spectralix_benchmark/` over adding one-off top-level scripts.
- Treat `benchmark/level_a.jsonl`, `benchmark/level_b.jsonl`, and `benchmark/level_c.jsonl` as generated build outputs.
- Prefer `benchmark/*_eval.jsonl` and `benchmark/benchmark_v3_eval.jsonl` for smoke runs and contract checks.
- If the same agent mistake happens twice, add a doc node or a tool entry in `docs/tools/README.md` instead of adding more prose here.

## Change Hygiene

- When package structure or runtime behavior changes, update:
  - [README.md](README.md)
  - [docs/architecture.md](docs/architecture.md)
  - [docs/tools/README.md](docs/tools/README.md)
  - [docs/security_runbook.md](docs/security_runbook.md) when sandbox/runtime policy changes
- When benchmark shape or taxonomy changes, update:
  - [docs/benchmark_ladder.md](docs/benchmark_ladder.md)
  - [docs/benchmark_construction.md](docs/benchmark_construction.md)
  - [docs/benchmark_taxonomy.md](docs/benchmark_taxonomy.md)
  - [benchmark/levels_manifest.yaml](benchmark/levels_manifest.yaml)
  - [benchmark/paper_eval_manifest.yaml](benchmark/paper_eval_manifest.yaml)
- After doc edits, verify links and commands still resolve.
