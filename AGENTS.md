# Agent Context TOC

This file is a navigation entrypoint for AI agents working in this repository.
Keep it short and follow links to `docs/*` for task-specific context.

## Read First

1. [README.md](README.md) - runtime and benchmark quick start
2. [docs/architecture.md](docs/architecture.md) - execution architecture
3. [docs/conventions.md](docs/conventions.md) - coding and data handling conventions
4. [docs/benchmark_ladder.md](docs/benchmark_ladder.md) - Level A/B/C benchmark model
5. [docs/benchmark_construction.md](docs/benchmark_construction.md) - source-by-source benchmark build rationale
6. [docs/g_eval.md](docs/g_eval.md) - rubric-based judge design and implementation
7. [docs/quality.md](docs/quality.md) - quality status and current gaps
8. [docs/tools/README.md](docs/tools/README.md) - operational commands and scripts
9. [docs/security_runbook.md](docs/security_runbook.md) - sandbox and security controls

## Task Routing

- Benchmark composition, ladder semantics, source mapping:
  - [docs/benchmark_ladder.md](docs/benchmark_ladder.md)
  - [docs/benchmark_construction.md](docs/benchmark_construction.md)
- Evaluation and runtime behavior:
  - [docs/architecture.md](docs/architecture.md)
  - [docs/g_eval.md](docs/g_eval.md)
  - [README.md](README.md)
- Data quality and prioritization:
  - [docs/quality.md](docs/quality.md)
- Operational commands:
  - [docs/tools/README.md](docs/tools/README.md)
- Security policy:
  - [docs/security_runbook.md](docs/security_runbook.md)

## Scope Rules

- Prefer modifying metadata/docs before regenerating large benchmark pools.
- Treat `benchmark/level_a.jsonl`, `benchmark/level_b.jsonl`, `benchmark/level_c.jsonl` as generated artifacts.
- Prefer `*_eval.jsonl` files for fast local checks and smoke runs.
- Default repository convention: content written to files should be in English unless explicitly requested otherwise.
- Keep user-facing dataset assumptions synchronized with:
  - `benchmark/levels_manifest.yaml`
  - `benchmark/paper_eval_manifest.yaml`

## Change Hygiene

- When benchmark shape changes, update:
  - [docs/benchmark_ladder.md](docs/benchmark_ladder.md)
  - [docs/benchmark_construction.md](docs/benchmark_construction.md)
  - [docs/quality.md](docs/quality.md)
  - [README.md](README.md) if quick-start behavior changes
- After editing docs, verify links resolve and commands are still valid.
