# Quality Status

This page tracks benchmark and runtime quality from an agent-execution perspective.

## Current Snapshot

- Runtime architecture: stable
- Security policy: documented
- Benchmark ladder model: implemented
- Paper eval subsets: implemented
- Smoke baseline: available
- `benchmark_v3` materialization path: documented as primary
- Resume mode (`--resume-existing`): available for student/judge/matrix

## Domain Grades

## A. Runtime and Eval Pipeline - Grade A-

Strengths:

- clear student -> judge -> matrix flow
- structured guard layer integrated
- explicit sandbox/security guidance

Gaps:

- local vs OpenShell behavior differences are not fully benchmarked across all paths
- tools vs tools+internet observability still needs stronger explicit per-row instrumentation

## B. Benchmark Ladder Documentation - Grade B+

Strengths:

- Level A/B/C semantics and manifests exist
- paper-eval subset strategy is documented

Gaps:

- legacy `benchmark_v1_0.jsonl` examples still exist for compatibility and may
  confuse users if mixed with ladder-first reporting
- keep `benchmark_v3_eval.jsonl` as default run/reporting entrypoint

## C. Dataset Curation Readiness - Grade B

Strengths:

- large pools and compact eval subsets both exist
- deterministic subset generation logic is in place

Gaps:

- mechanistic/selectivity depth in Level A remains partial
- Level C still has higher runtime cost per sample in smoke conditions

## D. Agent Harness Docs Graph - Grade B

Strengths:

- clear TOC in `AGENTS.md`
- dedicated nodes for conventions, ladder, quality, and tools

Gaps:

- no recurring doc freshness automation yet

## Prioritized Next Actions

1. Keep benchmark runbooks `benchmark_v3`-first and mark `benchmark_v1_0` as legacy.
2. Add explicit quality checks for score-source mix (`g_eval` vs `structured_fallback`).
3. Add a periodic doc freshness task (weekly) for drift detection.
