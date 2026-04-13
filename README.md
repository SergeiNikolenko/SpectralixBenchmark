# SpectralixBenchmark

`SpectralixBenchmark` is a chemistry benchmark and evaluation package for
agentic language models.

It is designed to measure performance across increasing planning depth rather
than as a single mixed chemistry test set.

## What The Benchmark Measures

`benchmark_v3` is organized as a three-level ladder:

- `Level A`: local reaction understanding
- `Level B`: single-step retrosynthesis
- `Level C`: route-level synthesis planning

The repository also includes an auxiliary grounding suite used for chemistry
role and condition extraction, but the primary paper-facing benchmark is the
`A/B/C` ladder.

The benchmark covers:

- organic chemistry reasoning
- retrosynthesis and route planning
- tandem mass spectrometry (`MS2`) structure tasks
- deterministic and rubric-based LLM evaluation

## Repository Layout

- `benchmark/`
  - public benchmark artifacts, eval subsets, and manifests
- `spectralix_benchmark/agents/`
  - OpenShell runtime, tool registry, and SGR reasoning layer
- `spectralix_benchmark/evaluation/`
  - student stage, judge stage, and matrix runner
- `spectralix_benchmark/guards/`
  - typed validation and retry helpers for student and judge calls
- `spectralix_benchmark/build/`
  - reproducible benchmark construction utilities
- `external_sources/`
  - source manifests, download links, and rebuild instructions
- `docs/`
  - architecture, ladder semantics, judging, taxonomy, and runtime policy

The public repository intentionally excludes generated runs, scratch analysis,
and legacy parsing pipelines that are not part of the final benchmark release.

## Public Benchmark Files

Primary runtime entrypoint:

- `benchmark/benchmark_v3_eval.jsonl`

Per-level public eval subsets:

- `benchmark/level_a_eval.jsonl`
- `benchmark/level_b_eval.jsonl`
- `benchmark/level_c_eval.jsonl`

Supporting manifests:

- `benchmark/levels_manifest.yaml`
- `benchmark/paper_eval_manifest.yaml`
- `benchmark/LEVELS.md`

Legacy compatibility/source layer retained for selected construction steps:

- `benchmark/benchmark_v1_0.jsonl`

## Installation

Requirements:

- Python `3.12+`
- `uv`
- Docker for OpenShell-backed runs
- an OpenAI-compatible endpoint or local proxy

Setup:

```bash
git clone https://github.com/SergeiNikolenko/SpectralixBenchmark.git
cd SpectralixBenchmark
uv sync
```

The package is importable as `spectralix_benchmark`.

## Runtime Setup

Export your model endpoint and API key:

```bash
export API_BASE_URL="http://127.0.0.1:8317/v1"
export CLIPROXY_API_KEY="<your-api-key>"
```

Quick health check:

```bash
curl -sS \
  -H "Authorization: Bearer $CLIPROXY_API_KEY" \
  "$API_BASE_URL/models"
```

If you run OpenShell-backed evaluation, make sure Docker and the OpenShell
gateway are available.

## Reproducing Benchmark Execution

### 1. Materialize the runtime-facing benchmark file

```bash
uv run python -m spectralix_benchmark.evaluation.materialize_benchmark_v3_eval \
  --output benchmark/benchmark_v3_eval.jsonl
```

### 2. Run student inference

```bash
uv run python -m spectralix_benchmark.evaluation.student_validation \
  --benchmark-path benchmark/benchmark_v3_eval.jsonl \
  --output-path runs/repro/student_output.jsonl \
  --api-base-url "$API_BASE_URL" \
  --model-name "gpt-5.4-mini" \
  --api-key "$CLIPROXY_API_KEY" \
  --agent-sandbox openshell \
  --agent-backend openshell_worker \
  --agent-tools-profile tools \
  --student-guard-enabled true \
  --student-guard-mode on_failure \
  --student-guard-retries 2
```

### 3. Run the judge

```bash
uv run python -m spectralix_benchmark.evaluation.llm_judge \
  --input-path runs/repro/student_output.jsonl \
  --gold-path benchmark/benchmark_v3_eval.jsonl \
  --judge-model "gpt-5.4-mini" \
  --judge-model-url "$API_BASE_URL" \
  --judge-api-key "$CLIPROXY_API_KEY" \
  --judge-method g_eval \
  --judge-g-eval-fallback-structured true \
  --judge-structured-retries 2 \
  --output-path runs/repro/llm_judge_output.jsonl
```

### 4. Run the full matrix pipeline

```bash
uv run python -m spectralix_benchmark.evaluation.run_full_matrix \
  --benchmark-path benchmark/benchmark_v3_eval.jsonl \
  --api-base-url "$API_BASE_URL" \
  --api-key "$CLIPROXY_API_KEY" \
  --models gpt-5.4-mini \
  --judge-model gpt-5.4-mini \
  --agent-sandbox openshell \
  --agent-backend openshell_worker \
  --agent-tools-profile tools \
  --student-guard-enabled true \
  --student-guard-mode on_failure \
  --student-guard-retries 2 \
  --judge-method g_eval \
  --judge-g-eval-fallback-structured true \
  --judge-structured-retries 2
```

Outputs are written into `runs/<run-id>/`.

## Rebuilding The Benchmark From Sources

The repository does not redistribute raw external datasets. To rebuild the
larger benchmark pools:

1. download the public sources listed in [external_sources/README.md](external_sources/README.md)
2. place them into the expected `external_sources/<group>/<source>/raw|extracted` layout
3. run:

```bash
uv run python -m spectralix_benchmark.build.level_benchmark_files
uv run python -m spectralix_benchmark.build.paper_eval_subsets
uv run python -m spectralix_benchmark.evaluation.materialize_benchmark_v3_eval \
  --output benchmark/benchmark_v3_eval.jsonl
```

Restricted/commercial sources are documented in `external_sources/blocked/` but
are not redistributed.

## Evaluation Notes

- The default integrity-oriented runtime is `OpenShell`.
- `local` sandbox mode is a development fallback, not the preferred evaluation mode.
- The student path includes a hidden SGR reasoning phase for supported runs.
- The judge combines deterministic scoring with rubric-based LLM evaluation.
- `score_method` in `llm_judge_output.jsonl` records whether a row was scored by:
  - `deterministic`
  - `g_eval`
  - `structured_fallback`
  - `llm_judge`

## Documentation Map

- [docs/architecture.md](docs/architecture.md) - runtime and package structure
- [docs/benchmark_ladder.md](docs/benchmark_ladder.md) - benchmark semantics
- [docs/benchmark_construction.md](docs/benchmark_construction.md) - source-to-benchmark construction
- [docs/benchmark_taxonomy.md](docs/benchmark_taxonomy.md) - taxonomy and reporting
- [docs/g_eval.md](docs/g_eval.md) - judge behavior
- [docs/tools/README.md](docs/tools/README.md) - operational commands
- [docs/security_runbook.md](docs/security_runbook.md) - runtime and sandbox policy
