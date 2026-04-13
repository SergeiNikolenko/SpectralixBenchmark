# G-Eval on `benchmark_v3`

This document explains how rubric-guided judging is implemented for
`benchmark_v3`, where it lives in the codebase, and how it interacts with the
evaluation pipeline.

## Purpose

Open-ended chemistry answers cannot be scored reliably with naive string
matching. `g_eval` addresses this by constraining the judge model with:

- explicit evaluation criteria
- ordered reasoning steps
- a fixed rubric scale (`0, 2, 4, 6, 8, 10`)
- a structured output schema

Deterministic answer types continue to use code-only scoring. `g_eval` is used
only where semantic grading is actually required.

## Pipeline Role

At a high level the flow is:

1. `student_validation.py` writes `student_output.jsonl`
2. `llm_judge.py` joins student rows with benchmark rows
3. deterministic answer types are scored directly in code
4. open-ended answer types go through `g_eval`
5. rubric scores (`0..10`) are normalized to `0.0..1.0`
6. normalized scores are multiplied by `max_score`

This keeps exact-match cases cheap and stable while still supporting semantic
judging for synthesis-style answers.

## Code Locations

The implementation is split across a small set of files:

- `scripts/evaluation/run_full_matrix.py`
  - selects `g_eval` as the default judge mode
- `scripts/evaluation/llm_judge.py`
  - routes rows between deterministic scoring and LLM judging
- `scripts/evaluation/judge_rubrics.py`
  - defines rubric templates and answer-type-specific criteria
- `scripts/pydantic_guard/judge_geval.py`
  - runs the rubric-guided model call and validates the result
- `scripts/pydantic_guard/schemas.py`
  - contains the structured response schema (`GEvalJudgeResult`)

## Routing Semantics

Deterministic scoring is used for exact or mostly exact answer types such as:

- `single_choice`
- `multiple_choice`
- `ordering`
- `numeric`
- `msms_structure_prediction`

`g_eval` is used for open-ended answer types such as:

- `text`
- `reaction_description`
- `property_determination`
- `full_synthesis`

This split is implemented in `scripts/evaluation/llm_judge.py`.

## Structured Judge Output

The judge model is required to return a structured object with fields such as:

- `criteria_steps`
- `step_findings`
- `rubric_score_0_to_10`
- `llm_comment`

Structured output matters for two reasons:

1. invalid responses can be retried or repaired deterministically
2. score decisions remain auditable in traces and verbose artifacts

## Score Computation

`g_eval` itself returns a rubric score on `0..10`. The evaluation pipeline then
computes:

- `llm_score = rubric_score_0_to_10 / 10.0`
- `final_score = max_score * llm_score`

Example:

- `max_score = 4`
- rubric score = `6`
- normalized score = `0.6`
- final score = `2.4`

## Fallback Behavior

If `g_eval` fails, the pipeline can fall back to the standard structured judge
when `--judge-g-eval-fallback-structured=true`.

That fallback is enabled by default in the current `benchmark_v3` pipeline so
that judge runs do not depend on every rubric-guided call succeeding on the
first attempt.

## Traceability

When trace logging is enabled, judge traces expose:

- judge mode
- score source (`g_eval`, `structured_fallback`, `deterministic`, etc.)
- normalized score
- final score
- rubric findings and comments

This makes judge decisions substantially easier to debug than a free-form
"assign a score" prompt.

## Current Implementation Note: Code-Driven vs Dataset-Driven Contracts

`benchmark_v3` rows include explicit contract metadata such as:

- `benchmark_suite`
- `benchmark_subtrack`
- `planning_horizon`
- `task_mode`
- `eval_contract_id`
- `expected_output_schema`
- `judge_rubric_id`

However, rubric selection still remains code-driven today:

- benchmark rows define the task, gold answer, and metadata
- `scripts/evaluation/judge_rubrics.py` still defines the rubric logic

In other words, the benchmark data exposes the intended contract, but the
current judge implementation is still primarily controlled by code rather than
fully by dataset-provided rubric definitions.

## Strengths

- deterministic scoring is preserved where strict rules are possible
- rubric-guided judging is used where semantic scoring is necessary
- structured outputs make retry and validation practical
- trace artifacts preserve a usable audit trail

## Limitations

- rubric granularity is still keyed mostly by `answer_type`
- judge quality still depends on the underlying model
- some task families may eventually need more specialized rubrics than the
  current type-level split
