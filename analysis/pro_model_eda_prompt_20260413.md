You are analyzing benchmark evaluation artifacts for a chemistry-agent paper.

Context:
- I am giving you an archive with `runs/` artifacts from multiple completed benchmark runs.
- The archive contains `summary.json`, `summary.csv`, `metrics.json`, `llm_judge_output.jsonl`, `student_output.jsonl`, `student_output_verbose.jsonl`, and some archived `.zip` run bundles.
- Focus on completed runs with valid judge outputs first.
- Treat clearly broken runs separately as failure cases, not as normal quality baselines.

Your task:
1. Inspect the archive contents and build a clean run inventory.
2. Identify completed valid runs versus broken or incomplete runs.
3. Produce a concise EDA report aimed at a scientific paper draft.
4. Generate publication-quality figures and save them with clear filenames.
5. Prefer reproducible Python analysis with pandas / seaborn / matplotlib.

Important analysis rules:
- Do not mix broken runs into the main quality ranking.
- Separate "quality comparison" from "failure analysis".
- If a run has `quality_normalized_score = null` or `reliability_ok_rate = 0`, treat it as an infra-failure case unless evidence shows otherwise.
- Use only metrics grounded in artifacts; do not invent missing values.
- If a metric is missing, mark it explicitly as missing.

Primary questions to answer:
1. Which completed run is the strongest overall baseline?
2. What is the quality / reliability / cost tradeoff across runs?
3. Does SGR help, hurt, or mainly change failure modes?
4. Which error modes dominate bad runs?
5. Which runs are best candidates to report in the paper main table?

Deliverables:
- `eda_report.md`
- `run_inventory.csv`
- `main_results_table.csv`
- `failure_cases_table.csv`
- `fig_quality_vs_cost.png`
- `fig_quality_vs_reliability.png`
- `fig_end_to_end_ranking.png`
- `fig_infra_error_breakdown.png`
- `fig_answer_type_breakdown.png` if answer-type breakdown data is available
- `fig_token_cost_breakdown.png` if token totals are available
- `analysis_notebook.ipynb` or `analysis.py`

Required tables:
1. Main results table with columns:
   - run_id
   - model_name
   - judge_model
   - quality_normalized_score
   - quality_end_to_end_score
   - reliability_ok_rate
   - estimated_cost_usd
   - key infra errors
   - run status (`valid`, `broken`, `incomplete`)
2. Failure table with columns:
   - run_id
   - dominant failure mode
   - evidence
   - whether failure is infra or modeling-related

Required figures:
1. Quality vs cost scatter plot
   - x = `estimated_cost_usd`
   - y = `quality_end_to_end_score`
   - point color = run family (`nosgr`, `sgr`, `mini`, `tools`, etc. if inferable)
   - point label = short run id
   - This is one of the most important paper-ready plots.
2. Quality vs reliability scatter plot
   - x = `reliability_ok_rate`
   - y = `quality_end_to_end_score`
   - highlight Pareto-efficient runs
3. Ranked bar chart for end-to-end score
   - sort by `quality_end_to_end_score`
   - annotate exact values
4. Stacked infra-error chart
   - one bar per run
   - stack by infra error type (`agent_step_error`, `sandbox_error`, `timeout`, `parse_error`, etc.)
   - broken runs should be visually obvious
5. Answer-type breakdown figure
   - if `breakdown_by_answer_type.json` exists, compare `reaction_description`, `text`, `full_synthesis`
   - use grouped bars or heatmap
6. Token/cost decomposition
   - student vs judge cost
   - student vs judge token totals
   - useful as supplementary figure

Recommended additional plots if supported by the data:
1. Bubble chart:
   - x = cost
   - y = quality_end_to_end
   - bubble size = reliability
2. Heatmap:
   - rows = runs
   - columns = normalized metrics (`quality`, `end_to_end`, `reliability`, `cost_inverse`)
3. Failure-case panel:
   - compact figure or table specifically showing broken SGR run versus fixed/non-SGR baseline

Figure style requirements for a paper:
- Use clean white background.
- Use large readable fonts.
- Use consistent palette across all plots.
- Avoid default matplotlib aesthetics if they look informal.
- Export high resolution (`dpi >= 300`).
- Keep titles concise and paper-like.
- Prefer legends that are easy to read in print.
- Ensure axis labels are publication quality.

Interpretation guidance:
- Distinguish between:
  - best raw quality,
  - best quality-cost tradeoff,
  - best reliability,
  - failure cases.
- Explicitly note if a run is not comparable because it failed before producing valid judged outputs.
- If one run dominates on quality but is much more expensive, say that clearly.
- If SGR evidence is inconclusive because one run was infra-broken and another is incomplete, say so explicitly.

Expected report structure:
1. Dataset inventory
2. Completed valid runs
3. Broken / incomplete runs
4. Main comparative findings
5. Recommended figures for the paper
6. Threats to validity / missing data

Final output style:
- Be concrete, numerical, and concise.
- Prefer direct statements over generic commentary.
- When claiming one run is better, cite the exact metric values.
