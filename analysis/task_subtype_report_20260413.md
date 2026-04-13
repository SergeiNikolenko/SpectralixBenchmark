# Task Subtype Results

Paper-facing summary of per-run task subtype metrics from local run artifacts.

## Runs

| run_id | model | overall_e2e | macro_depth_e2e | reliability | cost_usd |
|---|---|---:|---:|---:|---:|
| benchmark_v3_eval_nosgr_20260411_gpt54_tools_spectrum | gpt-5.4 | 0.5661 | 0.5755 | 0.9990 | 22.0203 |
| benchmark_v3_eval_sgr_20260412_gpt54_tools_spectrum_fixed_rescued | gpt-5.4 | 0.5875 | 0.6063 | 1.0000 | 50.0536 |
| spectrum_full_resumed_from_lobach_tunnel | gpt-5.4-mini | 0.5530 | 0.5700 | 0.9909 | 13.5168 |

## Task Subtype Breakdown

### benchmark_v3_eval_nosgr_20260411_gpt54_tools_spectrum (gpt-5.4)

| task_subtype | count | quality_score | end_to_end_score | ok_rate |
|---|---:|---:|---:|---:|
| condition_role_identification | 92 | 0.6761 | 0.6761 | 1.0000 |
| immediate_precursor_prediction | 210 | 0.4533 | 0.4533 | 1.0000 |
| immediate_precursor_with_disconnection | 210 | 0.2562 | 0.2562 | 1.0000 |
| mechanistic_classification | 76 | 0.4776 | 0.4776 | 1.0000 |
| reaction_center_identification | 76 | 0.9039 | 0.9039 | 1.0000 |
| reagent_role_identification | 112 | 0.7607 | 0.7607 | 1.0000 |
| reference_route_planning | 137 | 0.6081 | 0.6036 | 0.9927 |
| route_design | 13 | 0.8769 | 0.8769 | 1.0000 |
| transformation_classification | 64 | 0.8719 | 0.8719 | 1.0000 |

### benchmark_v3_eval_sgr_20260412_gpt54_tools_spectrum_fixed_rescued (gpt-5.4)

| task_subtype | count | quality_score | end_to_end_score | ok_rate |
|---|---:|---:|---:|---:|
| condition_role_identification | 92 | 0.6902 | 0.6902 | 1.0000 |
| immediate_precursor_prediction | 210 | 0.4676 | 0.4676 | 1.0000 |
| immediate_precursor_with_disconnection | 210 | 0.2752 | 0.2752 | 1.0000 |
| mechanistic_classification | 76 | 0.6461 | 0.6461 | 1.0000 |
| reaction_center_identification | 76 | 0.9539 | 0.9539 | 1.0000 |
| reagent_role_identification | 112 | 0.7509 | 0.7509 | 1.0000 |
| reference_route_planning | 137 | 0.6168 | 0.6168 | 1.0000 |
| route_design | 13 | 0.8538 | 0.8538 | 1.0000 |
| transformation_classification | 64 | 0.8344 | 0.8344 | 1.0000 |

### spectrum_full_resumed_from_lobach_tunnel (gpt-5.4-mini)

| task_subtype | count | quality_score | end_to_end_score | ok_rate |
|---|---:|---:|---:|---:|
| condition_role_identification | 92 | 0.6370 | 0.6370 | 1.0000 |
| immediate_precursor_prediction | 210 | 0.4519 | 0.4519 | 1.0000 |
| immediate_precursor_with_disconnection | 210 | 0.2510 | 0.2510 | 1.0000 |
| mechanistic_classification | 76 | 0.5711 | 0.5711 | 1.0000 |
| reaction_center_identification | 76 | 0.8974 | 0.8974 | 1.0000 |
| reagent_role_identification | 112 | 0.7473 | 0.7473 | 1.0000 |
| reference_route_planning | 137 | 0.5695 | 0.5321 | 0.9343 |
| route_design | 13 | 0.8538 | 0.8538 | 1.0000 |
| transformation_classification | 64 | 0.9516 | 0.9516 | 1.0000 |

