# Gemini 3 Flash Preview Minimal No-SGR Result

Generated from primary judge output. Repair-adjusted score is not reported because repair judge is incomplete.

| Run | Student rows | Judge rows | Quality score | End-to-end score | Reliability | Student errors | Empty answers | Repair status | Student tokens | Judge tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `benchmark_v3_gemini3flash_minimal_no-sgr` | 990 | 990 | 63.74% | 50.66% | 80.20% | 196 | 196 | 189 repaired, 0 pending, 7 exhausted | 17,843,322 | 3,219,320 |

## Paths

| Artifact | Path |
|---|---|
| Student output | `runs/benchmark_v3_gemini3flash_minimal_no-sgr/student_output.jsonl` |
| Student verbose output | `runs/benchmark_v3_gemini3flash_minimal_no-sgr/student_output_verbose.jsonl` |
| Primary judge output | `runs/benchmark_v3_gemini3flash_minimal_no-sgr/judge_gpt54mini/llm_judge_output.jsonl` |
| Metrics | `runs/benchmark_v3_gemini3flash_minimal_no-sgr/metrics.json` |
| Summary CSV | `runs/benchmark_v3_gemini3flash_minimal_no-sgr/summary.csv` |
| Repair merged output | `runs/benchmark_v3_gemini3flash_minimal_no-sgr/repair_failed/student_output_merged.jsonl` |
| Repair status | `runs/benchmark_v3_gemini3flash_minimal_no-sgr/repair_failed/repair_status.json` |

## Notes

- Primary run and primary judge are complete at 990 rows.
- Primary score includes original failed/empty rows; most were repairable, but repair judging is incomplete.
- Exhausted repair keys: 7.
