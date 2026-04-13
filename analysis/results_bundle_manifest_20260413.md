Spectralix local results bundle prepared on 2026-04-13.

Included:
- local `runs/` contents currently present in this workspace
- completed run directories that are already unpacked locally
- archived `.zip` run bundles already stored in `runs/`
- EDA prompt for a stronger external model

Important note:
- The new fixed SGR rerun on `Spectrum` is still in progress and is not part of this local archive snapshot.
- Remote in-progress run id:
  - `benchmark_v3_eval_sgr_20260412_gpt54_tools_spectrum_fixed`

Known completed local baselines at archive time:
- `benchmark_v3_eval_nosgr_20260411_gpt54_tools_spectrum`
- `benchmark_v3_eval_sgr_20260408_gpt54mini_tools_timeoutfix2`
- `spectrum_full_resumed_from_lobach_tunnel`
- `benchmark_v3_full_openshell_20260403_gpt54mini_minimal`
- `benchmark_v3_full_openshell_20260406_gpt54mini_tools_plus`

Known broken local run at archive time:
- `benchmark_v3_eval_sgr_20260412_gpt54_tools_spectrum`
  - this is the old failed SGR run with `agent_step_error=990`
