# Paper Evidence Tables

This directory is the paper-facing evidence registry. It records where each
candidate manuscript table row comes from, without copying raw `runs/` payloads
or episode-level dumps into `docs/paper`.

## How To Use

- Use `registry_index.csv` as the entry point.
- Use `*_raw_rows.csv` when a table number must be traced to a specific
  evaluation seed base and run output.
- Use `*_aggregate_rows.csv` or `*_best_rows.csv` for candidate manuscript rows.
- Keep raw run directories in `runs/` or `.local_guidance/phase*/data/`; do not
  paste large logs here.

## Required Trace Fields

Each row tries to record:

- `remote_run_dir`: original run directory on friday/server storage.
- `config`: config used for that row. Generated diagnostic configs may live
  inside the run directory.
- `checkpoint`: learned checkpoint path, blank for non-learning baselines.
- `training_seed`: learned-policy training seed when applicable.
- `eval_seed_base` / `eval_seed_bases`: evaluation seed base(s).
- `episodes_per_seed` and `total_episodes`: evaluation episode count.
- `local_metrics_file` and/or `remote_metrics_file`: CSV/JSON file containing
  the source metrics.
- `metrics_row_selector`: stable selector to locate the row inside the metrics
  file.

## Current Coverage

- Phase2 MC/bootstrap checkpoint-selection evidence.
- Phase3 same-scale zero-shot evidence.
- Phase3 nearby scale-transfer evidence.
- Phase3 boundary robustness evidence.
- 6UAV/80GU fixed-reference baseline and return-target diagnostic evidence.
- 6UAV/80GU auto queue-reference evaluation evidence.
- 6UAV/80GU bootstrap-GAE multi-seed retraining evidence.
- 6UAV load-transfer diagnostics using the new auto-reference multi-seed
  bootstrap-GAE checkpoints.
- 6UAV/40GU density/resource diagnostics for explaining why equal GU/UAV ratio
  does not imply source-scenario-equivalent service metrics.
- 6UAV/40GU density-controlled baseline-only resource sweep for identifying the
  minimum access/backhaul resource multiplier needed for serviceable metrics.
- 6UAV GU-load and 12UAV/80GU smoke diagnostics.
- Phase4 3UAV/20GU checkpoint-validation snapshot for RelCritic, GlobalCritic,
  and MAPPO-like source-scenario Section 5 planning.
- Phase4 3UAV/20GU formal held-out source-scenario main table for selected
  learned checkpoints and fixed baselines.
- Phase4 3UAV/20GU formal held-out raw rows, learned seed-level rows, and
  selected-versus-final companion rows for Section 5 checkpoint selection.
- Phase4 Section 5 training-validation curves, main performance bars,
  GlobalCritic seed-level displays, and runtime/resource summaries.
- Phase4 no-HA-PPO formal parameter sweeps for offered load and joint
  access/backhaul/service-capacity sensitivity. These rows cover STARS,
  STARS-GC, QCCS, Lyapunov, QBS, and Uniform across seven points per sweep.
- Phase4 nominal selected-evaluation rich metrics for queue-layer and
  access/backhaul/satellite flow decomposition. These rows reuse the nominal
  `nominal_load_x1p00` selected-checkpoint held-out protocol from the formal
  parameter sweep.
- Phase4 UAV-density collision smoke sweep for safety-stress triage. These rows
  vary $N_{\mathrm{U}}=2,\ldots,6$ in the 1500 m x 1500 m source area and compare
  STARS, STARS-GC, QCCS, and Lyapunov under three evaluation seed bases.

## Important Caveats

- Earlier 6UAV/80GU fixed-reference rows are historical diagnostics. They should
  not be mixed directly with auto queue-reference rows.
- 6UAV/80GU bootstrap-GAE multi-seed retraining is now registered, but should
  remain a larger-scale case-study candidate. The load-transfer diagnostics
  support the interpretation that 6UAV/80GU is a high-load stress setting, not
  a clean serviceable operating point.
- Smoke/diagnostic rows are included for traceability, but should not be promoted
  into final claims unless explicitly reclassified.
- The 6UAV/40GU density/resource rows are explanatory diagnostics. Treat them as
  evidence about scenario difficulty and resource sensitivity, not as final
  trained-policy benchmarks.
- The 6UAV/40GU resource-sweep rows are baseline-only capacity-calibration
  diagnostics. They indicate that x4 access/backhaul resources are the first
  serviceable point under the current thresholds, while x6 is the first
  3UAV/20GU-like point; they are not learned-policy performance claims.
- For final paper tables, prefer rows with multiple evaluation seed bases, and
  state when a learned policy still has only one training seed.
- Phase4 source-validation rows use checkpoint-eval seed base `910000` and
  should not be described as final held-out source-scenario results. They are
  planning evidence for Section 5 curves/tables.
- Phase4 formal held-out rows use seed bases `980000`, `981000`, and `982000`
  with 64 episodes per seed base. These are paper-facing source-scenario rows.
- Phase4 parameter-sweep rows registered on 2026-07-14 intentionally exclude
  HA-PPO because its flat learned evaluation path was much slower and caused
  instability under concurrent execution. Treat HA-PPO sweep completion as a
  separate companion task, not as a blocker for the main sensitivity figures.
- Phase4 nominal selected rich metrics currently have complete coverage for
  STARS, STARS-GC, QCCS, Lyapunov, QBS, and Uniform. The copied
  `nominal_load_x1p00` tree contains only one HA-PPO row, so do not use HA-PPO
  in queue/flow mechanism figures unless the missing selected-evaluation rows
  are rerun.
- Phase4 UAV-density collision rows are smoke/stress-test evidence. They are
  useful for identifying the UAV-density safety threshold, but should not be
  promoted into a final safety claim without deciding whether the high-density
  points are within the intended operating regime.
- For Section 5 main performance, use selected learned checkpoints plus fixed
  baselines in the main table. Treat final checkpoint rows as checkpoint
  stability/selection companion evidence or appendix material.
- MAPPO-like is a same-scenario, same held-out seed, same hybrid action
  interface learned adapter baseline. It should not be described as a faithful
  reproduction of a specific external MAPPO or MADDPG implementation.
