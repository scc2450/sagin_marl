# Phase 4 Formal Parameter Sweep Status

Snapshot time: 2026-07-14 03:19 CST.

Remote run root:

```text
friday:/home/sgy/workspace/sagin_marl_phase4_learning_ablation/runs/phase4_learning_ablation/3uav20gu_t250/formal_parameter_sweeps_20260714_full
```

Protocol:

- source scenario: `3uav20gu_t250`;
- policy mode: deterministic;
- eval seed bases: `980000`, `981000`, `982000`;
- episodes per eval seed base: 64;
- num envs per evaluation: 64;
- learned checkpoints: selected checkpoints used in the formal held-out main table;
- methods: `STARS`, `STARS-GC`, `HA-PPO`, `QCCS`, `Lyapunov`, `QBS`, `Uniform`.

Sweeps:

- offered-load sweep on GPU0: task arrival rate values
  `1.0`, `1.5`, `2.0`, `2.5`, `3.0`, `3.5`, `4.0`
  Mbit/(GU slot), corresponding to multipliers
  `0.50`, `0.75`, `1.00`, `1.25`, `1.50`, `1.75`, `2.00`
  relative to the nominal `lambda_0=2.0` Mbit/(GU slot).
- resource-scarcity sweep on GPU1: joint access/backhaul/service capacity
  values scale from the nominal
  `B_U=2 MHz`, `B_S=10 MHz`, `f_S=50 Gbit/s`.
  The plotted x-axis should use the actual capacity tuple or a clearly named
  physical value, not only the multiplier.

Execution layout:

- A first serial launch completed the initial `STARS` and `STARS-GC` rows at
  the first sweep point, then was stopped because `HA-PPO` evaluation was much
  slower than the structured methods.
- The active launch is sharded:
  - fastpath queues cover `STARS`, `STARS-GC`, `QCCS`, `Lyapunov`, `QBS`, and
    `Uniform`;
  - `HA-PPO` is split into one runner per sweep point for load and resource.
- Status files are written independently under the remote run root:
  `status_gpu0_load*.csv` and `status_gpu1_resource*.csv`.

Known runtime note:

- `HA-PPO` uses the same scenario, held-out seeds, and hybrid interface, but its
  flat learned policy path is much slower during evaluation. This should be
  reported as runtime/resource evidence rather than hidden.
- The initial HA-PPO shards were paused after confirming the overhead, so that
  the paper-facing sensitivity curves can first be completed for the main
  structured method and non-learning baselines.

Known transient issue:

- One resource fastpath row exited with `rc=-11` while GPU1 was running seven
  concurrent `HA-PPO` shards:
  `resource_x0p75_globalcritic_seed73129_selected_seedbase981000`.
- The row has no summary JSON and is excluded by aggregation.
- A watcher process is running on friday. After current runners finish, it will
  rerun the fastpath queues with skip-existing enabled and then aggregate all
  completed `rc=0` rows into:
  - `docs/paper/table_sources/phase4_formal_parameter_sweeps_raw_20260714.csv`;
  - `docs/paper/table_sources/phase4_formal_parameter_sweeps_aggregate_20260714.csv`.

Current completion:

- The no-HA-PPO sweep is complete and copied to the local paper evidence
  registry.
- Completed no-HA-PPO unique rows: 420/420.
- Aggregate rows: 84 = 2 sweeps x 7 points x 6 methods.
- Local paper-facing files:
  - `docs/paper/table_sources/phase4_formal_parameter_sweeps_nohappo_raw_20260714.csv`;
  - `docs/paper/table_sources/phase4_formal_parameter_sweeps_nohappo_aggregate_20260714.csv`.
- The two nonzero `rc=-11` status rows are retained in the remote status CSVs
  for traceability, but each affected label has a later successful `rc=0`
  summary and is included in the no-HA-PPO aggregate.

Local helper scripts:

- `scripts/experiments/phase4/run_formal_parameter_sweeps_20260714.py`
- `scripts/experiments/phase4/launch_formal_parameter_sweep_shards_20260714.sh`
- `scripts/experiments/phase4/watch_and_postprocess_formal_parameter_sweeps_20260714.sh`
- `scripts/analysis/phase4/aggregate_formal_parameter_sweeps_20260714.py`

Consolidation note, 2026-08-11:

- The launcher should default to the no-HA-PPO sweep. HA-PPO can still be run as
  a serial companion with `INCLUDE_HAPPO=1`, but it should not be mixed into the
  paper-facing full-sweep queue.
- The repository scripts are the canonical runner and aggregator paths. Earlier
  `/tmp/...` runner copies were temporary remote execution helpers only.
