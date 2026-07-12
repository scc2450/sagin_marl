# Phase 4 Formal Evaluation Matrix

Date: 2026-07-12

This note fixes the Section 5 source-scenario evaluation layout and separates
current checkpoint-validation evidence from the still-needed held-out formal
evaluation.

## Current Evidence Snapshot

Current evidence is checkpoint validation, not final held-out evaluation:

- scenario: `3UAV/20GU/T=250`;
- return target: `bootstrap_gae`;
- validation seed base: `910000`;
- checkpoint-eval episodes: `32`;
- checkpoint selection: last `model_improved=1` row, i.e. the actual
  `best_checkpoint.pt` selection rule, not necessarily the raw reward peak.

Local snapshot files:

```text
docs/paper/figure_sources/phase4_source_validation_20260712/phase4_source_validation_quicklook_20260712.png
docs/paper/figure_sources/phase4_source_validation_20260712/phase4_source_validation_quicklook_20260712.svg
docs/paper/table_sources/phase4_source_validation_checkpoint_summary_20260712.csv
docs/paper/table_sources/phase4_source_validation_checkpoint_aggregate_20260712.csv
docs/paper/table_sources/phase4_source_validation_checkpoint_curves_20260712.csv
```

Selected-checkpoint validation summary:

| Method | n | Reward | Processed | Drop | Backlog | D_sys | Collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| `queue_aware_bw` reference | 1 | 32.28 | 0.438 | 0.482 | 16.56 | 42.87 | 0.000 |
| MAPPO-like FlatActorCritic | 2 | 33.46 | 0.445 | 0.494 | 13.90 | 38.06 | 0.000 |
| GlobalCritic | 3 | 46.21 | 0.745 | 0.225 | 7.40 | 13.95 | 0.000 |
| RelCritic / proposed critic | 3 | 68.71 | 0.922 | 0.064 | 4.40 | 5.70 | 0.000 |

Interpretation:

- RelCritic is strong and stable across the three validation training seeds.
- GlobalCritic has one strong seed and two weak seeds, so its central result is
  high variance rather than consistent failure.
- MAPPO-like learns a modest policy, but only slightly improves reward over
  `queue_aware_bw` and has worse mean drop ratio. It should be presented as a
  weak flat learned baseline unless held-out results show otherwise.

## Figure And Table Plan

| Paper item | Purpose | Current evidence | Formal action before paper use |
|---|---|---|---|
| Fig. 5-1: training / validation curve | Show learning dynamics and mark selected best/final checkpoints. | Current quicklook figure and curve CSV are usable as a validation snapshot. | Redraw from final selected run set; keep selected-best star and final marker for each seed. |
| Table/Fig. 5-2: main source-scenario performance | Main source table across reward, processed, drop, backlog, `D_sys`, collision. | Only checkpoint-validation summary exists. | Run held-out deterministic evaluation with common seed bases for learned methods and fixed baselines. |
| Fig./Table 5-III: critic ablation | RelCritic vs GlobalCritic, optionally FlatFullStateCritic/global-linear if available. | RelCritic 3 seeds and GlobalCritic 3 seeds completed under validation protocol. | Evaluate selected and final checkpoints on the same held-out seed bases. Add optional critic variants only if smoke/formal runs are completed. |
| Fig./Table 5-IV: MAPPO-like comparison | Proposed structured method vs flat actor/flat critic MAPPO-like adapter. | MAPPO-like stabilized has 2 completed seeds under validation protocol. | Prefer one more MAPPO-like seed for a 3-seed table; otherwise label as `n=2`. Evaluate selected and final checkpoints under the same held-out protocol. |

## Held-Out Formal Protocol

Use one protocol for all Section 5 source-scenario rows:

```text
scenario: 3UAV/20GU/T=250 source scenario
policy mode: deterministic
episode seed bases: 980000, 981000, 982000
episodes per seed base: 64
total episodes per method/checkpoint: 192
metrics: reward_sum, processed_ratio_eval, drop_ratio_eval,
         pre_backlog_steps_eval, D_sys_report, collision_episode_fraction
```

Learned methods to evaluate:

- RelCritic selected checkpoint and final checkpoint, seeds `45211`, `73129`,
  `91457`;
- GlobalCritic selected checkpoint and final checkpoint, seeds `45211`,
  `73129`, `91457`;
- MAPPO-like FlatActorCritic selected checkpoint and final checkpoint, seeds
  `45211`, `73129`; add seed `91457` if compute allows before freezing the
  table.

Fixed baselines to evaluate under the same seed bases:

- `static_uniform`;
- `queue_aware_bw`;
- `cluster_center_queue_aware`;
- `observable_cluster_queue_aware`, if runtime is stable;
- MaxWeight/Lyapunov or DPP-style baseline only if the current native evaluator
  path is confirmed and the run can be traced in table sources.

Recommended reporting:

- main table: selected checkpoint rows plus fixed baselines;
- companion note or appendix rows: final checkpoint performance for each
  learned method, to address stability and peak-selection concerns;
- curves: validation best and final markers, not only peak reward.

## Evidence Boundaries

Do not describe the current quicklook figure as final held-out evidence. It is
useful for deciding the Section 5 structure and for checking whether the
baseline/ablation story is coherent before spending more evaluation budget.

The paper-facing claim should wait for the held-out source table:

```text
RelCritic improves learning stability and selected-checkpoint performance
relative to non-relational/global critic baselines, while a flat MAPPO-like
adapter learns only a modest policy under the same hybrid masked SAGIN action
interface.
```

