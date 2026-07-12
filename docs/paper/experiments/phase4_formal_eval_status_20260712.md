# Phase 4 Formal Held-Out Evaluation Status

Snapshot time: 2026-07-13 00:56 CST

Paper-facing Section 5 freeze:

- `docs/paper/experiments/section5_performance_assets_20260713.md`

Remote run root:

```text
friday:/home/sgy/workspace/sagin_marl_phase4_learning_ablation/runs/phase4_learning_ablation/3uav20gu_t250/formal_heldout_20260712_202812
```

Protocol:

- scenario: `3uav20gu_t250`;
- policy mode: deterministic;
- episode seed bases: `980000`, `981000`, `982000`;
- episodes per seed base: 64;
- target episodes per method/checkpoint/training seed: 192.

## Completed Blocks

- RelCritic selected/final: 3 training seeds, 192 held-out episodes per seed.
- GlobalCritic selected/final: 3 training seeds, 192 held-out episodes per seed.
- MAPPO-like selected/final: 3 training seeds, 192 held-out episodes per seed.
- Fixed baselines: `static_uniform`, `queue_aware_bw`,
  `cluster_center_queue_aware`, `observable_cluster_queue_aware`,
  `maxweight_lyapunov`, each with 192 held-out episodes.
- MAPPO-like seed `91457` training completed with `rc=0`; it stopped at update
  425 by checkpoint reward plateau.

Known non-result row:

- `status_gpu1.csv` contains one `rc=143` row for
  `mappo_like/seed45211/selected/seedbase980000`. This was a manually
  terminated compile/slow-path probe and must not be counted as a formal result.

## Current Queues

- No formal-evaluation queues are currently running.
- GPU0 queue finished at 2026-07-12 22:11:04 CST.
- GPU1 MAPPO-like queue finished at 2026-07-12 23:50:59 CST.

Final status accounting:

- `status_gpu1.csv`: 63 formal `rc=0` rows plus one manually terminated
  non-result row with `rc=143`.
- `status_gpu0.csv`: 6 MAPPO-like formal `rc=0` rows plus one training row.
- final formal evaluation rows used for aggregation: 69.

## Current Aggregate Snapshot

These numbers use only completed formal `rc=0` rows. The manually terminated
`rc=143` probe and the `train` row are excluded.

| Method | Checkpoint | n train seeds | Reward | Processed | Drop | Backlog | D_sys | Collision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RelCritic | selected | 3 | 71.017 +/- 0.824 | 0.911 +/- 0.012 | 0.073 +/- 0.012 | 4.463 +/- 0.444 | 6.287 +/- 0.527 | 0.005 +/- 0.000 |
| RelCritic | final | 3 | 62.244 +/- 10.300 | 0.855 +/- 0.054 | 0.127 +/- 0.047 | 4.711 +/- 0.913 | 7.740 +/- 2.242 | 0.014 +/- 0.012 |
| GlobalCritic | selected | 3 | 47.349 +/- 22.685 | 0.725 +/- 0.157 | 0.237 +/- 0.139 | 7.536 +/- 2.686 | 16.210 +/- 8.775 | 0.038 +/- 0.034 |
| GlobalCritic | final | 3 | 43.402 +/- 20.606 | 0.647 +/- 0.227 | 0.299 +/- 0.196 | 9.372 +/- 3.631 | 23.438 +/- 15.401 | 0.118 +/- 0.168 |
| MAPPO-like | selected | 3 | 27.477 +/- 1.591 | 0.508 +/- 0.028 | 0.376 +/- 0.017 | 16.742 +/- 0.266 | 37.213 +/- 2.440 | 0.302 +/- 0.059 |
| MAPPO-like | final | 3 | 26.807 +/- 0.272 | 0.501 +/- 0.014 | 0.377 +/- 0.016 | 16.716 +/- 0.064 | 37.964 +/- 1.690 | 0.333 +/- 0.000 |

| Fixed baseline | Reward | Processed | Drop | Backlog | D_sys | Collision |
|---|---:|---:|---:|---:|---:|---:|
| `cluster_center_queue_aware` | 50.448 | 0.893 | 0.079 | 6.179 | 8.512 | 0.005 |
| `maxweight_lyapunov` | 44.656 | 0.845 | 0.124 | 7.729 | 10.969 | 0.000 |
| `observable_cluster_queue_aware` | 31.903 | 0.634 | 0.281 | 10.475 | 19.443 | 0.245 |
| `queue_aware_bw` | 32.965 | 0.474 | 0.438 | 16.589 | 39.708 | 0.005 |
| `static_uniform` | 33.229 | 0.473 | 0.463 | 13.614 | 37.053 | 0.005 |

## Immediate Interpretation

- RelCritic selected is the current strongest and most stable held-out result.
- GlobalCritic has a high-variance profile: one strong seed and two weak seeds.
- The stronger fixed baselines are `cluster_center_queue_aware` and
  `maxweight_lyapunov`; both are below RelCritic selected on reward and system
  delay in the current completed snapshot.
- MAPPO-like is now a complete 3-seed learned baseline, but it is weak under the
  same hybrid masked SAGIN action interface: selected reward is lower than both
  `queue_aware_bw` and `static_uniform`, and collision fraction is much higher.
- For the main paper table, use selected checkpoints for learned methods and
  report final checkpoints as stability/selection evidence.
