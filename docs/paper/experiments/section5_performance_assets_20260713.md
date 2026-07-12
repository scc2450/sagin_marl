# Section 5 Performance Assets Freeze (2026-07-13)

Purpose: freeze the Section 5 main performance evidence after the formal held-out source-scenario evaluation completed on 2026-07-12/13.

## Protocol

- Scenario: `3uav20gu_t250`.
- Held-out policy mode: deterministic.
- Evaluation seed bases: `980000`, `981000`, `982000`.
- Episodes per seed base: 64, i.e. 192 held-out episodes per learned training seed or fixed baseline.
- Learned methods: RelCritic, GlobalCritic, and MAPPO-like, each with 3 training seeds.
- Fixed baselines in the main table: `cluster_center_queue_aware`, `maxweight_lyapunov`, `observable_cluster_queue_aware`, `queue_aware_bw`, and `static_uniform`.
- Excluded row: one `rc=143` MAPPO-like slow-path/manual probe in `status_gpu1.csv`; it is not counted in any table.

## Main Table Decision

Use selected checkpoints for the Section 5 main table. Put fixed baselines in the same table. Do not center the final checkpoint rows in the main table; use them as stability/selection companion evidence or appendix material.

Main-table source CSV:

- `docs/paper/table_sources/phase4_formal_heldout_source_selected_main_20260713.csv`

| Method | Checkpoint | n train seeds | Reward | Processed | Drop | Backlog | D_sys | Collision |
|---|---|---|---|---|---|---|---|---|
| RelCritic | selected | 3 | 71.017 +/- 0.824 | 0.911 +/- 0.012 | 0.073 +/- 0.012 | 4.463 +/- 0.444 | 6.287 +/- 0.527 | 0.005 +/- 0.000 |
| GlobalCritic | selected | 3 | 47.349 +/- 22.685 | 0.725 +/- 0.157 | 0.237 +/- 0.139 | 7.536 +/- 2.686 | 16.210 +/- 8.775 | 0.038 +/- 0.034 |
| MAPPO-like | selected | 3 | 27.477 +/- 1.591 | 0.508 +/- 0.028 | 0.376 +/- 0.017 | 16.742 +/- 0.266 | 37.213 +/- 2.440 | 0.302 +/- 0.059 |
| cluster_center_queue_aware | fixed |  | 50.448 +/- 0.335 | 0.893 +/- 0.007 | 0.079 +/- 0.005 | 6.179 +/- 0.115 | 8.512 +/- 0.336 | 0.005 +/- 0.009 |
| maxweight_lyapunov | fixed |  | 44.656 +/- 0.558 | 0.845 +/- 0.007 | 0.124 +/- 0.006 | 7.729 +/- 0.251 | 10.969 +/- 0.647 | 0.000 +/- 0.000 |
| observable_cluster_queue_aware | fixed |  | 31.903 +/- 0.848 | 0.634 +/- 0.006 | 0.281 +/- 0.005 | 10.475 +/- 0.312 | 19.443 +/- 0.730 | 0.245 +/- 0.009 |
| queue_aware_bw | fixed |  | 32.965 +/- 0.462 | 0.474 +/- 0.022 | 0.438 +/- 0.020 | 16.589 +/- 0.717 | 39.708 +/- 4.256 | 0.005 +/- 0.009 |
| static_uniform | fixed |  | 33.229 +/- 0.466 | 0.473 +/- 0.015 | 0.463 +/- 0.016 | 13.614 +/- 0.588 | 37.053 +/- 4.439 | 0.005 +/- 0.009 |

## Selected Versus Final Companion Table

Source CSV:

- `docs/paper/table_sources/phase4_formal_heldout_source_learned_selected_final_20260713.csv`

| Method | Checkpoint | n train seeds | Reward | Processed | Drop | Backlog | D_sys | Collision |
|---|---|---|---|---|---|---|---|---|
| RelCritic | selected | 3 | 71.017 +/- 0.824 | 0.911 +/- 0.012 | 0.073 +/- 0.012 | 4.463 +/- 0.444 | 6.287 +/- 0.527 | 0.005 +/- 0.000 |
| RelCritic | final | 3 | 62.244 +/- 10.300 | 0.855 +/- 0.054 | 0.127 +/- 0.047 | 4.711 +/- 0.913 | 7.740 +/- 2.242 | 0.014 +/- 0.012 |
| GlobalCritic | selected | 3 | 47.349 +/- 22.685 | 0.725 +/- 0.157 | 0.237 +/- 0.139 | 7.536 +/- 2.686 | 16.210 +/- 8.775 | 0.038 +/- 0.034 |
| GlobalCritic | final | 3 | 43.402 +/- 20.606 | 0.647 +/- 0.227 | 0.299 +/- 0.196 | 9.372 +/- 3.631 | 23.438 +/- 15.401 | 0.118 +/- 0.168 |
| MAPPO-like | selected | 3 | 27.477 +/- 1.591 | 0.508 +/- 0.028 | 0.376 +/- 0.017 | 16.742 +/- 0.266 | 37.213 +/- 2.440 | 0.302 +/- 0.059 |
| MAPPO-like | final | 3 | 26.807 +/- 0.272 | 0.501 +/- 0.014 | 0.377 +/- 0.016 | 16.716 +/- 0.064 | 37.964 +/- 1.690 | 0.333 +/- 0.000 |

Writing use: this companion table supports the checkpoint-selection narrative. RelCritic selected is both strongest and more stable than final; GlobalCritic remains high variance; MAPPO-like remains weak even after completing the third seed.

## Figures To Use

Source figure directory:

- `docs/paper/figure_sources/phase4_section5_performance_20260713/`

LaTeX-ready copies:

- `docs/paper/manuscript/figures/phase4_training_validation_curves_20260713.pdf`
- `docs/paper/manuscript/figures/phase4_main_performance_grouped_bars_20260713.pdf`
- `docs/paper/manuscript/figures/phase4_globalcritic_seed_level_20260713.pdf`

Figure roles:

- `phase4_training_validation_curves_20260713`: training/checkpoint-validation curves. Star marks selected best; square marks final.
- `phase4_main_performance_grouped_bars_20260713`: main performance grouped bars for reward, processed ratio, drop ratio, and `D_sys`; this avoids a reward-only presentation.
- `phase4_globalcritic_seed_level_20260713`: seed-level RelCritic versus GlobalCritic display. Use this when explaining the large GlobalCritic variance.

## GlobalCritic Seed-Level Interpretation

GlobalCritic should not be summarized only by the mean. The selected-checkpoint aggregate has large training-seed variance, so Section 5 should explicitly show the three seed-level points or error bars. The conservative claim is that global-only critic can occasionally find a usable seed, but it is not robust under this protocol.

## MAPPO-like Fairness Boundary

MAPPO-like is a same-scenario, same held-out seed, same hybrid/masked SAGIN interface learned baseline. It uses a flat learned actor/critic inside the same PPO/GAE-style training and evaluation pipeline, keeping reward, safety handling, action interface, and held-out protocol aligned.

Do not describe it as a faithful reproduction of a specific external MAPPO, MADDPG, or SAGIN paper. The paper-facing wording should be: MAPPO-like adapter baseline under our hybrid action interface, used to test whether a flat learned multi-agent baseline suffices when the topology-aware actor and relational critic are removed.

## Runtime And Resource Cost

Runtime/resource source CSV:

- `docs/paper/table_sources/phase4_runtime_resource_summary_20260713.csv`
- `docs/paper/table_sources/phase4_formal_eval_wallclock_summary_20260713.csv`

Training resource protocol: `num_envs=64`, `rollout_env_steps=250`, i.e. 16,000 environment transitions per update. Training was run on friday GPUs; exact GPU queue placement is retained in status logs, with the formal held-out queue split across GPU0/GPU1.

| Method | train wall-clock h | selected update | final update | completed updates |
|---|---:|---:|---:|---:|
| RelCritic | 1.15 +/- 0.30 | 342 | 458 | 458 |
| GlobalCritic | 0.60 +/- 0.04 | 250 | 392 | 392 |
| MAPPO-like | 0.43 +/- 0.04 | 400 | 450 | 450 |

Use these numbers to add a compact Section 5 paragraph or table reporting training cost, selected update, final update, and environment-step budget. For update-to-episode discussion, translate updates as `updates * 64` rollout episodes or `updates * 64 * 250` environment steps.

## Paper Claim Boundary

Supported now:

- RelCritic selected is the main source-scenario result and outperforms both fixed baselines and the learned ablations on reward, processed ratio, drop ratio, backlog, and `D_sys`.
- The relational critic ablation is strong evidence because GlobalCritic has high variance and worse mean performance under the same training/evaluation interface.
- MAPPO-like is complete at 3 seeds and provides a fair in-pipeline learned baseline, but only within the adapter-baseline boundary above.
- Final checkpoints are useful for stability/selection analysis, not for replacing the selected-checkpoint main result.

Not supported without additional experiments:

- Claims that this MAPPO-like baseline reproduces or defeats any particular external MAPPO/MADDPG implementation.
- Claims that final checkpoint behavior is the primary performance target.
- Broad larger-scale generalization claims from this Section 5 source-scenario evidence alone.
