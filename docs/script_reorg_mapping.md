# Scripts Reorganization Mapping

This is a planning and execution log for gradually reorganizing `scripts/`.

## Goals

- Keep the current training/evaluation/render path easy to find.
- Reduce top-level `scripts/` clutter without breaking imports.
- Move one-off diagnostics into clearly named buckets.
- Preserve historical scripts instead of deleting them.

## Important Constraints

- `scripts/train_joint_mcgae.py` imports helpers from top-level `scripts.*`. Moving those helpers requires import updates.
- Many Python scripts contain `Path(__file__).resolve().parents[1]`; moving them into subdirectories requires a bootstrap fix.
- Do not move current entrypoints in the first script-cleanup commit. Keep them at root until the import/path refactor is tested.

## Proposed Directory Buckets

```text
scripts/
  train_joint_mcgae.py                         # keep root for now
  evaluate_structured_mixed_heads_native.py    # keep root for now
  render_structured_episode.py                 # keep root for now
  run_autodl_cuda_smoke.sh                     # keep root for now
  run_mac_joint_smoke.sh                       # keep root for now
  analysis/
  benchmarks/
  diagnostics/
    audit/
    diagnose/
    probe/
  evaluation/
  experiments/
  legacy/
  runners/
  misc/
```

## Recommended Execution Order

1. P0 keep-root: do not move current entrypoints or imported core helpers yet.
2. P1 low-risk move: move analysis, benchmark, probes, shell runners after updating direct references.
3. P2 import refactor: move interdependent audits/diagnostics/legacy scripts only after updating `from scripts.*` imports and root bootstraps.
4. Run `py_compile`, shell syntax checks, and a tiny smoke load after each batch.

## P0 Root Keepers

| path | reason |
|---|---|
| `scripts/evaluate_structured_mixed_heads_native.py` | current entrypoint or imported helper; leave in root for first pass |
| `scripts/render_structured_episode.py` | current entrypoint or imported helper; leave in root for first pass |
| `scripts/run_autodl_cuda_smoke.sh` | current entrypoint or imported helper; leave in root for first pass |
| `scripts/run_mac_joint_smoke.sh` | current entrypoint or imported helper; leave in root for first pass |
| `scripts/train_joint_mcgae.py` | current entrypoint or imported helper; leave in root for first pass |
| `scripts/train_stage_mcgae.py` | current entrypoint or imported helper; leave in root for first pass |

## Summary By Proposed Top Bucket

| bucket | count |
|---|---:|
| `scripts/analysis` | 17 |
| `scripts/benchmarks` | 5 |
| `scripts/diagnostics` | 105 |
| `scripts/evaluation` | 9 |
| `scripts/experiments` | 23 |
| `scripts/legacy` | 8 |
| `scripts/root-keepers` | 6 |

## Import-Dependent Scripts

This table records the early planning snapshot. Executed move sections below supersede locations when a script has since been migrated.

| script | imported by |
|---|---|
| `scripts/audit_stage_ppo_credit_alignment.py` | 16: `audit_accel_one_update_lr_effect.py`, `audit_bw_real_update_alignment.py`, `audit_critic_artifact_eval.py`, `audit_critic_head_lr_sweep.py`, `audit_critic_heldout_seed_sweep.py`, `audit_critic_relearn_after_policy_update.py`, `audit_critic_vhat_artifact.py`, `audit_fixed_critic_benchmark.py`, `audit_sat_local_world_vhat_probe.py`, `audit_stage_credit_chain.py`, `audit_stage_critic_only_fit.py`, `audit_stage_qpi_action_credit.py`, `diagnose_bw_actor_update_repro.py`, `train_joint_mcgae.py`, `train_sat_mcgae.py`, `train_stage_mcgae.py` |
| `scripts/audit_stage_critic_only_fit.py` | 12: `audit_critic_artifact_eval.py`, `audit_critic_head_lr_sweep.py`, `audit_critic_heldout_seed_sweep.py`, `audit_critic_relearn_after_policy_update.py`, `audit_critic_vhat_artifact.py`, `audit_fixed_critic_benchmark.py`, `audit_sat_local_world_vhat_probe.py`, `diagnose_bw_actor_update_repro.py`, `profile_joint_critic_step.py`, `train_joint_mcgae.py`, `train_sat_mcgae.py`, `train_stage_mcgae.py` |
| `scripts/diagnose_reward_action_sensitivity.py` | 6: `audit_critic_relearn_after_policy_update.py`, `audit_critic_vhat_artifact.py`, `audit_stage_credit_chain.py`, `audit_stage_critic_only_fit.py`, `audit_stage_ppo_credit_alignment.py`, `audit_stage_qpi_action_credit.py` |
| `scripts/train_stage_mcgae.py` | 6: `audit_accel_one_update_lr_effect.py`, `audit_bw_real_update_alignment.py`, `diagnose_bw_actor_update_repro.py`, `diagnose_inductor_bw_cross_compile.py`, `profile_joint_critic_step.py`, `train_joint_mcgae.py` |
| `scripts/train_joint_mcgae.py` | 5: `audit_accel_one_update_lr_effect.py`, `audit_bw_real_update_alignment.py`, `diagnose_bw_actor_update_repro.py`, `diagnose_inductor_bw_cross_compile.py`, `profile_joint_critic_step.py` |
| `scripts/audit_fixed_critic_benchmark.py` | 4: `audit_critic_artifact_eval.py`, `audit_critic_head_lr_sweep.py`, `audit_critic_vhat_artifact.py`, `audit_sat_local_world_vhat_probe.py` |
| `scripts/audit_stage_qpi_action_credit.py` | 3: `audit_critic_relearn_after_policy_update.py`, `audit_stage_credit_chain.py`, `audit_stage_critic_only_fit.py` |
| `scripts/audit_stage_credit_chain.py` | 2: `audit_critic_relearn_after_policy_update.py`, `audit_stage_critic_only_fit.py` |
| `scripts/diagnose_frontend_critic.py` | 1: `diagnose_bw_perhead_vs_oldjoint.py` |
| `scripts/evaluate_thesis_native_methods.py` | 1: `analyze_thesis_fairness.py` |


## Executed P1 Low-Risk Batch

Moved in the first script cleanup batch because these files had no direct `scripts.*` import dependents, no detected root-bootstrap issue, and no non-mapping documentation references.

- `scripts/analyze_reward_scales.py -> scripts/analysis/analyze_reward_scales.py`
- `scripts/backfill_structured_tb.py -> scripts/analysis/export/backfill_structured_tb.py`
- `scripts/bench_structured_native_kernels.py -> scripts/benchmarks/bench_structured_native_kernels.py`
- `scripts/collect_stage_diagnostics.py -> scripts/diagnostics/collection/collect_stage_diagnostics.py`
- `scripts/debug_bw_audit_subproc_snapshot.py -> scripts/diagnostics/debug/debug_bw_audit_subproc_snapshot.py`
- `scripts/plot_action_head_ablation_preview.py -> scripts/analysis/plots/plot_action_head_ablation_preview.py`
- `scripts/plot_danger_imitation_ablation_preview.py -> scripts/analysis/plots/plot_danger_imitation_ablation_preview.py`
- `scripts/plot_learning_curve.py -> scripts/analysis/plots/plot_learning_curve.py`
- `scripts/plot_recomputed_positive_reward_curve.py -> scripts/analysis/plots/plot_recomputed_positive_reward_curve.py`
- `scripts/plot_reward_curve_from_eval_ratio.py -> scripts/analysis/plots/plot_reward_curve_from_eval_ratio.py`
- `scripts/plot_typical_uav_trajectory_preview.py -> scripts/analysis/plots/plot_typical_uav_trajectory_preview.py`
- `scripts/plot_uav_queue_imbalance_preview.py -> scripts/analysis/plots/plot_uav_queue_imbalance_preview.py`
- `scripts/probe_structured_sat_state.py -> scripts/diagnostics/probe/probe_structured_sat_state.py`

## Executed No-Internal-Import Batch

Moved in the second script cleanup batch because these files had no detected internal script import relationship, no detected root-bootstrap issue, no tracked text references outside this mapping file, and are not current train/eval/render entrypoints.

- `scripts/audit_bw_focus_leverage.py -> scripts/diagnostics/audit/audit_bw_focus_leverage.py`
- `scripts/audit_bw_snapshot_bank.py -> scripts/diagnostics/audit/audit_bw_snapshot_bank.py`
- `scripts/audit_bw_snapshot_reward_gap.py -> scripts/diagnostics/audit/audit_bw_snapshot_reward_gap.py`
- `scripts/audit_requested_items.py -> scripts/diagnostics/audit/audit_requested_items.py`
- `scripts/collect_bw_local_opportunity_bank.py -> scripts/experiments/bw_distill/collect_bw_local_opportunity_bank.py`
- `scripts/diagnose_abcd.py -> scripts/diagnostics/diagnose/diagnose_abcd.py`
- `scripts/diagnose_bw_old_logprob_replay.py -> scripts/diagnostics/diagnose/diagnose_bw_old_logprob_replay.py`
- `scripts/diagnose_native_flow_regime.py -> scripts/diagnostics/diagnose/diagnose_native_flow_regime.py`
- `scripts/diagnose_old_bootstrap_targets.py -> scripts/diagnostics/diagnose/diagnose_old_bootstrap_targets.py`
- `scripts/diagnose_structured_bootstrap_targets.py -> scripts/diagnostics/diagnose/diagnose_structured_bootstrap_targets.py`
- `scripts/diagnose_structured_bw_access_signal_curve.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_access_signal_curve.py`
- `scripts/diagnose_structured_bw_advantage_shape.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_advantage_shape.py`
- `scripts/diagnose_structured_bw_advantage_sources.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_advantage_sources.py`
- `scripts/diagnose_structured_bw_basin_gap.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_basin_gap.py`
- `scripts/diagnose_structured_bw_candidate_hit_rate.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_candidate_hit_rate.py`
- `scripts/diagnose_structured_bw_counterfactual_credit_probe.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_counterfactual_credit_probe.py`
- `scripts/diagnose_structured_bw_det_marginal_teacher.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_det_marginal_teacher.py`
- `scripts/diagnose_structured_bw_exploration_eval.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_exploration_eval.py`
- `scripts/diagnose_structured_bw_flatness_decomposition.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_flatness_decomposition.py`
- `scripts/diagnose_structured_bw_gap_geometry.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_gap_geometry.py`
- `scripts/diagnose_structured_bw_grad_split.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_grad_split.py`
- `scripts/diagnose_structured_bw_heuristic_perturb_impulse.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_heuristic_perturb_impulse.py`
- `scripts/diagnose_structured_bw_interpolation_curve.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_interpolation_curve.py`
- `scripts/diagnose_structured_bw_k2_alpha_curve.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_k2_alpha_curve.py`
- `scripts/diagnose_structured_bw_kstep_override.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_kstep_override.py`
- `scripts/diagnose_structured_bw_kstep_splice.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_kstep_splice.py`
- `scripts/diagnose_structured_bw_linear_probe.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_linear_probe.py`
- `scripts/diagnose_structured_bw_loc_probe.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_loc_probe.py`
- `scripts/diagnose_structured_bw_local_transport_search.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_local_transport_search.py`
- `scripts/diagnose_structured_bw_panel_execsources.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_panel_execsources.py`
- `scripts/diagnose_structured_bw_proxy_credit_alignment.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_proxy_credit_alignment.py`
- `scripts/diagnose_structured_bw_queue_reward_impulse.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_queue_reward_impulse.py`
- `scripts/diagnose_structured_bw_representation_probe.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_representation_probe.py`
- `scripts/diagnose_structured_bw_rule_probe.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_rule_probe.py`
- `scripts/diagnose_structured_bw_search_target_geometry.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_search_target_geometry.py`
- `scripts/diagnose_structured_bw_slot_order_sensitivity.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_slot_order_sensitivity.py`
- `scripts/diagnose_structured_bw_update.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_update.py`
- `scripts/diagnose_structured_bw_update_direction.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_update_direction.py`
- `scripts/diagnose_structured_bw_within_state_returns.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_within_state_returns.py`
- `scripts/diagnose_structured_critic_alignment.py -> scripts/diagnostics/diagnose/diagnose_structured_critic_alignment.py`
- `scripts/diagnose_structured_sat_bw_mechanisms.py -> scripts/diagnostics/diagnose/diagnose_structured_sat_bw_mechanisms.py`
- `scripts/diagnose_structured_step_bootstrap_bias.py -> scripts/diagnostics/diagnose/diagnose_structured_step_bootstrap_bias.py`
- `scripts/diagnose_structured_step_targets.py -> scripts/diagnostics/diagnose/diagnose_structured_step_targets.py`
- `scripts/diagnose_structured_target_decomposition.py -> scripts/diagnostics/diagnose/diagnose_structured_target_decomposition.py`
- `scripts/eval_fixed_frontsat_bw_variants.py -> scripts/evaluation/eval_fixed_frontsat_bw_variants.py`
- `scripts/eval_param_sweep.py -> scripts/evaluation/eval_param_sweep.py`
- `scripts/evaluate_action_head_ablation_best.py -> scripts/evaluation/evaluate_action_head_ablation_best.py`
- `scripts/evaluate_structured_fixed_policy.py -> scripts/evaluation/evaluate_structured_fixed_policy.py`
- `scripts/evaluate_structured_hybrid_heads.py -> scripts/evaluation/evaluate_structured_hybrid_heads.py`
- `scripts/prepare_structured_bw_reset_init.py -> scripts/experiments/bw_tools/prepare_structured_bw_reset_init.py`
- `scripts/sat_supervised_probe.py -> scripts/experiments/sat/sat_supervised_probe.py`
- `scripts/solve_bw_flow_bcd_general.py -> scripts/experiments/bw_tools/solve_bw_flow_bcd_general.py`

## Executed Diagnostics Entry Batch

Moved in the next diagnostics cleanup batch because these files had no detected code import dependents outside the mapping file. Shared helpers and interdependent BW/critic/stage audit clusters remain at top level for a later coordinated import refactor.

This batch also normalizes existing `scripts/diagnostics/**` repo-root bootstraps so scripts moved into nested directories can still be launched directly from the repository root.

- `scripts/audit_accel_one_update_lr_effect.py -> scripts/diagnostics/audit/audit_accel_one_update_lr_effect.py`
- `scripts/audit_accel_positive_signal_breakdown.py -> scripts/diagnostics/audit/audit_accel_positive_signal_breakdown.py`
- `scripts/audit_accel_ppo_mechanics.py -> scripts/diagnostics/audit/audit_accel_ppo_mechanics.py`
- `scripts/audit_accel_ppo_update_credit.py -> scripts/diagnostics/audit/audit_accel_ppo_update_credit.py`
- `scripts/audit_bw_real_update_alignment.py -> scripts/diagnostics/audit/audit_bw_real_update_alignment.py`
- `scripts/audit_critic_artifact_eval.py -> scripts/diagnostics/audit/audit_critic_artifact_eval.py`
- `scripts/audit_critic_heldout_seed_sweep.py -> scripts/diagnostics/audit/audit_critic_heldout_seed_sweep.py`
- `scripts/diagnose_accel_candidate_workload_reward.py -> scripts/diagnostics/diagnose/diagnose_accel_candidate_workload_reward.py`
- `scripts/diagnose_bw_actor_update_repro.py -> scripts/diagnostics/diagnose/diagnose_bw_actor_update_repro.py`
- `scripts/diagnose_inductor_bw_cross_compile.py -> scripts/diagnostics/diagnose/diagnose_inductor_bw_cross_compile.py`
- `scripts/diagnose_queue_regime.py -> scripts/diagnostics/diagnose/diagnose_queue_regime.py`
- `scripts/diagnose_sat_counterfactual_step.py -> scripts/diagnostics/diagnose/diagnose_sat_counterfactual_step.py`
- `scripts/diagnose_sat_credit_mismatch.py -> scripts/diagnostics/diagnose/diagnose_sat_credit_mismatch.py`
- `scripts/diagnose_sat_selection_gap.py -> scripts/diagnostics/diagnose/diagnose_sat_selection_gap.py`
- `scripts/diagnose_structured_bw_acf_deterministic_baseline.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_acf_deterministic_baseline.py`
- `scripts/diagnose_structured_bw_clean_candidate_sweep.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_clean_candidate_sweep.py`
- `scripts/diagnose_structured_bw_native_clean_fixed_fit.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_native_clean_fixed_fit.py`
- `scripts/diagnose_structured_bw_workload_proportional_fit.py -> scripts/diagnostics/diagnose/diagnose_structured_bw_workload_proportional_fit.py`

## Executed Stage Credit Batch

Moved the interdependent stage-credit audit helper cluster after updating all direct `scripts.*` imports and historical documentation command paths. These scripts are used to diagnose PPO/MC/QPI credit alignment and critic-fit behavior; moving them first keeps the later critic-audit migration smaller.

- `scripts/audit_stage_ppo_credit_alignment.py -> scripts/diagnostics/audit/audit_stage_ppo_credit_alignment.py`
- `scripts/audit_stage_qpi_action_credit.py -> scripts/diagnostics/audit/audit_stage_qpi_action_credit.py`
- `scripts/audit_stage_credit_chain.py -> scripts/diagnostics/audit/audit_stage_credit_chain.py`
- `scripts/audit_stage_critic_only_fit.py -> scripts/diagnostics/audit/audit_stage_critic_only_fit.py`
- `scripts/diagnose_reward_action_sensitivity.py -> scripts/diagnostics/diagnose/diagnose_reward_action_sensitivity.py`

## Executed Remaining Audit/Benchmark Batch

Moved the remaining root-level audit and benchmark scripts after updating direct helper imports, documentation command paths, and nested repo-root bootstraps. `audit_bw_broad2local_offline.py` is a shared helper for BW distillation/evaluation/training utilities, so its downstream imports were migrated in the same batch.

- `scripts/audit_bw_broad2local_offline.py -> scripts/diagnostics/audit/audit_bw_broad2local_offline.py`
- `scripts/audit_critic_head_lr_sweep.py -> scripts/diagnostics/audit/audit_critic_head_lr_sweep.py`
- `scripts/audit_critic_relearn_after_policy_update.py -> scripts/diagnostics/audit/audit_critic_relearn_after_policy_update.py`
- `scripts/audit_critic_vhat_artifact.py -> scripts/diagnostics/audit/audit_critic_vhat_artifact.py`
- `scripts/audit_fixed_critic_benchmark.py -> scripts/diagnostics/audit/audit_fixed_critic_benchmark.py`
- `scripts/audit_old_good_accel_python_credit.py -> scripts/diagnostics/audit/audit_old_good_accel_python_credit.py`
- `scripts/audit_sat_local_world_vhat_probe.py -> scripts/diagnostics/audit/audit_sat_local_world_vhat_probe.py`
- `scripts/bench_structured_training_system.py -> scripts/benchmarks/bench_structured_training_system.py`
- `scripts/bench_train.py -> scripts/benchmarks/bench_train.py`

## Full Mapping

| original path | proposed path | phase | risk | note |
|---|---|---|---|---|
| `scripts/access_control_imitation_common.py` | `scripts/experiments/access_control/access_control_imitation_common.py` | P2 experiment | low | access-control experiment family |
| `scripts/analyze_bw_update_hot_cold_direction.py` | `scripts/analysis/analyze_bw_update_hot_cold_direction.py` | P1 analysis | low | analysis/report helper |
| `scripts/analyze_metrics.py` | `scripts/analysis/analyze_metrics.py` | P1 analysis | low | analysis/report helper |
| `scripts/analyze_reward_scales.py` | `scripts/analysis/analyze_reward_scales.py` | P1 analysis | low | analysis/report helper |
| `scripts/analyze_thesis_fairness.py` | `scripts/analysis/analyze_thesis_fairness.py` | P1 analysis | update-docs | analysis/report helper |
| `scripts/audit_accel_one_update_lr_effect.py` | `scripts/diagnostics/audit/audit_accel_one_update_lr_effect.py` | P2 diagnostics | low | audit script; many imports require coordinated update |
| `scripts/audit_accel_positive_signal_breakdown.py` | `scripts/diagnostics/audit/audit_accel_positive_signal_breakdown.py` | P2 diagnostics | fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_accel_ppo_mechanics.py` | `scripts/diagnostics/audit/audit_accel_ppo_mechanics.py` | P2 diagnostics | fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_accel_ppo_update_credit.py` | `scripts/diagnostics/audit/audit_accel_ppo_update_credit.py` | P2 diagnostics | fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_bw_broad2local_offline.py` | `scripts/diagnostics/audit/audit_bw_broad2local_offline.py` | P2 diagnostics | low | audit script; many imports require coordinated update |
| `scripts/audit_bw_focus_leverage.py` | `scripts/diagnostics/audit/audit_bw_focus_leverage.py` | P2 diagnostics | low | audit script; many imports require coordinated update |
| `scripts/audit_bw_real_update_alignment.py` | `scripts/diagnostics/audit/audit_bw_real_update_alignment.py` | P2 diagnostics | fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_bw_snapshot_bank.py` | `scripts/diagnostics/audit/audit_bw_snapshot_bank.py` | P2 diagnostics | low | audit script; many imports require coordinated update |
| `scripts/audit_bw_snapshot_reward_gap.py` | `scripts/diagnostics/audit/audit_bw_snapshot_reward_gap.py` | P2 diagnostics | low | audit script; many imports require coordinated update |
| `scripts/audit_critic_artifact_eval.py` | `scripts/diagnostics/audit/audit_critic_artifact_eval.py` | P2 diagnostics | fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_critic_head_lr_sweep.py` | `scripts/diagnostics/audit/audit_critic_head_lr_sweep.py` | P2 diagnostics | fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_critic_heldout_seed_sweep.py` | `scripts/diagnostics/audit/audit_critic_heldout_seed_sweep.py` | P2 diagnostics | fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_critic_relearn_after_policy_update.py` | `scripts/diagnostics/audit/audit_critic_relearn_after_policy_update.py` | P2 diagnostics | fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_critic_vhat_artifact.py` | `scripts/diagnostics/audit/audit_critic_vhat_artifact.py` | P2 diagnostics | fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_fixed_critic_benchmark.py` | `scripts/diagnostics/audit/audit_fixed_critic_benchmark.py` | P2 diagnostics | imported-by-4, fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_old_good_accel_python_credit.py` | `scripts/diagnostics/audit/audit_old_good_accel_python_credit.py` | P2 diagnostics | fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_requested_items.py` | `scripts/diagnostics/audit/audit_requested_items.py` | P2 diagnostics | low | audit script; many imports require coordinated update |
| `scripts/audit_sat_local_world_vhat_probe.py` | `scripts/diagnostics/audit/audit_sat_local_world_vhat_probe.py` | P2 diagnostics | fix-ROOT-bootstrap | audit script; many imports require coordinated update |
| `scripts/audit_stage_credit_chain.py` | `scripts/diagnostics/audit/audit_stage_credit_chain.py` | P2 diagnostics | coordinated-import-update | stage-credit diagnostic helper cluster |
| `scripts/audit_stage_critic_only_fit.py` | `scripts/diagnostics/audit/audit_stage_critic_only_fit.py` | P2 diagnostics | coordinated-import-update | stage-credit diagnostic helper cluster |
| `scripts/audit_stage_ppo_credit_alignment.py` | `scripts/diagnostics/audit/audit_stage_ppo_credit_alignment.py` | P2 diagnostics | coordinated-import-update | stage-credit diagnostic helper cluster |
| `scripts/audit_stage_qpi_action_credit.py` | `scripts/diagnostics/audit/audit_stage_qpi_action_credit.py` | P2 diagnostics | coordinated-import-update | stage-credit diagnostic helper cluster |
| `scripts/backfill_structured_tb.py` | `scripts/analysis/export/backfill_structured_tb.py` | P1 analysis | low | export/report generation |
| `scripts/bench_structured_native_kernels.py` | `scripts/benchmarks/bench_structured_native_kernels.py` | P1/P2 benchmark | low | performance benchmark or profiler |
| `scripts/bench_structured_training_system.py` | `scripts/benchmarks/bench_structured_training_system.py` | P1/P2 benchmark | low | performance benchmark or profiler |
| `scripts/bench_train.py` | `scripts/benchmarks/bench_train.py` | P1/P2 benchmark | fix-ROOT-bootstrap | performance benchmark or profiler |
| `scripts/check_bw_distill_reload.py` | `scripts/experiments/bw_distill/check_bw_distill_reload.py` | P2 experiment | low | BW distillation/offline data family |
| `scripts/collect_bw_local_opportunity_bank.py` | `scripts/experiments/bw_distill/collect_bw_local_opportunity_bank.py` | P2 experiment | low | BW distillation/offline data family |
| `scripts/collect_stage_diagnostics.py` | `scripts/diagnostics/collection/collect_stage_diagnostics.py` | P1 diagnostics | low | diagnostic collection helper |
| `scripts/debug_bw_audit_subproc_snapshot.py` | `scripts/diagnostics/debug/debug_bw_audit_subproc_snapshot.py` | P1 diagnostics | low | debug helper |
| `scripts/diagnose_abcd.py` | `scripts/diagnostics/diagnose/diagnose_abcd.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_accel_candidate_workload_reward.py` | `scripts/diagnostics/diagnose/diagnose_accel_candidate_workload_reward.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_bw_actor_update_repro.py` | `scripts/diagnostics/diagnose/diagnose_bw_actor_update_repro.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_bw_best_of_n_readout.py` | `scripts/diagnostics/diagnose/diagnose_bw_best_of_n_readout.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_bw_credit_rank_corr.py` | `scripts/diagnostics/diagnose/diagnose_bw_credit_rank_corr.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_bw_old_logprob_replay.py` | `scripts/diagnostics/diagnose/diagnose_bw_old_logprob_replay.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_bw_perhead_vs_oldjoint.py` | `scripts/diagnostics/diagnose/diagnose_bw_perhead_vs_oldjoint.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_frontend_critic.py` | `scripts/diagnostics/diagnose/diagnose_frontend_critic.py` | P2 diagnostics | imported-by-1 | diagnostic script |
| `scripts/diagnose_inductor_bw_cross_compile.py` | `scripts/diagnostics/diagnose/diagnose_inductor_bw_cross_compile.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_joint_head_replay_timeline.py` | `scripts/diagnostics/diagnose/diagnose_joint_head_replay_timeline.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_native_flow_regime.py` | `scripts/diagnostics/diagnose/diagnose_native_flow_regime.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_old_bootstrap_targets.py` | `scripts/diagnostics/diagnose/diagnose_old_bootstrap_targets.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_queue_regime.py` | `scripts/diagnostics/diagnose/diagnose_queue_regime.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_reward_action_sensitivity.py` | `scripts/diagnostics/diagnose/diagnose_reward_action_sensitivity.py` | P2 diagnostics | coordinated-import-update | shared reward/action sensitivity diagnostic helpers |
| `scripts/diagnose_sat_counterfactual_step.py` | `scripts/diagnostics/diagnose/diagnose_sat_counterfactual_step.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_sat_credit_mismatch.py` | `scripts/diagnostics/diagnose/diagnose_sat_credit_mismatch.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_sat_selection_gap.py` | `scripts/diagnostics/diagnose/diagnose_sat_selection_gap.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_structured_action_effect_variance.py` | `scripts/diagnostics/diagnose/diagnose_structured_action_effect_variance.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_structured_bootstrap_targets.py` | `scripts/diagnostics/diagnose/diagnose_structured_bootstrap_targets.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_access_signal_curve.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_access_signal_curve.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_acf_deterministic_baseline.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_acf_deterministic_baseline.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_structured_bw_action_interface_leverage.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_action_interface_leverage.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_advantage_shape.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_advantage_shape.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_advantage_sources.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_advantage_sources.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_basin_gap.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_basin_gap.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_branch_linesearch.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_branch_linesearch.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_candidate_hit_rate.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_candidate_hit_rate.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_clean_candidate_sweep.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_clean_candidate_sweep.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_structured_bw_clean_gradient_conflict.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_clean_gradient_conflict.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_clean_horizon_sensitivity.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_clean_horizon_sensitivity.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_clean_plateau.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_clean_plateau.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_counterfactual_credit_probe.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_counterfactual_credit_probe.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_credit_decomposition.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_credit_decomposition.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_credit_mismatch.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_credit_mismatch.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_structured_bw_cross_follow_policy_consistency.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_cross_follow_policy_consistency.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_det_marginal_teacher.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_det_marginal_teacher.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_env_leverage.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_env_leverage.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_exploration_eval.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_exploration_eval.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_fixed_target_interpolation.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_fixed_target_interpolation.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_fixed_teacher_fit.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_fixed_teacher_fit.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_flatness_decomposition.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_flatness_decomposition.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_fused_head_compare.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_fused_head_compare.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_gap_geometry.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_gap_geometry.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_grad_split.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_grad_split.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_heuristic_perturb_impulse.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_heuristic_perturb_impulse.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_interpolation_curve.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_interpolation_curve.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_k2_alpha_curve.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_k2_alpha_curve.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_kstep_override.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_kstep_override.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_kstep_splice.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_kstep_splice.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_linear_probe.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_linear_probe.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_loc_probe.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_loc_probe.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_local_state_target_ambiguity.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_local_state_target_ambiguity.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_local_transport_search.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_local_transport_search.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_lookup_vs_shared_fit.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_lookup_vs_shared_fit.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_native_clean_fixed_fit.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_native_clean_fixed_fit.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_structured_bw_online_update_direction.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_online_update_direction.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_panel_execsources.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_panel_execsources.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_policy_gradient_alignment.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_policy_gradient_alignment.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_proxy_credit_alignment.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_proxy_credit_alignment.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_queue_reward_impulse.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_queue_reward_impulse.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_representation_probe.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_representation_probe.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_rule_probe.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_rule_probe.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_search_target_geometry.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_search_target_geometry.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_slot_order_sensitivity.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_slot_order_sensitivity.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_update.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_update.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_update_direction.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_update_direction.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_weighted_action_contrast.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_weighted_action_contrast.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_within_state_returns.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_within_state_returns.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_bw_workload_proportional_fit.py` | `scripts/diagnostics/diagnose/diagnose_structured_bw_workload_proportional_fit.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_structured_critic_alignment.py` | `scripts/diagnostics/diagnose/diagnose_structured_critic_alignment.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_critic_value_loss.py` | `scripts/diagnostics/diagnose/diagnose_structured_critic_value_loss.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_fixed_policy_rollout_variance.py` | `scripts/diagnostics/diagnose/diagnose_structured_fixed_policy_rollout_variance.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_sat_action_horizon.py` | `scripts/diagnostics/diagnose/diagnose_structured_sat_action_horizon.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_structured_sat_bw_mechanisms.py` | `scripts/diagnostics/diagnose/diagnose_structured_sat_bw_mechanisms.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_sat_local_swap_horizon.py` | `scripts/diagnostics/diagnose/diagnose_structured_sat_local_swap_horizon.py` | P2 diagnostics | fix-ROOT-bootstrap | diagnostic script |
| `scripts/diagnose_structured_step_bootstrap_bias.py` | `scripts/diagnostics/diagnose/diagnose_structured_step_bootstrap_bias.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_step_targets.py` | `scripts/diagnostics/diagnose/diagnose_structured_step_targets.py` | P2 diagnostics | low | diagnostic script |
| `scripts/diagnose_structured_target_decomposition.py` | `scripts/diagnostics/diagnose/diagnose_structured_target_decomposition.py` | P2 diagnostics | low | diagnostic script |
| `scripts/distill_bw_select_v1.py` | `scripts/experiments/bw_distill/distill_bw_select_v1.py` | P2 experiment | low | BW distillation/offline data family |
| `scripts/distill_bw_winner_bank_v0.py` | `scripts/experiments/bw_distill/distill_bw_winner_bank_v0.py` | P2 experiment | low | BW distillation/offline data family |
| `scripts/estimate_throughput.py` | `scripts/analysis/estimate_throughput.py` | P1 analysis | low | analysis/report helper |
| `scripts/eval_fixed_frontsat_bw_variants.py` | `scripts/evaluation/eval_fixed_frontsat_bw_variants.py` | P2 evaluation | low | evaluation helper or sweep |
| `scripts/eval_param_sweep.py` | `scripts/evaluation/eval_param_sweep.py` | P2 evaluation | low | evaluation helper or sweep |
| `scripts/evaluate.py` | `scripts/legacy/pettingzoo/evaluate.py` | P2 legacy | update-docs | old flat/PettingZoo workflow |
| `scripts/evaluate_action_head_ablation_best.py` | `scripts/evaluation/evaluate_action_head_ablation_best.py` | P2 evaluation | low | evaluation helper or sweep |
| `scripts/evaluate_joint_head_timeline.py` | `scripts/evaluation/evaluate_joint_head_timeline.py` | P2 evaluation | fix-ROOT-bootstrap | evaluation helper or sweep |
| `scripts/evaluate_partner_swap_matrix.py` | `scripts/evaluation/sat/evaluate_partner_swap_matrix.py` | P2 evaluation | low | SAT/partner evaluation helper |
| `scripts/evaluate_sat_slot_permutation.py` | `scripts/evaluation/sat/evaluate_sat_slot_permutation.py` | P2 evaluation | fix-ROOT-bootstrap | SAT/partner evaluation helper |
| `scripts/evaluate_structured.py` | `scripts/legacy/structured/evaluate_structured.py` | P2 legacy | update-docs | old structured/stage workflow; docs still reference it |
| `scripts/evaluate_structured_access_control_fixedrule.py` | `scripts/experiments/access_control/evaluate_structured_access_control_fixedrule.py` | P2 experiment | low | access-control experiment family |
| `scripts/evaluate_structured_access_control_learned.py` | `scripts/experiments/access_control/evaluate_structured_access_control_learned.py` | P2 experiment | low | access-control experiment family |
| `scripts/evaluate_structured_access_control_oracle.py` | `scripts/experiments/access_control/evaluate_structured_access_control_oracle.py` | P2 experiment | low | access-control experiment family |
| `scripts/evaluate_structured_bw_select.py` | `scripts/experiments/bw_training/evaluate_structured_bw_select.py` | P2 experiment | low | BW experimental training/eval |
| `scripts/evaluate_structured_fixed_policy.py` | `scripts/evaluation/evaluate_structured_fixed_policy.py` | P2 evaluation | low | evaluation helper or sweep |
| `scripts/evaluate_structured_hybrid_heads.py` | `scripts/evaluation/evaluate_structured_hybrid_heads.py` | P2 evaluation | low | evaluation helper or sweep |
| `scripts/evaluate_structured_mixed_heads_native.py` | `scripts/evaluate_structured_mixed_heads_native.py` | P0 keep-root | low-now; high-if-moved | current entrypoint or imported helper; do not move in first pass |
| `scripts/evaluate_thesis_native_methods.py` | `scripts/evaluation/evaluate_thesis_native_methods.py` | P2 evaluation | imported-by-1, update-docs | evaluation helper or sweep |
| `scripts/export_debug_episode.py` | `scripts/analysis/export/export_debug_episode.py` | P1 analysis | low | export/report generation |
| `scripts/export_tb_scalars.py` | `scripts/analysis/export/export_tb_scalars.py` | P1 analysis | update-docs | export/report generation |
| `scripts/generate_joint_mcgae_training_ppt.py` | `scripts/analysis/export/generate_joint_mcgae_training_ppt.py` | P1 analysis | update-docs | export/report generation |
| `scripts/offline_validate_sat_local_listwise.py` | `scripts/experiments/sat/offline_validate_sat_local_listwise.py` | P2 experiment | fix-ROOT-bootstrap | SAT offline/supervised experiment |
| `scripts/plot_action_head_ablation_preview.py` | `scripts/analysis/plots/plot_action_head_ablation_preview.py` | P1 analysis | low | plot generation |
| `scripts/plot_danger_imitation_ablation_preview.py` | `scripts/analysis/plots/plot_danger_imitation_ablation_preview.py` | P1 analysis | low | plot generation |
| `scripts/plot_learning_curve.py` | `scripts/analysis/plots/plot_learning_curve.py` | P1 analysis | low | plot generation |
| `scripts/plot_recomputed_positive_reward_curve.py` | `scripts/analysis/plots/plot_recomputed_positive_reward_curve.py` | P1 analysis | low | plot generation |
| `scripts/plot_reward_curve_from_eval_ratio.py` | `scripts/analysis/plots/plot_reward_curve_from_eval_ratio.py` | P1 analysis | low | plot generation |
| `scripts/plot_typical_uav_trajectory_preview.py` | `scripts/analysis/plots/plot_typical_uav_trajectory_preview.py` | P1 analysis | low | plot generation |
| `scripts/plot_uav_queue_imbalance_preview.py` | `scripts/analysis/plots/plot_uav_queue_imbalance_preview.py` | P1 analysis | low | plot generation |
| `scripts/prepare_structured_bw_reset_init.py` | `scripts/experiments/bw_tools/prepare_structured_bw_reset_init.py` | P2 experiment | low | BW search/solver/tooling |
| `scripts/probe_structured_bw_value_generalization.py` | `scripts/diagnostics/probe/probe_structured_bw_value_generalization.py` | P1 diagnostics | fix-ROOT-bootstrap | probe script |
| `scripts/probe_structured_sat_state.py` | `scripts/diagnostics/probe/probe_structured_sat_state.py` | P1 diagnostics | low | probe script |
| `scripts/profile_joint_critic_step.py` | `scripts/benchmarks/profile_joint_critic_step.py` | P1/P2 benchmark | fix-ROOT-bootstrap | performance benchmark or profiler |
| `scripts/profile_native_rollout_step_segments.py` | `scripts/benchmarks/profile_native_rollout_step_segments.py` | P1/P2 benchmark | low | performance benchmark or profiler |
| `scripts/render_episode.py` | `scripts/legacy/pettingzoo/render_episode.py` | P2 legacy | update-docs | old flat/PettingZoo workflow |
| `scripts/render_structured_episode.py` | `scripts/render_structured_episode.py` | P0 keep-root | low-now; high-if-moved | current entrypoint or imported helper; do not move in first pass |
| `scripts/run_autodl_cuda_smoke.sh` | `scripts/run_autodl_cuda_smoke.sh` | P0 keep-root | low-now; high-if-moved | current entrypoint or imported helper; do not move in first pass |
| `scripts/run_curriculum_stage123_formal.ps1` | `scripts/legacy/runners/run_curriculum_stage123_formal.ps1` | P2 legacy | low | old Windows runner script |
| `scripts/run_mac_joint_smoke.sh` | `scripts/run_mac_joint_smoke.sh` | P0 keep-root | low-now; high-if-moved | current entrypoint or imported helper; do not move in first pass |
| `scripts/run_structured_bw_fourway_ablation.ps1` | `scripts/legacy/runners/run_structured_bw_fourway_ablation.ps1` | P2 legacy | low | old Windows runner script |
| `scripts/sat_supervised_probe.py` | `scripts/experiments/sat/sat_supervised_probe.py` | P2 experiment | low | SAT offline/supervised experiment |
| `scripts/search_bw_sanity_gap.py` | `scripts/experiments/bw_tools/search_bw_sanity_gap.py` | P2 experiment | low | BW search/solver/tooling |
| `scripts/search_structured_bw_reward_alignment.py` | `scripts/experiments/bw_tools/search_structured_bw_reward_alignment.py` | P2 experiment | low | BW search/solver/tooling |
| `scripts/solve_bw_1uav2gu_flow_bcd.py` | `scripts/experiments/bw_tools/solve_bw_1uav2gu_flow_bcd.py` | P2 experiment | low | BW search/solver/tooling |
| `scripts/solve_bw_flow_bcd_general.py` | `scripts/experiments/bw_tools/solve_bw_flow_bcd_general.py` | P2 experiment | low | BW search/solver/tooling |
| `scripts/summarize_policy_kpi.py` | `scripts/analysis/summarize_policy_kpi.py` | P1 analysis | low | analysis/report helper |
| `scripts/train.py` | `scripts/legacy/pettingzoo/train.py` | P2 legacy | update-docs | old flat/PettingZoo workflow |
| `scripts/train_accel_python_simple_ppo.py` | `scripts/experiments/training/train_accel_python_simple_ppo.py` | P2 experiment | fix-ROOT-bootstrap | training experiment |
| `scripts/train_access_control_imitation.py` | `scripts/experiments/access_control/train_access_control_imitation.py` | P2 experiment | low | access-control experiment family |
| `scripts/train_bw_panel_advantage_v0.py` | `scripts/experiments/bw_training/train_bw_panel_advantage_v0.py` | P2 experiment | low | BW experimental training/eval |
| `scripts/train_joint_mcgae.py` | `scripts/train_joint_mcgae.py` | P0 keep-root | low-now; high-if-moved, imported-by-5 | current entrypoint or imported helper; do not move in first pass |
| `scripts/train_sat_mcgae.py` | `scripts/legacy/structured/train_sat_mcgae.py` | P2 legacy | fix-ROOT-bootstrap, update-docs | old structured/stage workflow; docs still reference it |
| `scripts/train_stage_mcgae.py` | `scripts/train_stage_mcgae.py` | P0 keep-root | low-now; high-if-moved, imported-by-6 | current entrypoint or imported helper; do not move in first pass |
| `scripts/train_structured.py` | `scripts/legacy/structured/train_structured.py` | P2 legacy | update-docs | old structured/stage workflow; docs still reference it |
| `scripts/train_structured_bw_actoronly_debug.py` | `scripts/experiments/bw_training/train_structured_bw_actoronly_debug.py` | P2 experiment | low | BW experimental training/eval |
| `scripts/train_structured_bw_imitation_sanity.py` | `scripts/experiments/bw_training/train_structured_bw_imitation_sanity.py` | P2 experiment | low | BW experimental training/eval |
| `scripts/train_structured_bw_search_distill_diagnostic.py` | `scripts/experiments/bw_training/train_structured_bw_search_distill_diagnostic.py` | P2 experiment | low | BW experimental training/eval |
| `scripts/train_structured_bw_sequence_distill_diagnostic.py` | `scripts/experiments/bw_training/train_structured_bw_sequence_distill_diagnostic.py` | P2 experiment | low | BW experimental training/eval |
| `scripts/validate_structured_long_rollout_acceptance.py` | `scripts/diagnostics/validation/validate_structured_long_rollout_acceptance.py` | P1 diagnostics | low | validation helper |
