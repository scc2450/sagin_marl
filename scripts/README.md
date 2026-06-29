# Scripts Directory Guide

This directory contains training, evaluation, rendering, diagnostics, analysis, benchmark, and historical helper scripts. The top-level is intentionally kept small: current entrypoints and a few legacy entrypoints remain at root, while most specialized scripts live in themed subdirectories.

## Start Here

Current mainline scripts:

```text
scripts/train_joint_mcgae.py
scripts/evaluate_structured_mixed_heads_native.py
scripts/render_structured_episode.py
```

Current mainline config:

```text
configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
```

Use the current line for native CUDA structured environment runs with joint MC-GAE critic fitting and accel / SAT / BW staged actor updates. The older `train.py`, `evaluate.py`, `train_structured.py`, `evaluate_structured.py`, and `render_episode.py` files are kept only for legacy compatibility.

## Top-Level Files

| File | Role | Current status |
|---|---|---|
| `train_joint_mcgae.py` | Current formal training entrypoint for joint MC-GAE. | Use for current training. |
| `train_stage_mcgae.py` | Shared stage MC-GAE training helpers and older stage entrypoint. | Keep at root because it is imported by current and diagnostic scripts. |
| `evaluate_structured_mixed_heads_native.py` | Current native deterministic evaluation entrypoint, including mixed-head/stage-best evaluation. | Use for current checkpoint evaluation. |
| `render_structured_episode.py` | Current structured policy rendering entrypoint. | Use for rendering current structured checkpoints. |
| `run_autodl_cuda_smoke.sh` | CUDA smoke runner for AutoDL-style GPU machines. | Use after remote setup or CUDA refactors. |
| `run_mac_joint_smoke.sh` | Mac/local smoke runner for import/config wiring. | Useful for CPU-side wiring checks; full native training still needs CUDA. |
| `train_sat_mcgae.py` | Older SAT MC-GAE training entrypoint. | Legacy/stage-specific; verify before using. |
| `train_structured.py` | Older structured training flow. | Legacy. Not the current paper-result path. |
| `evaluate_structured.py` | Older structured evaluation flow. | Legacy. Prefer `evaluate_structured_mixed_heads_native.py` for current results. |
| `train.py` | Older flat/PettingZoo MAPPO training flow. | Legacy. |
| `evaluate.py` | Older flat/PettingZoo evaluation flow. | Legacy. |
| `render_episode.py` | Older flat/PettingZoo render flow. | Legacy. Do not use for current `final.pt` structured checkpoints. |

## Directory Map

| Directory | Contents | When to use | Watch-outs |
|---|---|---|---|
| `analysis/` | Metric analysis, KPI summaries, reward-scale inspection, fairness analysis, plot/export helpers. | Post-processing runs and preparing figures/tables/slides. | These scripts usually assume existing run artifacts. |
| `benchmarks/` | Runtime and profiling scripts. | Measuring native kernels, training system speed, critic step cost, rollout segment timing. | Hardware and first-update compile effects can dominate results. |
| `diagnostics/` | Audits, probes, one-off diagnoses, validation checks, and debug scripts. | Investigating training failure modes or validating internal assumptions. | Many scripts are historical and may require specific run artifacts. |
| `evaluation/` | Evaluation helpers and method-comparison scripts beyond the main native evaluator. | Thesis/baseline comparison, parameter sweeps, fixed-policy evals, SAT-specific evaluation. | Current formal evaluation still starts from top-level `evaluate_structured_mixed_heads_native.py`. |
| `experiments/` | Non-mainline training/eval experiments grouped by topic. | Access-control, BW distillation/training/tools, SAT offline validation, and simple training experiments. | Experimental scripts may encode old assumptions; read their CLI help first. |
| `legacy/` | Historical runner scripts moved out of top-level. | Reproducing old Windows/PowerShell command lines. | Not current. Prefer current root scripts unless deliberately reproducing history. |

## Analysis

### `analysis/`

General analysis and summarization utilities:

```text
analyze_bw_update_hot_cold_direction.py
analyze_metrics.py
analyze_reward_scales.py
analyze_thesis_fairness.py
estimate_throughput.py
summarize_policy_kpi.py
```

Use these when examining completed runs rather than launching training. `analyze_thesis_fairness.py` is the thesis fairness/metric helper and imports the thesis evaluation utilities from `scripts/evaluation/`.

### `analysis/phase2/`

Phase 2 degradation-analysis summarizers:

```text
summarize_h2_probe.py
summarize_matched_step.py
```

Use these to summarize the BW branch-alignment / H2 probe and matched-step calibration run folders produced by `scripts/experiments/phase2/`.

### `analysis/export/`

Export and presentation helpers:

```text
backfill_structured_tb.py
export_debug_episode.py
export_tb_scalars.py
generate_joint_mcgae_training_ppt.py
```

Use these to export TensorBoard scalars, debug episodes, or presentation material from run artifacts.

### `analysis/plots/`

Plot generators for known paper/report figures:

```text
plot_action_head_ablation_preview.py
plot_danger_imitation_ablation_preview.py
plot_learning_curve.py
plot_recomputed_positive_reward_curve.py
plot_reward_curve_from_eval_ratio.py
plot_typical_uav_trajectory_preview.py
plot_uav_queue_imbalance_preview.py
```

These are mostly report/preview scripts. Check their arguments and expected input files before running.

## Benchmarks

### `benchmarks/`

Performance scripts:

```text
bench_structured_native_kernels.py
bench_structured_training_system.py
bench_train.py
profile_joint_critic_step.py
profile_native_rollout_step_segments.py
```

Use these for timing investigations. For stable comparisons, avoid drawing conclusions from cold-start compile/cache behavior; steady-state updates are more reliable.

## Diagnostics

### `diagnostics/audit/`

Audit scripts examine whether training mechanics, targets, critic values, PPO updates, or stage credit assignment match expectations. This includes stage-credit audits, critic artifact evaluation, BW real-update alignment, SAT local-world probes, and fixed-critic benchmarks.

Important family examples:

```text
audit_stage_ppo_credit_alignment.py
audit_stage_critic_only_fit.py
audit_stage_credit_chain.py
audit_bw_broad2local_offline.py
audit_bw_real_update_alignment.py
audit_fixed_critic_benchmark.py
```

Several downstream scripts import helpers from this folder. Treat it as a diagnostic helper library plus executable audit scripts.

### `diagnostics/diagnose/`

One-off or focused diagnostic entrypoints. This is the largest diagnostic area and includes many BW, SAT, critic, target, and reward-action investigations.

Common name patterns:

- `diagnose_bw_*`: BW-specific actor/update/credit diagnostics.
- `diagnose_sat_*`: SAT selection, credit, and counterfactual diagnostics.
- `diagnose_structured_bw_*`: structured BW geometry, target, reward, policy-gradient, and representation diagnostics.
- `diagnose_structured_critic_*`: critic value/alignment diagnostics.
- `diagnose_*target*` or `diagnose_*bootstrap*`: return-target and bootstrap behavior checks.

Use this folder when a training curve looks wrong and you need a targeted mechanism check.

### `diagnostics/probe/`

Lightweight probes:

```text
probe_structured_bw_value_generalization.py
probe_structured_sat_state.py
```

Use these for narrow questions about state representation or value generalization.

### `diagnostics/collection/`

Collection helper:

```text
collect_stage_diagnostics.py
```

Use this to gather stage diagnostics from runs.

### `diagnostics/debug/`

Debug helper:

```text
debug_bw_audit_subproc_snapshot.py
```

Use this when debugging subprocess/snapshot behavior in BW audit flows.

### `diagnostics/validation/`

Validation helper:

```text
validate_structured_long_rollout_acceptance.py
```

Use this for acceptance-style checks around long structured rollouts.

## Evaluation

### `evaluation/`

Evaluation helpers that are not the current root native evaluator:

```text
eval_fixed_frontsat_bw_variants.py
eval_param_sweep.py
evaluate_action_head_ablation_best.py
evaluate_structured_fixed_policy.py
evaluate_structured_hybrid_heads.py
evaluate_thesis_native_methods.py
```

`evaluate_thesis_native_methods.py` is the thesis/baseline comparison helper. It evaluates the proposed checkpoint, simple rule baselines, `maxweight_lyapunov`, the lightweight DPP/MaxWeight ablations, `topology_dpp`, `dpp_resource_hybrid`, and the native staged/joint variants `dpp_resource_hybrid_native`, `topology_dpp_native_bw_sat_cached`, and `full_topology_dpp_joint`. For current one-off deterministic evaluation, still prefer the top-level `evaluate_structured_mixed_heads_native.py`; `topology_dpp` and legacy `dpp_resource_hybrid` use structured Python fallback paths, while the native variants use source triples around `dpp_resource_bw`.

### `evaluation/sat/`

SAT/partner-selection evaluation helpers:

```text
evaluate_joint_head_timeline.py
evaluate_partner_swap_matrix.py
evaluate_sat_slot_permutation.py
```

`evaluate_joint_head_timeline.py` dynamically loads `evaluate_partner_swap_matrix.py`, so these files intentionally live together.

## Experiments

### `experiments/access_control/`

Access-control imitation and evaluation family:

```text
access_control_imitation_common.py
train_access_control_imitation.py
evaluate_structured_access_control_fixedrule.py
evaluate_structured_access_control_learned.py
evaluate_structured_access_control_oracle.py
```

Use this for access-bid/access-control experiments outside the current formal joint MC-GAE training path.

### `experiments/phase2/`

Phase 2 late-degradation experiment drivers:

```text
run_4090_bw_matrix.sh
run_4090_h2_probe.sh
run_4090_matched_step.sh
```

Use these on 4090-class CUDA machines to reproduce the BW continuation matrix, H2 branch-alignment probe, and matched-step calibration experiments. These are experiment drivers, not current mainline training entrypoints; they expect existing Phase 2 checkpoints under `runs/phase2/`.

### `experiments/bw_distill/`

BW distillation and offline data scripts:

```text
check_bw_distill_reload.py
collect_bw_local_opportunity_bank.py
distill_bw_select_v1.py
distill_bw_winner_bank_v0.py
```

Use this folder for local-opportunity/winner-bank collection and student distillation workflows.

### `experiments/bw_training/`

BW experimental training and evaluation scripts:

```text
evaluate_structured_bw_select.py
train_bw_panel_advantage_v0.py
train_structured_bw_actoronly_debug.py
train_structured_bw_imitation_sanity.py
train_structured_bw_search_distill_diagnostic.py
train_structured_bw_sequence_distill_diagnostic.py
```

These are BW-focused experimental paths, not the current full joint training entrypoint.

### `experiments/bw_tools/`

BW search, solver, and reset-init tools:

```text
prepare_structured_bw_reset_init.py
search_bw_sanity_gap.py
search_structured_bw_reward_alignment.py
solve_bw_1uav2gu_flow_bcd.py
solve_bw_flow_bcd_general.py
```

Use these when generating BW oracle/solver targets or searching reward-alignment/sanity gaps.

### `experiments/sat/`

SAT offline/supervised experiments:

```text
offline_validate_sat_local_listwise.py
sat_supervised_probe.py
```

Use this when validating SAT local listwise behavior or supervised SAT probes.

### `experiments/training/`

Single-purpose training experiments:

```text
train_accel_python_simple_ppo.py
```

This is an experimental accel-only/simple-PPO path. It is not the current native joint MC-GAE mainline.

## Legacy

### `legacy/runners/`

Historical PowerShell runners:

```text
run_curriculum_stage123_formal.ps1
run_structured_bw_fourway_ablation.ps1
```

These are preserved for reproducibility and command archaeology. They are not recommended starting points for new runs.

## Choosing the Right Script

Use this quick routing table:

| Goal | Start with |
|---|---|
| Current formal training | `scripts/train_joint_mcgae.py` |
| Current formal native eval | `scripts/evaluate_structured_mixed_heads_native.py` |
| Render current structured checkpoint | `scripts/render_structured_episode.py` |
| Check local/Mac import wiring | `scripts/run_mac_joint_smoke.sh` |
| Check remote CUDA setup | `scripts/run_autodl_cuda_smoke.sh` |
| Thesis/baseline comparison | `scripts/evaluation/evaluate_thesis_native_methods.py` |
| Fairness/KPI analysis | `scripts/analysis/analyze_thesis_fairness.py` or `scripts/analysis/summarize_policy_kpi.py` |
| Runtime profiling | `scripts/benchmarks/` |
| Debug critic/advantage/BW/SAT mechanism | `scripts/diagnostics/` |
| Phase 2 degradation experiments | `scripts/experiments/phase2/` plus `scripts/analysis/phase2/` |
| Access-control experiments | `scripts/experiments/access_control/` |
| BW distillation or BW-only experiments | `scripts/experiments/bw_*` |
| SAT offline/supervised checks | `scripts/experiments/sat/` or `scripts/evaluation/sat/` |
| Historical PettingZoo/old structured flows | top-level legacy files or `scripts/legacy/` |

## Adding New Scripts

Use this placement rule:

- Current mainline entrypoint: only keep at top-level if it is part of the primary user workflow.
- Analysis or plotting: `scripts/analysis/` or `scripts/analysis/plots/`.
- Phase-specific analysis helpers: `scripts/analysis/<phase-or-topic>/`.
- Export/report generation: `scripts/analysis/export/`.
- Benchmark/profiling: `scripts/benchmarks/`.
- Mechanism audit or diagnosis: `scripts/diagnostics/audit/`, `scripts/diagnostics/diagnose/`, or `scripts/diagnostics/probe/`.
- Evaluation helper: `scripts/evaluation/`, with SAT-specific helpers in `scripts/evaluation/sat/`.
- Non-mainline experiment: `scripts/experiments/<topic>/`.
- Historical runner or old workflow: `scripts/legacy/`.

Before moving a script, check for direct imports and hard-coded sibling paths. Many older scripts need a repo-root bootstrap based on `Path(__file__).resolve().parents` rather than fixed parent indexes.
