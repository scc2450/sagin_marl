# Configs Directory Guide

This directory contains YAML experiment configurations. Treat `configs/current/` as the source of truth for the current training line; most other folders are smoke tests, profiling configs, ablations, comparisons, or archived historical experiments.

## Start Here

Current mainline config:

```text
configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
```

This is the current native CUDA structured environment + joint MC-GAE + relational critic configuration for the 3-UAV / 20-GU / 250-step setting.

The matching main entrypoints are:

```text
scripts/train_joint_mcgae.py
scripts/evaluate_structured_mixed_heads_native.py
scripts/render_structured_episode.py
```

Important behavior to remember:

- `train_joint_mcgae.py` enforces joint training semantics for accel / sat / bw.
- `access_bw_decision_interval` is usually passed from the command line for the current formal runs; the current formal BW macro interval is `K=5`.
- `sat_decision_interval` is normally kept at `1` for the current line.
- Smoke or archive configs may load correctly but should not be assumed to represent the current paper/result path.

## Directory Map

| Directory | Contents | When to use | Watch-outs |
|---|---|---|---|
| `current/` | Current formal config. | Default starting point for training, evaluation, rendering, and new controlled variants. | Keep this small. Do not dump exploratory variants here. |
| `smoke/` | Minimal configs for import, compile, CUDA/native, and short pipeline checks. | Fast sanity checks after refactors or environment changes. | Passing smoke tests does not prove full training quality. |
| `perf/` | Speed probe, compile, CUDA graph, rollout metrics, and workload profiling configs. | Timing or bottleneck investigations. | These are not training-quality configs. Compare only with matching hardware/settings. |
| `stage_sanity/` | Per-stage sanity configs for accel, sat, and bw. | Isolating one stage before debugging joint behavior. | Many are intentionally simplified; do not mix them with formal joint results. |
| `comparison/` | Reference and baseline-comparison configs. | Running method comparisons against reference policies or sampling variants. | Keep comparison protocol aligned with evaluation scripts. |
| `ablations/` | Capacity and optimizer-strength ablation configs. | Controlled experiments on actor/critic capacity or PPO strength. | These should branch from a known baseline; record the base run in notes. |
| `variants/` | Temporary or near-current variants. | Short-lived edits close to current mainline. | Promote to `current/` only after the variant becomes the new mainline. |
| `archive/` | Historical configs from earlier phases and diagnostic campaigns. | Reproduction, archaeology, or understanding why a design changed. | Default assumption: not current. Verify script compatibility before running. |

## Current Config

### `current/`

Contains the current formal training config:

```text
structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
```

This is the config to reach for first when you want to:

- launch the current formal training path;
- reproduce the main 3-UAV / 20-GU / 250-step joint setup;
- derive a small controlled variant for an ablation;
- evaluate or render checkpoints trained with the current structured-native path.

If you create a new official config, prefer adding one clearly named file here only when it replaces or intentionally extends the current line. Otherwise use `variants/`, `ablations/`, `smoke/`, or `perf/`.

## Fast Checks

### `smoke/`

Contains tiny or reduced configs for quick validation:

```text
structured_actor_stage_width_query_smoke.yaml
structured_auto_min0_smoke.yaml
structured_bw_sanity_1uav_static_gap_debug_envreward_smoke.yaml
structured_compile_smoke.yaml
```

Use these when checking whether code still imports, compiles, or reaches the native structured path. These configs are useful after file moves, CUDA extension edits, actor/critic shape edits, or environment setup changes.

### `perf/`

Contains performance and profiling configs:

```text
structured_speed_probe_auto_expanded.yaml
structured_speed_probe_auto_expanded_nowarm.yaml
structured_speed_probe_auto_metrics.yaml
structured_speed_probe_auto_rollout_metrics.yaml
structured_speed_probe_auto_rollout_workload.yaml
structured_speed_probe_compile.yaml
structured_speed_probe_cudagraph_prepare.yaml
structured_speed_probe_cudagraphs.yaml
structured_speed_probe_no_actor_compile.yaml
```

Use these for runtime and compile investigations. They are meant to isolate timing behavior such as rollout speed, actor compilation, CUDA graph preparation, and metric collection overhead.

## Stage-Specific Sanity

### `stage_sanity/accel/`

Accel-only or accel-focused sanity configs. These are useful when checking whether the movement/acceleration actor has a learnable signal before involving SAT and BW.

Notable patterns:

- `structured_single_accel_*`: single-stage accel PPO line.
- `structured_accel_sanity_*positive*`: positive reward / current-env sanity variants.
- `*_criticwarmup_*` and `*_timebootstrap*`: historical checks around critic warmup or target construction.

### `stage_sanity/bw/`

BW-only sanity configs. Currently this folder contains the single-BW PPO sanity path:

```text
structured_single_bw_3uav_20gu_t250_ppo.yaml
```

Use this only when isolating bandwidth allocation behavior outside the full joint MC-GAE loop.

### `stage_sanity/sat/`

SAT-only or SAT-focused sanity configs:

```text
structured_sat_mcgae_3uav_20gu_t250_positive_relcritic.yaml
structured_single_sat_3uav_20gu_t250_ppo.yaml
```

Use these when debugging satellite/partner-selection behavior independently from full joint actor updates.

## Comparisons

### `comparison/ref/`

Reference comparison configs for structured joint runs and sampling variants:

```text
structured_joint_vs_ref_3uav_20gu_t250.yaml
structured_joint_vs_ref_3uav_20gu_t250_active_sampling.yaml
structured_joint_vs_ref_3uav_20gu_t250_uniform_sampling.yaml
```

### `comparison/ref/joint_accelppo_satbwvsref/`

Configs for comparing accel-PPO with SAT/BW reference behavior:

```text
structured_joint_accelppo_satbwvsref_active_mixture_rows16_std02.yaml
structured_joint_accelppo_satbwvsref_uniform_rows16_std02.yaml
```

These are protocol/configuration comparison tools, not the default training line.

## Ablations

### `ablations/opt_strength/`

Optimizer/PPO-strength ablations:

```text
baseline.yaml
actor_lr_6e4.yaml
lr_decay_off.yaml
lr_final_factor_05.yaml
ppo_epochs_5.yaml
```

Use these to isolate optimizer schedule and PPO update strength effects.

### `ablations/capacity/accel_actor/`

Actor-capacity ablations for the accel actor. File names use `A*` IDs to identify controlled changes such as embedding size, hidden size, attention heads, encoder depth, context depth, query width, and danger/std settings.

Examples:

```text
structured_accel_3uav20gu_A0_baseline.yaml
structured_accel_3uav20gu_A3_embed256.yaml
structured_accel_3uav20gu_A4_hidden512.yaml
structured_accel_3uav20gu_A10_interaction1_heads8.yaml
```

### `ablations/capacity/critic_accel_only/`

Critic-capacity ablations for accel-only critic experiments. File names use `C*` IDs for embedding, hidden, encoder, message MLP, message layers, and value-head changes.

Examples:

```text
structured_accel_3uav20gu_C0_baseline.yaml
structured_accel_3uav20gu_C2_embed256.yaml
structured_accel_3uav20gu_C4_valuehead512.yaml
structured_accel_3uav20gu_C8_value_head_layers3.yaml
```

## Variants

### `variants/`

Short-lived variants near the current mainline. Currently includes:

```text
structured_joint_mcgae_3uav_20gu_t250_positive_relcritic_lyap218d_tmp.yaml
```

Use this folder when a config is too close to current to be archive material, but not stable enough to become the official config.

## Archive

### `archive/`

Historical configs are grouped by experiment campaign. Use these for reproduction or context, not as defaults.

| Directory | Meaning |
|---|---|
| `bw_geom_live_phase1/` | Earlier BW geometry/live phase-1 variants, including Dirichlet/simplex-dimension tests. |
| `bw_sanity/` | Large BW diagnostic and sanity-search history, including static-gap, stronggap, current-schema, and teacher/probe variants. |
| `clean_per_user/` | Clean BW per-user experiments, mostly Beijing/resolution/reward-observation variants. |
| `clean_sat/` | Clean SAT/joint SAT comparison configs. |
| `phase1_joint/` | Older phase-1 joint action curriculum experiments and many structured/joint diagnostic variants. |
| `phase1_misc/` | Miscellaneous phase-1 action/effect tests. |
| `phase1_queuefix/` | Earlier queue-fix curriculum configs. |
| `phase1_stage1_accel/` | Historical stage-1 accel curriculum and safety/reward-shaping variants. |
| `phase1_stage2_bw/` | Historical stage-2 BW curriculum variants. |
| `phase1_stage3_sat/` | Historical stage-3 SAT curriculum variants. |
| `stage1_safety_followup/` | Follow-up configs for stage-1 safety/prealert/boundary behavior. |
| `structured_joint_dirichlet/` | Earlier structured-joint Dirichlet/simplex-dimension variants. |

Archive configs often depend on older training assumptions, older reward shaping, or older script paths. Before running one, check the intended script and compare key fields against `configs/current/`.

## Naming Conventions

Common name fragments:

- `structured`: uses the structured actor/critic/environment path rather than the older flat PettingZoo path.
- `joint`: trains or evaluates multiple stages together.
- `mcgae`: critic target uses finite-horizon MC return before GAE-style actor advantage computation.
- `3uav_20gu_t250`: 3 UAVs, 20 ground users, 250 environment steps per episode/rollout horizon.
- `positive_relcritic`: positive reward framing with relational critic.
- `ppo`: PPO-style actor update.
- `currentenv` or `current_schema`: intended to match a newer environment/schema at the time the file was created; still verify against `current/`.
- `smoke`, `probe`, `diagnose`, `sanity`: validation/debug configs, not final-result configs.

## Adding New Configs

Use this placement rule:

- New formal mainline: `configs/current/`.
- Small local experiment close to current: `configs/variants/`.
- Controlled ablation: `configs/ablations/<topic>/`.
- One-stage sanity check: `configs/stage_sanity/<stage>/`.
- Speed/compile/profiling work: `configs/perf/`.
- Smoke/import/CUDA check: `configs/smoke/`.
- Historical result preservation: `configs/archive/<campaign>/`.

When adding a config, prefer a descriptive filename that includes the task, scale, horizon, and intended method. If it is derived from another config, record the base config in the experiment notes or commit message.
