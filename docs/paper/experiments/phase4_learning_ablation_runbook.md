# Phase 4 Learning Ablation Runbook

Date: 2026-07-11

This runbook tracks two learned Section 5 comparison families:

> critic-structure ablations under the same staged actor, and a MAPPO-like
> flat actor/flat centralized critic learning baseline.

The purpose is to test whether the relational centralized critic improves value
estimation and training quality. `global_only` is an internal critic ablation.
`flat_mlp` is a stronger critic-side baseline than `global_only`: it keeps the
same staged actor and legal-action adapter, but replaces the topology-aware
critic with a flat centralized MLP value function. It is not a complete MAPPO
baseline because the actor is still the structured staged actor.

The MAPPO-like baseline is separate: it sets `structured_actor_backbone:
flat_mlp` and `critic_value_mode: flat_mlp`. It keeps the same hybrid/masked
environment action interface, reward, safety handling, and PPO/GAE loop, but
removes the actor-side token attention, subset scoring, and BW competition
modules. This should be described as a MAPPO-like adapter baseline, not as an
exact reproduction of an external MAPPO implementation.

## Branch

Implementation branch:

```bash
git switch erik/phase4-learning-ablation-baseline
```

## Canonical Status

Last consolidated: 2026-08-11.

This file is the canonical Phase 4 / Section 5 experiment runbook. Older
dated notes about formal evaluation status, figure design, parameter sweeps,
STARS-GC stability, and Section 5 claim placement have been absorbed here to
avoid maintaining several nearly-overlapping paper guidance files.

Keep:

- source tables under `docs/paper/table_sources/`;
- reproducible figure sources under `docs/paper/figure_sources/`;
- manuscript-facing figure copies under `docs/paper/manuscript/figures/`;
- generation and aggregation scripts under `docs/paper/experiments/` and
  `scripts/analysis/phase4/`.

Do not create a new dated markdown note for every small decision. Add short
updates to this runbook, or update `docs/paper/evidence_index.md` when a claim
becomes manuscript-facing.

Absorbed notes:

- `phase4_formal_eval_status_20260712.md`;
- `phase4_formal_evaluation_matrix_20260712.md`;
- `section5_performance_assets_20260713.md`;
- `section5_table_figure_claim_plan_20260713.md`;
- `performance_evaluation_figure_draft_20260714.md`;
- `performance_figure_design_guidelines_20260714.md`;
- `phase4_formal_parameter_sweep_status_20260714.md`;
- `phase4_parameter_sweep_figure_notes_20260714.md`;
- `phase4_stars_gc_stability_10seeds_status_20260714.md`;
- `phase4_baseline_consolidation_review_20260811.md`.

## Satellite-Control Consistency Audit

Audit date: 2026-08-11.

Issue: several active learned configs inherited `fixed_satellite_strategy: true`
while also setting `train_sat: true` and `exec_sat_source: policy`. In the
Python structured driver, `fixed_satellite_strategy=true` overrides the
satellite action and chooses the nearest visible satellite, so this combination
is not semantically valid for learned satellite selection.

Resolution:

- active learned configs under `configs/current`, `configs/experiments`,
  `configs/comparison/ref`, `configs/stage_sanity/sat`, `configs/smoke`, and
  `configs/variants` were normalized to `fixed_satellite_strategy: false`
  whenever satellite execution is policy-controlled;
- `train_joint_mcgae.py` and `evaluate_structured_actor_exec_sources` now reject
  `fixed_satellite_strategy=true` together with learned/policy satellite
  control;
- `tests/test_config_parsing.py` includes an active-config invariant so this
  conflict cannot be reintroduced outside `configs/archive`.

Impact on existing Phase 4 formal results: the old field value is a real config
hygiene problem, but the paper-facing friday runs were executed through
`GpuStructuredDriverGroup` with `structured_env_backend=native` and
`structured_env_tensor_backend=cuda`. The native CUDA rollout uses the source
mode code: `SOURCE_POLICY` decodes the actor's satellite subset action, while
nearest-visible satellite selection is used by the `SOURCE_ZERO` path. The
recorded RelCritic seed45211 training log also reports
`exec=(policy,policy,policy)`. Therefore the formal GPU/native Phase 4 learned
runs should not be described as fixed-satellite runs, but future Python fallback
or smoke/eval runs must use the corrected configs.

## Paper Evidence Freeze

Controlled source scenario: `3uav20gu_t250`, deterministic held-out
evaluation. The formal held-out protocol uses seed bases `980000`, `981000`,
and `982000`, with 64 episodes per seed base. Learned methods use three
training seeds, so each learned method has 576 held-out episodes in the main
selected-checkpoint table. Fixed baselines use 192 held-out episodes.

Primary table source:

```text
docs/paper/table_sources/phase4_formal_heldout_source_selected_main_20260713.csv
```

Selected-checkpoint source-scenario summary:

| Method | Family | Reward | Processed | Drop | Backlog | D_sys | Collision |
|---|---|---:|---:|---:|---:|---:|---:|
| RelCritic / STARS | learned | 71.017 | 0.911 | 0.073 | 4.463 | 6.287 | 0.005 |
| GlobalCritic / STARS-GC | learned ablation | 47.349 | 0.725 | 0.237 | 7.536 | 16.210 | 0.038 |
| MAPPO-like | learned adapter | 27.477 | 0.508 | 0.376 | 16.742 | 37.213 | 0.302 |
| QCCS | fixed heuristic | 50.448 | 0.893 | 0.079 | 6.179 | 8.512 | 0.005 |
| Lyapunov | fixed heuristic | 44.656 | 0.845 | 0.124 | 7.729 | 10.969 | 0.000 |
| Observable QCCS | fixed heuristic | 31.903 | 0.634 | 0.281 | 10.475 | 19.443 | 0.245 |
| QBS | fixed heuristic | 32.965 | 0.474 | 0.438 | 16.589 | 39.708 | 0.005 |
| Uniform | fixed heuristic | 33.229 | 0.473 | 0.463 | 13.614 | 37.053 | 0.005 |

Selected-versus-final companion table source:

```text
docs/paper/table_sources/phase4_formal_heldout_source_learned_selected_final_20260713.csv
```

Use selected checkpoints in the main table because checkpoint selection was
predefined and evaluated on held-out seeds. Report final checkpoints as
companion evidence to show stability and avoid a peak-only narrative.

10-seed critic stability sources:

```text
docs/paper/table_sources/phase4_stars_training_checkpoint_summary_10seed_20260714.csv
docs/paper/table_sources/phase4_stars_gc_training_checkpoint_summary_10seed_20260714.csv
docs/paper/table_sources/phase4_critic_convergence_best_so_far_10seed_20260714.csv
```

Stability interpretation:

- STARS selected validation reward is consistently high across 10 seeds
  (roughly 58.9--70.5 in the checkpoint summary).
- STARS-GC can occasionally match STARS on favorable seeds, but its seed spread
  is much larger and low-performing seeds are common.
- This supports a relational-critic stability claim, not a claim that the
  global critic always fails.

No-HA-PPO parameter sweep sources:

```text
docs/paper/table_sources/phase4_formal_parameter_sweeps_nohappo_raw_20260714.csv
docs/paper/table_sources/phase4_formal_parameter_sweeps_nohappo_aggregate_20260714.csv
```

The no-HA-PPO sweep has 420 raw rows and 84 aggregate rows: two sweeps, seven
points each, and six methods (`STARS`, `STARS-GC`, `QCCS`, `Lyapunov`, `QBS`,
`Uniform`). HA-PPO is intentionally excluded from paper-facing full sweeps
because its evaluation path is orders of magnitude slower and has shown
process-level instability. Keep HA-PPO only as a separately-labeled diagnostic
unless it is repaired and rerun under the same sweep protocol.

Current manuscript-facing Phase 4 figure copies are under:

```text
docs/paper/manuscript/figures/
```

Current source figures are under:

```text
docs/paper/figure_sources/performance_evaluation_20260714/
docs/paper/figure_sources/phase4_section5_performance_20260713/
```

Recommended Section 5 figure/table roles:

- training/checkpoint validation: mark selected and final checkpoints;
- main source-scenario table: reward plus processed/drop/backlog/D_sys/collision;
- critic ablation: show 10-seed best-so-far reward or selected seed spread;
- MAPPO-like comparison: same selected/final protocol and same metrics;
- load/resource sensitivity: use no-HA-PPO sweeps unless HA-PPO is repaired.

## Section 5 Writing Scaffold

Section 5 should answer four questions in this order:

1. Does STARS improve source-scenario performance over strong fixed scheduling
   heuristics under the same held-out protocol?
2. Does the relational critic improve stability and performance over a global
   critic while keeping the staged actor fixed?
3. Is a flat MAPPO-like learned adapter sufficient under the same hybrid masked
   SAGIN action interface?
4. What training/evaluation cost is required to obtain the selected policies?

Suggested structure:

- `5.1 Experimental protocol`: scenario, seed bases, episodes, checkpoint
  rule, hardware/runtime reporting, and vectorized interaction scale.
- `5.2 Comparative methods and metrics`: baseline taxonomy and metric
  directions.
- `5.3 Training dynamics and checkpoint selection`: selected/final explanation
  and validation curves.
- `5.4 Source-scenario performance`: main table and network metrics.
- `5.5 Learning-side ablation`: relational critic versus GlobalCritic and the
  MAPPO-like adapter.

Do not claim arbitrary scale generalization in Section 5. Scale transfer,
larger retraining, and stress-case behavior belong in Section 6 unless the page
budget later forces a merge.

## Figure Style

Use a stable color/marker mapping across Section 5:

| Method | Color | Marker |
|---|---|---|
| STARS | `#0072B2` | circle |
| STARS-GC | `#D55E00` | square |
| HA-PPO / MAPPO-like | `#CC79A7` | diamond |
| QCCS | `#009E73` | triangle |
| Lyapunov | `#E69F00` | inverted triangle |
| QBS | `#56B4E9` | pentagon |
| Uniform | `#666666` | x |

Prefer PDF in the manuscript. Keep PNG/SVG only for inspection or source
archives. Avoid reward-only figures when a network metric table can carry the
same claim more cleanly.

## Friday Run Storage

Remote worktree:

```text
/home/sgy/workspace/sagin_marl_phase4_learning_ablation
```

Remote run root:

```text
/home/sgy/workspace/sagin_marl_phase4_learning_ablation/runs/phase4_learning_ablation/3uav20gu_t250
```

Storage snapshot from 2026-08-11:

| Path under `3uav20gu_t250` | Size | Keep/Action |
|---|---:|---|
| `stability_10seeds_20260714` | 32G | Keep until Section 5 ablation figures are frozen |
| `relational_critic` | 11G | Keep selected evidence; old failed/diagnostic runs can later be quarantined |
| `mappo_like_flat_actor_critic_stabilized` | 9.0G | Keep until MAPPO-like table/curve evidence is frozen |
| `global_only_critic` | 6.2G | Keep until GlobalCritic evidence is frozen |
| `formal_parameter_sweeps_20260714_full` | 29M | Keep |
| `formal_heldout_20260712_202812` | 4.2M | Keep |
| `_failed_compile` | 26M | Can be quarantined after audit |
| `_queues`, `_logs`, small smoke folders | <10M each | Can be kept or archived |

Two old directories returned `Input/output error` during `du`:

```text
mappo_like_flat_actor_critic/seed45211_20260711_2258_bootstrapgae_native_nocompile
relational_critic/seed45211_20260711_213639_bootstrapgae_nocompile/diagnostics/bw_parity
```

Do not include these paths in paper evidence. Do not delete them casually from
the current shell. Treat them as suspicious filesystem remnants and inspect or
quarantine only after a separate disk-health pass.

Current filesystem capacity is not the immediate bottleneck: `/dev/nvme0n1p6`
has about 761G free and is roughly 31% used.

## Three-Way Git Sync

Use GitHub as the canonical sync mechanism for code, configs, docs, and
paper-facing table/figure assets:

```text
GitHub origin <-> local Mac clone <-> friday clone
```

Remote `origin` on friday currently points to:

```text
git@github.com:scc2450/sagin_marl.git
```

Rules:

- Commit repo changes on one side, push to `origin`, then pull with
  `--ff-only` on the other side.
- Do not use raw run directories as the cross-machine synchronization layer.
  Promote only selected CSV/JSON summaries, figure sources, and scripts into
  Git.
- Keep `runs/` out of Git unless a small paper-facing table source has been
  intentionally exported under `docs/paper/table_sources/`.
- If friday produces a table/figure source, copy it into the repo on friday,
  commit and push from friday, then `git pull --ff-only` locally.
- If local produces a paper doc/script change, commit and push locally, then
  `git pull --ff-only` on friday before launching any new run.

Useful checks:

```bash
git status --short --branch
git pull --ff-only
git push origin erik/phase4-learning-ablation-baseline
ssh sgy@100.80.212.103 'git -C /home/sgy/workspace/sagin_marl_phase4_learning_ablation status --short --branch'
```

Current remote sync note: friday had untracked scratch files before the
2026-08-11 sync. They were saved as a remote stash named
`phase4-untracked-before-sync-20260811`; do not pop it unless deliberately
auditing pre-sync scratch outputs.

## Main Integration Review

Do not merge `erik/phase4-learning-ablation-baseline` directly into `main` as
one PR. The branch is integration-heavy: relative to `main`, it contains 62
commits and roughly 900 file-level changes across docs, configs, scripts,
runtime code, tests, paper assets, and archives.

Primary risks:

- code changes and paper assets are interleaved;
- phase3 generalization work, paper workspace setup, phase4 ablation work, and
  script/archive reorganization are all present on the same branch lineage;
- generated assets and experiment evidence are useful for the manuscript but
  not all of them belong in the long-lived code mainline;
- a single PR would make review quality poor and make rollback difficult.

Recommended PR split:

1. Paper workspace / IEEEtran manuscript skeleton:
   `docs/paper/manuscript/`, Overleaf packaging, template audit, section files.
2. Core environment and evaluator fixes needed by later experiments:
   native/evaluator bug fixes, fixed-baseline support, and focused tests.
3. Phase3 generalization configs and table sources:
   keep separate from phase4 learning ablation; use
   `docs/paper/experiments/phase3_generalization_runbook.md` as the canonical
   review note.
4. Phase4 learning-ablation code and configs:
   RelCritic/GlobalCritic/MAPPO-like config support, checkpoint-eval resume
   fixes, early-stop state restore, and launch scripts.
5. Phase4 paper evidence assets:
   `docs/paper/table_sources/`, `docs/paper/figure_sources/`, and figure
   generation scripts. This may remain on the paper branch if `main` should not
   carry paper artifacts.

Review order:

1. Review core code/tests first.
2. Review experiment configs and launch scripts second.
3. Review paper table/figure assets third.
4. Keep raw remote `runs/` outside PR scope.

Minimum checks before each PR:

```bash
python -m py_compile <changed-python-files>
bash -n <changed-shell-files>
git diff --check
pytest tests/test_baselines.py tests/test_structured_action_modules.py tests/test_structured_critic_system_readout.py
```

The current branch should stay as the paper integration branch until these
splits are prepared. Use it as the working paper branch, not as a direct
candidate for `main`.

## Configs

Full method anchor:

```text
configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
```

Matched relational critic full method:

```text
configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_relational_critic.yaml
```

Global-only critic ablation:

```text
configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_global_only_critic.yaml
```

Flat full-state centralized critic baseline:

```text
configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_flat_full_state_critic.yaml
```

MAPPO-like flat actor/critic learning baseline:

```text
configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_mappo_like_flat_actor_critic.yaml
```

Only intended critic-ablation config difference:

```yaml
structured_actor_backbone: topology_aware
critic_value_mode: relational
```

vs.

```yaml
critic_value_mode: global_only
```

or, for the flat full-state critic baseline:

```yaml
structured_actor_backbone: topology_aware
critic_value_mode: flat_mlp
```

Keep fixed:

- actor architecture and staged action factorization;
- environment scenario: 3 UAV, 20 GU, T=250;
- reward, queue, safety, and mask settings;
- MC-GAE/return-target settings. Do not rely on the training script default:
  `scripts/train_joint_mcgae.py` defaults to `--return_target mc`. If the paper
  main line follows the stronger prior bootstrap-GAE result, the launch command
  must explicitly include `--return_target bootstrap_gae`;
- training seed list and evaluation seed bases;
- checkpoint selection rule.

MAPPO-like baseline intended learning-side difference:

```yaml
structured_actor_backbone: flat_mlp
critic_value_mode: flat_mlp
```

Keep the environment, reward, legal masks, safety handling, return target,
training seed list, validation cadence, and checkpoint selection rule aligned
with the full method. The actor architecture is intentionally changed for this
baseline.

## Algorithmic Delta

This ablation changes only the centralized critic value estimator. The learned
policy interface and execution logic remain the same.

Unchanged actor/execution components:

- acceleration actor;
- satellite subset actor;
- bandwidth allocation actor;
- staged action factorization: acceleration -> satellite selection ->
  bandwidth allocation;
- action masks and safety-aware execution;
- reward, queue dynamics, traffic model, and scenario parameters;
- PPO/MC-GAE training loop and stage-wise actor update order.

Full relational critic pathway:

```text
global scalars
  + GU/UAV/SAT node tokens
  + UAV-GU/UAV-SAT/UAV-UAV typed edge tokens
  + entity masks and local summaries
  -> typed relational message passing
  -> system token
  -> stage-specific value heads
```

Global-only critic pathway:

```text
global scalars
  -> global scalar encoder
  -> stage-specific value heads
```

Removed from the critic value pathway:

- GU/UAV/SAT node encoders;
- UAV-GU/UAV-SAT/UAV-UAV edge encoders;
- local summary encoders;
- typed relational message-passing blocks;
- entity/relation masks as critic-side relational structure.

Still present in `global_only`:

- trainable neural value functions;
- stage-specific value heads for acceleration, satellite selection, and
  bandwidth allocation;
- the same actor observations and action masks.

Therefore the comparison should be named:

```text
Relational critic vs global-only critic
```

It should not be named:

```text
Ours vs vanilla MAPPO
```

because the actor remains the same structured staged actor. This comparison
tests whether relational centralized value estimation improves learning and
evaluation quality under the same staged policy class.

For the MAPPO-like learning baseline, the changed components are:

- acceleration head: flat MLP over local accel observation and masks;
- satellite head: flat MLP logits over legal subset candidates;
- bandwidth head: flat MLP over local BW observation and masks, with the same
  masked Dirichlet simplex distribution;
- centralized critic: flat MLP over the fixed-size world state.

Removed from the actor-side value/action parameterization:

- token-level query attention in the acceleration actor;
- SAT competition/self-attention and item/count subset scoring;
- BW downlink attention and user competition self-attention.

Still retained because they are environment/interface requirements rather than
our topology-aware modeling contribution:

- staged submission of accel/SAT/BW actions to the simulator;
- legal action masks and safety-aware execution;
- same hybrid action distributions and PPO log-prob contracts.

## Local Smoke

Use the existing Mac/local smoke runner for wiring checks:

```bash
CONFIG=configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_global_only_critic.yaml \
RUN_DIR=runs/phase4_learning_ablation/smoke_global_only_critic \
UPDATES=1 \
NUM_ENVS=1 \
ROLLOUT_ENV_STEPS=2 \
scripts/run_mac_joint_smoke.sh
```

This is not evidence for the paper. It only checks that the config loads and the
joint training entrypoint accepts the ablation mode.

Verified locally on 2026-07-11:

```text
Update 1/1 completed with critic_value_mode=global_only.
```

Smoke artifacts were written under:

```text
runs/phase4_learning_ablation/smoke_global_only_critic
```

## Formal Run Shape

Use checkpoint-eval early stopping, not a fixed training length. The training
script still needs a budget cap, but `--max_updates` is only a hard upper bound.
The phase4 ablation should follow the same early-stop shape as the current
phase3 return-target runs: do not stop before 300 updates, validate every 25
updates, and stop on validation plateau before the 700-update cap when possible.
Patience counters are reset before `checkpoint_eval_min_stop_update`, so the
first validation at or after update 300 starts the stopping patience window
rather than inheriting pre-300 plateau counts.

```text
checkpoint_eval_interval_updates = 25
checkpoint_eval_min_stop_update = 300
checkpoint_eval_episodes = 32
checkpoint_eval_episode_seed_base = 910000
checkpoint_eval_reward_patience = 4
checkpoint_eval_reward_min_delta_rel = 0.005
hard_max_updates = 700
```

Suggested run directory naming:

```text
runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/seed45211
runs/phase4_learning_ablation/3uav20gu_t250/mappo_like_flat_actor_critic/seed45211
```

Suggested first seed:

```text
45211
```

If compute allows, use at least two aligned seeds for the paper table.

Suggested remote launch shape for `global_only`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
/home/sgy/workspace/sagin_marl/.venv/bin/python scripts/train_joint_mcgae.py \
  --config configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_global_only_critic.yaml \
  --run_dir runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/seed45211_<timestamp> \
  --device cuda \
  --num_envs 64 \
  --rollout_env_steps 250 \
  --return_target bootstrap_gae \
  --return_target_schedule fixed \
  --max_updates 700 \
  --seed 45211 \
  --save_every 25
```

Suggested remote launch shape for `flat_mlp`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
/home/sgy/workspace/sagin_marl/.venv/bin/python scripts/train_joint_mcgae.py \
  --config configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_flat_full_state_critic.yaml \
  --run_dir runs/phase4_learning_ablation/3uav20gu_t250/flat_full_state_critic/seed45211_<timestamp> \
  --device cuda \
  --num_envs 64 \
  --rollout_env_steps 250 \
  --return_target bootstrap_gae \
  --return_target_schedule fixed \
  --max_updates 700 \
  --seed 45211 \
  --save_every 25
```

This `flat_mlp` config is the critic-side baseline, not the MAPPO-like learning
baseline. It isolates the critic-side topology-aware inductive bias while
keeping the staged actor. Because `flat_mlp` flattens a fixed-size world state,
treat it as a source-scenario critic baseline unless a matching shape-specific
config is trained.

Suggested remote launch shape for MAPPO-like FlatActorCritic:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/sgy/workspace/sagin_marl/.venv/bin/python scripts/train_joint_mcgae.py \
  --config configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_mappo_like_flat_actor_critic.yaml \
  --run_dir runs/phase4_learning_ablation/3uav20gu_t250/mappo_like_flat_actor_critic/seed45211_<timestamp> \
  --device cuda \
  --num_envs 64 \
  --rollout_env_steps 250 \
  --return_target bootstrap_gae \
  --return_target_schedule fixed \
  --max_updates 700 \
  --seed 45211 \
  --save_every 25 \
  --structured_env_backend native \
  --structured_env_tensor_backend cuda \
  --disable_torch_compile
```

This is the paper-facing learning baseline to use when the reviewer question is
"does the structured/topology-aware actor-critic design beat a flatter
MAPPO-style policy class?" It is still an adapter baseline because the SAGIN
environment has hybrid actions and legal masks.

Implementation note: the MAPPO-like baseline uses a hybrid native path. The
environment rollout, local observation construction, and history recording stay
on the native CUDA backend, while the flat PyTorch actor writes accel/SAT/BW
live action tensors directly on the GPU. Do not use the old sync/CPU adapter for
formal timing or training evidence.

The selected checkpoint for downstream held-out evaluation should be
`best_checkpoint.pt`, chosen by source-scenario checkpoint evaluation. `final.pt`
is still useful for stability reporting, but it is not the selection rule.

Important protocol audit from 2026-07-11:

```text
runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/seed45211_20260711_153506_strictminstop
runs/phase4_learning_ablation/3uav20gu_t250/relational_critic/seed45211_20260711_162232_strictminstop
```

These completed runs used the default `return_target=mc`, as confirmed by their
`train.log` headers and `metrics.csv` return-target code. They are valid
MC-target critic diagnostics, but they are not comparable to the earlier
`bootstrap_seed45211` result that reached held-out reward about 71.48 with
`--return_target bootstrap_gae`. Do not use these two runs as the paper's
bootstrap-GAE critic-structure ablation.

Reproducibility audit from 2026-07-11:

The corrected bootstrap-GAE RelCritic/GlobalCritic reruns are protocol-aligned
with the old `seed45211` mainline at the config and return-target level, but a
single `seed45211` retrain should not be treated as an exact reproduction of the
old checkpoint. The RelCritic YAML differs from
`configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml`
only in `checkpoint_eval*` fields. However, short same-code/same-seed tests with
`--disable_checkpoint_eval` still diverged after the first update:

```text
runs/phase4_learning_ablation/debug_samecode_seed45211_u2_20260711_201529_a
runs/phase4_learning_ablation/debug_samecode_seed45211_u2_20260711_201529_b
```

Both runs match the first-rollout bootstrap-GAE targets
`a=0.786, s=0.815, b=0.866`, but their first-update critic EV and actor KL differ,
and their second-update targets already diverge. This localizes the
irreproducibility to the learning update path rather than to the scenario
generator or the return-target setting. Checkpoint evaluation can perturb run
state if not carefully isolated, but it is not the sole cause because divergence
also appears with checkpoint evaluation disabled.

Implication for paper-facing ablations: do not compare single retrains as if
same seed implies the same training trajectory. Use the corrected
RelCritic/GlobalCritic pair as preliminary evidence only, then run matched
multi-seed critic ablations and report mean/std over held-out evaluation. The
old `checkpoint_update0575.pt` remains a valid frozen historical mainline
artifact, but it should not be described as reproducible from scratch by seed
alone under the current training pipeline.

Aborted run note:

```text
runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/seed45211_20260711_143556
```

This run was launched with fixed `--updates 300` and stopped around update 14.
Do not use it as a paper result.

## 2026-07-11 Active Friday Launches

Remote worktree:

```text
friday:/home/sgy/workspace/sagin_marl_phase4_learning_ablation
```

MAPPO-like FlatActorCritic GPU0 trial:

```text
runs/phase4_learning_ablation/3uav20gu_t250/mappo_like_flat_actor_critic/seed45211_20260711_2258_bootstrapgae_native_nocompile
```

Launch notes:

- GPU: 0.
- Seed: `45211`.
- Command uses `--return_target bootstrap_gae`, `--max_updates 700`, and
  `--disable_torch_compile`.
- It uses `--structured_env_backend native --structured_env_tensor_backend
  cuda`. Native CUDA drives the environment; flat PyTorch actor heads write
  live action/logprob tensors on GPU.
- Native CUDA smoke runs completed with checkpoint evaluation both disabled and
  enabled:

```text
runs/phase4_learning_ablation/smoke_mappo_like_flat_actor_critic_native_gpu0_20260711_2250
runs/phase4_learning_ablation/smoke_mappo_like_flat_actor_critic_native_gpu0_ckpteval_20260711_2255
```

- The earlier sync/CPU trial was intentionally stopped because collection was
  too slow for formal training:

```text
runs/phase4_learning_ablation/3uav20gu_t250/mappo_like_flat_actor_critic/seed45211_20260711_213452_bootstrapgae
```

FlatActorCritic native compatibility fixes made on 2026-07-11:

- bypass native topology-aware actor ABI binding when
  `structured_actor_backbone: flat_mlp`;
- write flat accel/SAT/BW actor outputs directly into native runtime live
  tensors;
- ignore the uninitialized live `subset_mask` in the flat SAT policy and derive
  legal SAT subsets from `sat_mask`, `sat_valid_mask`, and subset members, as
  the topology-aware SAT policy does;
- keep a guard that replaces any illegal flat SAT subset with the first legal
  subset before writing live tensors, then recomputes its log-prob under the
  same policy.

MAPPO-like FlatActorCritic seed45211 first result:

```text
runs/phase4_learning_ablation/3uav20gu_t250/mappo_like_flat_actor_critic/seed45211_20260711_2258_bootstrapgae_native_nocompile
```

This run is not a paper-ready strong learning baseline. It stopped at update
375 by checkpoint-reward plateau. The best reward row in `checkpoint_eval.csv`
was only around `26.78` with processed ratio around `0.298` and drop ratio
around `0.626`, which is below the `queue_aware_bw` checkpoint reference
(`reward=32.28`, processed ratio `0.438`, drop ratio `0.482`). Treat it as an
underfit/unstable flat-learner diagnostic, not as a fair tuned MAPPO-like
baseline.

Stabilized MAPPO-like follow-up:

```text
configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_mappo_like_flat_actor_critic_stabilized.yaml
runs/phase4_learning_ablation/3uav20gu_t250/mappo_like_flat_actor_critic_stabilized/seed45211_20260712_0016_bootstrapgae_native_nocompile
```

Changes relative to the first FlatActorCritic config:

- enable flat actor and flat critic input normalization;
- increase flat actor/critic hidden size from 256 to 512;
- reduce actor learning rates to `(5e-5, 1e-4, 1e-4)` for accel/SAT/BW;
- reduce actor epochs from 5 to 3;
- tighten target KL from 0.02 to 0.01;
- add small entropy regularization.

This is still a MAPPO-like flat actor/flat centralized critic baseline: it does
not use heuristic teacher warm start and does not change the reward or legal
action interface.

Stabilized MAPPO-like result and follow-up tuning note, 2026-07-12:

```text
runs/phase4_learning_ablation/3uav20gu_t250/mappo_like_flat_actor_critic_stabilized/seed45211_20260712_0016_resume_u0100_bootstrapgae_native_nocompile
```

This run stopped normally at update 475 by checkpoint-reward plateau. The
selected `best_checkpoint.pt` corresponds to update 375 under the 0.5% relative
reward-improvement rule, even though update 450 has a slightly higher raw
reward. The selected checkpoint is stronger than the internal `queue_aware_bw`
checkpoint reference (`reward=33.67` vs `32.28`, `processed=0.462` vs `0.438`,
`drop=0.477` vs `0.482`, `pre_backlog=13.81` vs `16.56`) but remains far below
the RelCritic full method. Treat it as a credible single-seed learned baseline
only after held-out evaluation and multi-seed confirmation.

GPU0 lrmid follow-up launched on 2026-07-12:

```text
runs/phase4_learning_ablation/3uav20gu_t250/mappo_like_flat_actor_critic_stabilized_lrmid/seed45211_20260712_143753_bootstrapgae_native_nocompile
```

It reuses the stabilized config and changes only CLI optimization knobs:

```text
--accel_actor_lr 7.5e-5
--sat_actor_lr 2.0e-4
--bw_actor_lr 2.0e-4
--actor_epochs 4
```

Rationale: the stabilized run learned, but stage KLs were often very small,
suggesting under-aggressive actor updates. This lrmid run increases actor update
strength while keeping the same flat actor/critic architecture, reward,
environment, masks, return target, validation cadence, early-stop rule, and
700-update hard cap. Early updates reached nonzero but bounded KL
(`~0.004-0.014`) and checkpoint eval at update 25 was valid
(`reward=31.50`, `processed=0.359`, `drop=0.530`); wait for plateau/best result
before drawing conclusions.

The lrmid run was manually stopped after update 80 because its early
operational metrics were worse than the stabilized run: update 50 had
`reward=31.46`, `processed=0.235`, `drop=0.539`, `pre_backlog=42.25`,
`D_sys=312.60`, and update 75 had `reward=29.27`, `processed=0.270`,
`pre_backlog=33.75`, with nonzero collision fraction. Do not use this run as a
paper result. The next reasonable tuning direction, if needed, is gentler:
keep accel LR and actor epochs at the stabilized setting, and only raise SAT/BW
LR modestly (for example `1.5e-4`) or leave the stabilized setting unchanged.

Gentler SAT/BW-only LR follow-up launched on GPU0:

```text
runs/phase4_learning_ablation/3uav20gu_t250/mappo_like_flat_actor_critic_stabilized_sbw15e4/seed45211_20260712_144404_bootstrapgae_native_nocompile
```

It reused the stabilized config and changed only:

```text
--accel_actor_lr 5.0e-5
--sat_actor_lr 1.5e-4
--bw_actor_lr 1.5e-4
--actor_epochs 3
```

This run exited unexpectedly around update 41 without `training_stop.json` or a
Python traceback in the launch log. Its update-25 checkpoint eval was already
weaker than the stabilized run (`reward=27.88`, `processed=0.333`,
`drop=0.566`, `pre_backlog=16.25`, collision fraction `0.344`), so it was not
restarted. Current recommendation: keep the stabilized MAPPO-like baseline as
the best tuned single-seed flat baseline unless a more deliberate search budget
is explicitly allocated.

MAPPO-like stabilized multi-seed follow-up launched on GPU0:

```text
runs/phase4_learning_ablation/3uav20gu_t250/mappo_like_flat_actor_critic_stabilized/seed73129_20260712_150723_bootstrapgae_native_nocompile
```

This run keeps the stabilized configuration unchanged and changes only the
training seed from `45211` to `73129`. Its purpose is to test whether the
stabilized MAPPO-like baseline is consistently learnable across seeds, not to
tune the baseline further. Use the same validation/early-stop rule and later
run held-out evaluation before using it in the paper table.

Critic ablation GPU1 three-seed queue:

```text
runs/phase4_learning_ablation/3uav20gu_t250/_queues/critic_ablation_gpu1_3seeds_20260711_213639_nocompile.log
```

Queued runs:

```text
runs/phase4_learning_ablation/3uav20gu_t250/relational_critic/seed45211_20260711_213639_bootstrapgae_nocompile
runs/phase4_learning_ablation/3uav20gu_t250/relational_critic/seed73129_20260711_213639_bootstrapgae_nocompile
runs/phase4_learning_ablation/3uav20gu_t250/relational_critic/seed91457_20260711_213639_bootstrapgae_nocompile
runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/seed45211_20260711_213639_bootstrapgae_nocompile
runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/seed73129_20260711_213639_bootstrapgae_nocompile
runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/seed91457_20260711_213639_bootstrapgae_nocompile
```

Launch notes:

- GPU: 1.
- Seeds: `45211`, `73129`, `91457`.
- Runs are serial within the queue to avoid same-GPU contention.
- All runs use `--return_target bootstrap_gae`, `--max_updates 700`, and
  `--disable_torch_compile`.
- The first attempt without `--disable_torch_compile` failed immediately due
  to a torch-dynamo shape recompilation error in `_value_accel`. It was
  archived as:

```text
runs/phase4_learning_ablation/3uav20gu_t250/_failed_compile/relational_critic_seed45211_20260711_213452_bootstrapgae
```

## Metrics To Compare

Training side:

- validation reward curve;
- best/final validation reward;
- stage-wise critic loss;
- stage-wise explained variance, if logged;
- actor KL/clip/entropy or skipped-update indicators, if logged.

Evaluation side:

- reward;
- processed ratio;
- drop ratio;
- backlog or delay proxy;
- collision/early termination rate;
- runtime if easy to collect.

## Interpretation

If relational beats global-only:

- claim that relational value estimation improves learning quality or stability;
- tie the claim to critic/advantage quality, not to every actor-side design.

If global-only is close:

- keep the ablation and be honest;
- shift the main contribution emphasis toward staged hybrid-action scheduling,
  safety-aware execution, and protocol-level robustness.

If global-only wins:

- inspect critic overfitting, EV, and training horizon before treating the
  relational critic as final.
