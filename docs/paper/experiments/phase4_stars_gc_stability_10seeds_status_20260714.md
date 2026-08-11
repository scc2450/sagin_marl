# Phase 4 STARS vs STARS-GC Stability Extension

Date: 2026-07-14

Status: launched on `friday`.

Purpose: extend the critic-side stability evidence from three training seeds to
ten aligned training seeds for STARS and STARS-GC. The result should support
the Section 5 ablation/stability discussion, not replace the main held-out
source-scenario table until the full training and held-out evaluation pass is
complete.

## Scope

Existing formal training seeds:

- `45211`
- `73129`
- `91457`

New training seeds:

- `10331`
- `21893`
- `36467`
- `48761`
- `59023`
- `64217`
- `87539`

Methods:

- STARS: `relational_critic`
- STARS-GC: `global_only_critic`

The final stability set is therefore ten training seeds per method, assuming
all seven new seeds complete and pass held-out evaluation.

## Protocol

- Scenario: `3uav20gu_t250`
- Training entrypoint: `scripts/train_joint_mcgae.py`
- Configs:
  - `configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_relational_critic.yaml`
  - `configs/experiments/phase4_learning_ablation/structured_joint_mcgae_3uav20gu_t250_global_only_critic.yaml`
- `num_envs=64`
- `rollout_env_steps=250`
- `return_target=bootstrap_gae`
- `return_target_schedule=fixed`
- `max_updates=700`
- `save_every=25`
- `disable_torch_compile=true`

## Remote Run Root

```text
friday:/home/sgy/workspace/sagin_marl_phase4_learning_ablation/runs/phase4_learning_ablation/3uav20gu_t250/stability_10seeds_20260714
```

Remote execution environment:

```text
/home/sgy/workspace/sagin_marl/.venv/bin/python
```

Status files:

```text
runs/phase4_learning_ablation/3uav20gu_t250/stability_10seeds_20260714/status_gpu0.csv
runs/phase4_learning_ablation/3uav20gu_t250/stability_10seeds_20260714/status_gpu1.csv
```

Queue logs:

```text
runs/phase4_learning_ablation/3uav20gu_t250/stability_10seeds_20260714/_queues/gpu0_queue.log
runs/phase4_learning_ablation/3uav20gu_t250/stability_10seeds_20260714/_queues/gpu1_queue.log
```

Launcher:

```text
scripts/experiments/phase4/launch_stars_gc_stability_10seeds_20260714.sh
```

## Runtime Incident

At approximately `2026-07-14T07:56:42+08:00`, the first attempt of STARS seed
`87539` on GPU1 terminated with return code `139` after reaching update 15. The
training log showed normal startup and early updates before a native/CUDA
segmentation fault, so this is treated as an execution failure rather than a
completed training seed.

The launcher was patched to:

- skip method/seed pairs that already have a `DONE` status when a queue is
  resumed;
- record failed runs without terminating the remaining queue.

GPU1 was resumed at approximately `2026-07-14T09:52:17+08:00`; the resumed queue
skipped completed STARS seeds `59023` and `64217`, retried STARS seed `87539`,
and will continue to the STARS-GC runs even if an individual seed fails again.

## Queue Assignment

GPU0:

- STARS seed `10331`
- STARS seed `21893`
- STARS seed `36467`
- STARS seed `48761`
- STARS-GC seed `10331`
- STARS-GC seed `21893`

GPU1:

- STARS seed `59023`
- STARS seed `64217`
- STARS seed `87539`
- STARS-GC seed `36467`
- STARS-GC seed `48761`
- STARS-GC seed `59023`
- STARS-GC seed `64217`
- STARS-GC seed `87539`

## Paper-Facing Use

After completion, run the same held-out evaluation protocol used by the current
formal source-scenario table:

- deterministic policy mode
- seed bases `980000`, `981000`, `982000`
- 64 episodes per seed base
- selected checkpoint and final checkpoint rows kept separate

Recommended Section 5 figure after aggregation:

- seed-level point plot or box/violin plot for STARS vs STARS-GC;
- primary metrics: reward, processed ratio, drop ratio, and `D_sys`;
- show all training seeds, not only mean/error bars.

Conservative claim boundary:

- If the ten-seed result preserves the current pattern, state that relational
  critic training is more stable under the matched protocol.
- Do not write that GlobalCritic fails universally; the relevant claim is higher
  seed sensitivity and worse reliability under the same actor/execution
  interface.

## Completion and Data Check: 2026-07-14 15:30 CST

The ten-seed training artifacts are available for both STARS and STARS-GC.

Available per completed seed:

- `checkpoint_eval.csv`
- `training_stop.json`
- `best_checkpoint.pt`

The STARS-GC set uses successful retry outputs for seeds `36467` and `87539`.
Earlier failed `rc=139` attempts should be excluded from paper tables and
figures.

STARS selected-checkpoint validation reward over ten seeds:

- mean/std: `66.614 / 3.306`
- min/max: `58.871 / 70.524`
- final-checkpoint mean/std: `56.055 / 12.427`

STARS-GC selected-checkpoint validation reward over ten seeds:

- mean/std: `44.632 / 13.250`
- min/max: `26.823 / 66.830`
- final-checkpoint mean/std: `36.685 / 13.014`

Same-protocol QCCS checkpoint-validation reference:

- reward: `49.1207`
- source: `friday:/home/sgy/workspace/sagin_marl_phase4_learning_ablation/runs/phase4_learning_ablation/3uav20gu_t250/convergence_refs_20260714/qccs_checkpoint_validation_ref/qccs_checkpoint_validation_ref_summary.json`
- local copy: `docs/paper/table_sources/phase4_qccs_checkpoint_validation_ref_20260714_summary.json`

Paper-facing interpretation:

- The STARS convergence figure should remain focused on trainability: light
  individual-seed traces, a median best-so-far validation reward curve, and
  sparse interquartile-range error bars, truncated at update `525`.
- STARS-GC is usable for the critic ablation/stability figure. The current
  evidence supports a cautious claim that the global critic is more
  seed-sensitive under the same staged actor and execution interface, not that
  it universally fails.
- Selected and final checkpoints should stay separated. Final-checkpoint drift
  is real for both methods and should be discussed as checkpoint-selection
  motivation rather than folded into the main convergence figure.

## Figure Update: 2026-07-14 16:05 CST

The convergence figures were regenerated with an update cutoff of `525`.

Use the following distinction consistently:

- `best-so-far validation reward`: for each training seed, the curve tracks the
  best validation checkpoint observed up to the current update. This is the
  paper-facing convergence/checkpoint-selection curve and matches the
  selected-checkpoint evaluation protocol.
- `raw checkpoint validation reward`: the curve tracks the validation reward of
  the checkpoint at each update. This exposes late-training drift and should be
  used only for checkpoint-selection/stability discussion, not as the main
  convergence claim.

Generated paper-facing candidates:

- STARS only:
  `docs/paper/manuscript/figures/phase4_stars_convergence_summary_10seed_20260714.pdf`
- STARS vs STARS-GC combined:
  `docs/paper/manuscript/figures/phase4_critic_convergence_best_so_far_combined_10seed_20260714.pdf`
- STARS vs STARS-GC two-panel:
  `docs/paper/manuscript/figures/phase4_critic_convergence_best_so_far_panels_10seed_20260714.pdf`
- STARS raw checkpoint curve:
  `docs/paper/manuscript/figures/phase4_stars_training_curve_10seed_20260714.pdf`
- STARS raw-vs-best-so-far two-panel:
  `docs/paper/manuscript/figures/phase4_stars_checkpoint_vs_best_so_far_10seed_20260714.pdf`

Recommended use:

- Use the combined STARS vs STARS-GC best-so-far figure if Section 5 needs a
  compact critic-stability comparison.
- Use the two-panel figure if the combined figure becomes visually crowded in
  the final IEEE column layout.
- Use the STARS raw-vs-best-so-far two-panel figure if the convergence
  discussion needs to explicitly distinguish instantaneous checkpoint behavior
  from the selected-checkpoint envelope. This is the most transparent option
  for addressing late checkpoint drift without weakening the selected-checkpoint
  evaluation protocol.
- For the STARS-only figure, state that error bars denote the interquartile
  range across ten training seeds.
- The STARS-only figure also includes light individual-seed traces so the
  ten-seed aggregation is visible without turning the plot into a dense seed
  spaghetti plot.
- For the STARS vs STARS-GC comparison figures, state that shaded bands denote
  the interquartile range across ten training seeds.
- In all captions, explicitly state that the best-so-far curves track
  selected-checkpoint availability rather than raw checkpoint reward.

## Historical Live Status: 2026-07-14 13:33 CST

Training is not yet fully complete.

Completed successfully:

- STARS new seeds: 7/7 complete.
- STARS-GC new seeds: 4/7 complete.

Still active:

- STARS-GC seed `64217` is running on GPU1.
- STARS-GC seed `36467` is being retried on GPU0 after an earlier `rc=139`
  failure.
- STARS-GC seed `87539` remains queued after seed `64217`.

Observed runtime versus the earlier three-seed history:

- Historical RelCritic/STARS mean training wall-clock from the formal table
  source was about 1.15 h per seed.
- The seven new successful STARS seeds averaged about 1.80 h per seed, roughly
  1.56x the historical mean. This is mainly because several new seeds reached
  later stopping points, including one run that completed all 700 updates.
- Historical GlobalCritic/STARS-GC mean training wall-clock was about 0.60 h per
  seed. The completed new STARS-GC seeds remain close to that range, so the
  machine does not appear globally slower for the global-critic variant.

The original 6--8 h wall-clock estimate was too optimistic. It assumed the
three-seed historical mean would represent the added seeds and did not leave
enough margin for late early-stopping seeds or `rc=139` retries. With the current
retry in place, expected remaining training time is roughly 1--1.5 h if no
additional retry is needed.

## Mid-Run Status at 2026-07-14 12:51 CST

The initial 6--8 hour estimate was optimistic. It used the earlier
iteration-sum training-time summaries as a rough guide, whereas the current
queue elapsed time includes checkpoint evaluation overhead, early-stop
variation, and a GPU1 interruption.

Observed state:

- STARS new seeds: `7/7` completed successfully.
- STARS-GC new seeds:
  - completed: `48761`;
  - running: `10331`, `59023`;
  - failed once and still needs a successful retry: `36467`;
  - not yet started: `21893`, `64217`, `87539`.

Notable interruption:

- STARS seed `87539` first failed with `rc=139` after 286 s on GPU1. The queue
  was resumed later and the same seed then completed successfully.
- STARS-GC seed `36467` failed with `rc=139` after 1503 s and needs a retry
  after the active queues finish or on an idle GPU.

Observed STARS elapsed times for the seven successful new seeds:

```text
4980, 5329, 5564, 6187, 6941, 7069, 9410 seconds
```

The long tail is expected because early stopping fired at different checkpoint
updates. One seed reached update 700, while others stopped between update 375
and 525.

Observed completed STARS-GC seed `48761` took 2254 seconds, so the remaining
STARS-GC jobs should be faster than the completed STARS jobs if they do not
hit another failure.

## Historical Live Status Snapshot: 2026-07-14 10:37 CST

Training is not complete.

New-run accounting:

- total new training jobs: 14 = 7 STARS + 7 STARS-GC;
- completed: 5 STARS jobs;
- running: 2 STARS jobs (`48761` on GPU0 and restarted `87539` on GPU1);
- pending: 7 STARS-GC jobs;
- failed attempts: 1 STARS attempt for seed `87539` exited with `rc=139`
  after 286 s, then was relaunched from scratch.

Important correction: the initial 6--8 hour estimate was too optimistic. The
queue also was not interleaved by seed; it runs the STARS additions before the
STARS-GC additions on each GPU. The final ten-seed comparison remains aligned,
but early partial results are one-sided until STARS-GC starts.

Revised expectation at this snapshot: the training queue is likely closer to
10--11 hours total wall-clock from the 04:31 launch, with uncertainty from
early-stop behavior and possible native/CUDA failures. Held-out evaluation is a
separate pass after training completion.
