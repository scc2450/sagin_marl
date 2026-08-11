# Phase 4 Baseline Consolidation Review

Date: 2026-08-11

Scope: local checkout and friday worktree state for Phase 4 learning baselines,
critic ablations, fixed schedulers, and paper-facing Section 5 evidence.

## Local Checkout

- Branch: `erik/phase4-learning-ablation-baseline`.
- State before consolidation: local branch was one commit ahead of
  `origin/erik/phase4-learning-ablation-baseline`, with additional uncommitted
  paper evidence, plotting scripts, and figure archive updates.
- The branch is an integration branch, not a small patch branch. It contains
  paper workspace setup, phase3 generalization work, DPP/fixed-baseline support,
  and Phase 4 learning-ablation commits.
- Recommendation: do not merge this branch directly into `main` until the paper
  evidence files and experimental scripts are committed and reviewed as a
  coherent integration unit.

## Paper Evidence To Keep

- Keep the no-HA-PPO formal parameter sweep tables:
  - `phase4_formal_parameter_sweeps_nohappo_raw_20260714.csv`
  - `phase4_formal_parameter_sweeps_nohappo_aggregate_20260714.csv`
- Keep the nominal selected rich metrics used for queue and flow decomposition:
  - `phase4_nominal_selected_rich_raw_20260715.csv`
  - `phase4_nominal_selected_rich_aggregate_20260715.csv`
  - `phase4_nominal_selected_rich_summaries_20260715/`
- Keep the 10-seed STARS/STARS-GC convergence and checkpoint-validation
  summaries for critic stability.
- Keep the UAV-density collision smoke tables as diagnostic or appendix
  evidence only. They should not become a main safety claim unless the intended
  operating regime is explicitly defined.

## Figures

- Manuscript-facing figures should remain PDF-only under
  `docs/paper/manuscript/figures/`.
- Older 2026-07-13 figures have been moved out of the manuscript-facing root and
  into role-specific archive folders.
- The current main-text candidate set is the Section 5 single-panel PDF set
  documented in `docs/paper/manuscript/figures/README.md`.
- Draft sensitivity, scale-transfer, queue-flow variants, and safety-stress
  figures should remain archived or source-only unless the Section 5/6 narrative
  explicitly needs them.

## HA-PPO Sweep Boundary

- HA-PPO remains a source-scenario learned baseline.
- HA-PPO should not be part of the default full parameter sweep because its
  flat learned evaluation path is much slower and has previously polluted
  concurrent sweep stability.
- The default launcher now keeps the full sweep to STARS, STARS-GC, QCCS,
  Lyapunov, QBS, and Uniform.
- If HA-PPO sweep evidence is needed, run it separately with
  `INCLUDE_HAPPO=1` and treat it as a serial companion run, not a blocker for
  the main Section 5 sensitivity figures.

## Friday State

- Main friday checkout:
  `/home/sgy/workspace/sagin_marl` on `codex/phase3-scenario-generalization`,
  clean at inspection.
- Phase 4 friday worktree:
  `/home/sgy/workspace/sagin_marl_phase4_learning_ablation` on
  `erik/phase4-learning-ablation-baseline`.
- At inspection, the Phase 4 worktree was behind the local checkout and had
  untracked copies of some paper tables and Phase 4 scripts.
- No active Phase 4 runner process was detected.
- Remote run storage under
  `runs/phase4_learning_ablation/3uav20gu_t250` was about 59 GB.
- Two old run subdirectories returned `Input/output error` during `du`. Treat
  this as a friday storage-health issue and do not attempt destructive cleanup
  until the affected paths are copied or explicitly abandoned.

## Recommended Commit Plan

1. Commit the local paper evidence registry, Section 5 figure set, plotting
   scripts, and launcher hardening as one paper-evidence consolidation commit.
2. Push `erik/phase4-learning-ablation-baseline` so friday can fast-forward to
   the same code and scripts.
3. On friday, archive or delete only clearly temporary queue logs and failed
   HA-PPO sweep fragments after storage health is checked.
4. Leave `main` untouched until the branch has a clean integration review.
