# Run Artifact Roots Before Branch Consolidation

Updated: 2026-09-09

This file records ignored `runs/` artifact roots that must remain reachable while
the historical branches are gradually consolidated into
`erik/phase4-learning-ablation-baseline`. Raw run directories are intentionally
not tracked by Git.

## Current Code State

The consolidation center is `erik/phase4-learning-ablation-baseline`.
As of this note, the branch has absorbed the two remaining
`phase2-bw-degradation` diagnostic commits:

- `8626db6` (`Add BW actor geometry diagnostics`)
- `ca29d6f` (`Add BW policy direction diagnostics`)

`codex/phase3-scenario-generalization`, `codex/paper-submission-workspace`, and
`origin/mac-python-rollout` have no commits that are unique relative to this
phase4 branch. The `lyapunov-dpp` branch remains a separate function branch and
should be ported manually if needed.

## Artifact Roots

| Host | Path | Approx. size | Contents | Handling |
| --- | --- | ---: | --- | --- |
| local macOS | `/Users/erik/Documents/GitHub/sagin_marl/runs` | 1.1G | Local phase2 archives, small smoke/eval outputs, method-validation summaries. | Keep until phase2 reproducibility and archive paths are rechecked. |
| local macOS | `/Users/erik/Documents/GitHub/sagin_marl_phase2_bw_logs/runs` | not present | No local run root observed in this worktree during this pass. | No action. |
| friday | `/home/sgy/workspace/sagin_marl/runs` | 32G | Historical phase2 artifacts, method-validation runs, phase3 generalization runs, smoke/benchmark records. | Keep; this is the current broad evidence store. |
| friday | `/home/sgy/workspace/sagin_marl_phase2_bw_degradation/runs` | 2.1G | Phase2 BW direction/magnitude diagnostics and reruns tied to the two merged diagnostic commits. | Keep until the phase2 worktree is retired and useful reports are indexed. |
| friday | `/home/sgy/workspace/sagin_marl_phase4_learning_ablation/runs` | 63G | Phase4 formal training, selected held-out evaluation, parameter sweeps, convergence/stability runs, collision smoke runs. | Keep; do not delete when the phase4 branch is merged. |
| friday | `/home/sgy/workspace/sagin_marl_repro_940ffb7` | 17M | Detached old-code repro checkout; no `runs/` root observed. | Can be removed later if no longer needed for old-checkpoint reproduction. |

## Phase4 Subtree To Preserve

The main paper-facing phase4 subtree is:

```text
/home/sgy/workspace/sagin_marl_phase4_learning_ablation/runs/phase4_learning_ablation/3uav20gu_t250
```

Important subdirectories include:

- `relational_critic/`
- `global_only_critic/`
- `mappo_like_flat_actor_critic_stabilized/`
- `stability_10seeds_20260714/`
- `formal_heldout_20260712_202812/`
- `formal_parameter_sweeps_20260714_full/`
- `uav_density_collision_smoke_20260715/`

The paper-facing CSV/JSON summaries derived from these runs are tracked under
`docs/paper/evidence_tables/`, and figure regeneration scripts are tracked under
`docs/paper/reproduction/`.

## Cleanup Rules

Merging Git branches does not preserve or move ignored `runs/` artifacts.
Before deleting any worktree directory, first ensure that its ignored run roots
are either:

1. recorded in this manifest or `docs/run_registry.csv`;
2. copied or moved to an independent artifact root; or
3. explicitly classified as disposable scratch output.

Do not delete the known phase4 paths that previously showed `Input/output error`
from a normal cleanup pass. Recheck disk health and copy out readable summaries
before any quarantine or removal.

Future physical consolidation should use an artifact root outside any Git
worktree, for example:

```text
/home/sgy/workspace/sagin_marl_artifacts/runs/{phase2,phase3,phase4}
```

After that move, worktrees can optionally contain symlinks for convenience, but
the canonical raw artifacts should not depend on branch-specific checkout
directories.
