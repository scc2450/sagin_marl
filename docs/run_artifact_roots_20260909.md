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
phase4 branch. The historical `lyapunov-dpp` branch has one remaining unique
commit, `d771119` (`添加DPP策略，搜索最佳候选动作，更新配置和环境回调以实现链路质量重算`).
Current phase4 already contains the maintained `topology_dpp` and
`dpp_resource_hybrid` Python baselines, plus native staged sources including
`topology_dpp_accel`, `topology_dpp_sat`, and `topology_dpp_bw`. Do not merge
`lyapunov-dpp` directly into phase4; treat it as a superseded historical
reference unless a specific old interface must be recovered.

## Remote Stash Audit

Friday's `/home/sgy/workspace/sagin_marl` worktree still has one old stash:
`phase4-untracked-before-sync-20260811`. It contains untracked phase4 CSV
summaries and helper scripts:

- `docs/paper/table_sources/phase4_formal_parameter_sweeps_nohappo_aggregate_20260714.csv`
- `docs/paper/table_sources/phase4_formal_parameter_sweeps_nohappo_raw_20260714.csv`
- `docs/paper/table_sources/phase4_uav_density_collision_smoke_aggregate_20260715.csv`
- `docs/paper/table_sources/phase4_uav_density_collision_smoke_raw_20260715.csv`
- `scripts/analysis/phase4/aggregate_uav_density_collision_smoke_20260715.py`
- `scripts/experiments/phase4/launch_stars_gc_stability_10seeds_20260714.sh`
- `scripts/experiments/phase4/run_uav_density_collision_smoke_20260715.py`

Compared against
`/home/sgy/workspace/sagin_marl_phase4_learning_ablation` after syncing to
`c7c110a`, the launch script is an exact match. The two Python scripts differ
only because the current tracked versions explicitly set CSV
`lineterminator="\n"`. The four CSV files have matching normalized content after
removing carriage returns; the UAV-density CSV rows also match after sorting.
The stash content is therefore covered by current phase4, but the stash has not
been dropped because dropping it is a destructive cleanup action.

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
