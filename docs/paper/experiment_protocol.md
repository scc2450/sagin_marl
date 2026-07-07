# Experiment protocol for manuscript

This file tracks the paper-facing evaluation protocol. It should be updated
before numbers are copied into manuscript tables.

## Selection rule

- Select learned checkpoints using source-scenario validation metrics.
- Evaluate selected checkpoints on held-out or zero-shot scenarios.
- Do not select checkpoints based on target generalization results.

## Current scenario groups

| Group | Role | Status |
|---|---|---|
| 3UAV/20GU/T=250 source scenario | Main training and validation source | Active mainline |
| Same-scale perturbations | Zero-shot robustness evidence | Candidate paper result |
| Nearby scale-transfer settings | Scale-transfer evidence | Candidate paper result |
| 6UAV/80GU/T=250 | Larger-scale candidate main scenario | Needs more training seeds |

## Required table families

| Table | Purpose | Notes |
|---|---|---|
| Main comparison | Learned policy vs strongest baselines | Report reward and network metrics, not reward alone. |
| Ablation | Justify staged actor, critic, return target, safety, and K | Keep only paper-relevant ablations. |
| Generalization | Same-scale perturbation and nearby scale-transfer | Use fixed checkpoint-selection rule. |
| Runtime/resource cost | Training and inference practicality | Include hardware and episode/update settings. |

## Reporting rule

Every table row should be traceable to a run directory, checkpoint path,
configuration file, evaluation seed base, and commit hash when available.
