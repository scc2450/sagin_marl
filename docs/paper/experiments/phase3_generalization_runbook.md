# Phase 3 Generalization Runbook

This runbook is the tracked, canonical summary for Phase 3 generalization and
scale-transfer evidence. It consolidates the earlier local notes under
`.local_guidance/phase3/`; those local notes are now historical scratch records,
not the primary source for paper-facing claims.

## Scope

Phase 3 answers whether a policy selected on the controlled source scenario can
remain useful under nearby scenario changes, and whether larger-scale scenarios
are serviceable under the current physical and traffic assumptions.

Source scenario:

| Item | Value |
|---|---:|
| Config | `configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml` |
| UAVs | 3 |
| GUs | 20 |
| Satellites | 144 |
| Horizon | 250 slots |
| Mean arrival | `2.0e6` bit/GU/slot |
| Access BW decision interval | 5 |
| SAT decision interval | 1 |

Representative source-selected checkpoints:

| Label | Method | Checkpoint rule |
|---|---|---|
| `MC-best(seed45211,u400)` | finite-horizon MC target | selected by source validation, then evaluated zero-shot |
| `Bootstrap-best(seed45211,u575)` | bootstrap-GAE | selected by source validation, then evaluated zero-shot |

Do not select checkpoints using target generalization scores.

## Evidence Sources

All paper-facing summary data should be traced through
`docs/paper/table_sources/registry_index.csv`.

| Evidence family | Table sources |
|---|---|
| Same-scale zero-shot | `phase3_same_scale_zero_shot_*` |
| Nearby scale-transfer zero-shot | `phase3_scale_transfer_zero_shot_*` |
| Boundary robustness extra seeds | `phase3_boundary_scale_robustness_*` |
| 6UAV/80GU fixed queue-reference baseline/train | `phase3_6uav80gu_fixed_ref_*`, `phase3_6uav80gu_return_target_*` |
| 6UAV/80GU auto queue-reference evaluation/train | `phase3_6uav80gu_auto_queue_ref_*`, `phase3_6uav80gu_bootstrap_auto_ref_multiseed_*` |
| 6UAV load transfer | `phase3_6uav_load_transfer_new_multiseed_*` |
| 6UAV/40GU density/resource diagnostic | `phase3_6uav40_density_resource_diagnostic_*` |
| 6UAV/40GU resource calibration sweep | `phase3_6uav40_resource_sweep_baseline_*` |
| 12UAV/80GU map smoke | `phase3_12uav80gu_map_sweep_smoke_rows.csv` |

Raw `runs/` directories stay on friday or other run hosts. Only selected
CSV/JSON summaries, figure sources, and scripts are committed.

## Baseline Taxonomy

| Baseline | Role | Paper caveat |
|---|---|---|
| `queue_aware` | weak queue-aware rule | simple reference, not strongest baseline |
| `observable_cluster_queue_aware` | fair observable heuristic | estimates centers from current observable GU state |
| `cluster_center_queue_aware` | privileged strong heuristic | uses true cluster metadata; mark as privileged |
| `maxweight_lyapunov` | strong MaxWeight/Lyapunov rule | main non-learning comparator |
| `dpp_resource_hybrid_native` | DPP resource baseline | useful method-family baseline, not strongest |
| `topology_dpp_native_bw_sat_cached` | DPP SAT/BW baseline | stable native DPP-family comparator |
| `full_topology_dpp_joint` | full DPP-family ablation | currently too weak for strongest-baseline claims |

## Current Conclusions

1. Same-scale perturbation is the cleanest Phase 3 generalization evidence. The
   source-selected learned policies remain ahead of the current strong
   non-learning baselines under traffic, hotspot, satellite-visibility, and BW
   decision-interval perturbations.
2. Nearby scale transfer is positive but should be described as nearby
   zero-shot transfer, not unbounded scalability. The tested dimensions are
   `GU=10/30`, `UAV=2/4`, and `visible_sats_max=4/8`.
3. `GU=30` and `UAV=2` are the narrow-margin boundary cases. Extra evaluation
   seed bases reduced the risk that the positive margin is a seed accident.
4. `6UAV/80GU` is trainable with bootstrap-GAE under the auto queue-reference
   protocol, but it is not a clean serviceable main benchmark yet. Drop remains
   high and the strongest service metrics do not clearly dominate strong
   baselines.
5. The `6UAV/80GU` degradation is primarily a load-density and service-capacity
   stress effect. Auto queue references restore queue-step semantics, but they
   do not add physical service capacity.
6. `6UAV/40GU` is not a scale-isomorphic copy of `3UAV/20GU` even though both
   have the same average GU/UAV ratio. More UAVs create stronger spatial,
   interference, backhaul, and collision-avoidance coupling.
7. The resource multiplier result must be framed as capacity calibration, not
   as a literal UAV hardware requirement. The sweep scales both access and
   backhaul resources.

## Scenario Matrix

| Group | Scenarios | Current status | Paper role |
|---|---|---|---|
| Same-scale perturbation | load low/high, hotspot sparse/wide, harder SAT visibility, K=10 | formal zero-shot complete | primary generalization table |
| Nearby scale transfer | GU 10/30, UAV 2/4, visible SAT 4/8 | formal zero-shot complete | scale-transfer table |
| Boundary robustness | GU 30, UAV 2 with extra seed bases | extra seed evaluation complete | robustness footnote or appendix |
| 6UAV/80GU | fixed-ref and auto-ref variants | train/eval complete but service quality weak | high-load stress case study |
| 6UAV load ladder | 6UAV/40GU, 60GU, 80GU | evaluation complete | explains load-pressure trend |
| 6UAV/40GU density/resource | density control, 2x resource, combined | diagnostic complete | explains non-isomorphic scaling |
| 6UAV/40GU resource sweep | x1, x1.5, x2, x3, x4, x6, x8 | baseline-only complete | capacity calibration |
| 12UAV/80GU | temporary map smoke | smoke only | diagnostic, not a paper result yet |

## 6UAV/40GU Capacity Calibration

The baseline-only resource sweep used density-controlled 6UAV/40GU geometry and
scaled access/backhaul resources together:

```text
b_acc = 2.0e6 * multiplier
b_backhaul_per_sat = 1.0e7 * multiplier
```

Thresholds:

| Threshold | Criteria |
|---|---|
| serviceable | `processed >= 0.85`, `drop <= 0.15`, `D_sys <= 10` |
| 3UAV/20GU-like | `processed >= 0.90`, `drop <= 0.10`, `D_sys <= 8` |

Compact result:

| Multiplier | Best processed | Best drop | Best D_sys | Interpretation |
|---:|---:|---:|---:|---|
| x1 | 0.621 | 0.309 | 21.16 | under-provisioned |
| x2 | 0.778 | 0.159 | 13.52 | still under-provisioned |
| x3 | 0.836 | 0.119 | 9.45 | close, but processed is below serviceable threshold |
| x4 | 0.876 | 0.089 | 6.68 | first serviceable point |
| x6 | 0.931 | 0.051 | 4.26 | first 3UAV/20GU-like point |
| x8 | 0.954 | 0.033 | 3.00 | likely over-provisioned for minimum-resource claims |

Paper wording should say that x4 is the minimum serviceable calibration point,
and x6 is the first source-quality-matched calibration point. Do not state that
every UAV "requires 12 MHz" without the coupled access/backhaul and simulation
context.

## Larger-Scale Narrative

Conservative wording:

```text
Under fixed six-UAV resources, increasing the number of active ground users
from 40 to 60 and 80 consistently reduces processed ratio while increasing
drop, backlog, and system delay for both learned policies and non-learning
baselines. This indicates that the 6UAV/80GU setting is primarily a high-load
service-capacity stress test rather than a clean scale-isomorphic extension of
the 3UAV/20GU source setting.
```

Avoid:

```text
The learned policy scales directly from 3UAV/20GU to 6UAV/80GU.
```

Prefer:

```text
The learned policy shows robust zero-shot behavior under same-scale and nearby
scale perturbations. Larger six-UAV settings expose capacity and topology
coupling effects, so we report them as stress/capacity-calibration studies.
```

## Open Items

Before promoting any larger-scale result into a main claim:

1. Split access-only and backhaul-only capacity sweeps to identify the active
   bottleneck.
2. If using `6UAV/40GU` as a serviceable larger-scale benchmark, retrain on the
   selected resource point rather than relying on zero-shot transfer.
3. Prefer x4 for minimum-serviceability claims and x6 for source-quality matched
   claims.
4. Keep `6UAV/80GU` as high-load stress evidence unless service metrics improve
   with a clearly justified resource or traffic configuration.
5. Keep raw run directories out of PRs; register only paper-facing summaries in
   `docs/paper/table_sources/`.

## Three-Way Sync Rule

Use GitHub `origin` as the canonical sync layer for code, configs, tracked docs,
and paper-facing evidence:

```text
GitHub origin <-> local Mac clone <-> friday clone
```

Raw run directories are not part of Git sync. They should remain on the run
host and be referenced through table-source rows and evidence indexes.
