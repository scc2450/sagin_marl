# Manuscript evidence index

Use this file to connect manuscript claims to local or remote evidence. Do not
paste large raw logs here.

## Claim map

| Claim | Evidence source | Manuscript location | Status |
|---|---|---|---|
| Learned policies outperform strongest non-learning baselines under same-scale perturbations. | `.local_guidance/phase3/phase3_generalization_plan_20260630.md` and phase3 run directories | Experiments: zero-shot robustness | Candidate |
| Learned policies retain positive margins in nearby scale-transfer settings. | `.local_guidance/phase3/phase3_generalization_plan_20260630.md` and scale-transfer tables | Experiments: scale transfer | Candidate |
| GU=30 and UAV=2 are boundary cases but remain positive with extra evaluation seeds. | Phase3 boundary robustness records | Experiments: boundary robustness | Candidate |
| 6UAV/80GU is trainable with bootstrap-GAE under the current protocol. | `.local_guidance/phase3/phase3_6uav80gu_bootstrap_multiseed_review_20260711.md`, `.local_guidance/phase3/phase3_6uav_load_transfer_new_multiseed_20260711.md`, and related table sources | Experiments: larger-scale setting | Multi-seed complete; seed-aligned baseline/load-transfer evaluation complete |
| 6UAV/80GU reward degradation is primarily associated with load density and service-capacity pressure. | `.local_guidance/phase3/phase3_6uav_load_transfer_new_multiseed_20260711.md` and `docs/paper/table_sources/phase3_6uav_load_transfer_new_multiseed_*.csv` | Experiments: larger-scale stress setting | Candidate; frame as high-load stress evidence |
| 6UAV/40GU is not scale-isomorphic to 3UAV/20GU despite the same GU/UAV ratio. | `.local_guidance/phase3/phase3_6uav40_density_resource_diagnosis_20260711.md` and `docs/paper/table_sources/phase3_6uav40_density_resource_diagnostic_*.csv` | Experiments: larger-scale diagnostic / limitations | Diagnostic; do not promote as final benchmark without retraining |
| Density-controlled 6UAV/40GU needs calibrated access/backhaul resources before it behaves like a serviceable larger-scale scenario. | `.local_guidance/phase3/phase3_6uav40_density_resource_diagnosis_20260711.md` and `docs/paper/table_sources/phase3_6uav40_resource_sweep_baseline_*.csv` | Experiments: larger-scale capacity calibration | Baseline-only diagnostic; x4 first serviceable, x6 first 3UAV/20GU-like |
| MC target under 6UAV/80GU should not be described as universally failed. | Critic EV gate and actor skip-rate diagnostics | Limitations / ablation discussion | Needs gate ablation |
| Relational critic improves checkpoint-validation stability over global-only critic in the 3UAV/20GU source scenario. | `docs/paper/table_sources/phase4_source_validation_checkpoint_*.csv` and `docs/paper/figure_sources/phase4_source_validation_20260712/` | Experiments: critic ablation | Checkpoint-validation complete; held-out formal eval still required |
| MAPPO-like flat actor/critic baseline learns a modest policy under the same source scenario but remains far below RelCritic. | `docs/paper/table_sources/phase4_source_validation_checkpoint_*.csv` and `docs/paper/experiments/phase4_formal_evaluation_matrix_20260712.md` | Experiments: learning baseline comparison | Two-seed checkpoint-validation; prefer third seed or mark `n=2`; held-out formal eval still required |

## Evidence hygiene

- Keep raw `runs/` outputs out of this directory.
- Prefer CSV/JSON summaries and generated table sources under `docs/paper/table_sources/`.
- Record whether a result is based on training seeds, evaluation seeds, or both.
- Do not promote single-seed transition results into final claims without marking the limitation.
