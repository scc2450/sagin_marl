# Manuscript evidence index

Use this file to connect manuscript claims to local or remote evidence. Do not
paste large raw logs here.

## Claim map

| Claim | Evidence source | Manuscript location | Status |
|---|---|---|---|
| Learned policies outperform strongest non-learning baselines under same-scale perturbations. | `docs/paper/experiments/phase3_generalization_runbook.md` and `docs/paper/table_sources/phase3_same_scale_zero_shot_*.csv` | Experiments: zero-shot robustness | Candidate |
| Learned policies retain positive margins in nearby scale-transfer settings. | `docs/paper/experiments/phase3_generalization_runbook.md` and `docs/paper/table_sources/phase3_scale_transfer_zero_shot_*.csv` | Experiments: scale transfer | Candidate |
| GU=30 and UAV=2 are boundary cases but remain positive with extra evaluation seeds. | Phase3 boundary robustness records | Experiments: boundary robustness | Candidate |
| 6UAV/80GU is trainable with bootstrap-GAE under the current protocol. | `docs/paper/experiments/phase3_generalization_runbook.md` and `docs/paper/table_sources/phase3_6uav80gu_bootstrap_auto_ref_multiseed_*.csv` | Experiments: larger-scale setting | Multi-seed complete; seed-aligned baseline/load-transfer evaluation complete |
| 6UAV/80GU reward degradation is primarily associated with load density and service-capacity pressure. | `docs/paper/experiments/phase3_generalization_runbook.md` and `docs/paper/table_sources/phase3_6uav_load_transfer_new_multiseed_*.csv` | Experiments: larger-scale stress setting | Candidate; frame as high-load stress evidence |
| 6UAV/40GU is not scale-isomorphic to 3UAV/20GU despite the same GU/UAV ratio. | `docs/paper/experiments/phase3_generalization_runbook.md` and `docs/paper/table_sources/phase3_6uav40_density_resource_diagnostic_*.csv` | Experiments: larger-scale diagnostic / limitations | Diagnostic; do not promote as final benchmark without retraining |
| Density-controlled 6UAV/40GU needs calibrated access/backhaul resources before it behaves like a serviceable larger-scale scenario. | `docs/paper/experiments/phase3_generalization_runbook.md` and `docs/paper/table_sources/phase3_6uav40_resource_sweep_baseline_*.csv` | Experiments: larger-scale capacity calibration | Baseline-only diagnostic; x4 first serviceable, x6 first 3UAV/20GU-like |
| MC target under 6UAV/80GU should not be described as universally failed. | Critic EV gate and actor skip-rate diagnostics | Limitations / ablation discussion | Needs gate ablation |
| Relational critic improves source-scenario held-out performance and seed stability over global-only critic. | `docs/paper/table_sources/phase4_formal_heldout_source_selected_main_20260713.csv`, `docs/paper/table_sources/phase4_formal_heldout_source_learned_selected_final_20260713.csv`, and `docs/paper/figure_sources/phase4_section5_performance_20260713/` | Experiments: critic ablation / main performance | Formal held-out complete; use selected checkpoint in main table and final checkpoint as companion evidence |
| MAPPO-like flat actor/critic adapter baseline is much weaker than RelCritic under the same scenario, held-out seeds, and hybrid action interface. | `docs/paper/table_sources/phase4_formal_heldout_source_selected_main_20260713.csv`, `docs/paper/table_sources/phase4_formal_heldout_source_learned_seed_rows_20260713.csv`, and `docs/paper/experiments/phase4_learning_ablation_runbook.md` | Experiments: learning baseline comparison | Three-seed formal held-out complete; frame as in-pipeline MAPPO-like adapter, not an external-paper reproduction |

## Evidence hygiene

- Keep raw `runs/` outputs out of this directory.
- Prefer CSV/JSON summaries and generated table sources under `docs/paper/table_sources/`.
- Record whether a result is based on training seeds, evaluation seeds, or both.
- Do not promote single-seed transition results into final claims without marking the limitation.
