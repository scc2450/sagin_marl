# Manuscript evidence index

Use this file to connect manuscript claims to local or remote evidence. Do not
paste large raw logs here.

## Claim map

| Claim | Evidence source | Manuscript location | Status |
|---|---|---|---|
| Learned policies outperform strongest non-learning baselines under same-scale perturbations. | `.local_guidance/phase3/phase3_generalization_plan_20260630.md` and phase3 run directories | Experiments: zero-shot robustness | Candidate |
| Learned policies retain positive margins in nearby scale-transfer settings. | `.local_guidance/phase3/phase3_generalization_plan_20260630.md` and scale-transfer tables | Experiments: scale transfer | Candidate |
| GU=30 and UAV=2 are boundary cases but remain positive with extra evaluation seeds. | Phase3 boundary robustness records | Experiments: boundary robustness | Candidate |
| 6UAV/80GU is trainable with bootstrap-GAE under the current protocol. | Phase3 6UAV/80GU bootstrap run and aligned evaluation | Experiments: larger-scale setting | Needs more training seeds |
| MC target under 6UAV/80GU should not be described as universally failed. | Critic EV gate and actor skip-rate diagnostics | Limitations / ablation discussion | Needs gate ablation |

## Evidence hygiene

- Keep raw `runs/` outputs out of this directory.
- Prefer CSV/JSON summaries and generated table sources under `docs/paper/table_sources/`.
- Record whether a result is based on training seeds, evaluation seeds, or both.
- Do not promote single-seed transition results into final claims without marking the limitation.
