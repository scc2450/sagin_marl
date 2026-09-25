# 100-GU Paper Assets

本目录是当前 100-GU 场景论文图片、绘图脚本与轻量证据的唯一归档入口。
旧 20-GU 图、系统图和方法图继续保留在 `docs/paper/`；原始 runs、checkpoint
和大体量日志不进入本目录。正文仍以 Overleaf 为准。

## Directory Map

| Path | Contents |
| --- | --- |
| `manuscript_overleaf/figures/` | 28 separate manuscript-candidate PDFs; six GC panels are pending review. |
| `reproduction/` | Four 100-GU aggregation/plotting scripts; no training algorithms. |
| `evidence_tables/` | Registry, nine compact CSVs, source manifest and figure index. |
| `reproduction/generated_figures/` | Ignored PNG previews and PDF QA; not a second authoritative asset set. |

The original Section5 style and best-so-far helper remain shared dependencies at
`docs/paper/reproduction/generate_section5_single_panel_figures_20260714.py`.
Do not copy that helper here. Future scan launchers archive the new plotter path;
resume also accepts the historical path inside immutable old run snapshots.

## Script Index

| Script | Role |
| --- | --- |
| `generate_more_gus_paper_assets.py` | Canonical registration and single-panel export; excludes selected/final comparison. |
| `generate_more_gus_training_figures.py` | Original algorithm-review diagnostics, including endpoint comparison outside the manuscript archive. |
| `generate_more_gus_scan_figures.py` | Run-level scan aggregation and full-range diagnostic plots. |
| `generate_dqs_revision_figures.py` | Verified DQS-only overlay retaining all other policy evidence. |

## Current 100-GU Single-Panel Set (2026-09-25)

Run `python docs/paper_moreGUs/reproduction/generate_more_gus_paper_assets.py` from the
repository root. It uses only registered lightweight evidence; CUDA and raw run
directories are not required for rendering. All current filenames have prefix
`sec5_100gu_` and suffix `_20260925.pdf`:

| Family | Panel names between prefix and suffix | Count |
| --- | --- | ---: |
| Load scan | `load_{reward,processed_ratio,drop_ratio,delay_proxy,pre_backlog,queue_workload}` | 6 |
| Joint-resource scan | `resource_{reward,processed_ratio,drop_ratio,delay_proxy,pre_backlog,queue_workload}` | 6 |
| Training | `training_{reward,best_so_far_reward,processed_ratio,drop_ratio,delay_proxy,pre_backlog}` | 6 |
| Queue decomposition | `selected_queue_decomposition` | 1 |
| Episode illustration | `episode_{trajectory,queues,reward}` | 3 |
| Pending critic ablation | `critic_ablation_{training,reward,processed_ratio,drop_ratio,delay_proxy,collision}_review` | 6 |

These are individual panels, not a combined plate. Style follows the existing
Section5 PDFs: single-column size, serif type, consistent colors and compact
legends. PNG previews live only in ignored
`docs/paper_moreGUs/reproduction/generated_figures/more_gus_20260925/`.
The authoritative file/caption/status mapping is
`docs/paper_moreGUs/evidence_tables/more_gus_20260925/figure_index.json`.

Main scans contain STARS, DQS, Lyapunov, QBS and Uniform; GC is separate. The
y-range is based on all strong-method points, with open boundary triangles and
a note where weak methods are off-scale. Source data are never clipped.
STARS bars are SD across three training-seed means, not episode-level confidence
intervals. Each method/seed/point uses64paired screening episodes.

The GC panels are **PENDING REVIEW**, not ready to support a final superiority
claim. Three GC runs completed at u375, but degradation and collisions remain
unexplained. Training curves are not extended to700 after early stopping.

No selected/final comparison panel is included. Training curves still show their
actual endpoints, and the single seed45211 continuation after u500 is dashed.
Selected queue decomposition uses only STARS selected checkpoints. Historical
review figures and selected/final evidence remain at their original run paths.

## Reproduction

### 100-GU Manuscript Archive / 单图归档（2026-09-25）

```bash
# Refresh evidence from explicitly recorded completed runs (no new evaluation).
python docs/paper_moreGUs/reproduction/generate_more_gus_paper_assets.py --register --check_only
# Render using only registered CSV/JSON; raw runs and CUDA are not required.
python docs/paper_moreGUs/reproduction/generate_more_gus_paper_assets.py
# Verify registered table hashes without writing figures.
python docs/paper_moreGUs/reproduction/generate_more_gus_paper_assets.py --check_only
```

Use local conda `rl`; on Friday use `/home/sgy/workspace/.venvs/sagin-paper/bin/python`.
Do not install plotting dependencies into the native training environment.

Sources: revised-DQS scan `more_gus_scans/stars_dqs_revised_k1_20260925`, the three
original bootstrap runs, the one original-seed45211 continuation, the algorithm
review's checked episode/hotspot artifacts, and
`more_gus_training/global_critic_3seed_k1_20260924`. All paths are below
`runs/experiments/`. `--register` checks completed status, unique/ordered updates,
config differences, layer sums, matching deterministic K1/32-env evaluation
summaries and episode hashes before exporting compact evidence.

Outputs: 28 separate PDFs in `manuscript_overleaf/figures/`, nine CSV tables plus
source and figure manifests in `evidence_tables/more_gus_20260925/`, PNG previews
only in ignored `reproduction/generated_figures/more_gus_20260925/`. The figure
index records filenames, hashes, captions, review flags and all off-scale values.
No new prose runbook or duplicated authoritative figure directory is introduced.

Interpretation and exclusions:

- Main scan roster: STARS/DQS/Lyapunov/QBS/Uniform, not QCCS or GC. Lyapunov is a
  display-name change only. Raw historical rows keep their original method IDs.
- Each selected checkpoint/fixed policy uses64paired screening episodes:
  bases1980000/1981000,32each,num_envs32,T250,Kbw=Ksat=1. STARS error bars show
  descriptive sample SD across three training-seed means; fixed policies have
  no artificial training-seed replication. No significance claim is implied.
- Shared sweep-wide y-limits use STARS mean plus/minus SD, DQS and Lyapunov.
  Off-scale weak-method means are marked at the boundary and remain unchanged
  in registered tables. Resource scaling jointly changes access/backhaul/CPU,
  not bandwidth alone. D_sys is a queue-derived delay proxy, not direct latency.
- Training plots preserve actual points and show only seed45211's continuation
  after u500 as dashed. The continuation disables reward early stopping and is
  not a fourth seed. Best-so-far is descriptive; raw reward remains available.
- The selected/endpoint comparison is excluded from this manuscript archive.
  Original run-level diagnostic plots/data remain intact. Queue decomposition
  uses selected checkpoints only. Episode trajectories/queues/reward are one
  illustrative episode, not independent statistical evidence; state paths end
  at t249 while post-step traces cover t1..250.
- GC is a separate critic ablation, with six panels visibly marked PENDING
  REVIEW. Its three runs end at375, with selected checkpoints50/25/75; thin
  training lines show each actual run and aggregate bands require all3seeds.
  GC degradation/collision cases remain unresolved. Same cap700 and stopping
  rule do not imply equal realized training steps. Saved configs additionally
  differ by two later-serialized DQS defaults, documented in the manifest;
  these fields are not read by learned-policy execution. Do not call GC an
  independent MAPPO/learning baseline.

### 100 GU Result Review / 100 GU 结果整理（2026-09-23）

The original diagnostic review outputs are kept under
`runs/experiments/more_gus_training/analysis/`, separate from the historical
20-GU manuscript PDFs. `README.md` embeds all retained figures; `png/`, `pdf/`
and `data/` hold previews, vector figures and reproducibility evidence.
No SVG is generated. This review does not register a new baseline comparison.

```bash
base=runs/experiments/more_gus_training
python docs/paper_moreGUs/reproduction/generate_more_gus_training_figures.py \
  --run_dirs \
    "$base/bootstrap_phase4_3uav100gu_k1_seed45211_20260920_120135" \
    "$base/bootstrap_phase4_3uav100gu_k1_seed45210_20260920_183803" \
    "$base/bootstrap_phase4_3uav100gu_k1_seed61723_20260920_183803" \
  --continuation "$base/bootstrap_phase4_3uav100gu_k1_seed45211_u500_to_u700_20260922_132212" \
  --out_dir "$base/analysis" \
  --episode_json "$base/bootstrap_phase4_3uav100gu_k1_seed61723_20260920_183803/renders/u700_seed1980000_native/episode.json" \
  --hotspot_json "$base/bootstrap_phase4_3uav100gu_k1_seed61723_20260920_183803/renders/hotspot_tape_seed1980000.json"
```

Use local conda `rl` with Matplotlib, NumPy, pandas and PyYAML; Friday's training
venv currently lacks pandas. This is offline post-processing, not GPU training.
`--check_only` validates input coverage/configs without writing figures.

Only the original-seed 45211 continuation is joined after u500, with a dashed
line and the changed reward-stopping rule disclosed. The alternative rollout
seed continuation is untouched and excluded, not counted as another training
seed. Raw reward and best-so-far share axes; early-stop tails are not imputed.
Selected/final use paired 64-episode screening evaluations. Queue layer totals
use actual node counts from each saved config, not the old fixed 20-GU constant.
Queue decomposition is retained (5.54-73.68 Mbit); flow decomposition is omitted
(all stage/arrival ratios 99.137%-99.983%). The display filter is descriptive,
not a significance test. Source hashes and the exact command are in
`data/provenance.json`. Historical manuscript PDFs are not overwritten.

The reused Section 5 generator now derives training axes and best-so-far
endpoints from observed data instead of truncating at u525 or reward75.
Its legacy data sources and baseline labels remain separate from this review.

## Open Evidence Gaps

Archive validation: 27 focused tests passed locally after migration. All 28 PDFs
were rendered and visually inspected as single-page panels. Friday verified the
nine registered table hashes and matched the synchronized files by checksum.
Its separate paper venv lacks pytest; no dependencies were installed into either
remote runtime. Tests do not constitute a new GPU evaluation.

GC degradation and collision mechanisms remain unreviewed. GC is a critic
ablation, not an independent learning baseline. A matched independent learned
baseline for this 100-GU setting is still missing. These 64-episode screening
scans do not inherit the older 192-episode formal protocol by being archived.
