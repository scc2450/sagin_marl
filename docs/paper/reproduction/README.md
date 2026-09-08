# Paper Reproduction Notes / 论文复现与证据说明

This file supersedes the previous separate notes
`main_integration_review_20260811.md`, `phase3_generalization_runbook.md`, and
`phase4_learning_ablation_runbook.md`.

本文件合并并替代原先分散的
`main_integration_review_20260811.md`、`phase3_generalization_runbook.md` 和
`phase4_learning_ablation_runbook.md`。

## Scope / 范围

本目录记录论文写作阶段真正需要保留的复现信息：脚本入口、证据表来源、图表生成关系、
同步规则和结论边界。它不是原始训练日志归档目录。

This directory records the reproducibility information that is still useful for
paper writing: script entry points, evidence-table sources, figure generation
relationships, and claim boundaries. It is not an archive for raw training
logs, and it does not treat local LaTeX body text as the authoritative
manuscript copy.

## Directory Contract / 目录职责

| Path | 中文职责 | English role |
|---|---|---|
| `docs/paper/manuscript_overleaf/` | 投稿图、PPT 图源和少量 Overleaf 辅助文件。正文以云端版本为准。 | Submission figures, PPT sources, and limited Overleaf support files. The cloud manuscript is authoritative. |
| `docs/paper/manuscript_overleaf/figures/` | 正文实际引用的 PDF 图及少量 PPT 图源。 | PDF figures referenced by the manuscript and selected editable PPT sources. |
| `docs/paper/evidence_tables/` | 注册后的 CSV/JSON 证据表。 | Registered CSV/JSON evidence tables. |
| `docs/paper/reproduction/` | 聚合、绘图和证据复核脚本。 | Aggregation, plotting, and evidence-audit scripts. |

`figure_sources/` 不再维护。若脚本需要生成临时检查图，应使用
`docs/paper/reproduction/generated_figures/`，该目录应视为 scratch 输出，不作为稿件图的第二份副本。

`figure_sources/` is no longer maintained. If a script needs temporary
inspection outputs, use `docs/paper/reproduction/generated_figures/`; that
directory should be treated as scratch output rather than a second manuscript
figure copy.

## Script Index / 脚本索引

| Script | 中文用途 | English purpose |
|---|---|---|
| `aggregate_phase4_nominal_rich_metrics_20260715.py` | 聚合 nominal selected rich metrics，用于队列和流量分解图。 | Aggregates nominal selected rich metrics for queue and flow decomposition. |
| `generate_section5_single_panel_figures_20260714.py` | 生成 Sec.5 当前采用的单面板 load/resource/training 图。 | Generates the current Section 5 single-panel load/resource/training figures. |
| `generate_section5_nominal_mechanism_figures_20260715.py` | 生成 nominal 队列分解和流量分解图。 | Generates nominal queue and flow decomposition figures. |
| `generate_section5_critic_ablation_convergence_20260715.py` | 生成 STARS 与 STARS-GC 的 10-seed critic 收敛对比图。 | Generates the 10-seed STARS versus STARS-GC critic convergence figure. |
| `generate_phase4_parameter_sweep_figures_20260714.py` | 生成 formal no-HA-PPO 参数扫描图。 | Generates formal no-HA-PPO parameter-sweep figures. |
| `generate_performance_evaluation_figures_20260714.py` | 旧版综合性能图脚本，主要用于复核旧图。 | Legacy combined performance figure script, mainly for auditing old panels. |
| `generate_section5_performance_assets_20260713.py` | 旧版 Sec.5 资产脚本，保留用于追溯 20260713 图表。 | Legacy Section 5 asset script retained for tracing 20260713 figures. |
| `generate_stars_*_20260714.py` | STARS 10-seed training/checkpoint 曲线辅助脚本。 | Helper scripts for STARS 10-seed training/checkpoint curves. |
| `generate_critic_convergence_comparison_10seed_20260714.py` | 早期 critic 收敛比较脚本。 | Earlier critic convergence comparison script. |
| `generate_section5_ablation_seed_spread_20260715.py` | 可选 seed spread 稳定性图。 | Optional seed-spread stability figures. |
| `generate_uav_density_collision_smoke_figure_20260715.py` | UAV 密度碰撞 smoke/stress 图，仅作诊断。 | UAV-density collision smoke/stress figure, diagnostic only. |

## Current Figure Set / 当前图集

当前稿件采用的 Sec.5 PDF 图位于 `docs/paper/manuscript_overleaf/figures/`：

The current Section 5 manuscript PDFs are under
`docs/paper/manuscript_overleaf/figures/`:

- `sec5_stars_raw_checkpoint_reward_20260714.pdf`
- `sec5_stars_best_so_far_reward_20260714.pdf`
- `sec5_critic_ablation_best_so_far_reward_10seed_20260715.pdf`
- `sec5_load_reward_20260714.pdf`
- `sec5_load_processed_ratio_20260714.pdf`
- `sec5_load_drop_ratio_20260714.pdf`
- `sec5_load_queue_workload_20260715.pdf`
- `sec5_resource_reward_20260714.pdf`
- `sec5_resource_processed_ratio_20260714.pdf`
- `sec5_resource_system_delay_20260714.pdf`
- `sec5_resource_queue_workload_20260715.pdf`
- `sec5_nominal_queue_decomposition_20260715.pdf`
- `sec5_nominal_flow_decomposition_20260715.pdf`

系统图 `system_model.pdf` 来自 `system_model_20260817.pptx`。方法图
`taskflow.pdf` 和 `obs_and_actor.pdf` 来自同一个
`obs_and_actor_20260825.pptx`。

The system figure `system_model.pdf` comes from
`system_model_20260817.pptx`. The method figures `taskflow.pdf` and
`obs_and_actor.pdf` come from the same `obs_and_actor_20260825.pptx`.

## Phase 4 Evidence / Phase 4 证据

主实验场景为 `3uav20gu_t250`。正式 held-out evaluation 使用 seed bases
`980000`, `981000`, and `982000`，每个 seed base 64 episodes。学习方法使用三个训练
seed，因此每个学习方法主表覆盖 576 个 held-out episodes；固定 baseline 覆盖 192 个
held-out episodes。

The source scenario is `3uav20gu_t250`. Formal held-out evaluation uses seed
bases `980000`, `981000`, and `982000`, with 64 episodes per seed base. Learned
methods use three training seeds, giving 576 held-out episodes per learned
method in the main selected-checkpoint table; fixed baselines use 192 held-out
episodes.

Primary evidence tables:

- `docs/paper/evidence_tables/phase4_formal_heldout_source_selected_main_20260713.csv`
- `docs/paper/evidence_tables/phase4_formal_heldout_source_learned_selected_final_20260713.csv`
- `docs/paper/evidence_tables/phase4_formal_parameter_sweeps_nohappo_aggregate_20260714.csv`
- `docs/paper/evidence_tables/phase4_critic_convergence_best_so_far_10seed_20260714.csv`
- `docs/paper/evidence_tables/phase4_nominal_selected_rich_aggregate_20260715.csv`

Selected source-scenario summary:

| Method | Role | Reward | Processed | Drop | Backlog | `D_sys` | Collision |
|---|---|---:|---:|---:|---:|---:|---:|
| STARS | learned full method | 71.017 | 0.911 | 0.073 | 4.463 | 6.287 | 0.005 |
| STARS-GC | global-critic ablation | 47.349 | 0.725 | 0.237 | 7.536 | 16.210 | 0.038 |
| MAPPO-like | flat learned adapter | 27.477 | 0.508 | 0.376 | 16.742 | 37.213 | 0.302 |
| QCCS | fixed heuristic | 50.448 | 0.893 | 0.079 | 6.179 | 8.512 | 0.005 |
| Lyapunov | fixed heuristic | 44.656 | 0.845 | 0.124 | 7.729 | 10.969 | 0.000 |
| QBS | fixed heuristic | 32.965 | 0.474 | 0.438 | 16.589 | 39.708 | 0.005 |
| Uniform | fixed heuristic | 33.229 | 0.473 | 0.463 | 13.614 | 37.053 | 0.005 |

写作边界：STARS-GC 是“同 staged actor 下的 global critic ablation”，不是完整
MAPPO baseline；MAPPO-like 是同一 hybrid masked 环境接口下的 flat actor/critic
adapter baseline，不应写成某篇外部 MAPPO 实现的严格复现。

Writing boundary: STARS-GC is a global-critic ablation under the same staged
actor, not a complete MAPPO baseline. MAPPO-like is a flat actor/critic adapter
under the same hybrid masked environment interface, not an exact reproduction of
a specific external MAPPO implementation.

## Phase 3 Evidence / Phase 3 证据

Phase 3 的作用是支持“同尺度扰动”和“邻近尺度迁移”的稳健性讨论，而不是主张任意规模可扩展。
`6UAV/80GU` 当前应写成 high-load stress case，不应作为干净的大规模泛化主结论。

Phase 3 supports robustness under same-scale perturbations and nearby
scale-transfer settings. It should not be used to claim arbitrary scalability.
`6UAV/80GU` should currently be framed as a high-load stress case, not as a
clean large-scale generalization result.

Important evidence families:

- Same-scale zero-shot: `phase3_same_scale_zero_shot_*`
- Nearby scale transfer: `phase3_scale_transfer_zero_shot_*`
- Boundary robustness: `phase3_boundary_scale_robustness_*`
- Larger six-UAV diagnostics: `phase3_6uav*`

容量扫描写作边界：`6UAV/40GU` resource sweep 同时放大 access/backhaul resources，
因此应称为 capacity calibration，而不是单独的 UAV 硬件带宽需求。

Capacity-sweep boundary: the `6UAV/40GU` resource sweep scales access and
backhaul resources together, so it should be described as capacity calibration,
not as a standalone UAV hardware-bandwidth requirement.

## Sync And PR Policy / 同步与 PR 策略

GitHub `origin` 是代码、配置、论文轻量证据和脚本的同步层：

GitHub `origin` is the synchronization layer for code, configs, lightweight
paper evidence, and scripts:

```text
GitHub origin <-> local Mac clone <-> friday clone
```

原始 run 目录留在运行机器上，只把选择后的 CSV/JSON summary、必要图表和生成脚本纳入 Git。

Raw run directories stay on the machine where they were produced. Only selected
CSV/JSON summaries, necessary figures, and generation scripts should enter Git.

当前分支不应直接作为一个大 PR 合并到 `main`。推荐拆分为：核心代码/测试、Phase3 配置与证据、
Phase4 learning ablation、论文稿件与证据资产。

The current integration branch should not be merged into `main` as one large
PR. Prefer splitting it into core code/tests, Phase 3 configs/evidence, Phase 4
learning ablation, and paper manuscript/evidence assets.

Minimum checks before promoting a paper-evidence change:

```bash
git diff --check
python -m py_compile docs/paper/reproduction/*.py
```
