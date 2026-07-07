# Docs Guide / 文档导航

`docs/` 现在既是使用说明区，也是研究过程记录区。很多文件是某一次实验、诊断、方案讨论或汇报材料的沉积，不应该被默认理解为“当前主线”。

如果目标是运行项目，优先从仓库根目录入口开始：

- `../README.md`：当前训练、评估入口和常用命令。
- `../PROJECT_STRUCTURE.md`：当前源码、配置、脚本布局。
- `../agent.md`：给后续 coding agent 的工作提示，避免误入 legacy 路线。
- `../configs/README.md`：配置文件目录说明。
- `guides/script_reorg_mapping.md`：脚本整理映射表和 legacy 路径说明。

如果目标是论文写作与投稿准备，优先看：

- `paper/README.md`：论文工作台、Overleaf 免费版同步方式和本地资产组织。
- `paper/manuscript/`：可上传到 Overleaf 的 TMLCN LaTeX manuscript 源文件。
- `paper/experiment_protocol.md`：论文侧实验协议与表格族。
- `paper/evidence_index.md`：论文 claim 到实验证据的索引。

## 当前阅读顺序

想理解当前主线训练方法时，建议按这个顺序读：

1. `current/joint_mcgae_training_flow_summary_20260514.md`
   - 解释为什么从 bootstrap GAE target 转向有限时域 MC target 拟合 critic，以及 actor update 如何使用拟合后的 critic。
2. `current/joint_mcgae_macro_k5_nogrow_u300_rerun_eval_20260514.md`
   - 当前 K=5 joint MC-GAE 主结果记录。
3. `current/bw_access_macro_decision_interval_design_20260513.md`
   - BW macro decision interval 设计说明；当前正式训练使用 K=5。
4. `guides/torch_compile_cache_setup_20260510.md`
   - Windows / server 侧 torch.compile 与 Triton cache 路径说明。
5. `guides/autodl_cuda_runbook.md` 和 `guides/mac_cpu_rollout.md`
   - AutoDL/CUDA 与本机 Mac 可运行性的实践记录。

## 当前结果文档

这些文件最接近当前论文/实验主线：

- `current/joint_mcgae_macro_k5_nogrow_u300_rerun_eval_20260514.md`：K=5 主结果评估。
- `current/joint_mcgae_macro_k5_nogrow_best_u300_result_20260514.md`：K=5 best-head 相关结果记录。
- `current/joint_mcgae_macro_k10_nogrow_u300_eval_20260514.md`：K=10 对照。
- `current/joint_mcgae_k5_danger_imitation_ablation_20260515.md`：danger imitation 开关消融。
- `current/joint_stage_safety_speed_check_20260509.md`：joint stage 安全性与训练速度检查。
- `experiments/line3/line3_short_mainline_metric_bundle.md`：Line 3 指标包。
- `experiments/line3/line3_short_train_validation.md`：Line 3 short-train validation。

## 方法与设计说明

这些文件适合用来解释算法、模型结构、baseline 和 reward/credit 设计：

- `method/baseline_choose.md`：当前 baseline 分层、MaxWeight/Lyapunov 命名和 `topology_dpp` strong non-learning benchmark 定位。
- `method/heuristic_baselines.md`：rule-based / greedy baseline 说明，以及当前 structured/native baseline ID 映射。
- `method/model_architecture.md`：较早的模型结构说明；引用前要和当前代码核对。
- `guides/metrics_guide.md`：较早的指标说明；指标词汇仍有用，但部分配置路径已是历史。
- `method/problem_analysis_20260402.md`：早期问题拆解。
- `method/structured_joint_redesign_20260402.md`：structured joint-control redesign blueprint。
- `method/structured_joint_redesign_code_design_20260402.md`：structured joint redesign 的代码设计版。
- `method/reward_vs_training_signal_summary_20260504.md`：为什么 reward 看起来可用但训练信号仍可能弱。
- `method/reward_action_credit_diagnosis_20260504.md`：action credit 诊断。
- `method/reward_candidate_stage_probe_20260504.md`：stage-specific reward candidate probe。
- `method/ppo_credit_alignment_audit_20260504.md`：PPO credit alignment audit。
- `method/single_stage_ppo_action_effect_variance_20260502.md`：单阶段 PPO 动作效应方差诊断。

## 操作指南

这些文件更偏运行、调试、测试或汇报：

- `guides/autodl_cuda_runbook.md`：AutoDL CUDA runbook。
- `guides/cloud_compute_workflow.md`：较早的云端训练流程；执行命令前要核对当前入口。
- `guides/mac_cpu_rollout.md`：本机 Mac CPU rollout 可行性和限制。
- `guides/tests_overview.md`：测试文件概览。
- `guides/torch_compile_cache_setup_20260510.md`：torch.compile / Triton cache 设置。
- `guides/ppt_materials_20260409.md`：较早的 PPT 材料整理。
- `assets/slides/joint_mcgae_training_flow_summary_20260514.pptx`：joint MC-GAE 训练流程幻灯片。

## 实验与诊断簇

目前文件已按大类移动，目录就是主要地图：

| 目录 | 含义 | 使用场景 |
|---|---|---|
| `current/` | 当前或接近当前的 joint MC-GAE 主线记录。 | 主结果解释、K 对照、消融。 |
| `guides/` | 运行、调试、测试、脚本整理和汇报材料。 | 找操作步骤和维护说明。 |
| `method/` | 算法、模型、baseline、reward/credit 设计。 | 写论文方法或解释训练逻辑。 |
| `experiments/bw/` | BW action interface、BW-only sanity、teacher/student、macro interval、clean per-user 历史。 | 查 BW 方案演化，不作为默认入口。 |
| `experiments/sat/` | SAT clean/listwise/pairwise/local-swap/macro 历史。 | 查 SAT 方案和诊断。 |
| `experiments/line3/` | Line 3 evaluation/material bundles。 | 论文/汇报证据包。 |
| `diagnostics/native/` | native CUDA、single-GPU runtime、kernel、性能瓶颈。 | 调 native 执行和速度问题。 |
| `diagnostics/structured/` | structured actor/critic、输入尺度、训练失败、架构差异审计。 | 查 structured 模型和训练诊断。 |
| `archive/temp/` | 临时讨论和 chat-derived plans。 | 归档候选；只在追溯某个旧讨论时读。 |
| `archive/legacy/` | 早期总结、编码备份和旧阶段记录。 | 历史追溯。 |
| `assets/` | PPTX、PNG、drawio 等非 Markdown 资产。 | 汇报材料和图表资产。 |

## Native CUDA 与 structured runtime

调 native 执行、性能瓶颈、CUDA kernel 时优先看：

- `diagnostics/native/structured_single_gpu_target_architecture_20260417.md`
- `diagnostics/native/structured_single_gpu_native_main_kernel_rebuild_plan_20260420.md`
- `diagnostics/native/structured_single_gpu_native_main_kernel_status_and_delivery_20260424.md`
- `diagnostics/native/structured_native_prefetch_rollout_fix_plan_20260424.md`
- `diagnostics/native/structured_native_gpu_diagnosis_20260424.md`
- `diagnostics/native/structured_native_gpu_diagnosis_20260425_after_runtime_cleanup.md`
- `diagnostics/native/structured_native_gpu_final_kernel_landing_20260425.md`
- `diagnostics/native/native_speed_bottleneck_audit_20260507.md`
- `diagnostics/native/env_channel_native_changes_20260428.md`

## 资产文件

非 Markdown 文件集中在 `assets/`：

- `assets/ppt_formula_assets/*.png`：生成幻灯片时用到的公式图片。
- `assets/diagrams/actor_network_latest.drawio`：actor network 图。
- `assets/slides/joint_mcgae_training_flow_summary_20260514.pptx`：训练流程 PPT。
- `experiments/line3/line3_short_mainline_*.csv`：Line 3 指标/评估 CSV 包。

如果后续移动资产，要同步更新读写这些路径的脚本。`scripts/analysis/export/generate_joint_mcgae_training_ppt.py` 已经改为写入 `docs/assets/` 下的新路径。

## 仍在根目录的文档

仓库根目录仍保留 `README.md`、`PROJECT_STRUCTURE.md`、`agent.md` 作为入口。DPP 拓扑感知资料已归档到 `archive/legacy/`，对应 smoke 脚本在 `../scripts/experiments/dpp/smoke_topology_aware_dpp.py`；它们来自 `lyapunov-dpp` 分支，可作为当前 `topology_dpp` Python baseline 的历史参考，而不是新的评估入口。
`archive/legacy/topology_aware_dpp_summary_20260415.md` 和 `archive/legacy/topology_aware_dpp_quick_start_20260415.md`：`lyapunov-dpp` 分支的拓扑感知 DPP baseline 历史资料；当前可运行入口优先使用 `scripts/evaluate_structured_mixed_heads_native.py --baseline_policy topology_dpp`。
