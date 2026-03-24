# PROJECT_STRUCTURE

> 数据来源：基于本地磁盘实际文件扫描（不是 Git 历史）。  
> 扫描时排除：`.git/`、`.venv/`、`.pytest_cache/`、`__pycache__/`。

## 目录树（可点击跳转）

- 📂 `sagin_marl/`
  - 📄 [.gitignore](#f001)
  - 📂 `configs/`
    - 📄 [phase1_actions_curriculum_stage1_accel.yaml](#f002)
    - 📄 [phase1_actions_curriculum_stage2_bw.yaml](#f003)
    - 📄 [phase1_actions_curriculum_stage3_sat.yaml](#f003a)
  - 📂 `docs/`
    - 📄 [cloud_compute_workflow.md](#f003b)
    - 📄 [heuristic_baselines.md](#f004)
    - 📄 [metrics_guide.md](#f005)
    - 📄 [model_architecture.md](#f006)
    - 📄 [phase1_training_summary.md](#f007)
    - 📄 [tests_overview.md](#f008)
  - 📄 [PROJECT_STRUCTURE.md](#f009)
  - 📄 [README.md](#f010)
  - 📄 [requirements.txt](#f011)
  - 📂 `runs/`
    - 📂 `phase1_actions/`
      - 📂 `stage1_accel/`
        - 📄 [actor.pt](#f012)
        - 📄 [config.yaml](#f013)
        - 📄 [config_source.yaml](#f014)
        - 📄 [critic.pt](#f015)
        - 📄 [episode.gif](#f016)
        - 📄 [eval_baseline.csv](#f017)
        - 📄 [eval_baseline_fixed.csv](#f018)
        - 📂 `eval_tb/`
          - 📄 [events.out.tfevents.1770277583.LAPTOP-CC8EJ2BL.113988.0](#f019)
          - 📄 [events.out.tfevents.1770277600.LAPTOP-CC8EJ2BL.105768.0](#f020)
          - 📄 [events.out.tfevents.1770277619.LAPTOP-CC8EJ2BL.113196.0](#f021)
        - 📄 [eval_trained.csv](#f022)
        - 📄 [eval_trained_fixed.csv](#f023)
        - 📄 [eval_trained_hybrid.csv](#f024)
        - 📄 [events.out.tfevents.1770274344.LAPTOP-CC8EJ2BL.99940.0](#f025)
        - 📄 [metrics.csv](#f026)
      - 📂 `stage2_full/`
        - 📄 [actor.pt](#f027)
        - 📄 [config.yaml](#f028)
        - 📄 [config_source.yaml](#f029)
        - 📄 [critic.pt](#f030)
        - 📄 [eval_baseline.csv](#f031)
        - 📂 `eval_tb/`
          - 📄 [events.out.tfevents.1770274932.LAPTOP-CC8EJ2BL.112564.0](#f032)
          - 📄 [events.out.tfevents.1770274953.LAPTOP-CC8EJ2BL.109152.0](#f033)
        - 📄 [eval_trained.csv](#f034)
        - 📄 [events.out.tfevents.1770274612.LAPTOP-CC8EJ2BL.108328.0](#f035)
        - 📄 [metrics.csv](#f036)
    - 📂 `stage1_accel/`
      - 📄 [eval_fixed_n20.csv](#f078)
      - 📄 [eval_random_accel_n20.csv](#f079)
      - 📄 [eval_centroid_n20.csv](#f080)
  - 📂 `sagin_marl/`
    - 📄 [__init__.py](#f037)
    - 📂 `env/`
      - 📄 [__init__.py](#f038)
      - 📄 [channel.py](#f039)
      - 📄 [config.py](#f040)
      - 📄 [orbit.py](#f041)
      - 📄 [sagin_env.py](#f042)
      - 📄 [topology.py](#f043)
      - 📄 [vec_env.py](#f076)
    - 📂 `rl/`
      - 📄 [__init__.py](#f044)
      - 📄 [action_assembler.py](#f045)
      - 📄 [baselines.py](#f046)
      - 📄 [buffer.py](#f047)
      - 📄 [critic.py](#f048)
      - 📄 [distributions.py](#f050a)
      - 📄 [mappo.py](#f049)
      - 📄 [policy.py](#f050)
    - 📂 `utils/`
      - 📄 [__init__.py](#f051)
      - 📄 [logging.py](#f052)
      - 📄 [normalization.py](#f053)
      - 📄 [progress.py](#f054)
      - 📄 [seeding.py](#f055)
    - 📂 `viz/`
      - 📄 [__init__.py](#f056)
      - 📄 [plots.py](#f057)
  - 📂 `scripts/`
    - 📄 [analyze_metrics.py](#f058)
    - 📄 [estimate_throughput.py](#f059)
    - 📄 [evaluate.py](#f060)
    - 📄 [render_episode.py](#f061)
    - 📄 [run_curriculum_stage123_formal.ps1](#f061a)
    - 📄 [summarize_policy_kpi.py](#f081)
    - 📄 [train.py](#f062)
  - 📂 `tests/`
    - 📄 [conftest.py](#f063)
    - 📄 [test_action_masking.py](#f064)
    - 📄 [test_baselines.py](#f065)
    - 📄 [test_config_parsing.py](#f066)
    - 📄 [test_early_stopping.py](#f067)
    - 📄 [test_env_reset_shapes.py](#f068)
    - 📄 [test_env_step_invariants.py](#f069)
    - 📄 [test_run_dir_paths.py](#f070)
    - 📄 [test_sat_queue_tracking.py](#f071)
    - 📄 [test_smoke_train.py](#f072)
    - 📄 [test_smoke_train_vec.py](#f077)
    - 📄 [test_traffic_curriculum.py](#f075)
  - 📄 [数学模型.md](#f073)
  - 📄 [算法设计.md](#f074)

## <a id="f001"></a>📄 `.gitignore`
- 功能：定义本项目不应纳入版本管理的文件。
- 关键内容：排除了 Python 缓存、虚拟环境、测试缓存、IDE 配置，以及 `runs/` 和模型权重文件。
- 项目作用：保证仓库只保留源码与必要文档，避免训练产物污染提交记录。

## <a id="f002"></a>📄 `configs/phase1_actions_curriculum_stage1_accel.yaml`
- 功能：Stage 1 训练配置，重点学习 UAV 机动（`accel`），不启用带宽与卫星动作。
- 关键内容：`train_accel=true`、`train_bw=false`、`train_sat=false`，使用 `reward_mode=controllable_flow`、`append_action_masks_to_obs=true`、`actor_encoder_type=set_pool`、`checkpoint_eval_interval_updates=50`，并保留当前稳定的安全初始化与避碰配置。
- 项目作用：作为三阶段课程训练的起点，产出 `stage1_accel` 权重。

## <a id="f003"></a>📄 `configs/phase1_actions_curriculum_stage2_bw.yaml`
- 功能：Stage 2 带宽配置，在 Stage 1 权重基础上仅训练 `bw`。
- 关键内容：`train_accel=false`、`train_bw=true`、`train_sat=false`，冻结共享骨干与 accel 头；`enable_bw_action=true`，`bw_policy=dirichlet`，执行时 `accel/bw` 都直接来自当前策略，不再使用旧的 teacher / residual-bw 语义。
- 项目作用：在保持机动策略稳定的前提下学习最终带宽分配。

## <a id="f003a"></a>📄 `configs/phase1_actions_curriculum_stage3_sat.yaml`
- 功能：Stage 3 卫星选择配置，在 Stage 2 权重基础上解锁 `sat`，并做最终三动作评估。
- 关键内容：`train_accel=true`、`train_bw=true`、`train_sat=true`，`fixed_satellite_strategy=false`，`sat_policy=masked_categorical`，checkpoint 参考策略使用 `stage2_exec_fixed_sat` 以隔离卫星动作收益。
- 项目作用：完成三阶段递进训练链路（加速度 → 带宽 → 卫星选择）。

## <a id="f003b"></a>📄 `docs/cloud_compute_workflow.md`
- 功能：云端租算力训练实操手册（Windows 本地连接到 Linux 服务器）。
- 关键内容：覆盖选型建议（CPU/GPU）、SSH 连接、环境初始化、训练/评估命令、TensorBoard 远程查看、结果打包回传与关机止损。
- 项目作用：提供从“租机”到“拿回实验结果”的完整可执行流程，降低云端复现实验门槛。

## <a id="f004"></a>📄 `docs/heuristic_baselines.md`
- 功能：给出启发式基线（贪心/规则）设计指南。
- 关键内容：说明如何基于 `obs`、`cfg`、掩码和 `assemble_actions` 组织 `accel/bw/sat` 基线动作，并附最小代码骨架。
- 项目作用：为 `scripts/evaluate.py` 增加新 baseline 提供模板。

## <a id="f005"></a>📄 `docs/metrics_guide.md`
- 功能：说明训练与评估日志字段的定义及解读方式。
- 关键内容：覆盖 `A_ref` 归一化口径、`controllable_flow` 奖励、`metrics.csv`/`eval_*.csv`/`checkpoint_eval.csv` 的主字段、TensorBoard 页面映射、checkpoint 排序规则与 reward-plateau early stop。
- 项目作用：统一实验结果口径，减少“指标含义不一致”问题。

## <a id="f006"></a>📄 `docs/model_architecture.md`
- 功能：记录模型网络结构设计。
- 关键内容：描述当前 set-pool actor、全局状态 critic、`accel + bw + sat` hybrid action、`bw_valid_mask/sat_valid_mask`、以及 Stage 1/2/3 的训练语义。
- 项目作用：作为源码实现（`policy.py`、`critic.py`）的设计说明文档。

## <a id="f007"></a>📄 `docs/phase1_training_summary.md`
- 功能：汇总 Phase 1 训练阶段的过程与现状。
- 关键内容：包含阶段性结论、问题定位、后续改动方向等实验复盘信息。
- 项目作用：帮助快速理解“当前实验状态和下一步决策依据”。

## <a id="f008"></a>📄 `docs/tests_overview.md`
- 功能：测试文件说明文档。
- 关键内容：逐项解释 `tests/` 中各测试覆盖的功能点。
- 项目作用：作为测试导航，方便定位回归风险。

## <a id="f009"></a>📄 `PROJECT_STRUCTURE.md`
- 功能：项目结构索引文档（当前文件）。
- 关键内容：目录树、文件锚点、逐文件职责说明。
- 项目作用：降低项目上手与代码审阅成本。

## <a id="f010"></a>📄 `README.md`
- 功能：项目总入口文档。
- 关键内容：包含正式三阶段 YAML、训练/评估/混合策略/渲染命令、TensorBoard 与图片导出说明、批量正式训练脚本、输出文件说明和环境接口简介。
- 项目作用：提供从零到复现实验的操作入口。

## <a id="f011"></a>📄 `requirements.txt`
- 功能：Python 依赖清单。
- 关键内容：核心依赖包括 `torch`、`gymnasium`、`pettingzoo`、`numpy`、`tensorboard`、`pytest` 等。
- 项目作用：用于快速构建可运行环境。

## <a id="f012"></a>📄 `runs/phase1_actions/stage1_accel/actor.pt`
- 功能：Stage 1 Actor 权重文件（PyTorch checkpoint）。
- 关键内容：对应 accel-only 训练后策略网络参数。
- 项目作用：可用于评估、渲染或作为 Stage 2 初始化权重。

## <a id="f013"></a>📄 `runs/phase1_actions/stage1_accel/config.yaml`
- 功能：训练启动时固化的完整配置快照。
- 关键内容：包含默认值展开后的全参数和 `_config_source` 来源路径。
- 项目作用：确保实验可复现与配置可追溯。

## <a id="f014"></a>📄 `runs/phase1_actions/stage1_accel/config_source.yaml`
- 功能：原始配置文件副本。
- 关键内容：与 `configs/...stage1_accelonly.yaml` 对应，保留注释与原结构。
- 项目作用：对比“原始配置”与“展开配置”差异时非常关键。

## <a id="f015"></a>📄 `runs/phase1_actions/stage1_accel/critic.pt`
- 功能：Stage 1 Critic 权重文件。
- 关键内容：保存集中式价值网络参数。
- 项目作用：用于继续训练或做价值网络初始化。

## <a id="f016"></a>📄 `runs/phase1_actions/stage1_accel/episode.gif`
- 功能：单回合可视化渲染结果。
- 关键内容：通常由 `scripts/render_episode.py` 生成。
- 项目作用：直观看 UAV 轨迹与策略行为。

## <a id="f017"></a>📄 `runs/phase1_actions/stage1_accel/eval_baseline.csv`
- 功能：Stage 1 基线策略评估结果（CSV）。
- 关键内容：含 `reward_sum`、`queue_mean/max`、`drop`、`sat_flow` 等每回合统计。
- 项目作用：与训练策略做客观对照。

## <a id="f018"></a>📄 `runs/phase1_actions/stage1_accel/eval_baseline_fixed.csv`
- 功能：基线评估的额外结果快照（固定版本）。
- 关键内容：字段与 `eval_baseline.csv` 一致，便于横向对比。
- 项目作用：保留特定评估设置下的对照数据。

## <a id="f019"></a>📄 `runs/phase1_actions/stage1_accel/eval_tb/events.out.tfevents.1770277583.LAPTOP-CC8EJ2BL.113988.0`
- 功能：评估 TensorBoard 事件文件（二进制）。
- 关键内容：保存 `eval/*` 标量曲线，时间戳对应一次写入会话。
- 项目作用：支持图形化对比评估表现。

## <a id="f020"></a>📄 `runs/phase1_actions/stage1_accel/eval_tb/events.out.tfevents.1770277600.LAPTOP-CC8EJ2BL.105768.0`
- 功能：评估 TensorBoard 事件文件（二进制）。
- 关键内容：与同目录其它事件文件共同构成完整评估日志。
- 项目作用：保留不同评估批次的数据轨迹。

## <a id="f021"></a>📄 `runs/phase1_actions/stage1_accel/eval_tb/events.out.tfevents.1770277619.LAPTOP-CC8EJ2BL.113196.0`
- 功能：评估 TensorBoard 事件文件（二进制）。
- 关键内容：记录评估过程中各指标写入事件。
- 项目作用：供 TensorBoard 进行时序可视化。

## <a id="f022"></a>📄 `runs/phase1_actions/stage1_accel/eval_trained.csv`
- 功能：Stage 1 训练策略评估结果。
- 关键内容：与基线 CSV 同字段，便于同口径对比。
- 项目作用：衡量学习策略在评估环境下的实际收益。

## <a id="f023"></a>📄 `runs/phase1_actions/stage1_accel/eval_trained_fixed.csv`
- 功能：训练策略评估的固定版本快照。
- 关键内容：保存额外评估批次或修正后的结果。
- 项目作用：作为历史对照，便于复盘。

## <a id="f024"></a>📄 `runs/phase1_actions/stage1_accel/eval_trained_hybrid.csv`
- 功能：混合策略评估结果（通常为 actor accel + 规则 bw/sat）。
- 关键内容：指标字段与其它 `eval_*.csv` 对齐。
- 项目作用：用于隔离“机动策略”和“资源分配策略”的贡献。

## <a id="f025"></a>📄 `runs/phase1_actions/stage1_accel/events.out.tfevents.1770274344.LAPTOP-CC8EJ2BL.99940.0`
- 功能：训练期 TensorBoard 事件文件（二进制）。
- 关键内容：记录 `episode_reward`、loss、队列等训练曲线。
- 项目作用：用于监控收敛过程与异常。

## <a id="f026"></a>📄 `runs/phase1_actions/stage1_accel/metrics.csv`
- 功能：Stage 1 训练主日志（按 update 记录）。
- 关键内容：字段很全，覆盖奖励分解、队列统计、吞吐与时间性能。
- 项目作用：离线分析与论文出图的核心数据源。

## <a id="f027"></a>📄 `runs/phase1_actions/stage2_full/actor.pt`
- 功能：Stage 2 Actor 权重文件。
- 关键内容：对应全动作训练后的策略参数。
- 项目作用：用于最终评估与部署侧推理。

## <a id="f028"></a>📄 `runs/phase1_actions/stage2_full/config.yaml`
- 功能：Stage 2 运行时完整配置快照。
- 关键内容：记录 full-action 训练的全部参数展开结果。
- 项目作用：保证实验可复现、可追踪。

## <a id="f029"></a>📄 `runs/phase1_actions/stage2_full/config_source.yaml`
- 功能：Stage 2 原始配置副本。
- 关键内容：与 `configs/...stage2_full.yaml` 对应。
- 项目作用：便于回溯实验输入配置。

## <a id="f030"></a>📄 `runs/phase1_actions/stage2_full/critic.pt`
- 功能：Stage 2 Critic 权重文件。
- 关键内容：保存集中式价值网络参数。
- 项目作用：支持继续训练或迁移初始化。

## <a id="f031"></a>📄 `runs/phase1_actions/stage2_full/eval_baseline.csv`
- 功能：Stage 2 基线评估结果。
- 关键内容：每回合记录收益、队列、丢包、卫星流量等指标。
- 项目作用：和 `eval_trained.csv` 做直接对比。

## <a id="f032"></a>📄 `runs/phase1_actions/stage2_full/eval_tb/events.out.tfevents.1770274932.LAPTOP-CC8EJ2BL.112564.0`
- 功能：Stage 2 评估 TensorBoard 事件文件。
- 关键内容：保存 `eval/trained` 或 `eval/baseline` 下的标量。
- 项目作用：图形化比较不同策略表现。

## <a id="f033"></a>📄 `runs/phase1_actions/stage2_full/eval_tb/events.out.tfevents.1770274953.LAPTOP-CC8EJ2BL.109152.0`
- 功能：Stage 2 评估 TensorBoard 事件文件。
- 关键内容：与同目录其它事件文件共同组成评估历史。
- 项目作用：支持多批次评估的可视化追踪。

## <a id="f034"></a>📄 `runs/phase1_actions/stage2_full/eval_trained.csv`
- 功能：Stage 2 训练策略评估结果。
- 关键内容：包含 `reward_raw`、`assoc_ratio_mean`、`centroid_dist_mean` 等关键字段。
- 项目作用：验证 full-action 策略收益。

## <a id="f035"></a>📄 `runs/phase1_actions/stage2_full/events.out.tfevents.1770274612.LAPTOP-CC8EJ2BL.108328.0`
- 功能：Stage 2 训练 TensorBoard 事件文件。
- 关键内容：记录训练过程中全部标量曲线。
- 项目作用：诊断训练速度、稳定性与收敛。

## <a id="f036"></a>📄 `runs/phase1_actions/stage2_full/metrics.csv`
- 功能：Stage 2 训练日志 CSV。
- 关键内容：字段与 Stage 1 对齐，可做跨阶段横向分析。
- 项目作用：实验报告与结果复盘的核心依据。

## <a id="f078"></a>📄 `runs/stage1_accel/eval_fixed_n20.csv`
- 功能：固定策略（`baseline=fixed`）对照评估结果。
- 关键内容：在 `configs/phase1_actions_curriculum_stage1_accel.yaml` 下评估 20 个 episode，记录 `queue_total_active`、`arrival_sum`、`outflow_sum`、`drop_ratio` 等字段。
- 项目作用：作为“位置不动”基线，用于衡量队列 KPI 的参考下界。

## <a id="f079"></a>📄 `runs/stage1_accel/eval_random_accel_n20.csv`
- 功能：随机移动策略（`baseline=random_accel`）对照评估结果。
- 关键内容：与 fixed 组使用相同配置与 episode 种子序列，字段口径一致。
- 项目作用：用于验证“随机机动”对队列尾部指标（P95/P99）的影响。

## <a id="f080"></a>📄 `runs/stage1_accel/eval_centroid_n20.csv`
- 功能：追质心策略（`baseline=centroid`）对照评估结果。
- 关键内容：采用启发式“向用户质心移动”加速度策略，并输出与其它基线一致的 KPI 字段。
- 项目作用：用于比较“结构化启发式机动”相对固定/随机策略的队列控制效果。

## <a id="f037"></a>📄 `sagin_marl/__init__.py`
- 功能：顶层包初始化文件。
- 关键内容：当前为空实现。
- 项目作用：确保 `sagin_marl` 作为可导入 Python 包存在。

## <a id="f038"></a>📄 `sagin_marl/env/__init__.py`
- 功能：环境子包导出入口。
- 关键内容：导出 `AblationConfig`、`SaginConfig`、`load_config`、`SaginParallelEnv`，以及向量环境 `SyncVecSaginEnv`、`SubprocVecSaginEnv`、`make_vec_env`。
- 项目作用：简化外部调用环境模块的导入路径。

## <a id="f039"></a>📄 `sagin_marl/env/channel.py`
- 功能：通信信道与链路基础公式库。
- 关键函数：`los_probability`、`pathloss_db`、`rician_power_gain`、`atmospheric_loss_db`、`doppler_attenuation`、`snr_linear`、`spectral_efficiency`。
- 项目作用：为环境中的接入链路与回传链路速率计算提供物理层计算基元。

## <a id="f040"></a>📄 `sagin_marl/env/config.py`
- 功能：定义全局配置对象与配置加载流程。
- 关键内容：`SaginConfig` 与 `AblationConfig` dataclass 覆盖地图、流量分级、队列、信道、奖励、PPO 超参与阶段训练开关；当前正式配置重点包括 `reward_mode=controllable_flow`、`arrival_ref_mode`、`bw_policy`、`sat_policy`、`append_action_masks_to_obs` 与 checkpoint-eval 参数，同时仍保留旧实验项的兼容入口。
- 项目作用：统一参数入口，是所有模块共享的配置中心。

## <a id="f041"></a>📄 `sagin_marl/env/orbit.py`
- 功能：Walker-Delta 轨道模型实现。
- 关键类：`WalkerDeltaOrbitModel`，在 `get_states(t)` 计算卫星在 ECEF 坐标系的位置与速度。
- 项目作用：为卫星可见性、多普勒和回传速率计算提供时变卫星状态。

## <a id="f042"></a>📄 `sagin_marl/env/sagin_env.py`
- 功能：核心多智能体环境（PettingZoo `ParallelEnv`）。
- 关键流程：`reset` 初始化用户聚类、UAV 状态和缓存；`step` 依次执行 UAV 动力学、用户关联、接入速率、卫星选择、回传速率、三级队列更新、统一奖励计算和终止判断。
- 关键机制：支持候选用户模式（assoc/nearest/radius）、多普勒与可见性约束、可选衰落/干扰/大气损耗、能量模型与安全层、流量课程学习；当前正式版本固定 `A_ref`，显式暴露 `bw_valid_mask/sat_valid_mask`，记录 `x_acc/x_rel/g_pre/d_pre` 与 `processed_ratio_eval/drop_ratio_eval/pre_backlog_steps_eval/D_sys_report`，并让环境执行语义与策略采样保持一致。
- 项目作用：这是训练与评估的环境主体，几乎所有实验行为都由该文件定义。

## <a id="f043"></a>📄 `sagin_marl/env/topology.py`
- 功能：地面用户空间分布生成。
- 关键函数：`thomas_cluster_process`，按 Thomas cluster process 生成聚类用户坐标。
- 项目作用：提供可控的非均匀用户拓扑，提升任务场景真实性。

## <a id="f076"></a>📄 `sagin_marl/env/vec_env.py`
- 功能：多环境并行封装（向量化环境）。
- 关键内容：提供 `SyncVecSaginEnv`（单进程多实例）与 `SubprocVecSaginEnv`（多进程），统一 `reset/step/get_global_state_batch/close` 接口，并缓存每步统计供训练日志汇总。
- 项目作用：在不改动环境物理模型的前提下提升 rollout 采样吞吐。

## <a id="f044"></a>📄 `sagin_marl/rl/__init__.py`
- 功能：强化学习子包对外导出。
- 关键内容：导出 `ActorNet`、`CriticNet`、`train`。
- 项目作用：简化外部脚本对 RL 组件的导入。

## <a id="f045"></a>📄 `sagin_marl/rl/action_assembler.py`
- 功能：把网络输出拼装成环境可执行动作字典。
- 关键函数：`assemble_actions`，按阶段把 `accel`、最终 `bw_alloc` 与最终 `sat_select_mask` 打包成扁平环境动作，不再做 heuristic residual 融合。
- 项目作用：统一训练、评估、基线策略到环境动作接口的桥接层，并保证“策略优化的动作”和“环境真实执行的动作”一致。

## <a id="f046"></a>📄 `sagin_marl/rl/baselines.py`
- 功能：内置启发式基线策略。
- 关键函数：`zero_accel_policy`、`random_accel_policy`、`centroid_accel_policy`、`cluster_center_accel_policy` 与 `queue_aware_policy`，并支持阶段一致的 hybrid 对照，例如 Stage 1 actor + heuristic bw、Stage 2 actor + heuristic sat。
- 项目作用：用于评估对照、分动作消融与阶段化 baseline 比较。

## <a id="f047"></a>📄 `sagin_marl/rl/buffer.py`
- 功能：Rollout 经验缓存。
- 关键类：`RolloutBuffer`，支持固定容量数组模式和列表模式，存储 obs/action/logprob/reward/value/done/global_state/imitation，并为 hybrid action 额外保存 `sat_indices`。
- 项目作用：为 PPO 更新阶段提供批量化训练样本，并支持 Stage 3 重新计算 masked categorical `log_prob`。

## <a id="f048"></a>📄 `sagin_marl/rl/critic.py`
- 功能：集中式价值网络定义。
- 关键类：`CriticNet`，两层 MLP（可选 `LayerNorm`）输出标量值函数。
- 项目作用：为 MAPPO 提供全局状态价值估计。

## <a id="f050a"></a>📄 `sagin_marl/rl/distributions.py`
- 功能：hybrid action 分布封装。
- 关键内容：实现 `MaskedDirichlet`、`MaskedSequentialCategorical` 与 `HybridActionDist`，统一处理 `bw` simplex 动作、`sat` 无放回离散选择，以及三动作头的 `sample/mode/log_prob/entropy`。
- 项目作用：把当前正式策略的分布语义从网络层抽出来，供训练和评估共用。

## <a id="f049"></a>📄 `sagin_marl/rl/mappo.py`
- 功能：MAPPO 训练主循环实现。
- 关键函数：`compute_gae` 与 `train`。
- 关键实现：支持单环境与多环境并行 rollout、按阶段冻结动作头、hybrid action 采样与重算 `log_prob`、按环境独立 GAE、PPO clipped objective、checkpoint 评估与 early stop、warm start `strict=False`、梯度裁剪、NaN/Inf 防护与指标记录。
- 项目作用：整个训练流程的中枢控制文件。

## <a id="f050"></a>📄 `sagin_marl/rl/policy.py`
- 功能：Actor 策略网络与动作概率建模。
- 关键内容：`flatten_obs`/`batch_flatten_obs` 将结构化观测展平；`ActorNet` 采用 set-pool 编码器与共享上下文骨干，`accel` 用 squashed Gaussian，`bw` 用逐用户打分 + Dirichlet，`sat` 用逐卫星打分 + masked categorical；`evaluate_actions` 计算联合 logprob 与分头熵。
- 项目作用：决定策略表达能力，是训练与推理的核心网络。

## <a id="f051"></a>📄 `sagin_marl/utils/__init__.py`
- 功能：工具子包初始化占位。
- 关键内容：当前无导出实现。
- 项目作用：保持 `utils` 目录可作为包被导入。

## <a id="f052"></a>📄 `sagin_marl/utils/logging.py`
- 功能：训练日志落盘与 TensorBoard 写入。
- 关键类：`MetricLogger`，自动初始化 CSV 表头、写入标量，并定义训练自定义面板布局（Reward、Queue、Safety、Satellite、Performance 等），含尾部分位与安全权重曲线。
- 项目作用：统一训练指标记录格式。

## <a id="f053"></a>📄 `sagin_marl/utils/normalization.py`
- 功能：在线统计均值方差。
- 关键类：`RunningMeanStd`，支持批量更新并用矩方法合并统计量。
- 项目作用：用于奖励归一化，稳定 PPO 训练。

## <a id="f054"></a>📄 `sagin_marl/utils/progress.py`
- 功能：轻量终端进度条。
- 关键类：`Progress`，显示当前进度、速率与 ETA；当输出被重定向到文件时会改成按行落盘，避免交互式 `\r` 进度条丢失。
- 项目作用：训练/评估脚本的人机反馈组件，并支撑后续批量训练的 `console.log` 留档。

## <a id="f055"></a>📄 `sagin_marl/utils/seeding.py`
- 功能：统一随机种子设置。
- 关键函数：`set_seed`，同步 `random`、`numpy`、`torch`（含 CUDA）。
- 项目作用：提升实验可复现性。

## <a id="f056"></a>📄 `sagin_marl/viz/__init__.py`
- 功能：可视化子包初始化占位。
- 关键内容：当前无导出实现。
- 项目作用：支持 `viz` 包路径导入。

## <a id="f057"></a>📄 `sagin_marl/viz/plots.py`
- 功能：离线绘图工具。
- 关键函数：`plot_learning_curve`（读取 `metrics.csv` 画 reward 曲线）、`plot_trajectories`（绘制 GU 与 UAV 轨迹）。
- 项目作用：快速生成实验图表。

## <a id="f058"></a>📄 `scripts/analyze_metrics.py`
- 功能：训练日志快速分析脚本。
- 关键内容：实现滚动均值 `_rolling_mean` 与趋势斜率 `_slope`，支持按指标输出起止值、斜率、最值。
- 项目作用：在不打开 Notebook 的情况下快速判断训练走势。

## <a id="f059"></a>📄 `scripts/estimate_throughput.py`
- 功能：吞吐能力 sanity check。
- 关键内容：通过环境内部关联/速率函数估计接入、回传、计算瓶颈和利用率，输出“欠载/平衡/过载”判断。
- 项目作用：在正式训练前验证配置负载是否合理。

## <a id="f060"></a>📄 `scripts/evaluate.py`
- 功能：评估入口脚本。
- 关键内容：支持训练策略评估、`fixed`/`random_accel`/`centroid`/`cluster_center`/`zero_accel`/`queue_aware` 基线评估、stage-consistent hybrid 评估；支持 `--episode_seed_base` 做跨策略同种子对照；输出 `processed_ratio_eval`、`drop_ratio_eval`、`pre_backlog_steps_eval`、`D_sys_report` 及 `x_acc_mean/x_rel_mean/g_pre_mean/d_pre_mean`，并写入 `eval_tb` TensorBoard。
- 项目作用：统一产出可对比评估结果。

## <a id="f061"></a>📄 `scripts/render_episode.py`
- 功能：渲染单回合并导出 GIF。
- 关键内容：加载 actor 权重，以确定性动作执行环境并逐帧保存；支持 trained / baseline 渲染、按 `episode_seed` 复现评估回合；`_resolve_render_paths` 对旧调用补了默认 `baseline` 兼容。
- 项目作用：用于可视化策略行为与调试。

## <a id="f061a"></a>📄 `scripts/run_curriculum_stage123_formal.ps1`
- 功能：正式三阶段课程训练批处理脚本。
- 关键内容：顺序运行 `stage1 -> stage2 -> stage3`，统一传入 `num_envs/vec_backend/torch_threads/updates`，在每阶段结束后自动做 trained 与 baseline 评估，并把终端输出同步写入各阶段的 `console.log`。
- 项目作用：减少正式实验的人为操作差异，并方便事后追踪训练过程。

## <a id="f081"></a>📄 `scripts/summarize_policy_kpi.py`
- 功能：对多策略评估 CSV 进行 KPI 汇总对照。
- 关键内容：读取 `label=csv_path` 输入，输出 `queue_total_active` 的 mean/P95/P99，`outflow_arrival_ratio` 的 mean/P05，以及 `drop_ratio` 的 mean。
- 项目作用：用于固定/随机/追质心等对照实验的一键统计汇总。

## <a id="f062"></a>📄 `scripts/train.py`
- 功能：训练启动脚本。
- 关键内容：解析参数、加载配置、设置随机种子、创建单环境或向量环境、解析日志目录、保存配置快照并调用 `mappo.train`；支持 `--num_envs`、`--vec_backend`、`--torch_threads`、warm start 权重路径与正式三阶段训练流程。
- 项目作用：命令行训练总入口。

## <a id="f063"></a>📄 `tests/conftest.py`
- 功能：测试初始化配置。
- 关键内容：将项目根目录加入 `sys.path`，确保测试可直接导入本地包。
- 项目作用：稳定 pytest 运行环境。

## <a id="f064"></a>📄 `tests/test_action_masking.py`
- 功能：动作掩码基础测试。
- 关键内容：在固定卫星策略下执行一步环境交互，验证流程可运行且观测数量正确。
- 项目作用：防止动作封装/掩码逻辑导致基础回归。

## <a id="f065"></a>📄 `tests/test_baselines.py`
- 功能：基线策略单元测试。
- 关键内容：验证 `zero_accel_policy`、`random_accel_policy`、`centroid_accel_policy` 的输出形状/范围，并验证 `queue_aware_policy` 输出 shape、dtype、有限性和范围约束。
- 项目作用：保证评估对照策略稳定可用。

## <a id="f066"></a>📄 `tests/test_config_parsing.py`
- 功能：配置解析测试。
- 关键内容：验证 `load_config` 能把 YAML 字符串数值与布尔值正确转换为目标类型，并校验 `ablation`/`ablation_flags` 的嵌套解析。
- 项目作用：防止配置类型错误在训练中后期才暴露。

## <a id="f067"></a>📄 `tests/test_early_stopping.py`
- 功能：早停机制测试。
- 关键内容：构造易触发早停的配置，检查训练在 `total_updates` 前提前结束。
- 项目作用：保证收敛判定逻辑可用。

## <a id="f068"></a>📄 `tests/test_env_reset_shapes.py`
- 功能：环境 reset 观测形状测试。
- 关键内容：校验 `own/users/sats/nbrs` 与 mask 的维度与配置一致。
- 项目作用：保障网络输入接口稳定。

## <a id="f069"></a>📄 `tests/test_env_step_invariants.py`
- 功能：环境 step 不变量测试。
- 关键内容：连续执行多步验证 GU/UAV/SAT 队列非负且有限；并覆盖 active 尾部惩罚公式、线性避碰限幅与 centroid 交叉退火权重迁移的单元测试。
- 项目作用：防止队列更新出现数值异常。

## <a id="f070"></a>📄 `tests/test_run_dir_paths.py`
- 功能：脚本路径解析测试。
- 关键内容：覆盖 `train/evaluate/render` 中 run_dir、run_id、默认输出路径逻辑。
- 项目作用：避免日志路径和输出文件落错位置。

## <a id="f071"></a>📄 `tests/test_sat_queue_tracking.py`
- 功能：卫星队列收支一致性测试。
- 关键内容：检查 `incoming/processed` 形状与边界，并验证 `sat_queue` 更新满足计算容量约束。
- 项目作用：保证卫星计算队列模型正确。

## <a id="f072"></a>📄 `tests/test_smoke_train.py`
- 功能：最小训练冒烟测试。
- 关键内容：小配置（单环境）跑 1 次 update，验证训练主流程可执行。
- 项目作用：在 CI 或本地快速发现严重回归。

## <a id="f077"></a>📄 `tests/test_smoke_train_vec.py`
- 功能：并行训练冒烟测试。
- 关键内容：小配置下创建向量环境（`num_envs=2`）运行 1 次 update，验证并行 rollout 训练链路可执行。
- 项目作用：防止多环境并行相关改动引入回归。

## <a id="f075"></a>📄 `tests/test_traffic_curriculum.py`
- 功能：流量课程学习测试。
- 关键内容：验证 `traffic_level` 在 `reset` 后能正确映射为有效到达率（Level 0/1/2），并在 step 中生效。
- 项目作用：防止流量课程配置失效导致训练阶段负载不符合预期。

## <a id="f073"></a>📄 `数学模型.md`
- 功能：论文/设计中的系统数学建模文档。
- 关键内容：覆盖系统场景、通信模型、任务处理模型、时延分析、MDP/Dec-POMDP 建模与奖励构造。
- 项目作用：提供环境与算法实现的理论依据。

## <a id="f074"></a>📄 `算法设计.md`
- 功能：论文/设计中的算法章节文档。
- 关键内容：讨论 Actor/Critic 结构、约束感知动作掩码、优先级采样、课程学习流程与复杂度分析。
- 项目作用：为 `rl/` 与 `env/` 的工程实现提供方法论说明。
