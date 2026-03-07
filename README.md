# SAGIN-MARL

面向空天地一体化网络（SAGIN）的多智能体强化学习仿真与训练代码。环境采用 PettingZoo `ParallelEnv` 接口，建模 UAV（无人机）、GU（地面用户）与卫星的接入、回传和队列演化，并使用 MAPPO 进行训练。

**主要特性**
- 可配置的 SAGIN 环境（用户分布、卫星轨道、队列、信道与多种物理约束开关）
- MAPPO 训练管线，自动保存模型与日志
- 支持多环境并行 rollout（`sync`/`subproc`），可利用多核 CPU 提升采样吞吐
- 评估脚本与渲染脚本（GIF）
- 训练与评估日志包含队列状态、丢包与卫星处理量、用时与吞吐统计
- 训练支持早停（基于奖励滑动均值的收敛判定）

**目录结构**
- `configs/`：实验配置（YAML）
- `sagin_marl/env/`：环境与物理模型
- `sagin_marl/rl/`：MAPPO、策略与训练组件
- `scripts/`：训练、评估与渲染入口
- `tests/`：基础单元测试

**安装**
1. 创建虚拟环境（可选）
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. 安装依赖
```powershell
python -m pip install -r requirements.txt
```

**快速开始**
完整流程（训练 → 评估 → 查看结果）：
1. 激活虚拟环境（如果已创建）
```powershell
.\.venv\Scripts\Activate.ps1
```
说明：阶段一当前默认使用 `configs/phase1_actions_curriculum_stage1_accel.yaml`，关键条件/开关为：
- `enable_bw_action=false`（仅训练加速度）
- `fixed_satellite_strategy=true`（卫星策略固定）
- `avoidance_enabled=true`（避障安全层）

可选：吞吐估算（判断到达率是否合理）
```powershell
python scripts/estimate_throughput.py --config configs/phase1_actions_curriculum_stage1_accel.yaml
```
说明：脚本会同时输出 `Arrival raw`（`num_gu * task_arrival_rate * tau0`）与 `Arrival eff`（与环境 `effective_task_arrival_rate` 一致的训练口径）。
2. 训练（自动生成独立目录，避免多次流程数据堆在一起）
```powershell
python scripts/train.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --log_dir runs/phase1_actions --run_id auto --num_envs 8 --vec_backend subproc --torch_threads 8 --updates 400
```
说明：终端会输出 `Run dir: runs/phase1_actions/20260204_121530`，下文用 `<RUN_DIR>` 指代该目录。
如需手动指定目录，可用 `--run_dir runs/phase1_actions/exp1`。
调试时可先用 `--num_envs 2 --vec_backend sync`，确认流程后再切到 `subproc`。
3. 评估（训练模型）
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20
```
可选：混合策略评估（accel 用训练模型，bw/sat 用 queue_aware）
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20 --hybrid_bw_sat queue_aware
```
4. 评估（启发式基线，推荐）
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20 --baseline queue_aware
```
可选：零加速度基线（更弱，但便于对照）
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20 --baseline zero_accel
```
5. 查看训练结果
- 训练指标：`<RUN_DIR>/metrics.csv`
- TensorBoard（训练 + 评估）：`tensorboard --logdir <RUN_DIR>`
- 训练指标分析脚本（滚动均值 + 斜率）：`python scripts/analyze_metrics.py --run_dir <RUN_DIR> --window 20`
6. 查看评估结果
- 评估指标：`<RUN_DIR>/eval_trained.csv`, `<RUN_DIR>/eval_baseline.csv`
- 评估 TensorBoard：`<RUN_DIR>/eval_tb`（tags: `eval/trained`, `eval/baseline`）

**当前推荐配置**
- Stage 1（加速度）：`configs/phase1_actions_curriculum_stage1_accel.yaml`
- Stage 2（带宽）：`configs/phase1_actions_curriculum_stage2_bw.yaml`
- Stage 3（卫星选择）：`configs/phase1_actions_curriculum_stage3_sat.yaml`

**三阶段训练（递进）**
```powershell
python scripts/train.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --updates 400 --log_dir runs/phase1_actions --run_id stage1_accel
python scripts/train.py --config configs/phase1_actions_curriculum_stage2_bw.yaml --updates 500 --log_dir runs/phase1_actions --run_id stage2_bw --init_actor runs/phase1_actions/stage1_accel/actor.pt --init_critic runs/phase1_actions/stage1_accel/critic.pt
python scripts/train.py --config configs/phase1_actions_curriculum_stage3_sat.yaml --updates 500 --log_dir runs/phase1_actions --run_id stage3_sat --init_actor runs/phase1_actions/stage2_bw/actor.pt --init_critic runs/phase1_actions/stage2_bw/critic.pt
```

**TensorBoard 查看建议（当前采用方案）**
1. 启动：
```powershell
tensorboard --logdir runs/phase1_actions
```
2. 训练曲线（Train）：
在 Runs 列表勾选 `stage1_accel` / `stage2_bw` / `stage3_sat`，查看 Scalars：
`episode_reward`, `reward_raw`, `gu_queue_mean`, `centroid_dist_mean`, `service_norm`。
3. 评估曲线（Eval）：
评估日志在 `<RUN_DIR>/eval_tb`，TensorBoard 左侧会出现 `eval_tb` 子目录。  
重点查看 `eval/trained/*` 与 `eval/baseline/*`，或在 `Custom Scalars` 中查看分组：
`Eval/Queues`, `Eval/Association`, `Eval/Reward`。
4. 只看某个方案：
在 Runs 中仅勾选对应 run（如 `stage1_accel`），并展开 `eval_tb` 查看评估曲线。

以下命令默认使用 `<RUN_DIR>`（上一步输出的 Run dir），如未设置请替换为具体目录。

训练：
```powershell
python scripts/train.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --log_dir runs/phase1_actions --run_id auto --num_envs 8 --vec_backend subproc --torch_threads 8 --updates 400
```

评估（输出 CSV）：
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20
```
评估（混合策略：accel=训练模型，bw/sat=queue_aware）：
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20 --hybrid_bw_sat queue_aware
```

评估（启发式基线，推荐）：
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20 --baseline queue_aware
```

评估（零加速度基线）：
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20 --baseline zero_accel
```

渲染一条轨迹（输出 GIF）：
```powershell
python scripts/render_episode.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR>
```

TensorBoard：
```powershell
tensorboard --logdir <RUN_DIR>
```
- 分组图在 `Custom Scalars` 标签页（训练 + 评估）。

吞吐估算（判断到达率是否合理）：
```powershell
python scripts/estimate_throughput.py --config configs/phase1_actions_curriculum_stage1_accel.yaml
```
说明：优先使用 `Arrival eff` / `Util eff` 与训练日志（`arrival_sum`、`service_norm`）对照。

**训练与评估输出**
训练输出：
- `<RUN_DIR>/actor.pt`、`<RUN_DIR>/critic.pt`
- `<RUN_DIR>/metrics.csv`（同时写入 TensorBoard）

评估输出：
- `<RUN_DIR>/eval_trained.csv`、`<RUN_DIR>/eval_baseline.csv`
- 指标说明文档：`docs/metrics_guide.md`
- 模型结构说明：`docs/model_architecture.md`

日志包含关键指标（部分示例）：
- `episode_reward`、`policy_loss`、`value_loss`、`entropy`
- `reward_raw`、`service_norm`、`drop_norm`、`centroid_dist_mean`
- `gu_queue_mean`、`uav_queue_mean`、`sat_queue_mean`
- `gu_queue_max`、`uav_queue_max`、`sat_queue_max`
- `gu_drop_sum`、`uav_drop_sum`
- `sat_processed_sum`、`sat_incoming_sum`
- `update_time_sec`、`rollout_time_sec`、`optim_time_sec`
- `env_steps_per_sec`、`update_steps_per_sec`
- 训练指标分析脚本：`python scripts/analyze_metrics.py --run_dir <RUN_DIR> --window 20 --metrics episode_reward,policy_loss,value_loss,entropy`

**进度条与早停**
训练与评估会显示进度条。训练支持早停（基于奖励滑动均值），相关参数在配置文件中：
- `early_stop_enabled`
- `early_stop_min_updates`
- `early_stop_window`
- `early_stop_patience`
- `early_stop_min_delta`

**环境接口**
环境类：`SaginParallelEnv`（PettingZoo Parallel API）

观测（每个 UAV 一个字典）：
- `own`: `(7,)`，自身状态（位置、速度、能量、队列、时间）
- `users`: `(users_obs_max, 5)`，候选 GU 特征
- `users_mask`: `(users_obs_max,)`
- `sats`: `(sats_obs_max, 9)`，可见卫星特征
- `sats_mask`: `(sats_obs_max,)`
- `nbrs`: `(nbrs_obs_max, 4)`，邻居 UAV 特征
- `nbrs_mask`: `(nbrs_obs_max,)`

动作（每个 UAV 一个字典）：
- `accel`: `(2,)`，二维加速度（归一化后再乘以 `a_max`）
- `bw_logits`: `(users_obs_max,)`，带宽分配权重（`enable_bw_action=true` 生效）
- `sat_logits`: `(sats_obs_max,)`，卫星选择权重（`fixed_satellite_strategy=false` 生效）

**候选 GU 机制（重要）**
- `candidate_mode=assoc`：候选集合由“当前关联到该 UAV 的 GU”组成。即使全局接入率接近 1，每个 UAV 看到的仍是自己服务的子集。
- `candidate_mode=nearest|radius`：候选集合按距离筛选，视野更广，但**带宽分配仍只对 `assoc==u` 的 GU 生效**，不会出现多个 UAV 同时给同一 GU 分配带宽。
- `candidate_k`：控制候选数量上限（不改变 `users_obs_max`，网络输入维度保持不变，超出部分由 mask 置零）。

**配置说明**
默认配置见 `sagin_marl/env/config.py`，课程阶段建议从 `configs/phase1_actions_curriculum_stage1_accel.yaml` 开始。常用参数：
- 规模：`num_uav`、`num_gu`、`num_sat`
- 时域：`tau0`、`T_steps`
- 观测截断：`users_obs_max`、`sats_obs_max`、`nbrs_obs_max`
- 物理约束：`v_max`、`a_max`、`d_safe`、`boundary_mode`
- 通信与噪声：`b_acc`、`b_sat_total`、`gu_tx_power`、`uav_tx_power`、`noise_density`、`pathloss_mode`
- 天线增益与卫星算力：`uav_tx_gain`、`sat_rx_gain`、`sat_cpu_freq`
- 机制开关：`enable_bw_action`、`fixed_satellite_strategy`、`doppler_enabled`、`energy_enabled`
- 奖励与候选：`reward_tanh_enabled`、`candidate_mode`、`candidate_k`、`candidate_radius`、`queue_topk_local`
- 基线启发式（queue_aware）：`baseline_accel_gain`、`baseline_assoc_bonus`、`baseline_sat_queue_penalty`、`baseline_repulse_gain`、`baseline_repulse_radius_factor`、`baseline_energy_low`、`baseline_energy_weight`
- 训练超参：`buffer_size`、`num_mini_batch`、`ppo_epochs`、`actor_lr`、`critic_lr`
- 并行训练参数（命令行）：`--num_envs`、`--vec_backend`、`--torch_threads`

**测试**
- 测试文件说明：`docs/tests_overview.md`
```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

**最小使用示例**
```python
from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv

cfg = load_config("configs/phase1_actions_curriculum_stage1_accel.yaml")
env = SaginParallelEnv(cfg)
obs, infos = env.reset()
```

## 策略对照实验（固定 vs 随机 vs 追质心）

目标：在同一环境配置下，比较三种机动策略对队列 KPI 的影响。

1. 激活虚拟环境
```powershell
.\.venv\Scripts\Activate.ps1
```

2. 在同一配置、同一 episode 种子序列下分别评估三种策略（示例 `N=20`）
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline fixed --episode_seed_base 42000 --out runs/stage1_accel/eval_fixed_n20.csv
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline random_accel --episode_seed_base 42000 --out runs/stage1_accel/eval_random_accel_n20.csv
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline centroid --episode_seed_base 42000 --out runs/stage1_accel/eval_centroid_n20.csv
```

3. 汇总对照指标（`queue_total_active` 的 mean/P95/P99，`outflow_arrival_ratio` 的 mean/P05，`drop_ratio` 的 mean）
```powershell
python scripts/summarize_policy_kpi.py --input fixed=runs/stage1_accel/eval_fixed_n20.csv random=runs/stage1_accel/eval_random_accel_n20.csv centroid=runs/stage1_accel/eval_centroid_n20.csv
```

4. 关键输出文件
- `runs/stage1_accel/eval_fixed_n20.csv`
- `runs/stage1_accel/eval_random_accel_n20.csv`
- `runs/stage1_accel/eval_centroid_n20.csv`
- `scripts/summarize_policy_kpi.py`

## 补充策略对照：带宽分配与卫星选择
同上，分别评估 `queue_aware_bw` 和 `queue_aware_sat` 基线，比较带宽分配和卫星选择策略对 KPI 的影响。三种带宽策略（uniform_bw、random_bw、queue_aware_bw）和三种卫星策略（uniform_sat、random_sat、queue_aware_sat）：平均分配、随机分配、基于队列状态的分配。

```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline uniform_bw --episode_seed_base 42000 --out runs/stage1_accel/eval_uniform_bw_n20.csv
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline random_bw --episode_seed_base 42000 --out runs/stage1_accel/eval_random_bw_n20.csv
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline queue_aware_bw --episode_seed_base 42000 --out runs/stage1_accel/eval_queue_aware_bw_n20.csv
```

```powershell
python scripts/summarize_policy_kpi.py --input uniform=runs/stage1_accel/eval_uniform_bw_n20.csv random=runs/stage1_accel/eval_random_bw_n20.csv queue_aware=runs/stage1_accel/eval_queue_aware_bw_n20.csv
```
```
---
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline uniform_sat --episode_seed_base 42000 --out runs/stage1_accel/eval_uniform_sat_n20.csv
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline random_sat --episode_seed_base 42000 --out runs/stage1_accel/eval_random_sat_n20.csv
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline queue_aware_sat --episode_seed_base 42000 --out runs/stage1_accel/eval_queue_aware_sat_n20.csv
```

```powershell
python scripts/summarize_policy_kpi.py --input uniform=runs/stage1_accel/eval_uniform_sat_n20.csv random=runs/stage1_accel/eval_random_sat_n20.csv queue_aware=runs/stage1_accel/eval_queue_aware_sat_n20.csv
```