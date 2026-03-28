# SAGIN-MARL

面向空天地一体化网络（SAGIN）的多智能体强化学习仿真与训练代码。环境采用 PettingZoo `ParallelEnv` 接口，建模 UAV（无人机）、GU（地面用户）与卫星的接入、回传、计算与队列演化，并使用 MAPPO 进行三阶段课程训练。

**主要特性**
- 可配置的 SAGIN 环境（用户分布、卫星轨道、队列、信道与多种物理约束开关）
- MAPPO 训练管线，自动保存模型与日志
- 支持多环境并行 rollout（`sync`/`subproc`），可利用多核 CPU 提升采样吞吐
- 评估脚本与渲染脚本（GIF）
- 训练与评估日志包含队列状态、丢包、卫星处理量、吞吐与耗时统计
- 当前正式策略使用 hybrid action：连续 `accel` + Dirichlet `bw` + masked categorical `sat`
- 当前正式奖励与主评估指标统一按 `A_ref` 归一化

**目录结构**
- `configs/`：实验配置（YAML）
- `sagin_marl/env/`：环境与物理模型
- `sagin_marl/rl/`：MAPPO、策略、分布与训练组件
- `scripts/`：训练、评估、渲染与批量训练入口
- `docs/`：模型结构、指标口径、基线与测试说明
- `tests/`：基础单元测试与 smoke tests

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

**当前推荐配置**
- Stage 1（加速度）：`configs/phase1_actions_curriculum_stage1_accel.yaml`
- Stage 2（带宽）：`configs/phase1_actions_curriculum_stage2_bw.yaml`
- Stage 3（卫星）：`configs/phase1_actions_curriculum_stage3_sat.yaml`

这三份正式 YAML 当前共享以下设定：
- `actor_encoder_type: set_pool`
- `reward_mode: controllable_flow`
- `append_action_masks_to_obs: true`
- `bw_policy: dirichlet`
- `sat_policy: masked_categorical`
- `checkpoint_eval_interval_updates: 50`
- `checkpoint_eval_start_update: 200`

**快速开始**
完整流程（训练 → 评估 → 查看结果）：
1. 激活虚拟环境
```powershell
.\.venv\Scripts\Activate.ps1
```
2. 可选：吞吐估算（判断到达率是否合理）
```powershell
python scripts/estimate_throughput.py --config configs/phase1_actions_curriculum_stage1_accel.yaml
```
说明：脚本会同时输出 `Arrival raw` 与 `Arrival eff`，后者和环境里的有效训练口径一致。
3. 训练 Stage 1
```powershell
python scripts/train.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --log_dir runs/phase1_actions --run_id auto --num_envs 12 --vec_backend subproc --torch_threads 2 --updates 400
```
说明：
- 终端会输出 `Run dir: ...`，下文统一用 `<RUN_DIR>` 指代
- 如需手动指定目录，可用 `--run_dir runs/phase1_actions/exp1`
- 调试时可先用 `--num_envs 2 --vec_backend sync`
4. 评估训练策略
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20
```
5. 可选：混合策略评估（accel 用训练模型，bw/sat 用 heuristic）
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20 --hybrid_bw_sat queue_aware
```
6. 评估启发式基线
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20 --baseline queue_aware
```
可选：
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20 --baseline zero_accel
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20 --baseline cluster_center
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --run_dir <RUN_DIR> --episodes 20 --baseline centroid
```
7. 渲染某个评估 episode
```powershell
python scripts/render_episode.py --config configs/phase1_actions_curriculum_stage3_sat.yaml --run_dir <RUN_DIR> --episode_seed 82003 --out <RUN_DIR>/episode_seed82003.gif
```
说明：
- 若 `evaluate.py` 使用 `--episode_seed_base B`，则第 `e` 个 episode 的渲染 seed 为 `B + e`
- baseline 渲染可直接加 `--baseline queue_aware`

baseline 渲染示例：
```powershell
python scripts/render_episode.py --config configs/phase1_actions_curriculum_stage3_sat.yaml --run_dir <RUN_DIR> --baseline queue_aware --episode_seed 82003 --out <RUN_DIR>/episode_queue_aware_seed82003.gif
```

**三阶段训练（递进）**
```powershell
python scripts/train.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --updates 1500 --log_dir runs/phase1_actions --run_id stage1_accel --num_envs 12 --vec_backend subproc --torch_threads 2
python scripts/train.py --config configs/phase1_actions_curriculum_stage2_bw.yaml --updates 1500 --log_dir runs/phase1_actions --run_id stage2_bw --num_envs 12 --vec_backend subproc --torch_threads 2 --init_actor runs/phase1_actions/stage1_accel/actor.pt --init_critic runs/phase1_actions/stage1_accel/critic.pt
python scripts/train.py --config configs/phase1_actions_curriculum_stage3_sat.yaml --updates 1600 --log_dir runs/phase1_actions --run_id stage3_sat --num_envs 12 --vec_backend subproc --torch_threads 2 --init_actor runs/phase1_actions/stage2_bw/actor.pt --init_critic runs/phase1_actions/stage2_bw/critic.pt
```

一键顺序执行正式课程训练：
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_curriculum_stage123_formal.ps1
```
说明：
- 脚本会顺序运行 `stage1 -> stage2 -> stage3`
- 每阶段结束后自动做 trained / baseline 评估
- 新启动的批量训练会在阶段目录额外写出 `console.log`

**TensorBoard 查看建议**
1. 启动
```powershell
tensorboard --logdir runs/phase1_actions
```
分组图在 `Custom Scalars` 标签页里，训练和评估都可以在那里按主题查看。
2. 训练曲线重点看
- `x_acc`
- `x_rel`
- `g_pre`
- `d_pre`
- `episode_reward`
- `collision_rate`
3. PPO 诊断重点看
- `entropy`
- `entropy_accel`
- `entropy_bw`
- `entropy_sat`
- `approx_kl`
- `clip_frac`
- `explained_variance`
4. checkpoint 评估重点看
- `processed_ratio_eval`
- `drop_ratio_eval`
- `pre_backlog_steps_eval`
- `reward_plateau_streak`
- `early_stop_triggered`

批量导出 TensorBoard 曲线：
```powershell
python scripts/export_tb_scalars.py --logdir <RUN_DIR> --outdir <RUN_DIR>/tb_plots --format png --dpi 180
```
- 默认导出该 run 下全部 scalar tag，每个 tag 一张图。
- 可用 `--tag` 过滤指定 tag，可重复传入，例如：`--tag q_norm_* --tag queue_total_active*`
- 可选平滑：`--ema_alpha 0.9`

跨多个 run 叠加导出：
```powershell
python scripts/export_tb_scalars.py --logdir runs/phase1_actions --run stage1_* --tag q_norm_active --tag q_norm_tail_hit_rate --overlay
```
- `--overlay` 会额外生成同一 tag 的多 run 对比图。
- 结果目录包含：
- `by_run/`：每个 run 的单独曲线图
- `overlay/`：同一 tag 的多 run 叠加图

**训练与评估输出**
训练输出：
- `<RUN_DIR>/actor.pt`、`<RUN_DIR>/critic.pt`
- `<RUN_DIR>/metrics.csv`
- `<RUN_DIR>/checkpoint_eval.csv`
- `<RUN_DIR>/train_state.pt`

评估输出：
- `<RUN_DIR>/eval_trained_final.csv`
- `<RUN_DIR>/eval_queue_aware_final.csv`
- `<RUN_DIR>/eval_tb`

新批量脚本的文本日志：
- `<RUN_DIR>/console.log`

补充分析：
- 训练指标分析：`python scripts/analyze_metrics.py --run_dir <RUN_DIR> --window 20`
- 指标口径说明：`docs/metrics_guide.md`
- 模型结构说明：`docs/model_architecture.md`

查看训练结果：
- 训练指标：`<RUN_DIR>/metrics.csv`
- TensorBoard：`tensorboard --logdir <RUN_DIR>`

查看评估结果：
- 评估指标：`<RUN_DIR>/eval_trained_final.csv`、`<RUN_DIR>/eval_queue_aware_final.csv`
- 评估 TensorBoard：`<RUN_DIR>/eval_tb`

**进度条与早停**
训练与评估会显示进度条。

当前正式配置的 checkpoint-eval early stop 逻辑：
- 从 `checkpoint_eval_start_update` 开始
- 每 `checkpoint_eval_interval_updates` 做一次 checkpoint eval
- 若 `reward_sum` 连续若干次未达到最小相对改进
- 且 `collision_episode_fraction` 不高于门限
- 则提前停止

更详细的字段与判定说明见 `docs/metrics_guide.md`。

**环境接口**
环境类：`SaginParallelEnv`（PettingZoo Parallel API）

观测（每个 UAV 一个字典）：
- `own`: `(7,)`
- `danger_nbr`: `(5,)`，最危险邻机摘要（若启用）
- `users`: `(users_obs_max, 5)`
- `users_mask`: `(users_obs_max,)`
- `bw_valid_mask`: `(users_obs_max,)`
- `sats`: `(sats_obs_max, 12)`
- `sats_mask`: `(sats_obs_max,)`
- `sat_valid_mask`: `(sats_obs_max,)`
- `nbrs`: `(nbrs_obs_max, 4)`
- `nbrs_mask`: `(nbrs_obs_max,)`

动作（每个 UAV 一个字典）：
- `accel`: `(2,)`
- `bw_alloc`: `(users_obs_max,)`
- `sat_select_mask`: `(sats_obs_max,)`

说明：
- `bw_alloc` 是最终带宽份额，不再是 residual logit
- `sat_select_mask` 是最终多热选择，不再是环境二次采样输入

**候选 GU 机制（重要）**
- `candidate_mode=assoc`：候选集合由当前已关联 GU 组成
- `candidate_mode=nearest|radius`：候选集合按空间规则筛选
- 无论候选集如何形成，真正执行带宽分配时都只会在有效关联子集上生效
- `candidate_k` 控制候选数量上限，不改变网络输入维度，超出部分由 mask 置零

**配置说明**
默认配置见 `sagin_marl/env/config.py`。常用参数包括：
- 规模：`num_uav`、`num_gu`、`num_sat`
- 时域：`tau0`、`T_steps`
- 观测截断：`users_obs_max`、`sats_obs_max`、`nbrs_obs_max`
- 物理约束：`v_max`、`a_max`、`d_safe`、`boundary_mode`
- 通信与噪声：`b_acc`、`b_sat_total`、`gu_tx_power`、`uav_tx_power`、`noise_density`
- 卫星算力：`sat_cpu_freq`、`task_cycles_per_bit`
- 动作与训练开关：`enable_bw_action`、`fixed_satellite_strategy`、`train_accel`、`train_bw`、`train_sat`
- 正式动作分布：`bw_policy`、`bw_alpha_floor`、`sat_policy`、`sat_num_select`
- 正式奖励：`reward_mode`、`reward_w_access`、`reward_w_relay`、`reward_w_pre_drop`、`reward_w_pre_growth`
- 启发式基线：`baseline_accel_gain`、`baseline_assoc_bonus`、`baseline_sat_queue_penalty`、`baseline_repulse_gain`
- PPO 超参：`buffer_size`、`num_mini_batch`、`ppo_epochs`、`actor_lr`、`critic_lr`
- 并行参数（命令行）：`--num_envs`、`--vec_backend`、`--torch_threads`

**最小使用示例**
```python
from sagin_marl.env.config import load_config
from sagin_marl.env.sagin_env import SaginParallelEnv

cfg = load_config("configs/phase1_actions_curriculum_stage1_accel.yaml")
env = SaginParallelEnv(cfg)
obs, infos = env.reset()
```

**策略对照实验**
Stage 1 accel 对照：
```powershell
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline fixed --episode_seed_base 62000 --out runs/stage1_accel/eval_fixed_n20.csv
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline random_accel --episode_seed_base 62000 --out runs/stage1_accel/eval_random_accel_n20.csv
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline centroid --episode_seed_base 62000 --out runs/stage1_accel/eval_centroid_n20.csv
python scripts/evaluate.py --config configs/phase1_actions_curriculum_stage1_accel.yaml --episodes 20 --baseline cluster_center --episode_seed_base 62000 --out runs/stage1_accel/eval_cluster_center_n20.csv
```

Stage 2/3 资源动作对照：
- Stage 2 更建议比较“同时有 `accel + bw` 的方法”，例如 `stage1 actor + queue_aware_bw`
- Stage 3 更建议比较“同时有 `accel + bw + sat` 的方法”，例如 `stage2 actor + queue_aware_sat`

**相关文档**
- `docs/metrics_guide.md`
- `docs/model_architecture.md`
- `docs/heuristic_baselines.md`
- `docs/tests_overview.md`
- `PROJECT_STRUCTURE.md`

**测试**
```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests -q --import-mode=importlib
```
