# 3UAV/20GU 单阶段 PPO 与动作效应方差诊断

日期：2026-05-02

## 1. 配置口径

基准环境使用当前 3UAV/20GU/250step 配置族，三份单阶段配置为：

- `configs/tmp/structured_single_accel_3uav_20gu_t250_ppo.yaml`
- `configs/tmp/structured_single_sat_3uav_20gu_t250_ppo.yaml`
- `configs/tmp/structured_single_bw_3uav_20gu_t250_ppo.yaml`

共同口径：

- `reward_mode: positive_weighted_workload_level`
- `bw_return_mode: gae`
- `structured_step_bootstrap_stage: accel`
- `critic_value_mode: relational`
- `actor_advantage_normalize_enabled: true`
- 只训单阶段 actor，但 PPO 时 critic 三个 stage head 都训练。

执行源：

| 训练阶段 | accel 执行 | sat 执行 | bw 执行 |
|---|---|---|---|
| accel-only | policy | queue_aware | queue_aware |
| sat-only | cluster_center_queue_aware | policy | queue_aware |
| bw-only | cluster_center_queue_aware | queue_aware | policy |

## 2. 50-update 训练结果

运行目录：

- `runs/diagnostics/single_stage_3uav20gu_accel_ppo_positive_u50`
- `runs/diagnostics/single_stage_3uav20gu_sat_ppo_positive_u50`
- `runs/diagnostics/single_stage_3uav20gu_bw_ppo_positive_u50`

| 阶段 | update1 reward | update50 reward | update1 ep_len | update50 ep_len | update50 EV |
|---|---:|---:|---:|---:|---:|
| accel | 0.1404 | 0.1372 | 250.0 | 203.0 | 0.597 |
| sat | 0.1806 | 0.1783 | 250.0 | 250.0 | 0.750 |
| bw | 0.1815 | 0.1752 | 250.0 | 250.0 | 0.939 |

结论：三个单阶段 PPO 都没有显示出稳定提升。critic EV 能升起来，尤其 BW 到 0.94，但 actor 端 reward 仍基本横盘或略降。

注意：本配置 `ppo_epochs=1`，`approx_kl_*`/`clip_frac_*` 为 0 不能直接解释为 actor 完全没更新，因为很多实现是在唯一一次 minibatch step 前计算 old/new logprob 差。

## 3. 动作效应方差诊断方法

新增脚本：

`scripts/diagnose_structured_action_effect_variance.py`

诊断口径：

1. 采样 stage snapshot：`t=0,100,200`，4 个 episode seed。
2. 对每个 snapshot 保存完整 runtime/RNG state。
3. 在同一个 snapshot 上反复 reload，保证外生随机一致。
4. 第一拍只替换当前 stage 的动作：deterministic ref action vs stochastic sampled actions。
5. 后续 20 步使用 deterministic current policy / baseline follow。
6. 比较两种方差：
   - `action_var`：同一状态、同一未来随机 tape，只换第一拍动作带来的 return 方差。
   - `trajectory_var`：同一 t、不同 episode seed 下 ref return 的方差。

选择 `horizon_steps=20` 的原因：当前 `gamma=0.995, gae_lambda=0.95`，`1 / (1 - gamma * lambda) ≈ 18.5`，20 步基本覆盖 PPO/GAE 的主要信用长度，同时避免 full-episode return 被剩余时长主导。

输出目录：

`runs/diagnostics/action_effect_variance_3uav20gu_positive_u50`

## 4. 诊断结果

| 阶段 | rows | ref return std | action return std | action_var / trajectory_var | mean abs delta / ref std |
|---|---:|---:|---:|---:|---:|
| accel | 8 | 0.790 | 0.0319 | 0.00586 | 0.0505 |
| sat | 12 | 0.327 | 0.0359 | 0.02235 | 0.1383 |
| bw | 12 | 4.846 | 0.0730 | 0.00054 | 0.0265 |

同一 t 分组下：

- accel：`action_var / trajectory_var` 均值约 0.015，中位数约 0.0013。
- sat：整体约 0.022；`t=200` 因 ref return 跨 seed 方差极小，比例被放大，不能单独代表主结论。
- bw：同一 t 均值约 0.0013，中位数约 0.00049，最清楚地显示动作影响被轨迹/状态波动淹没。

## 5. 解释

BW 最符合“PPO 高方差、动作效应被淹没”的判断。随机 BW 动作和 deterministic ref 的 L1 距离并不小，平均约 0.79，但 20-step return 的动作方差只有轨迹方差的约 0.054%。这说明 PPO 的 sampled-action advantage 很容易主要反映“这个状态/traffic/队列轨迹难不难”，而不是“这次 BW 动作好不好”。

accel 也被淹没得比预期明显。随机 accel 动作距离 ref 不小，平均 L2 约 1.50，但动作方差只有轨迹方差的约 0.6%。这说明在当前 positive workload reward + queue-aware sat/bw follow 下，20 步内移动动作对 workload return 的边际影响也很弱。

sat 的动作效应相对最大，但整体仍小于轨迹波动很多。它可能比 BW 更接近 PPO 可学区域，但当前 50 updates 仍没有稳定 reward 提升。

整体判断：当前三阶段里，PPO 的问题不只是 critic EV 低。即使 critic 能解释大量 state-level return 结构，sampled action 对 return 的边际影响仍比状态/轨迹波动小很多，导致 PPO gradient 的有效信噪比很低。BW 最严重，accel 也明显，sat 稍好但仍不强。
