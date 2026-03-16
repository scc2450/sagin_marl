# Metrics Guide

本文档说明当前代码版本实际会写出的指标，只覆盖下面三类输出：

- 训练日志：`runs/<log_dir>/metrics.csv` 和训练 TensorBoard
- 评估日志：`eval_*.csv` 和评估 TensorBoard
- checkpoint 评估：`runs/<log_dir>/checkpoint_eval.csv`

如果你打开的是更早版本的 run，可能会看到更多旧字段；那些旧字段不属于本文档讨论范围。

## 1. 先看横坐标

当前项目里，不同指标确实不该共用同一种横坐标。先把这个约定看清楚，后面读表会轻松很多。

### 1.1 横坐标类别

- `update`
  - 指 PPO 参数更新次数。
  - 适合优化器和 PPO 诊断类指标。
- `total_env_steps`
  - 指累计环境交互步数，已经把并行环境都加总进去了。
  - 适合吞吐、队列、安全等“策略在环境里实际跑出来的表现”。
- `eval_episode`
  - 指评估脚本里的 episode 序号。
  - 适合单次评估输出。
- `checkpoint_update`
  - 指 checkpoint 对应的训练 update 编号。
  - 适合 `checkpoint_eval.csv`。

### 1.2 一个很重要的细节

训练 CSV 里的首列 `step` 仍然始终是 `update`，不会因为指标类型不同而改变。

也就是说：

- `metrics.csv` 里所有行索引都是 `update`
- 但训练 TensorBoard 里，环境表现类指标已经改成用 `total_env_steps`
- 训练 TensorBoard 里，PPO 诊断类指标仍然用 `update`

因此，本文后面的“横坐标”列说的是“你应该怎么理解和看它”，不是说所有文件都真的换了同一个索引列。

## 2. 聚合口径总览

为了避免后面反复写长句，先约定几个聚合口径名称：

- `最近完整 episode 滚动均值`
  - 在训练中，对最近 `train_episode_stat_window` 个已完成 episode 做滚动平均。
  - 默认窗口大小是 `100`。
- `当前 rollout step 均值`
  - 当前 update 这一整块采样数据里，对所有 env-step 求平均。
- `当前 update 优化统计`
  - 当前 update 里，跨 PPO epoch 和 minibatch 聚合出来的诊断量。
- `单个 eval episode`
  - 一条评估 episode 的结果。
- `checkpoint 多 episode 均值`
  - 一个 checkpoint 在多条 eval episode 上的平均结果。

## 3. 当前指标汇总

### 3.1 训练指标总表

| 指标 | 输出位置 | 聚合口径 | 横坐标 | 含义一句话 |
| --- | --- | --- | --- | --- |
| `episode_reward` | CSV + TB | 最近完整 episode 滚动均值 | CSV:`update` / TB:`total_env_steps` | 完整 episode 的 undiscounted return 滚动均值 |
| `episode_length_mean` | CSV | 最近完整 episode 滚动均值 | `update` | 完整 episode 长度滚动均值 |
| `completed_episode_count` | CSV | 当前 update 计数 | `update` | 当前 update 内完成了多少条 episode |
| `rollout_reward_per_step` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均每步 reward |
| `episode_term_throughput_access` | CSV + TB | 最近完整 episode 滚动均值 | CSV:`update` / TB:`total_env_steps` | 完整 episode 上接入吞吐奖励项求和后的滚动均值 |
| `episode_term_throughput_backhaul` | CSV + TB | 最近完整 episode 滚动均值 | CSV:`update` / TB:`total_env_steps` | 完整 episode 上回传吞吐奖励项求和后的滚动均值 |
| `throughput_access_norm` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均每步接入吞吐归一化值 |
| `throughput_backhaul_norm` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均每步回传吞吐归一化值 |
| `gu_queue_mean` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均 GU 单节点队列长度 |
| `uav_queue_mean` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均 UAV 单节点队列长度 |
| `queue_total_active` | CSV + TB | 当前 rollout step 均值 | CSV:`update` / TB:`total_env_steps` | 当前 rollout 的平均活动前端总队列量，等于 GU 队列和 UAV 队列之和 |
| `collision_rate` | CSV + TB | 最近完整 episode 滚动均值 | CSV:`update` / TB:`total_env_steps` | 最近完成 episode 中发生过碰撞的比例 |
| `policy_loss` | CSV + TB | 当前 update 优化统计 | `update` | PPO actor 损失 |
| `value_loss` | CSV + TB | 当前 update 优化统计 | `update` | critic 的 MSE 损失 |
| `explained_variance` | CSV + TB | 当前 update 优化统计 | `update` | critic 对 return target 的解释度 |
| `entropy` | CSV + TB | 当前 update 优化统计 | `update` | 当前策略熵 |
| `approx_kl` | CSV + TB | 当前 update 优化统计 | `update` | PPO 近似 KL |
| `clip_frac` | CSV + TB | 当前 update 优化统计 | `update` | PPO 裁剪比例 |
| `danger_imitation_loss` | CSV + TB | 当前 update 优化统计 | `update` | 危险模仿损失 |
| `danger_imitation_coef` | CSV + TB | 当前 update 当前值 | `update` | 危险模仿损失系数 |
| `danger_imitation_active_rate` | CSV + TB | 当前 rollout step 均值 | `update` | rollout 中危险模仿掩码激活的频率 |
| `actor_lr` | CSV | 当前 update 当前值 | `update` | actor 学习率 |
| `critic_lr` | CSV | 当前 update 当前值 | `update` | critic 学习率 |
| `update_time_sec` | CSV | 当前 update 运行统计 | `update` | 一次 update 总耗时 |
| `rollout_time_sec` | CSV | 当前 update 运行统计 | `update` | rollout 采样耗时 |
| `optim_time_sec` | CSV | 当前 update 运行统计 | `update` | PPO 优化耗时 |
| `env_steps_per_sec` | CSV | 当前 update 运行统计 | `update` | rollout 阶段环境步吞吐率 |
| `update_steps_per_sec` | CSV | 当前 update 运行统计 | `update` | 整个 update 的环境步吞吐率 |
| `total_env_steps` | CSV | 累积计数 | `update` | 截止当前 update 的累计环境步数 |
| `total_time_sec` | CSV | 累积计时 | `update` | 从训练开始或恢复后累计的墙钟时间 |

### 3.2 评估指标总表

| 指标 | 输出位置 | 聚合口径 | 横坐标 | 含义一句话 |
| --- | --- | --- | --- | --- |
| `episode` | CSV | 单个 eval episode | `eval_episode` | 当前评估 episode 序号 |
| `reward_sum` | CSV + TB | 单个 eval episode | `eval_episode` | 该 episode 的 undiscounted reward sum |
| `steps` | CSV | 单个 eval episode | `eval_episode` | 该 episode 实际运行步数 |
| `throughput_access_norm` | CSV + TB | 单个 eval episode | `eval_episode` | 该 episode 的平均每步接入吞吐归一化值 |
| `throughput_backhaul_norm` | CSV + TB | 单个 eval episode | `eval_episode` | 该 episode 的平均每步回传吞吐归一化值 |
| `sat_processed_norm` | CSV + TB | 单个 eval episode | `eval_episode` | 该 episode 的平均每步卫星处理量归一化值 |
| `gu_queue_mean` | CSV + TB | 单个 eval episode | `eval_episode` | 该 episode 的平均 GU 单节点队列长度 |
| `uav_queue_mean` | CSV + TB | 单个 eval episode | `eval_episode` | 该 episode 的平均 UAV 单节点队列长度 |
| `gu_queue_arrival_steps_p95` | CSV + TB | 单个 eval episode | `eval_episode` | GU 总队列的 step 级 p95，再换算成“平均到达量等价步数” |
| `uav_queue_arrival_steps_p95` | CSV + TB | 单个 eval episode | `eval_episode` | UAV 总队列的 step 级 p95，再换算成“平均到达量等价步数” |
| `terminated_early` | CSV + TB | 单个 eval episode | `eval_episode` | 是否早于时间上限结束 |
| `collision` | CSV + TB | 单个 eval episode | `eval_episode` | 该 episode 是否发生过碰撞 |
| `gu_queue_drift_ratio` | CSV + TB | 单个 eval episode | `eval_episode` | GU 总队列的净漂移速度，相对平均到达量归一化 |
| `uav_queue_drift_ratio` | CSV + TB | 单个 eval episode | `eval_episode` | UAV 总队列的净漂移速度，相对平均到达量归一化 |

### 3.3 Checkpoint 评估指标总表

`checkpoint_eval.csv` 只有在启用了 checkpoint eval 时才会存在。

| 指标 | 输出位置 | 聚合口径 | 横坐标 | 含义一句话 |
| --- | --- | --- | --- | --- |
| `update` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 该行对应的训练 update 编号 |
| `checkpoint_suffix` | CSV | checkpoint 元信息 | `checkpoint_update` | checkpoint 文件名后缀，例如 `u0030` |
| `episodes` | CSV | checkpoint 元信息 | `checkpoint_update` | 本次 checkpoint eval 使用的 episode 数 |
| `reward_sum` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | checkpoint 的平均 episode return |
| `throughput_access_norm_mean` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | checkpoint 的平均接入吞吐归一化值 |
| `throughput_backhaul_norm_mean` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | checkpoint 的平均回传吞吐归一化值 |
| `gu_queue_arrival_steps_mean` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | checkpoint 的平均 GU 队列等价到达步数 |
| `uav_queue_arrival_steps_mean` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | checkpoint 的平均 UAV 队列等价到达步数 |
| `sat_queue_arrival_steps_mean` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | checkpoint 的平均卫星队列等价到达步数 |
| `sat_drop_ratio` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | checkpoint 的平均卫星丢包占到达量比例 |
| `collision_episode_fraction` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | checkpoint 评估中发生碰撞的 episode 比例 |
| `fixed_reward_sum` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 固定参考策略的平均 episode return |
| `fixed_throughput_access_norm_mean` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 固定参考策略的平均接入吞吐归一化值 |
| `fixed_throughput_backhaul_norm_mean` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 固定参考策略的平均回传吞吐归一化值 |
| `fixed_gu_queue_arrival_steps_mean` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 固定参考策略的平均 GU 队列等价到达步数 |
| `fixed_uav_queue_arrival_steps_mean` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 固定参考策略的平均 UAV 队列等价到达步数 |
| `fixed_sat_queue_arrival_steps_mean` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 固定参考策略的平均卫星队列等价到达步数 |
| `fixed_sat_drop_ratio` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 固定参考策略的平均卫星丢包占到达量比例 |
| `fixed_collision_episode_fraction` | CSV | checkpoint 多 episode 均值 | `checkpoint_update` | 固定参考策略的碰撞 episode 比例 |
| `gu_improved` | CSV | checkpoint 状态标志 | `checkpoint_update` | 相比历史最好值，GU 队列是否达到“足够改善” |
| `uav_improved` | CSV | checkpoint 状态标志 | `checkpoint_update` | 相比历史最好值，UAV 队列是否达到“足够改善” |
| `front_improved` | CSV | checkpoint 状态标志 | `checkpoint_update` | GU 或 UAV 任一前端队列是否改善 |
| `sat_drop_worsened` | CSV | checkpoint 状态标志 | `checkpoint_update` | 相比上一 checkpoint，卫星丢包是否恶化 |
| `sat_drop_worse_streak` | CSV | checkpoint 状态标志 | `checkpoint_update` | “卫星丢包恶化且前端未改善”的连续次数 |
| `early_stop_triggered` | CSV | checkpoint 状态标志 | `checkpoint_update` | 是否触发 checkpoint-eval 早停条件 |

## 4. 训练指标详细说明

### 4.1 完整 episode 统计

#### `episode_reward`

- 定义：最近 `train_episode_stat_window` 个已完成 episode 的 undiscounted reward sum 的滚动均值。
- 为什么现在更标准：它对应的是 RL 里更常见的 episodic return，而不是旧版的“rollout 平均每步 reward”。
- 细节：
  - 统计窗口跨并行环境共享。
  - 环境是自动 reset 的，所以一个 update 内可能完成 0 条、1 条或多条 episode。
  - 如果训练刚开始且还没有任何 episode 完成，这个值为 `0`。
- 推荐横坐标：`total_env_steps`。

#### `episode_length_mean`

- 定义：与 `episode_reward` 使用同一个滚动窗口，对完整 episode 长度取均值。
- 用途：区分“return 变差”究竟是每步表现变差，还是 episode 因碰撞等原因变短。
- 推荐横坐标：`update` 或配合 `episode_reward` 一起按 `total_env_steps` 看。

#### `completed_episode_count`

- 定义：当前 update 内完成的 episode 数量。
- 细节：
  - 因为是并行环境加 auto-reset，这个值不一定等于 `num_envs`。
  - 它可能是 `0`，也可能大于 `num_envs`。
- 用途：判断某个 update 的 episode 级统计是否由足够多的新 episode 支撑。
- 推荐横坐标：`update`。

#### `rollout_reward_per_step`

- 定义：当前 update rollout 中，所有采样 env-step 的 reward 平均值。
- 公式：`当前 rollout 总 reward / 当前 rollout 总 step 数`。
- 说明：这就是旧版 `episode_reward` 的真实含义，现在单独保留并改了名字。
- 用途：它仍然有诊断价值，但不应再拿来当主奖励曲线。
- 推荐横坐标：`total_env_steps`。

#### `episode_term_throughput_access`

- 定义：最近完整 episode 上，`term_throughput_access` 的逐 episode 求和，再做滚动均值。
- 说明：
  - 它是“奖励项”的 episode 级统计。
  - 如果不是 `throughput_only` 模式，它已经包含了配置里的吞吐奖励系数。
- 用途：看 return 上升是否真的来自接入吞吐奖励项。
- 推荐横坐标：`total_env_steps`。

#### `episode_term_throughput_backhaul`

- 定义：最近完整 episode 上，`term_throughput_backhaul` 的逐 episode 求和，再做滚动均值。
- 说明：口径与 `episode_term_throughput_access` 相同，只是对象换成回传链路。
- 用途：看 return 上升是否来自回传吞吐奖励项。
- 推荐横坐标：`total_env_steps`。

### 4.2 当前 rollout 的环境表现

#### `throughput_access_norm`

- 定义：当前 rollout 中，`throughput_access_norm` 的 step 均值。
- 单步定义：
  - 每个环境步先算 `outflow_sum / arrival_scale`
  - 其中 `arrival_scale = effective_task_arrival_rate * num_gu * tau0`
- 说明：
  - 这是“归一化接入吞吐”，不是原始吞吐量。
  - 当前实现里它与 `service_norm` 的数值定义相同，但输出名称更贴近当前关注目标。
- 推荐横坐标：`total_env_steps`。

#### `throughput_backhaul_norm`

- 定义：当前 rollout 中，`throughput_backhaul_norm` 的 step 均值。
- 单步定义：每个环境步的 `backhaul_sum / arrival_scale`。
- 含义：衡量回传链路把前端流量送入卫星处理链的强度，已经按到达量尺度归一化。
- 推荐横坐标：`total_env_steps`。

#### `gu_queue_mean`

- 定义：当前 rollout 中，GU 层“单节点平均队列长度”的 step 均值。
- 单步定义：每个环境步先对 `env.gu_queue` 做 `mean`，再跨 rollout 求平均。
- 单位：队列长度本身，不是比例。
- 推荐横坐标：`total_env_steps`。

#### `uav_queue_mean`

- 定义：当前 rollout 中，UAV 层“单节点平均队列长度”的 step 均值。
- 单步定义：每个环境步先对 `env.uav_queue` 做 `mean`，再跨 rollout 求平均。
- 单位：队列长度本身，不是比例。
- 推荐横坐标：`total_env_steps`。

#### `queue_total_active`

- 定义：当前 rollout 中，活动前端总队列量的 step 均值。
- 单步定义：`q_total_active = sum(GU 队列) + sum(UAV 队列)`。
- 说明：
  - 不包含卫星队列。
  - 这是总量指标，不是 per-node mean。
- 推荐横坐标：`total_env_steps`。

#### `collision_rate`

- 定义：最近 `train_episode_stat_window` 个已完成 episode 中，发生过至少一次碰撞的 episode 比例。
- 说明：
  - 这是 episode 级碰撞概率。
  - 它不再是旧版的 step 级 `collision_event` 平均值。
- 推荐横坐标：`total_env_steps`。

### 4.3 PPO 训练诊断

#### `policy_loss`

- 定义：当前 update 中，PPO actor 损失在所有 minibatch 上的平均值。
- 用途：观察策略更新是否稳定，但它本身不一定要越小越好，重点看是否异常抖动或爆炸。
- 推荐横坐标：`update`。

#### `value_loss`

- 定义：当前 update 中，critic 对 return target 的 MSE 损失平均值。
- 用途：观察 value function 是否失真或发散。
- 推荐横坐标：`update`。

#### `explained_variance`

- 定义：critic 预测值对当前 return target 的解释方差比例。
- 解释：
  - 越接近 `1` 通常越好。
  - 接近 `0` 表示 critic 基本没学到。
  - 明显为负时通常代表 critic 很不靠谱。
- 推荐横坐标：`update`。

#### `entropy`

- 定义：当前 update 中策略分布熵的平均值。
- 用途：看策略是否过早塌缩、探索是否迅速消失。
- 推荐横坐标：`update`。

#### `approx_kl`

- 定义：当前 update 中旧策略和新策略之间的近似 KL 平均值。
- 用途：判断每次 PPO 更新步子是否过大或过小。
- 推荐横坐标：`update`。

#### `clip_frac`

- 定义：当前 update 中，被 PPO clip 约束裁掉的样本比例平均值。
- 用途：长期很高通常意味着更新过猛，长期过低又可能说明更新很弱。
- 推荐横坐标：`update`。

### 4.4 危险模仿干预

#### `danger_imitation_loss`

- 定义：当前 update 中危险模仿附加损失的平均值。
- 说明：这是额外加到策略优化里的安全相关约束项，不是主 PPO 损失本身。
- 用途：看这个安全约束是否在主导训练。
- 推荐横坐标：`update`。

#### `danger_imitation_coef`

- 定义：当前 update 使用的危险模仿损失系数。
- 说明：如果这个值不变，那就只是一个常数监视项；如果后续配置支持调度，它会反映当下有效权重。
- 推荐横坐标：`update`。

#### `danger_imitation_active_rate`

- 定义：当前 rollout 中，环境返回的 `danger_imitation_active_rate` 的 step 均值。
- 更直白地说：它表示危险模仿掩码在 rollout 里有多频繁被触发。
- 用途：需要和 `danger_imitation_loss` 一起看。
  - `loss` 大但 `active_rate` 很低，说明是少数高强度干预。
  - `loss` 不大但 `active_rate` 很高，说明约束长期处于开启状态。
- 推荐横坐标：`update`。

### 4.5 训练运行统计

#### `actor_lr`

- 定义：当前 update 的 actor 优化器学习率。
- 推荐横坐标：`update`。

#### `critic_lr`

- 定义：当前 update 的 critic 优化器学习率。
- 推荐横坐标：`update`。

#### `update_time_sec`

- 定义：一个完整 update 的总耗时。
- 组成：包括 rollout、优化和其他 update 内部开销。
- 推荐横坐标：`update`。

#### `rollout_time_sec`

- 定义：当前 update 收集 rollout 样本的耗时。
- 推荐横坐标：`update`。

#### `optim_time_sec`

- 定义：当前 update 做 PPO 优化的耗时。
- 推荐横坐标：`update`。

#### `env_steps_per_sec`

- 定义：`当前 rollout 步数 / rollout_time_sec`。
- 含义：只看采样阶段的环境步吞吐率。
- 推荐横坐标：`update`。

#### `update_steps_per_sec`

- 定义：`当前 rollout 步数 / update_time_sec`。
- 含义：看完整 update 粒度的环境步吞吐率。
- 推荐横坐标：`update`。

#### `total_env_steps`

- 定义：截至当前 update，累计收集的环境步总数。
- 说明：并行环境的步数已经合并到一起。
- 用途：它本身也是训练 TensorBoard 里环境表现类曲线的横坐标来源。
- 推荐横坐标：`update`。

#### `total_time_sec`

- 定义：从训练开始或从 checkpoint 恢复后，累计消耗的总墙钟时间。
- 推荐横坐标：`update`。

## 5. 评估指标详细说明

### 5.1 基础结果

#### `episode`

- 定义：评估脚本中当前 episode 的序号，从 `0` 开始。
- 用途：只是索引，不是质量指标。

#### `reward_sum`

- 定义：该 eval episode 的 undiscounted reward sum。
- 说明：这是单条评估 episode 的完整 return，不做滚动平均。
- 推荐横坐标：`eval_episode`。

#### `steps`

- 定义：该 eval episode 实际运行了多少个环境步。
- 说明：
  - 若小于 `T_steps`，说明 episode 提前结束了。
  - 提前结束的原因通常可以再结合 `terminated_early` 和 `collision` 判断。
- 推荐横坐标：`eval_episode`。

### 5.2 吞吐和处理能力

#### `throughput_access_norm`

- 定义：该 episode 中，单步 `throughput_access_norm` 的平均值。
- 单步口径与训练相同：`outflow_sum / arrival_scale`。
- 含义：评估策略在接入段的平均归一化吞吐能力。

#### `throughput_backhaul_norm`

- 定义：该 episode 中，单步 `throughput_backhaul_norm` 的平均值。
- 单步口径与训练相同：`backhaul_sum / arrival_scale`。
- 含义：评估策略在回传段的平均归一化吞吐能力。

#### `sat_processed_norm`

- 定义：该 episode 中，单步 `sat_processed_norm` 的平均值。
- 单步口径：`sat_processed_sum / arrival_scale`。
- 含义：卫星处理段的平均归一化处理量。
- 备注：它不是主目标，但在怀疑瓶颈已经转移到卫星段时很有价值。

### 5.3 队列表现

#### `gu_queue_mean`

- 定义：该 episode 中，GU 层单节点平均队列长度的 episode 平均值。
- 单步做法：先对 `env.gu_queue` 取 `mean`，再跨 episode 所有步取平均。

#### `uav_queue_mean`

- 定义：该 episode 中，UAV 层单节点平均队列长度的 episode 平均值。
- 单步做法：先对 `env.uav_queue` 取 `mean`，再跨 episode 所有步取平均。

#### `gu_queue_arrival_steps_p95`

- 定义：该 episode 内，GU 总队列量的 step 级 `p95`，再除以该 episode 的平均外部到达量。
- 公式：
  - 每步先取 `sum(env.gu_queue)`
  - 对整条 episode 的这些值取 `p95`
  - 再除以 `arrival_per_step = arrival_sum_ep / steps`
- 单位解释：可以把它理解成“这个高位拥塞水平，大约等价于多少步平均到达量”。
- 说明：这是总量尾部指标，不是单节点均值尾部指标。

#### `uav_queue_arrival_steps_p95`

- 定义：与 `gu_queue_arrival_steps_p95` 相同，只是层级换成 UAV。
- 用途：观察 UAV 层是否在高拥塞尾部出现明显积压。

### 5.4 安全和终止

#### `terminated_early`

- 定义：如果 episode 的 `steps < T_steps`，则为 `1`，否则为 `0`。
- 说明：它表示“是否提前结束”，不区分提前结束的具体原因。

#### `collision`

- 定义：该 episode 中只要出现过一次 `collision_event`，就记为 `1`，否则为 `0`。
- 说明：这是 episode 级碰撞标签，不是按步平均的碰撞率。

### 5.5 队列漂移

#### `gu_queue_drift_ratio`

- 定义：GU 总队列从 episode 开始到结束的净变化速度，再按平均到达量归一化。
- 公式：
  - `((episode_end_gu_queue_sum - episode_start_gu_queue_sum) / steps) / arrival_per_step`
- 解释：
  - 大于 `0` 表示整体在积压。
  - 小于 `0` 表示整体在消化积压。
  - 绝对值越大，净变化越快。

#### `uav_queue_drift_ratio`

- 定义：与 `gu_queue_drift_ratio` 相同，只是对象换成 UAV 总队列。
- 用途：看 UAV 层是越积越多，还是在持续清空。

## 6. Checkpoint 评估指标详细说明

### 6.1 这个文件到底是什么

`checkpoint_eval.csv` 记录的是：训练进行到指定 update 时，对当前 checkpoint 做一次固定 episode 数量的评估，并把结果汇总成一行。

有两个要点：

- 当前 checkpoint actor 是按 `deterministic=True` 评估的
- `fixed_*` 字段对应的是配置里的固定参考策略，不一定总是零动作，也可能是 `queue_aware`

### 6.2 元信息字段

#### `update`

- 定义：这次 checkpoint eval 对应的训练 update 编号。
- 也是 `checkpoint_eval.csv` 最自然的横坐标。

#### `checkpoint_suffix`

- 定义：这次保存出来的 checkpoint 后缀，例如 `u0030`。
- 用途：方便直接对应回 `actor_u0030.pt`、`critic_u0030.pt`。

#### `episodes`

- 定义：本次 checkpoint eval 实际使用的评估 episode 数量。

### 6.3 当前 checkpoint 的平均表现

#### `reward_sum`

- 定义：当前 checkpoint 在多条评估 episode 上的平均 `reward_sum`。

#### `throughput_access_norm_mean`

- 定义：当前 checkpoint 在多条评估 episode 上的平均接入吞吐归一化值。

#### `throughput_backhaul_norm_mean`

- 定义：当前 checkpoint 在多条评估 episode 上的平均回传吞吐归一化值。

#### `gu_queue_arrival_steps_mean`

- 定义：当前 checkpoint 在多条评估 episode 上的平均 GU 队列等价到达步数。
- 单条 episode 内的做法：
  - 先取 GU 总队列量的 episode 平均值
  - 再除以该 episode 的 `arrival_per_step`
- 说明：这是“平均队列水平”的 arrival-steps 版本，不是 `p95`。

#### `uav_queue_arrival_steps_mean`

- 定义：与 `gu_queue_arrival_steps_mean` 相同，只是层级换成 UAV。

#### `sat_queue_arrival_steps_mean`

- 定义：与上面相同，只是层级换成卫星队列。

#### `sat_drop_ratio`

- 定义：当前 checkpoint 在多条评估 episode 上的平均卫星丢包比例。
- 单条 episode 口径：`sat_drop_sum_ep / arrival_sum_ep`。

#### `collision_episode_fraction`

- 定义：当前 checkpoint 的评估 episode 中，发生过碰撞的 episode 比例。

### 6.4 固定参考策略字段

下面这些字段和上面一一对应，只是对象从“当前 checkpoint”换成了“固定参考策略”：

- `fixed_reward_sum`
- `fixed_throughput_access_norm_mean`
- `fixed_throughput_backhaul_norm_mean`
- `fixed_gu_queue_arrival_steps_mean`
- `fixed_uav_queue_arrival_steps_mean`
- `fixed_sat_queue_arrival_steps_mean`
- `fixed_sat_drop_ratio`
- `fixed_collision_episode_fraction`

用途很直接：让你在同一行里就能看出当前 checkpoint 和参考基线谁更好。

### 6.5 早停辅助状态

#### `gu_improved`

- 定义：当前 `gu_queue_arrival_steps_mean` 是否相对历史最好 GU 队列值取得了“足够明显”的改善。
- 判定阈值来自配置 `checkpoint_eval_front_queue_rel_improve_tol`。

#### `uav_improved`

- 定义：当前 `uav_queue_arrival_steps_mean` 是否相对历史最好 UAV 队列值取得了“足够明显”的改善。

#### `front_improved`

- 定义：`gu_improved` 或 `uav_improved` 只要有一个为真，就记为真。
- 含义：前端链路是否至少在一层队列上有明显改善。

#### `sat_drop_worsened`

- 定义：当前 `sat_drop_ratio` 是否相对上一 checkpoint 明显恶化。
- 判定阈值来自配置 `checkpoint_eval_sat_drop_worsen_delta`。

#### `sat_drop_worse_streak`

- 定义：连续多少次出现“卫星丢包恶化，并且前端没有改善”。
- 用途：给 checkpoint-eval 早停提供耐心计数。

#### `early_stop_triggered`

- 定义：当 `sat_drop_worse_streak` 达到配置 `checkpoint_eval_worsen_patience` 时置为 `1`。
- 含义：checkpoint-eval 逻辑认为训练已经出现“卫星丢包持续恶化且前端没有实质改善”的模式。

## 7. TensorBoard 页面对应关系

当前自定义 TensorBoard 页面已经收缩为下面几页：

### 7.1 训练页

- `Training/Main`
  - `episode_reward`
  - `rollout_reward_per_step`
  - `episode_term_throughput_access`
  - `episode_term_throughput_backhaul`
  - `throughput_access_norm`
  - `throughput_backhaul_norm`
  - `gu_queue_mean`
  - `uav_queue_mean`
  - `queue_total_active`
  - `collision_rate`
- `Training/PPO`
  - `policy_loss`
  - `value_loss`
  - `entropy`
  - `approx_kl`
  - `clip_frac`
  - `explained_variance`
- `Training/Imitation`
  - `danger_imitation_loss`
  - `danger_imitation_coef`
  - `danger_imitation_active_rate`

### 7.2 评估页

- `Eval/Main`
  - `reward_sum`
  - `throughput_access_norm`
  - `throughput_backhaul_norm`
  - `gu_queue_mean`
  - `uav_queue_mean`
  - `gu_queue_arrival_steps_p95`
  - `uav_queue_arrival_steps_p95`
  - `terminated_early`
  - `collision`
- `Eval/Drift`
  - `gu_queue_drift_ratio`
  - `uav_queue_drift_ratio`
  - `sat_processed_norm`

## 8. 怎么看这些图最省力

如果只想抓主线，建议按下面顺序看：

1. 先看训练 `Training/Main`
   - `throughput_access_norm`
   - `throughput_backhaul_norm`
   - `gu_queue_mean`
   - `uav_queue_mean`
   - `queue_total_active`
   - `collision_rate`
2. 再看训练 `Training/PPO`
   - `entropy`
   - `approx_kl`
   - `clip_frac`
   - `explained_variance`
3. 然后看训练 `Training/Imitation`
   - `danger_imitation_loss`
   - `danger_imitation_coef`
   - `danger_imitation_active_rate`
4. 最后用评估结果确认
   - `reward_sum`
   - `throughput_access_norm`
   - `throughput_backhaul_norm`
   - `gu_queue_mean`
   - `uav_queue_mean`
   - `gu_queue_arrival_steps_p95`
   - `uav_queue_arrival_steps_p95`
   - `collision`
   - `terminated_early`

这套顺序的核心原则是：

- 训练主图看趋势
- PPO 图看有没有训崩
- 模仿图看安全约束有没有压过主目标
- 评估图看 checkpoint 最终到底好不好
