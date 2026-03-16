当前**真正关心的目标**已经很明确了：

1. 主目标：`throughput_access`、`throughput_backhaul`
2. 辅助状态：`GU/UAV` 队列是否在变坏
3. 训练约束：PPO 本身有没有训崩
4. 额外项：`danger_imitation_loss` 有没有过强干扰

所以没必要继续背着 100 多个指标看图。你现在最需要的是：

* **训练侧只保留少数“过程诊断指标”**
* **评估侧只保留少数“结果指标”**
* **不同指标换对横坐标**

---

# 一、先说最关键的：横坐标怎么改

你现在是并行 `vec env`，而且某个子环境一旦碰撞会立刻 reset。
这会带来一个核心问题：

**一个 update 收集到的是“固定步数的 rollout 样本块”，不是“固定数量的完整 episode”。**

所以：

* `updates` 只表示**参数更新次数**
* 不表示真实交互量
* 更不表示经历了多少完整 episode

这意味着：

## 1）哪些指标不适合用 `updates`

凡是和**环境交互表现**有关的量，都更应该用：

* `total_env_steps` 作为横坐标

比如：

* `throughput_access_norm`
* `throughput_backhaul_norm`
* `queue_total_active`
* `gu_queue_mean`
* `uav_queue_mean`
* `drop_ratio_step`
* `collision_rate`（这个不是，我想看的是一个episode的碰撞概率，因为碰撞了这个episode就终止了）


因为这些东西本质上是在描述：
**策略在环境里跑出来的行为质量，应该随“采样量/交互量”看，而不是随“更新次数”看。**

---

## 2）哪些指标继续用 `updates`

凡是和**优化器/PPO训练过程**直接相关的量，继续用 `updates` 最合适：

* `policy_loss`
* `value_loss`
* `entropy`
* `approx_kl`
* `clip_frac`
* `explained_variance`
* `adv_*`
* `danger_imitation_loss`
* `actor_lr`
* `critic_lr`

因为这些量本来就是“每次 update 计算一次”的优化统计。



---

# 二、训练侧到底保留哪些指标

我建议训练侧只保留 **12 个左右**，够用了。

---

## A. 训练主面板：业务结果

这是你最该看的面板。

### 必留

* `throughput_access_norm`
* `throughput_backhaul_norm`
* `gu_queue_mean`
* `uav_queue_mean`

### 选留一个总量指标

二选一：

* `queue_total_active`
* `q_norm_active`

我更建议留 **`queue_total_active` + `gu_queue_mean` + `uav_queue_mean`**，因为更直观。



---

## B. 训练诊断面板：PPO 是否正常

### 必留

* `policy_loss`
* `value_loss`
* `entropy`
* `approx_kl`
* `clip_frac`
* `explained_variance`

这 6 个基本够判断 PPO 是否训崩。

其中你最该看的是：

* `entropy`：策略是否过早塌缩
* `approx_kl`：每次更新是否过猛/过弱
* `clip_frac`：PPO clipping 是否长期过高
* `explained_variance`：critic 有没有完全失效

---

## C. 训练附加面板：模仿干预

既然你还有一个 `danger_imitation_loss`，这个要单独看，不要混进普通 PPO loss。

### 必留

* `danger_imitation_loss`
* `danger_imitation_coef`

### 可选

* `danger_imitation_active_rate`

这个有用，因为它能告诉你：
不是 loss 本身大不大，而是“这个约束到底多频繁地在起作用”。

我记得danger_imitation的开启关闭和别的量有关，这个量也要一并看

---

## D. 训练安全/失败辅助

如果你这阶段仍然会碰撞早停，我建议只留一个：

* `collision_rate`（我想看的是某个episode碰撞的概率）

别再留一堆 filter / fallback / pairwise 统计了。那些更适合调安全模块时专门看，不适合日常主训练面板。

---

## E. 训练里建议降级或删除的

### 1）直接从 TB 主面板移除

这些你当前主线几乎没必要看：


* `reward_raw`
* 几乎所有 `r_*`
* 几乎所有 `r_term_*`，除了两个 throughput 项，这个要留着，但应该和episode_reward同样处理，考虑完整episode
* `arrival_sum`
* `outflow_sum`
* `service_norm`
* 各种 `drop_*` 总和量
* 各种 `sat_*` 几何量
* 各种运行时分解
* 各种 `adv_*`
* `log_std_mean`, `action_std_mean`

原因很简单：
这些不是没信息，而是**相对于你的当前目标，边际信息太低**。

---

### 2）保留在 CSV，但不进 TB

这些可以留作排障，不必日常看图：

* `env_steps_per_sec`
* `update_steps_per_sec`
* `rollout_time_sec`
* `optim_time_sec`
* `update_time_sec`
* `total_env_steps`
* `total_time_sec`

以及你所有运行时细分：

* `env_obs_time_sec`
* `policy_forward_time_sec`
* `env_step_time_sec`
* ...

这些只有在你做性能优化时才重要。

---

# 三、评估侧保留哪些指标

评估侧比训练侧更重要，因为评估才真正回答：

> 当前 checkpoint 的策略到底有没有比之前更好。

我建议评估侧保留 **8～12 个核心量** 就够了。

---

## A. 评估主指标：结果成败

### 必留

* `throughput_access_norm`
* `throughput_backhaul_norm`
* `gu_queue_mean`
* `uav_queue_mean`

这是和训练主面板一一对应的，最重要。

---

## B. 评估稳态质量

### 队列尾部指标

GU和UAV的分开看

---

## C. 评估安全性

### 必留

* `collision`
* `terminated_early`

如果你一个 checkpoint 经常提前撞死，那 throughput 再高也没意义。


---

## D. 评估工况方向

### 必留一个 drift 指标

三选一：

* `gu_queue_drift_ratio`
* `uav_queue_drift_ratio`
* `total_net_drift_per_step`

如果你现在主要关心 GU/UAV，我建议留：

* `gu_queue_drift_ratio`
* `uav_queue_drift_ratio`

它们比单纯看 mean queue 更能说明系统是在“积压增加”还是“逐渐消化”。

---

## E. 可选补充

* `sat_processed_norm`

只有你怀疑瓶颈已经从接入段转到卫星处理段时才看。

---


# 四、给你一个实际可执行的面板重组方案

你现在 TB 页太多了，我建议收成下面 5 页。

## Training/Main

* `throughput_access_norm`
* `throughput_backhaul_norm`
* `gu_queue_mean`
* `uav_queue_mean`
* `queue_total_active`
* `collision_rate`



---

## Training/PPO

* `policy_loss`
* `value_loss`
* `entropy`
* `approx_kl`
* `clip_frac`
* `explained_variance`



---

## Training/Imitation

* `danger_imitation_loss`
* `danger_imitation_coef`
* `danger_imitation_active_rate`



---

## Eval/Main

* `throughput_access_norm`
* `throughput_backhaul_norm`
* `gu_queue_mean`
* `uav_queue_mean`
* 队列尾部指标
* `terminated_early`
* `collision`


---

## Eval/Drift

* `gu_queue_drift_ratio`
* `uav_queue_drift_ratio`
* `sat_processed_norm`



