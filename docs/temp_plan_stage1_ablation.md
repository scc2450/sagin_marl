下面给你一份按优先级排好的执行方案。目标不是“再试试看”，而是**尽快判断问题到底在优化器，还是在奖励设计本身**。

---

# 总判断

到目前为止，你已经基本确认了三件事：

1. **位置动作确实能显著影响队列**
   fixed、random、centroid 的队列高分位差了一个数量级，这说明“动作→队列”这条链是存在的。

2. **当前完整奖励里，`r_term_centroid` 仍然主导总奖励趋势**
   即使你把 queue/tail 项拉进来了，它们现在更多贡献的是噪声和负项，而不是新的平滑主目标。

3. **PPO 主更新长期偏弱**
   你已经看到：

   * baseline 的 `approx_kl` 很小
   * `clip_frac` 经常 0
   * entropy 主要由 `log_std` 上升驱动
     这说明“策略学不出来”里，既有**优化强度不足**，也有**奖励信号不够可学**两层问题。

所以接下来不要再同时改很多东西，而是按下面顺序做。

---

# 第一阶段：先固定一个“更公平的训练底座”

先别跑 H1/S1。先把训练底座定成一个更合理的版本，不然去掉 centroid 后就算失败，你也不知道到底是 reward 不行，还是 PPO 根本没在更新。

## 先用这个作为新的统一底座

在你当前配置基础上，只改这三项：

```yaml
lr_decay_enabled: false
ppo_epochs: 5
entropy_coef: 0.001
```

其它先不动，包括：

* `adv_clip: 5.0`
* `reward_norm_enabled: false`
* `queue_norm_K: 1.0`
* `q_norm_tail_q0: 0.005`
* `omega_q_tail: 10.0`
* `avoidance_enabled: true`
* `avoidance_adaptive_enabled: true`

## 为什么是这三项

因为你的消融已经说明：

* `ppo_epochs=5` 能把 KL 和 clip_frac 拉起来
* `lr_decay_off` 更稳，碰撞率更低
* entropy 上升主要是 `log_std` 在涨，所以熵系数应该降一点

## 跑法

先跑 **100 updates，12 envs**：

```bash
python scripts/train.py \
  --config configs/你的新底座配置.yaml \
  --log_dir runs/ablation \
  --run_id base_ep5_nolrdecay_u100 \
  --num_envs 12 \
  --vec_backend subproc \
  --torch_threads 12 \
  --updates 100
```

## 这一轮只看这些指标

不要重点看 episode_reward，看：

* `approx_kl`
* `clip_frac`
* `entropy`
* `log_std_mean`
* `collision_rate`
* `q_norm_tail_excess`
* `q_norm_tail_hit_rate`
* `queue_total_active`
* `centroid_dist_mean`

## 这一轮的判定标准

### 说明“底座可用”

满足其中大部分就算可用：

* `approx_kl` 明显高于旧 baseline
* `clip_frac` 持续非零
* `collision_rate` 没明显恶化
* `q_norm_tail_excess` 不比 baseline 更差
* entropy 不再持续单调明显上升
* `log_std_mean` 不再一路飙

### 说明“底座还不行”

如果还是：

* `approx_kl≈0`
* `clip_frac≈0`
* entropy / `log_std_mean` 继续单调涨
  那就先不要做 H1/S1，继续优先查优化器。

---

# 第二阶段：在新底座上做 H1 和 S1

这一阶段的目的不是“训出最好结果”，而是回答：

* **centroid 是不是必须的**
* **imitation 是不是必须的**
* **队列/安全信号本身够不够支撑学习**

---

## H1：半简化版

### 目的

先去掉 imitation，但保留 centroid 和 accel。
这个实验回答：

> imitation 去掉后，当前 reward 还能不能学？

### 配置改动

在“新底座”上改：

```yaml
imitation_enabled: false
imitation_coef: 0.0
imitation_coef_final: 0.0
imitation_coef_decay_updates: 0
```

其余保留：

```yaml
eta_centroid: 0.45
eta_centroid_final: 0.15
eta_centroid_decay_steps: 50000
eta_accel: 0.02
centroid_cross_anneal_enabled: true
```

### 跑法

先跑 **100 updates**。

### 看什么

和第一阶段一样，再额外看：

* `r_term_centroid`
* `r_term_q_delta`
* `r_term_queue`

### H1 成功的标志

* KL/clip_frac 没塌
* collision_rate 没明显恶化
* tail 指标不恶化
* 相比完整版本，没有明显更差

### H1 失败的标志

* entropy 更快上升
* KL 更低
* 碰撞率显著更高
* tail 指标明显更差

---

## S1：简化版

### 目的

去掉 imitation 和 centroid，只保留 queue/safety/accel。
这个实验直接回答：

> 没有 centroid，现有 queue/safety 信号本身够不够支撑学习？

### 配置改动

在“新底座”上改：

```yaml
eta_centroid: 0.0
eta_centroid_final: 0.0
eta_centroid_decay_steps: 0
centroid_cross_anneal_enabled: false

imitation_enabled: false
imitation_coef: 0.0
imitation_coef_final: 0.0
imitation_coef_decay_updates: 0
```

### 先保留

```yaml
eta_accel: 0.02
```

不要一开始就去掉 accel。

### 跑法

先跑 **100 updates**。

### 重点看

* `approx_kl`
* `clip_frac`
* `entropy`
* `log_std_mean`
* `collision_rate`
* `q_norm_tail_excess`
* `q_norm_tail_hit_rate`
* `queue_total_active`

### S1 的解释

#### 如果 S1 能学

说明：

* centroid 不是必须的
* 现有 queue/safety 信号是足够可学的
* 后面可以考虑把 centroid 完全退场

#### 如果 S1 学不起来

说明更可能是：

* 最终 queue 目标本身不够“局部可学”
* 你需要一个比最终队列更贴近位置控制的中间信号来桥接
* centroid 只是其中一种桥接，不一定是最终版本，但某种 bridge reward 仍然需要

---

# 第三阶段：只有 S1 能学时，才做 S2

## S2：在 S1 基础上去掉 accel

### 目的

回答：

> accel regularization 是不是也不是必须的？

### 配置改动

```yaml
eta_accel: 0.0
```

### 什么时候才跑

只有 S1 已经显示“没有 centroid 也能学”，才有必要跑 S2。
否则你会把“没有 centroid”和“没有动作平滑”两个问题混在一起。

---

# 第四阶段：修正你现在的监控方式

这一步很重要，不然你会一直被 reward 曲线误导。

## 不要再把 `episode_reward` 当主要进步指标

因为你的 reward 本身带 schedule（centroid 退火），所以它天然会跟 `r_term_centroid` 走。

以后主看：

### 安全类

* `collision_rate`

### 队列/尾部类

* `q_norm_tail_excess`
* `q_norm_tail_hit_rate`
* `queue_total_active`
* 如果有精力，加：

  * `q_norm_active_nonzero_rate`
  * `q_norm_active_max`

### 训练动态类

* `approx_kl`
* `clip_frac`
* `entropy`
* `log_std_mean`

### 参考类

* `centroid_dist_mean`
* `r_term_centroid`

---

# 第五阶段：补充日志，避免再误判

## Advantage 日志

把现在的 `adv_norm_mean` 拆成两套：

```python
adv_preclip = (adv_raw - adv_raw_mean) / (adv_raw_std + 1e-8)
adv_preclip_mean = np.mean(adv_preclip)
adv_preclip_std  = np.std(adv_preclip)

adv_postclip = np.clip(adv_preclip, -5.0, 5.0)
adv_postclip_mean = np.mean(adv_postclip)
adv_postclip_std  = np.std(adv_postclip)
```

以后解释：

* `adv_preclip_mean` 用来检查标准化实现是否正常（应接近 0）
* `adv_postclip_mean` 用来看剪裁后的偏斜
* `adv_raw_mean` 继续看训练动态

## 队列 tail 稀疏性日志

再加几个指标：

* `q_norm_active_max`
* `q_norm_active_nonzero_rate`
* `q_norm_tail_hit_rate`（你已经有了）
* 如果方便，再加：

  * `q_norm_active_p90`
  * 因为 p95/p99 在稀疏尖峰场景可能长期为 0

---

# 第六阶段：如果 S1 失败，下一步就不是调权重，而是承认需要“中间层目标”

这是最关键的认知转变。

如果在“更公平的训练底座”上，S1 仍然失败，那么你可以比较有把握地得出：

> 单靠“最终队列负值 + 安全项”，在你这个位置控制问题上，不够可学。

这时不要继续沉迷于：

* 再调一点 `omega_q`
* 再调一点 `eta_q_delta`
* 再改一点 centroid 退火

而是应该转向：

> 引入一个比最终队列更直接受位置影响、又和通信目标一致的中间层 reward。

你现在先不想加服务项没关系，但至少可以承认这个方向可能最终必须考虑。

---

# 你接下来具体怎么排期

## 第 1 步：先跑新的统一底座

* `lr_decay_enabled=false`
* `ppo_epochs=5`
* `entropy_coef=0.001`
* 跑 100 updates

## 第 2 步：如果底座可用，跑 H1

* 去 imitation
* 100 updates

## 第 3 步：如果 H1 不崩，跑 S1

* 去 imitation + 去 centroid
* 保留 accel
* 100 updates

## 第 4 步：只有 S1 成功，再跑 S2

* 去掉 accel
* 100 updates

## 第 5 步：对值得继续的版本，再拉到 200 updates

不是每个都 400。
先 100 看趋势，再 200 看稳定性，最后只有最有希望的才跑 400。

---

# 你现在最不该做的事

1. 不要再同时改很多 reward 项和优化器项
2. 不要再把 `episode_reward` 当主要判断依据
3. 不要在“PPO 还没动起来”时，就下结论说队列奖励一定不行
4. 也不要在 S1 失败前，就断言 centroid 必须存在

---

# 最后一版最小执行清单

### 新底座配置改动

```yaml
lr_decay_enabled: false
ppo_epochs: 5
entropy_coef: 0.001
```

### H1 配置改动（在新底座上）

```yaml
imitation_enabled: false
imitation_coef: 0.0
imitation_coef_final: 0.0
imitation_coef_decay_updates: 0
```

### S1 配置改动（在新底座上）

```yaml
eta_centroid: 0.0
eta_centroid_final: 0.0
eta_centroid_decay_steps: 0
centroid_cross_anneal_enabled: false

imitation_enabled: false
imitation_coef: 0.0
imitation_coef_final: 0.0
imitation_coef_decay_updates: 0
```

### S2 配置改动（在 S1 上）

```yaml
eta_accel: 0.0
```

---

如果你愿意，我下一条可以直接把这四套配置写成“相对于你当前 yaml 的最小 diff 清单”，你几乎可以直接照着改。
