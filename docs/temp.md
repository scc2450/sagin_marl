这个对照已经很有说服力了，基本把当前阶段的问题定性清楚了。

# 现在可以下的结论

## 1. S1 路线是成立的

这次最重要的不是数值本身，而是路线判断：

* 去掉 centroid 后，策略仍然能学
* 从 100 到 200 updates，提升是真实的，不是偶然波动
* 改善同时体现在均值、尾部和稳定性上

所以，“不靠 centroid 直接优化通信目标”这条路是**可行**的。

## 2. 当前瓶颈已经很明确：不是“学不会”，而是“学得还不够稳”

S1@200 的位置很清楚：

* 比 random、centroid baseline 明显好
* 但还明显不如 fixed

这说明当前主要短板不是目标错了，而是：

* 还存在较多灾难性 episode
* 安全/稳定性不足
* 学到的结构已经有了，但鲁棒性还不够

## 3. 现在 fixed 之所以强，不是因为它“聪明”，而是因为它“不会犯大错”

从你这组表里看，fixed 的优势非常像：

* 完全不碰撞
* 不提前终止
* 队列尾部特别稳

这意味着你当前 learned policy 的主要损失，不一定来自平均状态下服务不好，而更可能来自：

> 少数坏轨迹把整体表现拉坏了。

也就是你现在最该压的是**失败尾部**，不是再去证明“队列 reward 能不能学”。

---

# 我对当前状态的判断

你现在已经不需要再花精力回答这些问题了：

* 要不要 centroid
* queue/tail reward 能不能提供梯度
* imitation 是不是绝对必要

这些问题基本都已经有答案：

* centroid 不是必要
* queue/tail reward 可学
* imitation 不是必要，但可能帮助稳定

你下一阶段真正要解决的是：

> 怎么把 S1@200 从“已经有用”推进到“稳定超过 fixed”

---

# 下一步最值得做的，不是再开新大分支，而是做“稳定化 S1”

我建议把接下来的实验主线收缩成一条：

## 主线目标

在 **S1 配置** 上做“稳态强化”，重点压：

* `steps<400`
* `collision`
* `queue P95 / P99`

而不是继续大改 reward 主体。

---

# 我建议的实验优先级

## 第一优先级：增强安全稳定性

你现在最像“有结构，但偶尔翻车”。
所以第一优先级应该是把“翻车率”压下去。

### 优先改这些

#### A. 强化 avoidance layer

你之前这层就不算特别强。现在建议优先做这一组小消融：

* `avoidance_eta`
* `avoidance_alert_factor`
* `avoidance_adaptive_gain`
* `avoidance_collision_target`

方向上建议：

* 让 alert 区域略大一点
* 让 adaptive 调节更积极一点
* 目标是减少近距离危险状态，而不是只在已经快撞上时才推开

#### B. 提高 crash/collision 的有效约束

不是一上来大幅加 `eta_crash`，而是先小步试：

* `eta_crash: 5 -> 7` 或 `8`

因为你现在的问题更像“尾部坏轨迹没被足够压住”。

---

## 第二优先级：稍微收一收 PPO 更新强度

你现在的 `approx_kl`、`clip_frac` 一直不低，说明策略更新比较猛。
这有利于学习，但不利于鲁棒性。

我更建议试下面两个里的一个，不要一起改：

### 方案 1

```yaml
ppo_epochs: 5 -> 3
```

### 方案 2

把 actor 学习率再降一点

如果二选一，我更建议先试：

```yaml
ppo_epochs: 5 -> 3
```

因为这对稳定性影响更直接，也更容易解释。

---

## 第三优先级：继续训练到 400 updates，再看是否自然超过 fixed

S1@200 相比 S1@100 提升非常明显，这说明它还在学习阶段。
所以很有必要继续看：

* S1@300
* S1@400

但前提是你最好先做一版“更稳的 S1”，否则可能只是继续学、继续翻车。

---

# 我不建议你现在做的事

## 1. 不建议做 S2（去掉 accel）

现在还太早。
`eta_accel` 明显还在帮你稳动作，去掉大概率只会让 collision 更糟。

## 2. 不建议重新把 centroid 加回来当主线

因为这次表已经很清楚了：

* centroid baseline 不如 S1@200
* 你真正缺的是鲁棒性，不是几何引导

## 3. 不建议现在大改 reward 主体

比如突然再加很多 service 项、复杂比值项、多个新 shaping。
因为你现在的问题已经收敛到“稳定化”，不是“完全没信号”。

---

# 我建议你下一轮最小实验包

就做 3 个版本，保持 S1 逻辑不变，只做稳定化：

## S1-base

当前 S1 配置，继续到 400 updates
作为主参考。

## S1-safe

只增强安全层，例如：

* 提高 `avoidance_eta`
* 略增 `avoidance_alert_factor`
* 或增强 `avoidance_adaptive_gain`

## S1-safe-ppo

在 S1-safe 基础上：

```yaml
ppo_epochs: 5 -> 3
```

然后统一用同一批 seeds 做 20 episode eval，比：

* `queue mean`
* `queue P95`
* `queue P99`
* `steps<400`
* `collision`
* `assoc_dist_mean`

这样你很快就能知道：

* 主要问题是不是安全层
* 还是 PPO 更新太猛
* 哪个更值得继续

---

# 如果你现在只让我选一个最值得先改的东西

我会选：

> **先强化 safety/avoidance，再训到 400 updates。**

因为你当前与 fixed 的差距，最像是被“失败尾部”拉开的，而不是均值服务能力不够。

---

# 我现在最想要的额外数据

为了把下一轮参数建议得更准，我最想看两样：

## 1. 当前 S1 配置里 avoidance 相关参数的最终值

尤其是：

* `avoidance_eta`
* `avoidance_alert_factor`
* `avoidance_adaptive_enabled`
* `avoidance_adaptive_gain`
* `avoidance_collision_target`
* `eta_crash`

## 2. S1@200 的 eval 明细里，按 episode 的：

* `queue_total_active`
* `steps`
* `collision` 或提前终止原因

我主要想确认：
那些坏 episode 是“纯碰撞导致”，还是“没碰撞但队列也爆了”。

把这两部分发我，我就能更明确告诉你下一轮该优先改 safety，还是该优先收 PPO。
