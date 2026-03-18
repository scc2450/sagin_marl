我先直接给你结论：**bw 配置至少要先把 accel-only 里已经证明“更稳”的那批训练项同步过去**，否则你是在拿“改好后的 accel”去接“没改好的 bw 模板”，这样对比本身就不公平。你给的两份配置里，这种“不一致”很明显。 

你这两份配置对照下来，我建议分成三类看。

## 一、应该直接从 accel 配置迁到 bw 配置的项

这些我认为是**优先级最高**的，因为它们不是“accel 专属”，而是更像通用的 PPO 稳定化设置。

### 1) PPO 时序相关参数要补齐

在 accel 配置里有：

* `gamma: 0.9975`
* `gae_lambda: 0.95`
* `kl_stop: true`
* `target_kl: 0.02`

而你这份 bw 配置里这些项没有出现。 

这很可疑，因为如果代码默认值不是你现在 accel 这套，那 bw 实际训练口径就和 accel 不一致了。
**我建议 bw 先显式加上这四个：**

```yaml
gamma: 0.9975
gae_lambda: 0.95
kl_stop: true
target_kl: 0.02
```

尤其是 `kl_stop + target_kl=0.02`，你前面已经说 accel 改完后训练才比较好，这一项很可能就是关键稳定器之一。

---

### 2) checkpoint eval / early stop 逻辑要同步

accel 配置里有比较完整的 checkpoint-eval 和 reward plateau early-stop 逻辑，例如：

* `checkpoint_eval_interval_updates: 50`
* `checkpoint_eval_start_update: 200`
* `checkpoint_eval_fixed_policy: zero`
* `checkpoint_eval_early_stop_enabled: true`
* `checkpoint_eval_reward_early_stop_enabled: true`
* `checkpoint_eval_reward_patience: 5`
* `checkpoint_eval_reward_min_delta_rel: 0.005`
* `checkpoint_eval_reward_collision_threshold: 0.01`

而 bw 配置里是旧式的：

* `checkpoint_eval_interval_updates: 10`
* `checkpoint_eval_start_update: 10`
* `checkpoint_eval_fixed_policy: queue_aware`
* `checkpoint_eval_early_stop_enabled: false`

这说明 bw 这份配置还没吸收你后面对 accel 训练流程的修正。 

这里我建议分开说：

* **checkpoint eval 的频率/开始点**：bw 不一定要照搬 50/200，但至少要有一套你认可的逻辑，不要还是“每 10 update 就 eval 一次”的旧口径。
* **early stop**：如果 accel 上有用，bw 也建议接上。
* **baseline 口径**：bw 用 `queue_aware` 当固定 baseline 是合理的，这一点不一定改成 `zero`。因为 bw 的固定对照本来就更适合是一个 hand-crafted bandwidth baseline。

所以这里不是“全盘照搬”，而是：

* **训练稳定逻辑照搬**
* **baseline 口径按 bw 任务保留 `queue_aware`**

---

### 3) 初始化/生成场景的稳定化设置要同步

accel 配置里有这些，而 bw 没有：

* `assoc_unfair_gu_threshold: 15`
* `uav_spawn_curriculum_enabled: false`
* `uav_safe_random_init_enabled: true`
* `uav_init_boundary_margin_steps: 3.0`
* `uav_init_speed_frac: 0.2`
* `uav_init_min_spacing: 20.0`

bw 配置里却还是：

* `uav_spawn_curriculum_enabled: true`
* 没有 safe init 这一组
* 没有 `assoc_unfair_gu_threshold`

这说明 **环境分布本身也没同步**。 

这一点我非常建议你重视，因为你现在是想做：

> 在现有 accel-only 基础上训练 bw

那就意味着 **bw 阶段的环境分布最好尽量和“accel 已经学好的那套运行分布”一致**，否则 accel 学到的位置控制规律，和 bw 训练时看到的轨迹/碰撞/初始化分布不一致，会把后续 credit assignment 搅乱。

所以我建议 bw 先改成更接近 accel：

```yaml
assoc_unfair_gu_threshold: 15

uav_spawn_curriculum_enabled: false
uav_safe_random_init_enabled: true
uav_init_boundary_margin_steps: 3.0
uav_init_speed_frac: 0.2
uav_init_min_spacing: 20.0
```

---

### 4) 学习率拆分保留

这项你 bw 已经和 accel 一致了：

* `actor_lr: 1e-4`
* `critic_lr: 2e-4`

这个保留就行。 

---

### 5) reward norm 关闭保留

这项也已经一致：

* `reward_norm_enabled: false`

继续保持。 

---

## 二、不要机械照搬 accel，而是要按“bw 任务”单独判断的项

### 1) danger imitation 不建议直接照搬

accel 配置里：

* `danger_imitation_enabled: true`
* `danger_imitation_coef: 0.1`

bw 配置里：

* `danger_imitation_enabled: false`

这里我**不建议你直接照搬成 true**。 

原因很简单：
danger imitation 本质上更像是**碰撞/近距离风险下的 accel 安全先验**。如果你现在训练的是 bw，而 accel 是固定执行的，那么这个 imitation loss 很可能：

* 要么根本不作用在 bw head 上
* 要么通过共享 backbone 间接扰动表示学习
* 要么只会制造额外梯度噪声

所以这项必须看代码后才能定，不能光看 yaml 决定。

---

### 2) `exec_accel_source` 不能再用 `zero`

你现在这份 bw 配置里写的是：

```yaml
train_accel: false
train_bw: true
exec_accel_source: zero
exec_bw_source: policy
```

如果你真的是“在现有 accel-only 基础上训练 bw”，那这里大概率不应该是 `zero`，而应该让 accel 来自**已训练好的 accel policy**。

这是我目前看到的**最关键问题之一**。

因为 `exec_accel_source: zero` 的意思很可能是：

* UAV 不再执行你训练好的机动策略
* 而是用“零加速度 / no-op accel”跑环境

那这样训练出来的 bw，学到的是“静态/近静态 accel 行为下的最优带宽分配”，**不是你想要的“建立在已学好 accel 基础上的 bw”**。

这里我怀疑你真正需要的是类似下面这种口径：

```yaml
train_accel: false
train_bw: true
train_sat: false

exec_accel_source: teacher   # 或 policy / frozen_policy，具体看代码定义
exec_teacher_actor_path: <stage1_accel_actor.pt>
exec_teacher_deterministic: true

exec_bw_source: policy
exec_sat_source: zero
```

但具体 `exec_accel_source` 可选值是什么，必须看代码。

---

### 3) `train_shared_backbone: true` 要不要保留，要看实现

你现在 bw 配置是：

```yaml
train_shared_backbone: true
```

如果 actor 的 accel head 和 bw head 共享一个 backbone，那么当你只训练 bw 时，有两种可能：

* **可能是好的**：bw 能利用 accel 阶段学到的空间结构特征
* **也可能有坑**：bw 更新把原来 accel 已经学好的表示冲坏，之后再做 joint 或 sat stage 会出问题

所以这一项不能只靠经验判断，必须看代码里：

* “只训练 bw”时，shared backbone 是否仍然参与反传
* accel head 是否被冻结
* shared encoder 是否被冻结
* optimizer 参数组是怎么构造的

这一点我会把它列进“必须看代码”的第一优先级。

---

## 三、我认为你现在最可能需要的 bw 配置方向

如果你的目标真的是：

> 先训好 accel，再在这个基础上训 bw

那我建议你的 bw 配置方向不是“from scratch bw-only”，而是“**frozen accel execution + train bw**”。

也就是概念上改成：

```yaml
enable_bw_action: true
train_accel: false
train_bw: true
train_sat: false

# accel 用已训练好的 stage1 actor 执行
exec_accel_source: <teacher/frozen_policy/loaded_actor>
exec_teacher_actor_path: runs/.../actor.pt
exec_teacher_deterministic: true

# bw 仍由当前策略学习
exec_bw_source: policy
exec_sat_source: zero

# 同步 accel 中验证过的训练稳定项
gamma: 0.9975
gae_lambda: 0.95
reward_norm_enabled: false
actor_lr: 1e-4
critic_lr: 2e-4
kl_stop: true
target_kl: 0.02

# 同步环境分布
uav_spawn_curriculum_enabled: false
uav_safe_random_init_enabled: true
uav_init_boundary_margin_steps: 3.0
uav_init_speed_frac: 0.2
uav_init_min_spacing: 20.0
assoc_unfair_gu_threshold: 15
```

也就是说，**最重要的不是把所有字段抄过去，而是把“执行的 accel 来源”从 zero 改成已训练 accel**。

---

## 四、我还需要看的代码

为了把这个事判断准，我下一步最需要你给我这几部分代码。按优先级排：

### 1) action 执行源的分发逻辑

我最想先看这部分，确认这些配置到底怎么生效：

* `exec_accel_source`
* `exec_bw_source`
* `exec_sat_source`
* `exec_teacher_actor_path`
* `exec_teacher_deterministic`
* `train_accel / train_bw / train_sat`

对应通常会在这些位置之一：

* `train.py`
* `mappo.py`
* `policy.py`
* `action_assembler.py`
* 和环境 `step()` 前组装 action 的那段代码

我想确认的是：

* `zero` 到底是不是“全零动作”
* 是否支持“accel 用 teacher、bw 用 policy”
* teacher actor 是整网输出还是只输出 accel head
* 单独训 bw 时，采样的 logprob 是不是只算 bw 那部分

---

### 2) policy / actor 的多头结构

我需要看：

* shared backbone
* accel head
* bw head
* sat head
* forward 时哪些 head 被调用
* logprob / entropy 怎么分解
* `train_shared_backbone` 怎么控制

这里主要判断：

* bw 训练会不会把 accel 表征带坏
* 只训 bw 时 entropy 和 PPO ratio 是不是只对 bw 分量算
* 是否存在“未训练 head 仍参与 loss/entropy”的问题

---

### 3) PPO loss 组装代码

我需要看 `mappo.py` 里：

* actor loss
* entropy loss
* imitation / danger imitation loss
* mask 哪些 action 维度参与训练
* checkpoint eval / early stop 逻辑

重点确认：

* 只训 bw 时，danger imitation 是否还会影响 shared encoder
* KL stop 是按总 action 还是只按可训练 action 统计
* entropy 是全动作熵还是只算 bw 熵

---

### 4) 环境里 bw 动作的实际语义

你前面说过 bw 头不是直接 softmax，而是环境里再转成分配权重。这个地方我也需要再看一次完整链路：

* policy 输出什么
* action_assembler 怎么打包
* env 里怎么从 `bw_logits` 变成带宽比例
* 是否有裁剪、归一化、mask、无效候选处理

因为 bw 训练是否稳定，常常就死在这里，比如：

* 某些维度经常无效
* softmax 温度过硬
* 候选数变化导致梯度尺度乱跳
* 分配后大部分带宽落到无意义链路上

---

## 五、你现在可以直接先改的地方

不等代码，我建议你先把 bw yaml 至少改这几项：

```yaml
# 先补齐 PPO 稳定项
gamma: 0.9975
gae_lambda: 0.95
kl_stop: true
target_kl: 0.02

# 先对齐环境分布
assoc_unfair_gu_threshold: 15
uav_spawn_curriculum_enabled: false
uav_safe_random_init_enabled: true
uav_init_boundary_margin_steps: 3.0
uav_init_speed_frac: 0.2
uav_init_min_spacing: 20.0
```

然后最关键的是，把这项别再留成：

```yaml
exec_accel_source: zero
```

这项我目前判断**大概率必须改**，否则它不是“基于已学好 accel 训 bw”。

---

把下面几段代码贴给我就够了，我可以直接帮你给出“bw 阶段最终该怎么配”的精确版本：

1. `policy.py` 里 actor 多头 forward、logprob、entropy 那几段
2. `mappo.py` 里 action sampling / PPO loss / danger imitation 那几段
3. `action_assembler.py` 全文件
4. 环境里把 `bw_logits` 变成实际带宽分配的那段
5. 配置解析里 `exec_accel_source / train_bw / train_shared_backbone` 这些字段是怎么用的那段
