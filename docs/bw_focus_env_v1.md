你说得对。下面这份只做一件事：**改环境 leverage，让 bw 的可改空间真的变大**。
我不再让你去做 reward / target / entropy / aux 的微调实验；那些你已经测过，信息已经够了。你现在的证据是：在当前 fixed assoc / fixed `b_acc` / `candidate_mode=nearest` / `min(q_before, rate*tau0)` 裁剪的设定下，bw 对 `x_acc` 的单独边际空间通常只有 `1e-3 ~ 1e-2`，而且 `K=1` 基本没用、`K=2` 才开始有明显收益；你后来把 bw 链路洗得更干净，也没有让 bw 明显变得更可学。继续在 target 上抠，不值当了。 

把同 λ Poisson 改成**有持续性的局部 burst**，以及把用户活跃集/非均匀流量引入 MEC / UAV-MEC，本身就是正统建模，不是为了救算法而硬编环境。MEC 文献里有 random user arrivals，UAV-MEC 文献里有 random movements and task arrivals；Markov-modulated Poisson 和简单 ON/OFF 模型本来就是 bursty arrival 的标准建模；而时空非均匀流量会放大总到达率波动，并改变队列稳定性和时延。([arXiv][1])

---

## 1. 先把这次方案的边界定死

这次方案不是“最终 benchmark 环境”，而是一个**专门把 bw 练出来的环境变体**，我下面叫它：

**`bw_focus_env_v1`**

这版环境里，先**不改**这几件事：

* 不改 association：还是 GU 连路径损耗最小的 UAV
* 不改 `candidate_mode=nearest`
* 不改 `tau0`
* 不改总 GU 数：仍然固定 20
* 不加 active/inactive mask
* 不改 reward / target / actor / critic
* 不改训练脚本口径

原因很简单：你现在要回答的是“**环境 leverage 够不够**”。
如果第一刀就把 GU 数、active mask、association、reward、target 一起动了，最后你根本分不清是谁起作用。

---

## 2. 这份方案要解决的根因

你当前系统里，bw 的动作语义是：

* 不能改关联图
* 不能创造接入资源
* 只能在“已经关联到本 UAV 的用户”之间重分配固定 `b_acc`
* 即使把更多带宽给某个用户，`x_acc` 还会被 `min(q_before, rate*tau0)` 裁剪

所以 bw 现在经常只是在一个**很平的局部平面**里挪份额。你已有诊断已经证明了这一点。

因此这次环境 leverage 的设计目标只有三个：

1. **让同一架 UAV 的局部候选集内部，持续出现不均衡需求**
   不是全局都更难，而是**同一架 UAV 里**真的要决定先救谁。

2. **让这种不均衡持续 10~20 个 step，而不是一闪而过**
   这样 bw 才能拿到你已经观察到的那种“短前缀多步 credit”。

3. **让 downstream 不要把 access 侧差异全吞掉**
   不然前面 bw 做出的差异，后面又被回传/处理瓶颈抹平。

---

## 3. 完整方案：`bw_focus_env_v1`

### 3.1 环境核心思路

一句话：

**固定 20 个 GU，不改接入规则；把到达过程改成“均值守恒的 sticky 局部 hotspot burst”；把 GU/UAV 前两层 buffer 设得紧一些，把 SAT 侧设得松一些；再加局部 preload。**

这样做以后，bw 的作用不再是“对所有用户均匀分一点点更好”，而是：

* 某 3~4 个局部 GU 连续 10~20 步高压
* 它们大概率落在同一架 UAV 的已关联集合里
* 如果你不给它们更多 bw，它们会持续堆积，甚至在 GU/UAV 前两层出问题
* 但 SAT 不会立刻把这个信号吞掉

这就是你要的 leverage。

---

### 3.2 第一步：把到达过程改成 `sticky_subset_hotspot`

#### 你要新增的环境模式

```yaml
traffic_model: sticky_subset_hotspot
```

#### 先固定这些不变

```yaml
num_gu: 20
dynamic_gu_count: false
association_rule: nearest_pathloss
candidate_mode: nearest
tau0: <保持当前值>
```

#### 3.2.1 先构造“局部 hotspot 子集”

不要直接“整片区域一起高流量”，也不要“20 个 GU 各自独立 burst”。

你要的是：**每次只让 3~4 个空间上相近、并且在参考几何下大概率归到同一 UAV 的 GU 一起 burst。**

最实用的构造法：

1. 用你现在的 GU 位置，先跑一小批参考 episode。
   accel / sat 用你现在最稳的 base policy 或 heuristic，跑到几何基本稳定的阶段。
2. 记录每个 GU 大多数时候归到哪架 UAV。
3. 在每架 UAV 的“常归属 GU 集”里，按空间最近邻组 2 个 subset。
4. 每个 subset 大小设成 **3 或 4**。
   你现在总 GU=20，通常每架 UAV 会有几名关联 GU，所以这一步是够用的。
5. 最终得到大约 **6 个 hotspot subset**。

如果你不想依赖参考 policy，也可以退一步：
直接用“每个 GU + 它最近的 2~3 个 GU”构 subset，然后去重，保留 6 个最局部、最分散的子集。

#### 3.2.2 再定义 sticky hotspot 状态

环境内部维护一个 hotspot 状态 `h_t`：

* `h_t = 0` 表示当前没有热点
* `h_t = k` 表示第 `k` 个 hotspot subset 正在 burst

这个状态用 Markov 链更新，保持“粘性”：

* hotspot 持续平均长度：**15 steps**
* quiet 持续平均长度：**8 steps**

实现上可以直接用：

```text
P(stay_on)  = 1 - 1/15
P(stay_off) = 1 - 1/8
```

也就是：

* 当前在热点里，大概率继续留在原 hotspot
* 当前无热点，也会连续安静几步
* 切换时再随机挑下一个 hotspot subset

#### 3.2.3 每个 GU 的到达率怎么生成

每个 episode 开始时，先给每个 GU 一个轻微异质的 base scale：

```text
s_i ~ Uniform[0.85, 1.15]
```

然后每个 step 生成 raw multiplier：

```text
r_i(t) = rho_hot    if i in active_hotspot_subset
         1          otherwise
```

建议先从：

```text
rho_hot = 4.0
```

开始。

最后做**均值守恒归一化**，让总 offered load 不变，只改局部分布：

```text
lambda_i(t) = lambda_bar * s_i * r_i(t) / mean_j[s_j * r_j(t)]
A_i(t) ~ Poisson(lambda_i(t) * tau0)
```

这个归一化非常关键。
它保证你改的是“流量的局部分布和 burstiness”，不是把系统整体变得更重或更轻。

#### 为什么这一步能直接放大 bw leverage

因为现在不是“20 个 GU 平均变忙”，而是：

* **同一架 UAV 的局部几个用户**持续变忙
* 其他用户相对没那么忙
* bw 的每一步分配，都会影响接下来 10~20 步里谁积压得更快、谁更可能掉、谁还能继续出流

这正好对准了你已经看到的 “K=2 比 K=1 有用”。

---

### 3.3 第二步：把前两层 buffer 设紧，把后层设松

这一步同样重要。
只改 arrivals 还不够；如果 buffer 和 downstream 都太松，burst 也不会转化成 bw 的 leverage。

#### 3.3.1 用“base-arrival steps”来定义 queue_max

你现在本来就喜欢用“相当于几步进入量”来定 queue，这很好。
这次继续这么做，而且**按 base load 定，不按 burst peak 定**。

对每层定义：

* `mu_gu_step`：单个 GU 在 base traffic 下的平均每步到达量
* `mu_uav_step`：单个 UAV 在 base traffic 下的平均每步入流
* `mu_sat_step`：单个 SAT 在 base traffic 下的平均每步入流

然后设：

```text
queue_max_gu  = 6  * mu_gu_step
queue_max_uav = 8  * mu_uav_step
queue_max_sat = 24 * mu_sat_step
```

如果你当前代码接口里 `queue_max_*` 是标量，就继续保持标量，不要为了这一步去改成 per-user / per-node。

#### 为什么 SAT 要设得明显更松

因为这次是 `bw_focus_env`。
你不是要最终 benchmark 一步到位，而是要先把 bw 练出来。

如果 SAT 侧也很紧：

* 前端 bw 做出的 access 差异
* 很容易被 backhaul / SAT 处理瓶颈吞掉

于是你又回到“bw 怎么调都差不多”的状态。

所以这版环境里，**前两层要更紧，SAT 要更松**。

---

### 3.4 第三步：把 downstream 处理能力放宽一档

只把 `queue_max_sat` 变大还不一定够。
如果 UAV→SAT 回传或 SAT 处理速率本身经常是主瓶颈，bw 的 access 差异仍然会被后面抹掉。

所以在 `bw_focus_env_v1` 里，我建议同时放宽：

```yaml
b_sat_total_scale_bwfocus: 1.5
sat_cpu_scale_bwfocus: 2.0
```

也就是：

* `b_sat_total` 乘 **1.5**
* `sat_cpu_freq` 乘 **2.0**

你不需要永久改 benchmark，只在 `bw_focus_env_v1` 里这样做。

#### 这一刀的意义

它不是“把问题改没了”，而是故意让**前半段成为主瓶颈**：

* GU 队列 / UAV 队列会对 bw 更敏感
* access 侧差异更容易显化
* 你才能判断 bw 头本身到底能不能学

等 bw 学出来，再把 downstream 逐步收回去。

---

### 3.5 第四步：加局部 preload，让 bw 从 step 1 就有事做

burst 过程会逐步积压，但为了让训练一开始就有 leverage，我建议再加一个 reset 时的局部 preload。

#### 配置建议

```yaml
preload_enabled: true
preload_prob: 0.6
preload_hot_gu_steps: 3.0
preload_bg_gu_steps: 0.2
preload_hot_uav_steps: 2.0
preload_sat_steps: 0.0
```

含义：

* 60% episode 会做 preload
* 选一个 hotspot subset 作为本局初始热点
* hotspot 里的 GU 初始队列 = **3 个 base-arrival step**
* 其他 GU 初始队列 = **0.2 step**
* 对应 UAV 初始队列 = **2 个 base-uav-inflow step**
* SAT 初始队列 = **0**

#### 为什么 preload 要只加在前两层

因为你的目标是先把 bw 练出来。
所以初始压力要放在：

* GU 队列
* UAV 队列

不要一上来就把 SAT 灌满，那又会把信号往 sat 侧推。

---

### 3.6 第五步：v1 先不改观测

这次 v1 我建议**先不改 actor 输入**。

也就是说：

* 不暴露 hotspot id
* 不暴露真实 `lambda_i(t)`
* 不额外加 active mask
* 不新增 arrival history feature

原因是你这次要先回答“**只改环境 leverage，本来这套观察和算法能不能立起来**”。

如果一开始就改观测，你又会把因果混掉。

等到这套环境 leverage 确认有效，但 policy 还是不稳定时，再加一个现实可解释的字段：

```text
demand_ema_i = EMA(最近几步该 GU 的上报需求 / buffer 增量)
```

这个可以解释成 GU 的 buffer status report，不算泄露隐藏状态。
但这一步放到第二轮，不放在 v1。

---

## 4. 这版环境为什么会真正放大 bw 的可改空间

它解决的是三个具体问题。

### 第一，原来 bw 看到的是“平均世界”，现在看到的是“局部持续冲突”

过去同 λ Poisson 下，同一 UAV 的关联 GU 很多时候压力差不大。
现在同一 hotspot subset 会连续 10~20 步高压，bw 终于需要反复决定“先给谁”。

### 第二，原来差异容易被 queue clip 和 downstream 吃掉，现在更容易显化

过去就算多给一点 bw，也常常因为：

* 队列不够大
* 或后面 SAT 更堵

导致 `x_acc` / 队列状态差异出不来。
现在有持续 burst + 前两层紧 buffer + 后层放松，bw 的决定更容易在前半段留下痕迹。

### 第三，原来很多差异只在一步里看不到，现在会变成短前缀多步效应

你的旧诊断已经说明 `K=2` 比 `K=1` 更能看出 bw 作用。
这套 sticky hotspot + preload 正是在给这种“短前缀多步信用”制造训练分布。

---

## 5. 你可以直接照着加的配置

```yaml
env_variant: bw_focus_v1

# 不改这些
num_gu: 20
dynamic_gu_count: false
association_rule: nearest_pathloss
candidate_mode: nearest
tau0: <保持当前值>

# 到达过程
traffic_model: sticky_subset_hotspot
arrival_base_hetero: 0.15
hotspot_num_subsets: 6
hotspot_subset_size: 4      # 若单 UAV 常关联用户偏少，可改成 3
hotspot_rho: 4.0
hotspot_on_mean_steps: 15
hotspot_off_mean_steps: 8
arrival_mean_preserve: true

# buffer：单位都是 base-flow 的“步数”
queue_max_gu_steps_base: 6
queue_max_uav_steps_base: 8
queue_max_sat_steps_base: 24

# downstream 放松
b_sat_total_scale_bwfocus: 1.5
sat_cpu_scale_bwfocus: 2.0

# preload
preload_enabled: true
preload_prob: 0.6
preload_hot_gu_steps: 3.0
preload_bg_gu_steps: 0.2
preload_hot_uav_steps: 2.0
preload_sat_steps: 0.0

# 先不改观测
expose_hotspot_id: false
add_arrival_ema_obs: false
```

其中 `queue_max_*_steps_base` 的换算方式是：

```text
queue_max_gu  = queue_max_gu_steps_base  * mu_gu_step
queue_max_uav = queue_max_uav_steps_base * mu_uav_step
queue_max_sat = queue_max_sat_steps_base * mu_sat_step
```

这里的 `mu_*_step` 用你当前 baseline / heuristic 在**非 burst**环境下的平均每步流量估出来就行。

---

## 6. 如何验证这套方案是否真的解决了 leverage 小的问题

这一步必须分两段做。

### 6.1 先做“环境本身”的 leverage 审计，不训练

这一步最关键。
先别训，先看环境是不是已经把 bw 的可改空间放大了。

#### 做法

固定 accel / sat，用你现成的 base policy 或 heuristic。
从 old env 和 `bw_focus_env_v1` 各采样一批同口径 episode，然后在每个 bw 决策点做 counterfactual：

* `uniform_bw`
* `heuristic_bw`
* 当前 policy bw（可选）

#### 记录三类量

1. `x_acc`
2. `pre_drop = gu_drop + uav_drop`
3. `pre_backlog_steps = (sum_q_gu + sum_q_uav) / arrival_ref`

再做你已经有的：

* one-step 替换
* `K=2` override

#### 我建议的通过标准

新环境至少要满足下面三条里的两条：

1. **`heuristic_bw` 相对 `uniform_bw` 的 one-step 中位 gap 至少比旧环境大 3 倍**
   这个 gap 不要求只看 `x_acc`，`pre_drop` 和 `pre_backlog_steps` 也算。

2. **`x_acc` 的可分性不再停留在 `1e-3 ~ 1e-2` 量级**
   你现在旧环境里 अक्सर就在这个量级。新环境里希望中位数至少进到 `>= 1e-2`，高分位能到几 `1e-2` 甚至更高。

3. **`K=2` override 的增益显著放大**
   至少比旧环境的 `K=2` 中位增益大 3 倍。

如果这一步没过，就说明还没真正把 leverage 做出来，别急着训。

---

### 6.2 再做训练验证，但训练口径不要再改

这里不要再发明新 target 了。
直接复用你已经有的那套 **bw-only 训练 harness**，保持训练代码完全不变。

也就是：

* 还是 fixed accel/sat
* 还是你现在最熟的那套 bw-only 脚本
* 只换环境配置，从 old env 换成 `bw_focus_env_v1`

这样比较最干净。

#### 训练后看什么

还是看你已有的 matched eval：

* deterministic `policy/policy/policy`
* stochastic `policy/policy/policy`
* heuristic bw 对照

#### 我建议用这个指标

定义 gap closure：

```text
gap_closed =
(policy_det - reset_det) / max(heur_det - reset_det, eps)
```

这里 higher-better 的指标用这个公式；
如果是 lower-better 的 `pre_drop` / `pre_backlog_steps`，就把分子分母的方向反过来。

#### 通过标准

在 `bw_focus_env_v1` 上，到了你平时看的 `u100` 或 `u200`：

1. `policy_det` 不能再“和 reset 同量级”
2. `gap_closed` 至少要到 **0.7**
3. stochastic 和 deterministic 的差距要明显缩小
   不应该再出现“sample 出来的动作明显比 deterministic 好很多”的现象

如果这三条都做不到，但 6.1 已经证明环境 leverage 变大了，那就不是环境问题了。

---

## 7. 如果这套方案没效果，下一步怎么分流

### 情况 A：环境审计都没变大

也就是 6.1 没过。

这说明你的 burst 还不够强，或者太分散，没有真正落到 bw 的作用域里。

#### 这时按这个顺序加码

1. `hotspot_subset_size` 从 4 改到 3
   让热点更局部。

2. `hotspot_rho` 从 4 提到 6
   提高局部 burst 强度。

3. `queue_max_gu_steps_base` 从 6 降到 4
   `queue_max_uav_steps_base` 从 8 降到 6
   让前两层更容易进入 pressure 区。

4. `preload_hot_gu_steps` 从 3 提到 4

#### 如果还不够，再开一个更强的环境杠杆：动态 access budget

也就是 `bw_focus_env_v1.1`：

* 每步随机选 1 架 UAV 成为“低 access budget”状态
* 这架 UAV 的 `b_acc` 乘 **0.6**
* 其余 UAV 保持 1.0，或者做总量守恒再微调
* 这个状态也做 sticky，平均持续 10 steps

这一步非常直接。
因为你当前 bw 的根因之一，就是它只是在固定 `b_acc` 里挪份额。
给 `b_acc` 本身加局部时变稀缺性，bw 的 leverage 会再放大一截。

---

### 情况 B：环境审计明显变大了，但 RL 还是学不出来

也就是：

* 6.1 过了
* 6.2 没过

这时就不要再改环境、不要再改 reward 了。

结论会很清楚：

**主问题已经不是 leverage，而是算法本身。**

也就是说，到那时你才该回去看 actor / critic / optimization。
但那一步必须在 leverage 已经被环境证明确实放大之后再做，不然又会回到之前那种混乱状态。

---

### 情况 C：所有策略都崩了，环境太难

表现通常是：

* heuristic 也很差
* uniform 也很差
* drop 全爆
* 各种策略差不多烂

这说明你不是“放大 leverage”，而是把环境推到失稳区了。

#### 回调方式

* `rho_hot` 降一点
* `queue_max_gu/uav_steps_base` 加一点
* `preload_prob` 从 0.6 降到 0.4
* `b_sat_total_scale_bwfocus` 和 `sat_cpu_scale_bwfocus` 保持不变
  先别把 downstream 再收紧

---

### 情况 D：bw 在 `bw_focus_env_v1` 里能学，但一回 joint 就掉

这个其实是最好的失败。

因为它说明：

* bw 头本身能学
* 环境 leverage 方向是对的
* 问题在于“从 bw-focus 迁移回 joint benchmark 时，bw 又被别的瓶颈吞掉了”

这时不要回去改 reward。
直接做一个桥接环境：

**`joint_bridge_env_v1`**

做法：

1. 保留 sticky hotspot arrivals
2. 保留局部 preload
3. 逐步把 `b_sat_total_scale_bwfocus` 从 1.5 调回 1.0
4. 逐步把 `sat_cpu_scale_bwfocus` 从 2.0 调回 1.0
5. 逐步把 `queue_max_sat_steps_base` 从 24 调回你 benchmark 的值
6. 再开 joint 训练

这样你是在**保留 bw leverage 的前提下**把 sat relevance 慢慢放回来。

---

## 8. 为什么我这次不建议先做 GU 数变化 / active mask

random user arrivals 和 dynamic active-user sets 确实很常见；bursty on-off traffic 下，active user set 本来就会变化。([arXiv][2])

但**第一刀不要做这个**，原因不是它不合理，而是它会同时改：

* 输入分布
* 总 offered load
* 关联图
* 训练难度

这样你又会很难判断，到底是 leverage 变大了，还是任务 simply 变了。

所以：

* **v1：不改 GU 总数，不加 active mask**
* **v2：如果 v1 有效，再加 episode-level active session**

  * 每局 `N_active ∈ [14,20]`
  * 先 episode 内固定，不做 step 内 churn
* **v3：最后再做慢速 on/off**

---

## 9. 你现在应该按什么顺序做

按这个顺序，别跳：

1. 实现 `bw_focus_env_v1`

   * sticky subset hotspot arrivals
   * 前两层紧 buffer
   * SAT 放松
   * 局部 preload

2. 先做环境 leverage 审计
   不训练，先看 one-step / K=2 gap 是否变大。

3. 如果 leverage 审计没过
   先按 7A 加码，不要训。

4. leverage 审计过了以后
   用你现有 **同一套 bw-only harness** 直接训，训练代码不改。

5. 如果这时还是学不出来
   停止改环境，转去算法。

6. 如果学得出来
   再做 `joint_bridge_env_v1`，把 sat relevance 逐步加回来。

---

最后压成一句话：

**这次不要再改 target，也不要再在旧废案上打补丁。先造一个专门的 `bw_focus_env_v1`：局部 sticky hotspot arrivals + 前两层紧 buffer + SAT 放松 + 局部 preload。先用环境审计证明 bw 的 counterfactual gap 被放大了至少 3 倍，再用你现有 bw-only 训练口径复测。只有如果“环境 gap 已经明显变大但 RL 仍然不学”，才回去看算法。**

如果你愿意，我下一条可以直接把这个方案再压成“应该改哪些配置项、环境里要加哪几个状态变量、reset/step 各加哪段伪代码”的落地清单。

[1]: https://arxiv.org/abs/2004.07553?utm_source=chatgpt.com "Scheduling for Mobile Edge Computing with Random User Arrivals: An Approximate MDP and Reinforcement Learning Approach"
[2]: https://arxiv.org/abs/2004.07553 "https://arxiv.org/abs/2004.07553"
