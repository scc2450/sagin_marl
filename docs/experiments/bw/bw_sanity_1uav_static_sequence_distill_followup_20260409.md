# 1 UAV Static BW Sanity: Sequence-Level Distill Follow-up (2026-04-09)

## 1. 目的

这份文档总结 2026-04-09 针对 `1 UAV / BW-only` 简化 benchmark 做的 `sequence-level distill` 验证。

这轮验证的核心问题不是：

- PPO 还能不能继续打补丁
- 或者再换一条一步 reward 能不能凑巧训出来

而是更直接的问题：

> 如果我们承认这个 `BW` 问题更像“连续很多步都要做对”的控制问题，  
> 那么把训练对象从“一步动作”改成“一段动作序列”，会不会明显不同？

## 2. 这次验证的对象

### 2.1 脚本

- `scripts/experiments/bw_training/train_structured_bw_sequence_distill_diagnostic.py`

这个脚本是新加的 sequence-level 诊断版，和之前的一步版

- `scripts/experiments/bw_training/train_structured_bw_search_distill_diagnostic.py`

相对应。

### 2.2 训练目标

这次 `sequence-level distill` **没有**用 `gu_queue_level` 或 `weighted_workload_delta` 来给候选序列打分。

候选序列打分直接累加的是：

- `float(next(iter(step_result.rewards.values())))`

也就是环境 step 返回的 team reward，也就是主任务 `env reward`。

所以这次回答的问题是：

> 如果直接按最终任务 reward 做 sequence-level target，训练表现会不会不同？

而不是：

> 如果再换一条 BW shaping reward，会不会不同？

### 2.3 fixed bank 的定义

脚本支持：

- `rolling bank`
- `fixed bank`

其中 `fixed bank` 的语义是：

- 先加载 `init_actor`
- 用 **这个初始化 actor** 收一批 `BW` snapshot
- 后续所有 update 都复用这同一批起始状态

也就是说，fixed bank **不是**用 teacher 收的，也不是每轮更新后重新收的。

## 3. 为什么要做 sequence-level 而不是一步 target

前面的诊断已经证明：

- 好策略的优势主要不是来自“单独第一步特别神奇”
- 而是来自“后续很多步都一起对”

最直接的证据是：

- `runs/structured_bw_kstep_splice_bad_to_good/summary.json`

平均结果：

| K | score |
| --- | ---: |
| 0 | `-10.130` |
| 1 | `-9.993` |
| 2 | `-9.794` |
| 5 | `-9.513` |
| 10 | `-9.558` |
| 20 | `-8.489` |
| 100 | `8.833` |

对应提升：

- `K=1`: `+0.138`
- `K=20`: `+1.641`
- `K=100`: `+18.964`

这说明：

- 一步 improvement 很弱
- `K=1~10` 的局部修补远远不够
- 真正的收益来自长时间连续一致地做对

这正是 sequence-level 验证的动机。

## 4. 具体做法

### 4.1 序列搜索

对每个起始 `BW` snapshot：

1. 从当前 actor 出发，生成若干条长度 `H` 的候选 `BW` 动作序列
2. 候选动作不是启发式规则，而是围绕当前策略中心做 wide simplex 采样
3. 每条候选序列都从该起始状态 rollout 到 episode 结束
4. 用整回合 `env reward` 给整条序列打分
5. 选 reward 最高的一条序列作为 target

### 4.2 序列蒸馏

不是只取 target sequence 的第一个动作。

而是把这条 best sequence 里的整段：

- `(state_t, action_t)`
- `(state_{t+1}, action_{t+1})`
- ...
- `(state_{t+H-1}, action_{t+H-1})`

都拿出来，训练 actor 的 `det_mean` 去拟合这些 target action。

所以这次的训练对象是：

- sequence prefix 上的整段 `(state, action)` 样本

不是：

- 坏 continuation 下的一步 deviation action

## 5. 关键结果

### 5.1 坏初始化 + rolling bank

运行目录：

- `runs/structured_bw_sequence_distill_env_badinit_h20_c8_u5`

结果：

- `update 1`: `eval_reward = 0.123`
- `update 5`: `eval_reward = -6.409`

最终指标：

- `reward = -6.409`
- `processed = 0.695`
- `drop = 0.209`
- `backlog = 12.406`

结论：

- sequence-level 相比一步版已经不是“完全没信号”
- 但如果每轮都重新用**当前坏 actor**收 on-policy state bank，训练会继续变差

### 5.2 坏初始化 + fixed bank（4 个起点）

运行目录：

- `runs/structured_bw_sequence_distill_env_badinit_h20_c8_u5_fixedbank`

结果：

- `update 1`: `eval_reward = 0.123`
- `update 5`: `eval_reward = 3.219`

最终指标：

- `reward = 3.219`
- `processed = 0.763`
- `drop = 0.160`
- `backlog = 10.690`

结论：

- 同样的 sequence-level 更新，只把 state bank 固定住，就能稳定变好
- 说明问题不在“actor 学不会 sequence target”

### 5.3 坏初始化 + fixed bank（8 个起点）

运行目录：

- `runs/structured_bw_sequence_distill_env_badinit_h20_c4_u5_fixedbank_s8`

最终结果：

- `reward = 8.895`
- `processed = 0.797`
- `drop = 0.120`
- `backlog = 10.744`

相对坏初始化 eval-only 基线：

- `runs/structured_bw_search_distill_evalonly_badinit/summary.json`
- 基线 `reward = -5.653`

结论：

- 只增加 fixed bank 覆盖，不增加总候选预算量级，在线泛化就能继续明显提高
- 说明 state bank coverage 是当前关键杠杆之一

### 5.4 坏初始化 + fixed bank（8 个起点）+ 更长 horizon `H=50`

运行目录：

- `runs/structured_bw_sequence_distill_env_badinit_h50_c4_u5_fixedbank_s8`

最终结果：

- `reward = 14.480`
- `processed = 0.830`
- `drop = 0.0769`
- `backlog = 11.43`

和 `H=20` 对比：

| Setting | reward | processed | drop |
| --- | ---: | ---: | ---: |
| `H=20, fixed bank, 8 states` | `8.895` | `0.797` | `0.120` |
| `H=50, fixed bank, 8 states` | `14.480` | `0.830` | `0.0769` |

结论：

- 更长 horizon 明显有效
- 这和前面的 `K-step splice` 结论一致：这个问题确实需要多步一致性

### 5.5 好初始化 + rolling bank

运行目录：

- `runs/structured_bw_sequence_distill_env_imitationinit_h20_c8_u5`

最终结果：

- `reward = 25.881`
- `processed = 0.906`
- `drop = 0.0307`
- `backlog = 9.328`

对照固定 baseline：

- `queue_aware_bw` 固定参考：`25.933`

并且：

- `sequence_gap_mean < 0`

结论：

- 好初始化附近，局部 sequence search 已经几乎找不到更好的序列了
- 这说明 sequence-level oracle 本身不是乱的

### 5.6 面向正式模型成本的近似：anchored replay bank

为了验证“正式模型可承受”的近似方案，这次还做了一个折中版本：

- 固定底座 bank：`8` 个起点
- 每轮追加新状态：`4` 个
- 动态 bank 容量：`16`
- 每轮训练样本：`8` 个
  - `4` 个来自底座 bank
  - `4` 个来自动态 bank

这个近似方案比纯 fixed bank 更贴近正式模型可承受的做法，因为它不要求每轮都在全固定大 bank 上做搜索，也不要求每轮都完全 rolling 重收状态。

#### `H=20`

运行目录：

- `runs/structured_bw_sequence_distill_env_badinit_h20_c4_u5_anchored_s8_b4a4d16`

最终结果：

- `reward = 4.817`
- `processed = 0.772`
- `drop = 0.1498`
- `backlog = 10.75`

对照：

| Setting | reward |
| --- | ---: |
| bad init eval-only | `-5.653` |
| sequence, rolling bank, `H=20` | `-6.409` |
| sequence, fixed bank 4, `H=20` | `3.219` |
| sequence, fixed bank 8, `H=20` | `8.895` |
| sequence, anchored replay bank, `H=20` | `4.817` |

结论：

- anchored replay bank 明显比完全 rolling 更稳
- 但在 `H=20` 下还不如纯 fixed bank 8-state

#### `H=50`

运行目录：

- `runs/structured_bw_sequence_distill_env_badinit_h50_c4_u5_anchored_s8_b4a4d16`

最终结果：

- `reward = 15.238`
- `processed = 0.835`
- `drop = 0.0768`
- `backlog = 10.82`

对照：

| Setting | reward |
| --- | ---: |
| sequence, fixed bank 8, `H=50` | `14.480` |
| sequence, anchored replay bank, `H=50` | `15.238` |
| good init + rolling bank + `H=20` | `25.881` |

结论：

- 在更长 horizon 下，anchored replay bank 不仅有效，而且已经略优于纯 fixed bank
- 但即使这样，距离好初始化/启发式水平仍有明显 gap
- 所以当前近似方案的瓶颈不再是“完全 rolling 导致崩”，而是：
  - horizon 仍然不够长
  - bank 覆盖仍然不够宽
  - search 预算仍然不够强

## 6. 现在可以确定什么

### 6.1 可以确定的

现在已经可以比较明确地说：

1. `BW` 这个问题更像多步一致控制问题，而不是一步动作选择问题
2. sequence-level target 比一步 target 更匹配这个问题
3. actor 能学 sequence target；问题不在“表达力不够”
4. `rolling on-policy bank` 会把 sequence target 带偏
5. `fixed + wider bank` 会明显改善这个问题
6. `longer horizon` 会继续明显改善这个问题

### 6.2 还不能确定的

还不能说：

- sequence-level distill 已经是最终答案
- 只要继续加 horizon 和加 bank，就一定能追上好初始化

因为当前最好的坏初始化 run 仍然和好初始化有明显 gap：

| Setting | reward |
| --- | ---: |
| bad init eval-only | `-5.653` |
| bad init + fixed bank 8 + `H=20` | `8.895` |
| bad init + fixed bank 8 + `H=50` | `14.480` |
| bad init + anchored replay bank + `H=50` | `15.238` |
| good init + rolling bank + `H=20` | `25.881` |

所以 sequence-level 这条线已经证明“方向是对的”，但还没有逼近最终上限。

## 7. “moving state bank 会把 target 带偏”是什么意思

这里“带偏”的意思不是：

- target 计算错了
- 或 reward 公式错了

而是：

- 每一轮都用当前 actor 自己 rollout 出来的起始状态来做 sequence search
- actor 一变，这批起始状态也跟着变
- 所以 target 不是在逼近一个稳定对象，而是在追一个会跟着当前坏策略漂移的目标

这点的最直接证据是：

- 坏初始化 + rolling bank：`reward` 最后掉到 `-6.409`
- 坏初始化 + fixed bank：`reward` 提升到 `3.219`
- 坏初始化 + wider fixed bank：`reward` 提升到 `8.895`

所以当前 sequence-level 结果说明：

> 在线自举的 state distribution shift，比“一步 PPO 的形式”更像主要阻碍之一。

## 8. 对正式模型的意义：成本是不是太高

### 8.1 是，太高，不适合作为正式模型的最终在线训练方案

这是当前最重要的工程结论。

即使在这个极简 benchmark 上，sequence-level distill 要做的也是：

- 多个起始状态
- 每个起始状态多条候选序列
- 每条候选序列 rollout 到 episode 结束
- 再把整段 prefix 都蒸馏回来

这在正式模型里会更贵，因为正式模型还有：

- 多 UAV
- `accel / bw / sat` 三个动作头共同作用
- 更复杂的状态分布
- 更高的交互成本

所以：

> `长 horizon + 稳定/足够宽的 state bank`
>
> 在这个简化模型里有很强的诊断价值，  
> 但对正式模型来说，作为最终在线训练方案，成本确实太高。

不过这次 anchored replay bank 的结果也说明：

- 完全 rolling 不行
- 纯 fixed 也不是唯一选项
- “固定底座 + 渐进扩 bank” 是一个更现实的中间路线

它仍然不算便宜，但至少已经开始接近“正式模型可承受的近似”。

### 8.2 它更像什么

它更像：

- 诊断工具
- 机制验证工具
- 生成更可靠 teacher / replay bank 的工具
- 或 warm-start / offline improvement 的工具

而不是：

- 正式模型直接在线跑的主训练算法

## 9. 当前最可信的结论

到目前为止，最收敛的结论是：

1. 之前的一步 PPO/AWR 不是对的训练接口
2. 这个 `BW` 简化问题需要多步一致 target
3. sequence-level distill 已经证明“多步一致 target”这条方向是有效的
4. 但真正限制效果的，不只是 horizon，还包括 on-policy moving bank 带来的 state-distribution drift
5. 对正式模型，不能直接照搬“长 horizon + 宽固定 bank”的在线搜索蒸馏，因为成本过高

所以这条线的价值主要是：

> 它已经把问题定位到了  
> “训练接口与状态分布自举方式不匹配”  
> 而不是 reward、critic、或 actor 表达力本身。

## 10. 目前最合理的后续方向

如果继续往正式模型可用的方向推进，当前更合理的是：

1. 保留 sequence-level 作为诊断/teacher 生成工具
2. 不把它当最终在线训练主循环
3. 接下来寻找更便宜的近似：
   - 固定底座 bank + 逐步扩 bank
   - offline / replay-based policy iteration
   - 先 sequence-level warm-start，再轻量在线 fine-tune

而不是再回到：

- 一步 PPO/AWR
- 一步 deviation oracle
- rolling on-policy one-step target
