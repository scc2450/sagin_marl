你说得对：`lr=0` 对这个问题不是主证据。不更新当然不能解释“为什么 c78 学坏”。现在应该按“等价性断点”查，而不是继续看结果猜。

**核心判断**
c78 现在可以先当成“稳定差”处理：seed42/43 下 update5 都比 good 明显差，不像单次波动。既然 c78 理论上只是把 rollout/obs/reset/部分计算搬到 GPU，那排查标准应该是：

如果同一个状态、同一个 actor、同一个动作、同一个外生随机量，good 和 c78 的下一步不一样，那就是 GPU 化实现不等价。

**具体怎么查**
第一步查“环境一步转移是否等价”，这是最硬的。

不用固定规则策略，而是用真实 accel actor 产生的动作。做法是：

1. 从同一个初始状态开始。
2. 固定 arrival/fading 等外生随机量。
3. 用同一份 actor 动作喂给 good 和 c78。
4. 每一步比较：
   - actor 输入 obs/token/mask 是否一致
   - sampled/deterministic accel action 是否一致
   - safety/danger 后的最终执行 accel 是否一致
   - UAV pos/vel 是否一致
   - GU/UAV/SAT queue 是否一致
   - reward 分项是否一致
   - collision/done/reset 是否一致

这不是“看一两步”，而是跑完整 250 step，自动定位第一个不一致的字段和时间步。只要第一个大差异出现，就能直接指向：obs builder、actor action、safety layer、dynamics、reward、queue update、collision/reset 里的哪一段。

第二步查“actor 输入和分布是否等价”。

如果环境状态还没分叉，但训练行为已经不同，就比较每个 stage 的：

- actor raw input tensor
- mask
- distribution mean/std 或 logits
- sampled action
- old_logprob
- entropy

如果 input 一样但 action/logprob 不一样，就是 native actor / PyTorch actor / sampling 口径问题。如果 input 已经不一样，就是上游 obs/state builder 问题。

第三步查“PPO update 是否等价”。

如果 rollout buffer 内容一致，再比：

- rewards/dones
- values
- returns
- advantages
- normalized advantages
- policy loss
- value loss
- entropy loss
- danger imitation loss
- grad norm
- 每个 update 后 accel_policy 参数 delta

如果 rollout 一样但参数更新方向不一样，就是 PPO/update/optimizer/aux loss 口径问题。

**下一步最应该做**
先做第一步：`teacher-forced full-trajectory equivalence probe`。

也就是用真实 actor 动作，不用固定规则，固定外生随机量，good/c78 各跑同一条 250-step 轨迹，然后报告第一个不一致的位置。这个能把“native rollout 状态演化不等价”这种虚话变成具体字段，比如：

`step 37 env 6: exec_accel differs after danger layer`

或者：

`step 0 env 3: access_rate differs while action/state same`

或者：

`step 68 env 1: collision flag differs while pos/vel nearly same`

只有这样才能知道 c78 是哪里 GPU 化错了，而不是继续被 reward 曲线绕晕。