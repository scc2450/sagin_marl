# 地面用户热点模式与重要性采样实验总结

## 1. 地面用户热点模式

当前主要实验场景是 `3UAV / 20GU`。地面用户不是均匀分布，而是按空间簇生成：

```text
GU 簇数量 = 3
簇内标准差 = 80 m
簇中心最小间距 = 700 m
```

在此基础上，业务到达过程使用持续热点模式。热点子集不是固定的：

- 每个 episode reset 时，GU 位置会重新生成。
- 系统会基于当局 GU 的空间邻近关系重新构造候选热点子集。
- episode 运行过程中，当前激活的热点子集也会随时间切换。

当前热点参数为：

```text
候选热点子集数量 = 20
每个热点子集大小 = 10 个 GU
热点强度 rho = 20
热点平均持续时间 = 20 step
热点平均关闭时间 = 1 step
arrival_mean_preserve = true
```

含义是：某一时刻约 10 个空间邻近 GU 处于高业务状态；该热点平均持续 20 步，然后关闭，并平均 1 步后切换到另一个热点子集。

`arrival_mean_preserve = true` 表示保持全体 GU 的平均到达强度不变，只改变业务在 GU 之间的分布。因此该设置制造的是局部持续拥塞，而不是单纯增加系统总业务量。

需要强调的是：之前的 `K=5 / no-grow` 主结果已经使用这套热点模式；它不是重要性采样实验才新引入的条件。

## 2. 重要性采样方法

重要性采样用于 actor update。原始做法是每个 stage 使用整个 batch 的全部 row 更新；本次实验改成每轮 actor update 只从当前 batch 中抽取一半 row，并用重要性权重修正抽样分布。

样本优先级定义为：

```text
priority_i = |A_i| * score_norm_i
```

其中：

```text
A_i 是该 row 的 advantage。
|A_i| 越大，说明该 row 对 actor loss 的权重信号越强。

score_norm_i 是 log pi(a_i | s_i) 对策略分布输出参数的梯度范数。
它衡量的是：在当前策略分布下，这个 row 的 logprob 对分布参数有多敏感。
```

这里不是对整个 actor 网络参数求梯度，而是只对 action distribution 的直接参数求梯度：

```text
Accel:
distribution 参数是高斯的 mu 和 log_std。
score_norm = || d log pi / d(mu, log_std) ||

SAT:
distribution 参数是合法卫星子集的 categorical logits。
score_norm = || d log pi / d logits ||

BW:
distribution 参数是 Dirichlet / simplex policy 的直接输出：
valid GU score logits、tau raw、kappa raw。
score_norm = || d log pi / d(score logits, tau raw, kappa raw) ||
```

因此，该 priority 的含义是：优先抽取 advantage 大、并且 logprob 对当前策略分布参数敏感的 row。

本次实验参数：

```text
sample_frac = 0.5
alpha = 0.5
weight_clip = [0.25, 4.0]
uniform_frac = 0
```

也就是每轮 actor update 抽取当前 batch 的一半 row，全部按 priority 重采样，再用重要性权重校正。

## 3. 实验结果

300 updates 对比结果如下：

| 方法 | episode reward 最高值 |
| --- | ---: |
| K=5 / no-grow 基线 | 51.96 |
| 加重要性采样 | 44.94 |

速度对比是在相同条件下测得：

```text
64 env
250 step
1 update
BW K = 5
```

| 方法 | collect | critic | actor | total |
| --- | ---: | ---: | ---: | ---: |
| 重要性采样关闭 | 13.3 s | 117.4 s | 10.0 s | 140.7 s |
| 重要性采样开启 | 12.8 s | 117.5 s | 11.7 s | 142.1 s |

速度没有变快的原因是：重要性采样虽然只让 actor loss 使用一半 row，但仍然需要先对整个 batch 计算 priority 和 score norm；同时总耗时主要在 critic，actor 只占较小部分。因此抽半数 actor row 并没有带来总训练时间收益。

## 4. 结论

这版重要性采样没有提升训练效果，也没有带来速度收益，因此默认关闭。

当前默认配置为：

```text
stage_actor_importance_sampling_enabled = false
```

