# Structured 单 GPU 原生架构目标方案（2026-04-17，2026-05-04 重写）

本文档原文件出现了 mojibake 编码损坏，已在 2026-05-04 重写为干净 UTF-8 文档。损坏原文备份在：

```text
docs/structured_single_gpu_target_architecture_20260417.mojibake.bak
```

这份重写版保留原主题：在只有一张 GPU 的前提下，`structured` 环境、训练和评估应该收敛到怎样的目标架构。

## 1. 文档目的

目标不是“为了兼容旧实现继续打补丁”，而是明确最终架构应该长什么样。

唯一优先级有两条：

```text
1. 执行速度要快
2. 计算结果要正确且稳定
```

如果现有对象模型、进程模型或兼容接口会拖慢主路径，或者增加状态不同步风险，就应该把它降级为 reference / debug / compatibility path，而不是让它继续决定主架构。

## 2. 核心结论

单 GPU 下，`structured` 环境的合理主架构应当是：

```text
一个单进程、单 GPU、batched 的环境内核
```

训练和正式评估都应走同一条 GPU 原生主路径：

```text
config / factory
  -> GPU native env group
  -> batched rollout kernel / tensor core
  -> rollout buffer batch view
  -> PyTorch actor/critic update
```

旧 CPU env、旧 driver、旧 subproc worker 可以保留，但只能作为：

```text
reference implementation
compatibility shell
debug / equivalence check
legacy script fallback
```

它们不应该再作为 structured 训练和正式评估的默认主路径。

## 3. 不应该继续采用的架构

以下形式不应作为最终目标：

```text
subproc 多进程 + 每个 env 一个 Python 对象
CPU env step 完再把结果搬到 GPU
外层 Python loop 拼 batch
训练热路径频繁 CPU/GPU 往返
为了兼容旧脚本，让旧 env/driver 继续主导 runtime state
```

这些形式的问题不是“代码不好看”，而是会直接制造：

```text
性能瓶颈
状态同步错误
reset / history / snapshot 口径不一致
native path 与 Python path 行为不等价
```

这些问题已经多次表现为训练结果异常、rollout 速度异常、actor/critic 输入口径不一致。

## 4. 目标组件边界

### 4.1 Runtime State

运行时真源应只有一份：

```text
GPU tensor runtime state
```

旧的 numpy / Python env state 只能是镜像或导出视图，不能反向覆盖 GPU runtime。

状态包括：

```text
UAV/GU/SAT position and velocity
queues and drops
arrival / fading / hotspot tape
association / candidate / mask
SAT selection prefix
BW allocation prefix
reward part metrics
done / reset state
history snapshots
```

### 4.2 Env Group

`GpuStructuredEnvGroup` / `GpuStructuredDriverGroup` 应直接持有 GPU core，而不是先构造一批旧 `SaginParallelEnv`。

旧 env/driver 只在显式兼容接口被访问时 lazy materialize。

### 4.3 Rollout Buffer

rollout buffer 应保存 batched stage view，而不是逐条 append `StageTransition`。

训练 update 应直接消费：

```text
training_view
return_view
bootstrap_view
stage_batches[accel/sat/bw]
```

### 4.4 Actor / Critic

actor 和 critic 的输入构造必须来自同一套 stage/world schema。

不能出现：

```text
native actor 看到一套 obs
PyTorch update 看到另一套 obs
critic history 看到 action 之后的偷看 state
eval writer 又写第三套 world feature
```

actor old logprob、update new logprob、critic value、GAE return 必须有清晰一致的口径。

## 5. 三阶段执行语义

structured 控制仍然是三阶段：

```text
accel stage
sat stage
bw stage
```

但这三阶段应在同一个 GPU runtime 里连续推进。

每个 stage 的 snapshot 必须表达“当前 stage 决策前的 prefix state”，不能把当前动作执行后的结果写成当前 stage 输入。

正确时序是：

```text
prepare accel stage state
  -> accel actor action
  -> dynamics / association update

prepare sat stage state
  -> sat actor action
  -> sat prefix / load update

prepare bw stage state
  -> bw actor action
  -> access / backhaul / processing / queue / reward update
```

history、world、local obs、critic state 都应遵守这个 prefix 语义。

## 6. Native Kernel / Tensor Core 原则

“native kernel”在这里不是单指某一个 CUDA kernel 文件，而是指训练主路径里不回到 per-env Python 对象的 batched GPU 执行内核。

优先级是：

```text
1. 先保证语义正确
2. 再减少 Python 调度和小 Torch op
3. 最后才考虑更底层 custom CUDA/Triton 融合
```

不要为了“看起来高级”过早写 custom op。每个 native kernel 或 fused tensor function 都必须有 reference 口径。

## 7. 性能优化顺序

推荐顺序：

```text
1. 消除主路径 CPU/GPU 往返
2. 消除 per-env Python loop
3. 消除按字段循环发射的小 Torch op
4. 合并 history / snapshot / world writer
5. 用 profiler 找真正热点
6. 对热点尝试 torch.compile 或 fused tensor function
7. 只有 compile/fusion 仍不够时，再考虑 Triton/CUDA custom op
```

优化不能改变语义。任何性能改动都需要能回答：

```text
改动前后同 seed / 同 action / 同外生随机量下，
reward、done、queue、position、obs、history 是否一致？
```

## 8. 等价性验收

迁移到 GPU 原生主路径时，不能只看训练曲线。

必须保留以下验收：

```text
single-step equivalence
forced-action full-trajectory equivalence
actor input equivalence
old_logprob / new_logprob equivalence
value / return / advantage equivalence
reward parts equivalence
reset and mid-rollout reset equivalence
```

如果 good/reference 和 native 训练曲线不同，第一反应不应是调网络，而是定位哪一层不等价。

## 9. Reset / Random Tape 原则

arrival、hotspot、fading、reset 初始状态都必须由 runtime 明确管理。

允许两种实现：

```text
CPU 预生成 tape 后上传 GPU
GPU 上复刻同分布生成
```

但无论哪种，都必须清楚：

```text
训练主路径用的是哪一份 tape
branch replay 用的是哪一份 tape
mid-rollout reset 后如何继续
同一个 snapshot 的 branch 是否共享同一外生随机性
```

随机性不需要 bit-exact 复刻旧 Python 路径，但如果目标是排查等价性，就必须能固定随机 tape 做严格对照。

## 10. 兼容层定位

兼容层可以存在，但不能污染主路径。

允许：

```text
旧脚本通过 slot adapter 调用 GPU group
reference/debug 时 materialize 旧 env shell
导出 numpy 结果给分析脚本
```

不允许：

```text
训练 hot path 每步回到旧 env
旧 env cache 反向覆盖 runtime tensor state
脚本直接 new SaginParallelEnv 作为 structured 默认入口
native path 与 Python path 分别维护不同语义
```

## 11. 对 custom kernel 的定位

custom kernel 不是第一阶段工作。

只有在下面条件都满足时才值得做：

```text
batched GPU env 主路径已经稳定
reference 语义已经清楚
profiler 证明某个融合热点长期占主导
torch.compile / tensor fusion 仍不够
```

custom kernel 应优先服务这些高频混合链路：

```text
visible SAT filtering / geometry
access and backhaul rate
queue transition
reward / done / reset finalize
world / local obs writer
history snapshot writer
```

不要为很小的点操作单独下沉 kernel，否则 launch overhead 可能比计算本身更大。

## 12. 当前检查清单

主仓库应持续满足：

```text
训练默认 backend 是 GPU native / sync，不是 subproc
正式 eval 和训练共用 GPU native rollout path
旧 CPU env 只作为 reference/fallback
stage snapshot 是 action-before prefix state
actor/critic/world 输入来自同一 schema
BW/SAT/accel 的 mask 与 action slot 对齐
reward parts 在 native/Python/tensor path 语义一致
history 写入没有 env offset / stale prefix / post-action 偷看错误
```

## 13. 最终建议

本项目的 structured 主架构应收敛为：

```text
单进程
单 GPU
batched runtime state
native rollout
batched rollout buffer
PyTorch update
reference path 显式 opt-in
```

换句话说：

```text
旧 CPU env 是 gold reference，不是生产主实现。
兼容接口是边界工具，不是热路径骨架。
native GPU runtime 是训练和正式评估的唯一主路径。
```

后续如果训练表现异常，优先查等价性、输入语义、history 时序和 credit 口径；不要在没有证据时先重做 actor/critic 网络。
