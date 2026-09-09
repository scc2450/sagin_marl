# Structured SAT 局部 Swap Trace 检查（2026-04-16）

## 1. 这次补了什么

在上一版 [sat_local_swap_horizon_check_20260416.md](/d:/研三上/毕设/sagin_marl/docs/sat_local_swap_horizon_check_20260416.md:1) 的基础上，
把局部 swap 诊断脚本继续补成了“可追路径”的版本：

- 脚本：
  [scripts/diagnostics/diagnose/diagnose_structured_sat_local_swap_horizon.py](/d:/研三上/毕设/sagin_marl/scripts/diagnostics/diagnose/diagnose_structured_sat_local_swap_horizon.py:1)
- 新增输出：
  - [runs/sat_local_swap_trace_check_20260416/summary.json](/d:/研三上/毕设/sagin_marl/runs/sat_local_swap_trace_check_20260416/summary.json:1)
  - [runs/sat_local_swap_trace_check_20260416/context_rows.csv](/d:/研三上/毕设/sagin_marl/runs/sat_local_swap_trace_check_20260416/context_rows.csv:1)
  - [runs/sat_local_swap_trace_check_20260416/context_traces.json](/d:/研三上/毕设/sagin_marl/runs/sat_local_swap_trace_check_20260416/context_traces.json:1)

`context_traces.json` 里每个 `context / horizon` 现在都包含：

- `baseline_trace_steps`
- `best_trace_steps`
- 每步 `sat_action_ids_text`
- 每步 `bw_action_text`
- 每步 `queue_before / queue_after`
- 每步 `reward`
- `gu/uav/sat/total queue sum delta path`

也就是现在已经可以直接拆开看：

- 后续 `sat` 动作是不是变了
- 后续 `bw` 动作是不是变了
- 队列差是在扩大、缩小还是来回波动
- 累计回报差和队列差是不是同一回事

## 2. 干预口径

这次仍然保持最干净的局部干预定义：

- 第 0 步基线：`queue_aware_sat_policy`
- 第 0 步干预：只允许一个 UAV 把当前已选的一颗星换成另一颗可见星
- 第 1 步以后：两边都回到固定 `queue_aware_sat + queue_aware_bw`
- 所有分支都从同一个 `sat-stage snapshot + RNG` 恢复

所以这次 trace 里看到的后续差异，不是“随机性没对齐”，而是“第 0 步 sat 干预后，闭环状态变了，固定 heuristic 按不同状态继续出动作”。

## 3. 这次跑出来的结果

运行参数：

- `episodes = 1`
- `contexts = 3`
- `H = 1 / 2 / 5 / 10`

主汇总和上一版一致：

- `H=1` best gap 均值：`0.912`
- `H=2`：`46.614`
- `H=5`：`101.577`
- `H=10`：`313.704`

## 4. 新 trace 里最重要的观察

### 4.1 后续 sat 动作确实会分叉

例如 `context 0 / H=10`：

- baseline 前 4 步 sat 路径：
  `28|7;28|7;28|7 -> 28|8;28|8;28|8 -> 28|7;28|7;28|7 -> 28|8;28|8;28|8`
- best 前 4 步 sat 路径：
  `28|7;28|8;28|7 -> 28|7;28|8;28|7 -> 28|7;28|8;28|7 -> 28|7;28|8;28|7`

这说明“后续固定 sat heuristic”不是“动作也固定不变”，而是会根据不同状态走出不同动作序列。

### 4.2 后续 bw 动作一开始往往很接近

同一个 `context 0 / H=10` 里，baseline 和 best 的前 2 步 `bw_action_text` 是一致的。

这说明：

- 累计回报差并不一定来自“后续马上换了完全不同的 BW 分配”
- 至少在前几步里，差异更像是 sat 干预改变了状态轨迹，然后这个差异逐步传下去

### 4.3 “累计回报差变大”不等于“总队列差一直放大”

这是这次 trace 最关键的补充。

例如 `context 0 / H=10`：

- `delta_path`：
  `[1.065, 7.103, 3.588, 6.464, 0.280, 3.054, 0.624, 4.351, 1.165, 6.031]`
- `total_queue_sum_delta_path`：
  `[-1760727.0, -8475314.5, -4962820.0, -6246368.5, -51893.0, 0.0, 0.0, 0.0, 0.0, 0.0]`

也就是说：

- reward delta 后面还在持续为正
- 但总队列和的差值并不是一直越拉越大，后面甚至会回到 `0`

这说明当前 `weighted_workload_level` 下，后效不只是“总 backlog 多少”，还和“backlog 落在哪一层、哪几个节点上”有关。

这和 reward 定义是一致的，因为它本来就是加权 workload level，不是简单的总队列和。

### 4.4 有的 context 第 0 步甚至先亏，后面再赚

例如 `context 2 / H=10`：

- `delta_path`：
  `[-3.345, 106.152, 1.959, 125.616, 7.241, 144.686, 9.946, 161.601, 11.468, 179.002]`

这说明：

- 只看 `H=1` 会把这类动作判成坏动作
- 但它在后续闭环里会明显转优

所以 `sat` teacher 只看 `1-step` 仍然是不够的，这一点被新 trace 进一步确认了。

## 5. 现在能回答什么，不能回答什么

现在能比较明确地回答：

1. 第 0 步局部换星会改变后续闭环动作序列。
2. 这个影响会穿过后续固定 `queue_aware_sat + queue_aware_bw` 继续存在。
3. 累计回报差变大，并不意味着总队列差一定持续放大。

现在还不能直接下结论的是：

1. 哪一层队列分布变化最关键。
2. 是 `GU/UAV/SAT` 哪一层在主导 `weighted_workload_level` 的后效。
3. 多步 teacher 最合适用 `H=5`、`H=10`，还是“`1-step panel + short-horizon rerank`”。

## 6. 对后续 teacher 设计的启发

从这次 trace 看，下一步更合理的是：

- 保留当前“局部可控干预 + 同 snapshot replay”的诊断框架
- teacher 仍然不要只看 `H=1`
- 更适合做成：
  `1-step 或局部候选生成 -> short-horizon rerank`

而不是继续把 `1-step` 当最终 teacher 定义。
