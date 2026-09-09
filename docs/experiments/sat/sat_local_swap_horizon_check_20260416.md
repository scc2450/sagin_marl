# Structured SAT 局部 Swap 时效检查（2026-04-16）

## 1. 目的

上一版 `sat_action_horizon` 诊断里，第 0 步候选动作来自 top-M subset panel，再按真实 return 重排。
这会混入“候选动作生成器”和“最终评价器不是同一套规则”的问题。

这版改成更干净的口径：

- 基线第 0 步动作：`queue_aware_sat_policy`
- 干预第 0 步动作：只允许 **单个 UAV 把当前已选的一颗星替换成另一颗可见星**
- 第 1 步以后：两边都回到固定 `queue_aware_sat + queue_aware_bw`
- 所有分支都从同一个 `sat-stage snapshot + RNG` 恢复

脚本：
- [scripts/diagnostics/diagnose/diagnose_structured_sat_local_swap_horizon.py](/d:/研三上/毕设/sagin_marl/scripts/diagnostics/diagnose/diagnose_structured_sat_local_swap_horizon.py:1)

## 2. 运行结果

产物：
- [runs/sat_local_swap_horizon_check_20260416/summary.json](/d:/研三上/毕设/sagin_marl/runs/sat_local_swap_horizon_check_20260416/summary.json:1)
- [runs/sat_local_swap_horizon_check_20260416/context_rows.csv](/d:/研三上/毕设/sagin_marl/runs/sat_local_swap_horizon_check_20260416/context_rows.csv:1)

设置：
- `episodes = 1`
- `contexts = 3`
- `H = 1 / 2 / 5 / 10`

结果摘要：
- `H=1` 的 best gap 均值：`0.912`
- `H=2`：`46.614`
- `H=5`：`101.577`
- `H=10`：`313.704`

`step0_abs_share`：
- `H=2 ≈ 0.053`
- `H=5 ≈ 0.026`
- `H=10 ≈ 0.013`

`same_best_as_h1`：
- `H=2 ≈ 1.0`
- `H=5 ≈ 0.667`
- `H=10 ≈ 0.667`

## 3. 这版到底改了什么动作

明细见：
- [context_rows.csv](/d:/研三上/毕设/sagin_marl/runs/sat_local_swap_horizon_check_20260416/context_rows.csv:1)

三个 context 的基线都是：
- `28|7;28|7;28|7`

第 0 步最优局部 swap 示例：
- `UAV1: 7 -> 8`，动作变成 `28|7;28|8;28|7`
- `UAV0: 7 -> 8`，动作变成 `28|8;28|7;28|7`
- `UAV1: 7 -> 29`，动作变成 `28|7;28|29;28|7`

所以这版已经不是“top-M panel 里随便挑另一个 subset”，而是明确的单局部 swap。

## 4. 结论

即使把干预限制成“只换一颗星”，结果仍然显示：

1. `sat` 动作改变的影响不是只停留在当步
2. 长 horizon 下累计 gap 还是明显大于 `H=1`
3. 但这只能说明“闭环轨迹的累计回报差会持续存在”
4. 还不能直接推出“raw state gap 一定在放大”

下一步如果要进一步拆开看，应当额外记录：
- 每步 `sat` 后续动作
- 每步 `bw` 后续动作
- `gu/uav/sat queue` 差分路径
- 每步 reward delta 路径
