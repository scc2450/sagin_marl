# Structured SAT 动作影响时效检查（2026-04-16）

## 1. 目的

在继续推进 `sat clean joint teacher` 之前，先回答一个更基础的问题：

- 改一次 `sat` 动作的影响，主要集中在**当前 step**，还是会通过队列/后续固定 partner 继续往后传？
- 如果影响明显跨多步，`1-step teacher` 就只能当 `v1`，不能把它当成最终形态。

## 2. 检查口径

使用配置：
- [configs/clean_sat/structured_sat_clean_joint_beijing_hotspot_res200.yaml](/d:/研三上/毕设/sagin_marl/configs/clean_sat/structured_sat_clean_joint_beijing_hotspot_res200.yaml:1)

固定执行源：
- `accel_source = cluster_center_queue_aware`
- `current_sat_source = queue_aware`
- `followup_sat_source = queue_aware`
- `bw_source = queue_aware`

关键口径：
- 在某个 `sat-stage snapshot` 上，只替换**第 0 步**的 `sat` 联合动作
- 后续步骤都回到固定 `queue_aware_sat + queue_aware_bw`
- 所有候选都从**同一个 snapshot + 同一个 RNG 状态** replay
- 回报用 `weighted_workload_level`

也就是说，这里量的是：

“改这一次 `sat`，后面会拖出多长的真实影响”

而不是：

“两条不同 sat policy 连续跑很多步谁更好”

脚本：
- [scripts/diagnose_structured_sat_action_horizon.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_sat_action_horizon.py:1)

## 3. 两组检查

### 3.1 第一组：top-3 per UAV

运行产物：
- [runs/sat_action_horizon_check_20260416/summary.json](/d:/研三上/毕设/sagin_marl/runs/sat_action_horizon_check_20260416/summary.json:1)
- [runs/sat_action_horizon_check_20260416/context_rows.csv](/d:/研三上/毕设/sagin_marl/runs/sat_action_horizon_check_20260416/context_rows.csv:1)

设置：
- `topm_per_uav = 3`
- joint panel = `3^3 = 27`
- `contexts = 6`
- `H = 1 / 2 / 5 / 10`

结果摘要：
- `H=1` 的 best gap 均值只有 `0.281`
- `H=2` 提到 `143.236`
- `H=5` 提到 `279.074`
- `H=10` 提到 `745.021`

同一步影响占比：
- `step0_abs_share@H=2 ≈ 0.107`
- `step0_abs_share@H=5 ≈ 0.048`
- `step0_abs_share@H=10 ≈ 0.020`

`H=1` 最优动作稳定性：
- `same_best_as_h1@H=2 ≈ 0.667`
- `same_best_as_h1@H=5 ≈ 0.500`
- `same_best_as_h1@H=10 ≈ 0.500`

### 3.2 第二组：top-4 per UAV 稳健性检查

运行产物：
- [runs/sat_action_horizon_check_top4_20260416/summary.json](/d:/研三上/毕设/sagin_marl/runs/sat_action_horizon_check_top4_20260416/summary.json:1)

设置：
- `topm_per_uav = 4`
- joint panel = `4^3 = 64`
- `contexts = 3`

结果摘要：
- `H=1` 的 best gap 均值 `1.140`
- `H=2` 提到 `47.403`
- `H=5` 提到 `99.677`
- `H=10` 提到 `310.082`

同一步影响占比：
- `step0_abs_share@H=2 ≈ 0.065`
- `step0_abs_share@H=5 ≈ 0.035`
- `step0_abs_share@H=10 ≈ 0.014`

`H=1` 最优动作稳定性：
- `same_best_as_h1@H=2 ≈ 0.667`
- `same_best_as_h1@H=5 ≈ 0.333`
- `same_best_as_h1@H=10 ≈ 0.333`

## 4. 结论

当前这条 structured fixed-partner 口径下，`sat` 动作改变的影响**不是短时的当步效应**，而是明显会通过后续状态继续放大：

1. `H=1` 的 gap 很小，但 `H=2/5/10` 明显变大。
2. 在长 horizon 下，绝大多数收益差并不来自第 0 步本身。
3. `H=1` 最优动作到 `H=5/10` 经常会换，这说明一步排序并不稳定。

所以当前判断是：

- `1-step teacher` 可以作为 `v1` 启动版本
- 但不能把它当成最终版
- 后续至少应考虑：
  - `1-step panel -> top-2 rerank with short horizon`
  - 或直接把 teacher 打分改成 `k-step`

## 5. 直接影响到实现方案的地方

这次检查更支持下面这个推进顺序：

1. 先保留当前已经实现的 `1-step sat clean joint` 作为可运行版本
2. 下一步优先补：
   - `sat_clean_teacher_horizons`
   - `sat_clean_rerank_topk`
   - `sat_clean_teacher_return_mode = discounted_k_step`
3. 不建议把 `1-step` 直接视为最终 teacher 定义

## 6. 备注

这次检查有两个边界：

1. 当前 `current_sat_source` 用的是 `queue_aware`，不是 learned sat actor
2. panel 是 `top-M` 候选，不是全 `21^3` exact search

但这两点并不影响这次检查最核心的结论：

`sat` 的动作影响在这条环境口径里具有明显的跨步时效，不能只看一步。 
