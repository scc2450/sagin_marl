# SAT Macro Decision Interval

## 目的

`sat_decision_interval = K` 表示 SAT selection 每 `K` 个 primitive environment step 决策一次；中间 step 继续使用 macro-start 的选星结果。

它和 `access_bw_decision_interval` 的口径一致：

- 环境仍然每个 primitive step 推进。
- reward / return 仍然按 primitive step 计算。
- actor/critic 训练时，SAT actor 只使用 macro-start row。
- `K=1` 也走同一套 macro path，应等价于逐步决策。

## 当前实现

- `SaginConfig.sat_decision_interval` 新增配置字段，默认 `1`。
- native ABI 新增 `kParamSatDecisionInterval`，Python runtime int params 同步传入。
- `sat_to_bw_live_kernel(...)` 在 continuation step 从 history 恢复 macro-start 的 `sat_action_indices` / subset action / old logprob，并用恢复后的 SAT selection 构造 BW stage。
- `StructuredNativeRolloutTrainingBatchView` 保存 `sat_decision_interval`。
- rollout view 构造时，SAT stage batch 使用 macro-start `transition_indices = base + 1`；primitive return chain 不压缩，MC target 仍从完整 primitive return 里 gather。
- `scripts/train_joint_mcgae.py` 新增 `--sat_decision_interval`，并记录 `sat_macro_duration_*` 指标。

## 已检查

- Python compile 通过。
- `K=1` smoke 能跑通，SAT/BW macro duration 均为 `1`，row count 等于 primitive rows。
- `K=5` smoke 能跑通，20 steps × 8 envs 下 SAT/BW macro row count 均为 `32`，duration 全为 `5`。

## 注意

SAT macro 不是 SMDP reward 压缩。不要把 `K` 步 reward 合成一个新 reward 再训练；当前实现只在 macro-start row 上取 primitive return/advantage。
