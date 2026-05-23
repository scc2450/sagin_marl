# SAT 本地 Listwise 离线验证

## 目的

验证下面这件事是否至少在一个小规模离线 setting 下成立：

- 从同一个 `sat-stage snapshot` 出发；
- 把原始 rollout 的执行动作当作 baseline；
- 只对少量局部替代动作重放到 episode 结束；
- 用候选集内的完整尾回报做本地 listwise 目标；
- 看 structured SAT actor 能不能在候选集内学到更合理的排序。

这不是在线训练结果，也不是全动作空间结论，只是一个局部候选集上的可证伪离线检查。

## 脚本

- 验证脚本： [scripts/offline_validate_sat_local_listwise.py](/d:/研三上/毕设/sagin_marl/scripts/offline_validate_sat_local_listwise.py:1)

脚本口径：

- baseline rollout 用冻结的当前 SAT actor
- `accel` 固定 `cluster_center_queue_aware`
- `bw` 固定 `queue_aware`
- 每个 context 的候选集是：
  - 当前执行动作
  - 单 UAV、单换星的局部 swap 候选
- oracle 分数用“从同一 snapshot、同一 RNG 出发，跑到各自 episode 结束”的完整尾回报
- 训练目标是候选集内的 listwise loss

## 主要实验

命令：

```powershell
.venv\Scripts\python.exe scripts\offline_validate_sat_local_listwise.py `
  --episodes 3 `
  --step-stride 80 `
  --max-contexts 6 `
  --holdout-size 2 `
  --epochs 20 `
  --eval-every 5 `
  --min-best-gap 1 `
  --out-dir runs\sat_offline_listwise_positivegap_v2_20260416
```

结果文件：

- 汇总： [summary.json](/d:/研三上/毕设/sagin_marl/runs/sat_offline_listwise_positivegap_v2_20260416/summary.json:1)
- 训练曲线： [history.json](/d:/研三上/毕设/sagin_marl/runs/sat_offline_listwise_positivegap_v2_20260416/history.json:1)
- 样本表： [entries.csv](/d:/研三上/毕设/sagin_marl/runs/sat_offline_listwise_positivegap_v2_20260416/entries.csv:1)

## 结果

这轮只保留 `best_gap > 1` 的 context，最终拿到 `3` 个正 gap 样本：

- `oracle_best_gap` 均值：`2.72e7`
- 中位数：`7.70e6`
- 最大值：`7.36e7`

离线训练前：

- holdout `top1_hit = 0.0`
- holdout `det_gap_to_best = 3.97e6`
- holdout `det_gain_vs_executed = 0.0`

离线训练后（按 holdout `listwise_loss` 选最佳 checkpoint）：

- holdout `top1_hit = 0.5`
- holdout `det_gap_to_best = 3.85e6`
- holdout `det_gain_vs_executed = 1.16e5`
- holdout `listwise_loss` 从 `2.3067` 降到 `2.2974`

训练集也能被明显推向 oracle：

- train `top1_hit` 从 `0.0` 到 `1.0`
- train `det_gap_to_best` 从 `7.36e7` 到 `0.0`

## 解读

这轮离线验证说明两件事：

1. 当前 SAT actor 的原始执行动作，在局部 swap 候选集内确实经常不是最优。
2. 用“原始 rollout 当 baseline + 局部替代分支完整尾回报 + 本地 listwise loss”，至少可以在这个小样本上把 SAT actor 往更优候选上推。

但它还不能说明：

- 这种方法在线训练一定能稳定成功；
- 小候选集外的动作也会一起变好；
- 当前样本量下的 holdout 改善已经足够稳健。

## 当前结论

最保守的结论是：

- 这条 `local listwise` 路线 **不是空的**；
- 它在离线小样本上已经表现出“能把当前动作往更优局部候选移动”的迹象；
- 但现在证据还偏小，下一步更合理的是先扩充正 gap 数据量，而不是立刻替换整个在线 SAT 训练主线。
