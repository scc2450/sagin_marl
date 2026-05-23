# Context Handoff 2026-04-01

这个文件用于在新对话里无缝继续当前工作。

## 用户约束

- 不要并行读取文件或并行做代码搜索。
- 执行任何 python 命令前先激活虚拟环境：
  - `& .\.venv\Scripts\Activate.ps1; ...`
- Windows 有时会拒绝一次操作；先原样重试一遍。
- 如果重试后仍失败，而且任务需要该操作，直接申请权限，不要先来回问。

## 当前代码改动

### 1. `cluster_center_queue_aware` 执行源已接通

已改文件：

- [config.py](d:/研三上/毕设/sagin_marl/sagin_marl/env/config.py)
- [mappo.py](d:/研三上/毕设/sagin_marl/sagin_marl/rl/mappo.py)
- [evaluate.py](d:/研三上/毕设/sagin_marl/scripts/evaluate.py)

现在 `exec_accel_source / exec_bw_source / exec_sat_source` 都支持：

- `policy`
- `teacher`
- `heuristic`
- `queue_aware`
- `cluster_center_queue_aware`

`mappo.py` 和 `evaluate.py` 里都加了 `_resolve_exec_heuristic_triplet(...)` 一类逻辑，用来按 head 选 `queue_aware` 或 `cluster_center_queue_aware`。

### 2. `bw-only + fixed accel/sat execution source` 实验配置已加

配置文件：

- [bwonly_clustercenterexec.yaml](d:/研三上/毕设/sagin_marl/configs/phase1_actions_curriculum_joint_3heads_fading_interference_vsat_precomp_joint_puremappo_criticdecoupled_bwonly_clustercenterexec.yaml)

关键设置：

- 环境不变
- reward 不变
- critic 不变
- `danger_imitation_enabled: false`
- `train_accel: false`
- `train_bw: true`
- `train_sat: false`
- `train_shared_backbone: true`
- `exec_accel_source: cluster_center_queue_aware`
- `exec_bw_source: policy`
- `exec_sat_source: cluster_center_queue_aware`

用户明确要求：

- 不要冻住 shared backbone
- stage2 的旧 bw heuristic 太老，不用
- accel/sat 固定执行源用 `cluster_center_queue_aware`

### 3. `bw private trunk` 实验已实现

已改文件：

- [config.py](d:/研三上/毕设/sagin_marl/sagin_marl/env/config.py)
- [policy.py](d:/研三上/毕设/sagin_marl/sagin_marl/rl/policy.py)
- [mappo.py](d:/研三上/毕设/sagin_marl/sagin_marl/rl/mappo.py)

新增 config 字段：

- `bw_private_trunk_enabled: bool = False`
- `bw_private_trunk_init_from_shared: bool = True`

实现内容：

- `ActorNet` 可选复制一套 `bw` 专用 trunk
- `bw` 可使用独立 `bw_ctx`
- 初始可从 shared trunk `load_state_dict` 复制
- 从旧 checkpoint 加载时，如果旧权重缺 private trunk，会自动从 shared trunk 补初始化
- `bw_head_grad_scale` / head freeze / head copy 的逻辑都已覆盖 private trunk

配置文件：

- [joint_privatebwtrunk_from_sharedinit.yaml](d:/研三上/毕设/sagin_marl/configs/phase1_actions_curriculum_joint_3heads_fading_interference_vsat_precomp_joint_puremappo_criticdecoupled_joint_privatebwtrunk_from_sharedinit.yaml)

## 已完成的重要实验与结论

### A. `bw-only cluster_center exec` 训练

run：

- [bwonly_clustercenterexec_u600_env12_subproc_20260401](d:/研三上/毕设/sagin_marl/runs/phase1_actions/bwonly_clustercenterexec_u600_env12_subproc_20260401)

说明：

- 实际没有机械地跑满 600 updates
- checkpoint eval 在 `u0500` 左右 plateau/早停
- 用户明确说“看这个早停结果就行”

最佳 checkpoint：

- `u0350`
- 文件见 [checkpoint_eval.csv](d:/研三上/毕设/sagin_marl/runs/phase1_actions/bwonly_clustercenterexec_u600_env12_subproc_20260401/checkpoint_eval.csv)

最佳结果：

- `reward_sum = 226.0517`
- `processed_ratio_eval = 0.9113`
- `drop_ratio_eval = 0.1258`
- `pre_backlog_steps_eval = 10.8761`

对应固定 heuristic 参考：

- `reward_sum = 248.4329`
- `processed_ratio_eval = 0.9352`
- `drop_ratio_eval = 0.1042`
- `pre_backlog_steps_eval = 10.0308`

结论：

- 固定 accel+sat，只学 bw，这条线能学到东西
- 但目前还没超过全 heuristic 参考
- 训练尾部 `clip_frac_bw` 仍很高，几何压力还在

### B. `joint + private bw trunk` 50-update 实验

smoke run：

- [joint_privatebwtrunk_from_sharedinit_smoke_u1](d:/研三上/毕设/sagin_marl/runs/phase1_actions/joint_privatebwtrunk_from_sharedinit_smoke_u1)

正式 run：

- [joint_privatebwtrunk_from_sharedinit](d:/研三上/毕设/sagin_marl/runs/privatebwtrunk_50u/joint_privatebwtrunk_from_sharedinit)

评估：

- [eval_trained_n20_seed42000.csv](d:/研三上/毕设/sagin_marl/runs/privatebwtrunk_50u/joint_privatebwtrunk_from_sharedinit/eval_trained_n20_seed42000.csv)

对照：

- [a0_joint](d:/研三上/毕设/sagin_marl/runs/bw_geom_credit_50u/a0_joint)

结果对照：

- control `reward_sum = 185.5533`
- private trunk `reward_sum = 129.0924`

- control `processed_ratio_eval = 0.8760`
- private trunk `processed_ratio_eval = 0.7983`

- control `drop_ratio_eval = 0.1422`
- private trunk `drop_ratio_eval = 0.1860`

- control `pre_backlog_steps_eval = 19.0018`
- private trunk `pre_backlog_steps_eval = 22.7091`

训练几何最后 10 updates 均值：

- control `actor_minibatches_executed = 7.8`
- private trunk `actor_minibatches_executed = 18.3`

- control `clip_frac_bw = 0.9425`
- private trunk `clip_frac_bw = 0.9271`

- control `approx_kl_bw = 0.0110`
- private trunk `approx_kl_bw = 0.0107`

- control `log_ratio_var_bw = 36.42`
- private trunk `log_ratio_var_bw = 37.85`

结构诊断输出：

- [summary.json](d:/研三上/毕设/sagin_marl/runs/privatebwtrunk_50u/bw_diag_compare_20260401_privatetrunk/summary.json)

结论：

- `bw` 从 shared ctx 解耦后，局部结构变化方向更像 queue-aware
- 但 50 updates 下没有转成更好的 joint 性能
- 说明 shared context 干扰可能是问题的一部分，但不是单独拆 trunk 就能直接救活

### C. `geometry vs credit` 判别实验

四条 run：

- [a0_joint](d:/研三上/毕设/sagin_marl/runs/bw_geom_credit_50u/a0_joint)
- [a1_bwgrad025](d:/研三上/毕设/sagin_marl/runs/bw_geom_credit_50u/a1_bwgrad025)
- [b0_headwisesharedadv](d:/研三上/毕设/sagin_marl/runs/bw_geom_credit_50u/b0_headwisesharedadv)
- [b1_headwisebwcredit](d:/研三上/毕设/sagin_marl/runs/bw_geom_credit_50u/b1_headwisebwcredit)

统一评估结果（20 seeds）：

- A0: `reward_sum=103.4709`, `processed=0.7699`, `drop=0.2296`, `pre_backlog=22.2594`
- A1: `reward_sum=106.5802`, `processed=0.7745`, `drop=0.2170`, `pre_backlog=25.2593`
- B0: `reward_sum=-296.8718`, `processed=0.1728`, `drop=0.6553`, `pre_backlog=69.9531`
- B1: `reward_sum=-273.7731`, `processed=0.1953`, `drop=0.6248`, `pre_backlog=71.6652`

训练动态结论：

- A1 相比 A0 有正向信号
- B1 只是在坏载体 B0 上轻微回升
- 当前证据更偏向：`bw` 更新几何比单独 credit 更值得优先抓

## `bw` 动作路径 / PPO 审计

汇总文件：

- [bw_path_ppo_audit_20260401.md](d:/研三上/毕设/sagin_marl/runs/privatebwtrunk_50u/bw_path_ppo_audit_20260401.md)

这个文件已经汇总：

- `policy.py` 里 `bw` 的 `forward / act / evaluate_actions_parts`
- `distributions.py` 里 `MaskedDirichlet.sample / log_prob`
- [action_assembler.py](d:/研三上/毕设/sagin_marl/sagin_marl/rl/action_assembler.py) 打包方式
- [sagin_env.py](d:/研三上/毕设/sagin_marl/sagin_marl/env/sagin_env.py) 中 `bw_alloc -> 实际分配` 核心路径
- PPO actor loss 里 `ratio / clip / clip_frac_bw / log_ratio_var_bw`
- joint vs bw-only 前 50 updates 的同口径统计
- 若干真实 step 的 `bw_score_raw / bw_alpha / valid_mask / sampled_bw_action / executed_bw_alloc`

关键结论：

- 当前 `bw` 不是 softmax logits，而是 `bw_score -> softplus + alpha_floor -> Dirichlet(alpha)`
- env 侧对 `bw` 没有 top-k / hard selection
- env 侧本质是 mask + renorm
- `clip_frac_bw` 和 `log_ratio_var_bw` 都基于一个“每样本一个标量 joint bw logprob”
- 抓到的真实 step 里，`executed_bw_alloc == sampled_bw_action`
- 所以如果有几何问题，更像是 Dirichlet / joint logprob 的尺度问题，不是 env 又离散化了一次

## 最新验证：`|part_log_ratio_bw|` 是否随 `valid_count` 上升，而执行分配变化没有同比例增大

汇总文件：

- [report.md](d:/研三上/毕设/sagin_marl/runs/privatebwtrunk_50u/validcount_logratio_execcheck_20260401/report.md)
- [report.json](d:/研三上/毕设/sagin_marl/runs/privatebwtrunk_50u/validcount_logratio_execcheck_20260401/report.json)

定义：

- `abs_log_ratio_bw = |logprob_new_bw - logprob_old_bw|`
- `delta_exec_l1 = old/new bw_alpha` 在同一个 valid mask 下归一化后 share 的 `L1` 变化
- 由于 env 对 `bw` 没有 top-k/hard selection，这个 `delta_exec_l1` 就是执行 share 变化的直接代理

对两个 run 做了 replay 检查：

- `joint_a0_u50`
- `bwonly_clustercenter_u50`

### `joint_a0_u50`

- `corr(valid_count, |log_ratio_bw|)`:
  - Pearson `0.1937`
  - Spearman `0.2616`
- `corr(valid_count, delta_exec_l1)`:
  - Pearson `0.1336`
  - Spearman `0.2392`
- `1-4 -> 13+`：
  - `|log_ratio_bw|` 均值放大 `2.08x`
  - `delta_exec_l1` 只放大 `1.44x`
  - `delta_exec_top1` 反而变成 `0.66x`
- `delta_exec_l1 / |log_ratio_bw|`：
  - `1-4` bin: `0.0313`
  - `13+` bin: `0.0216`

### `bwonly_clustercenter_u50`

- `corr(valid_count, |log_ratio_bw|)`:
  - Pearson `0.3546`
  - Spearman `0.4504`
- `corr(valid_count, delta_exec_l1)`:
  - Pearson `0.0556`
  - Spearman `0.2683`
- `1-4 -> 13+`：
  - `|log_ratio_bw|` 均值放大 `4.47x`
  - `delta_exec_l1` 只放大 `1.10x`
  - `delta_exec_top1` 只剩 `0.34x`
- `delta_exec_l1 / |log_ratio_bw|`：
  - `1-4` bin: `0.0227`
  - `13+` bin: `0.0056`

当前判断：

- 这个假设基本成立
- `|part_log_ratio_bw|` 确实带有明显的 `valid_count` 尺度效应
- 执行分配变化没有按同样比例同步放大
- 所以当前的 `clip_frac_bw` / `|log_ratio_bw|` 很可能混入了 action dimension / valid dimension 的统计尺度问题
- 尤其是在 `bw-only` 那条线，这个现象更强

## 新对话里建议的起手点

如果要继续做研究，优先级建议：

1. 继续围绕 `bw` joint logprob 的尺度问题做验证
2. 重点检查“按 valid_count 归一化的 bw logprob / ratio 统计”是否能更真实反映更新几何
3. 如果要做新实验，先做“统计口径修正/额外日志”，再考虑大改 reward 或 critic

比较自然的下一步有两条：

- 路线 A：加一个只用于日志的 `bw_logprob_per_valid_dim` / `bw_log_ratio_per_valid_dim`
- 路线 B：做一个实验版 surrogate，对 `bw` 的联合 logprob 按 `valid_count` 或 `sqrt(valid_count)` 归一化，看 `clip_frac_bw` 与真实执行变化是否更一致

## 关键文件索引

- [policy.py](d:/研三上/毕设/sagin_marl/sagin_marl/rl/policy.py)
- [mappo.py](d:/研三上/毕设/sagin_marl/sagin_marl/rl/mappo.py)
- [distributions.py](d:/研三上/毕设/sagin_marl/sagin_marl/rl/distributions.py)
- [action_assembler.py](d:/研三上/毕设/sagin_marl/sagin_marl/rl/action_assembler.py)
- [sagin_env.py](d:/研三上/毕设/sagin_marl/sagin_marl/env/sagin_env.py)
- [evaluate.py](d:/研三上/毕设/sagin_marl/scripts/evaluate.py)
- [diagnose_bw_perhead_vs_oldjoint.py](d:/研三上/毕设/sagin_marl/scripts/diagnose_bw_perhead_vs_oldjoint.py)

## 本文件用途

把这个文件路径直接贴到新对话里即可：

- [context_handoff_20260401.md](d:/研三上/毕设/sagin_marl/context_handoff_20260401.md)
