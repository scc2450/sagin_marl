# Agent Guide

本文件给后续在本仓库并行工作的 agent 使用。目标是让多个窗口快速对齐当前状态，避免重复评估、误用 legacy 脚本、覆盖已有运行结果，或把已经修正过的算法结论又退回旧说法。

更新时间：2026-06-04
项目：`SAGIN-MARL`
本地仓库：`/Users/erik/Documents/GitHub/sagin_marl`
长期 GPU 机器：`ssh s9`
s9 项目目录：`D:\sagin_marl`
s9 Python：`D:\anaconda3\envs\sagin-rl\python.exe`

## 0. 当前任务状态

当前 Phase 2 的核心问题已经从：

```text
MC vs bootstrap 谁天然更好？
```

转为：

```text
哪些 return target 能形成强中期策略，以及如何避免 late-stage policy degradation？
```

最重要的最新发现：

```text
bootstrap-GAE final 很弱，但 bootstrap-GAE update150 是当前 best-representative 横评第一。
```

因此不要再写：

```text
bootstrap-GAE 不可用 / 训练不出来
```

应该写：

```text
bootstrap-GAE 可以学到很强中期策略，但后期稳定性最差，final/update300 严重退化。
```

## 1. 当前代码状态

当前本地工作树有未提交源码改动：

```text
sagin_marl/rl/stage_mcgae.py
scripts/train_joint_mcgae.py
scripts/train_stage_mcgae.py
```

这些改动用于 Phase 2 return-target variants，主要包括：

```text
MC baseline
bootstrap-GAE baseline
A: MC cold-start -> bootstrap-GAE tracking
B: n-step / truncated MC target
C: MC/bootstrap-GAE mixed target
D: bootstrap-GAE main target + MC auxiliary critic loss
```

不要随手 revert 这三个文件。若需要改动，先读 diff 并说明会影响哪个方案。

私有工作指导文件：

```text
.local_guidance/phase2_hybrid_return_targets_plan_20260603.md
```

该目录被 ignore，不进入 git，但它是当前 Phase 2 的详细实验/结论文档。多个 agent 开工前建议先读它。

## 2. 当前主线入口

正式训练入口：

```text
scripts/train_joint_mcgae.py
```

正式评估入口：

```text
scripts/evaluate_structured_mixed_heads_native.py
```

当前主配置：

```text
configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
```

当前正式规模：

```text
num_envs = 64
rollout_env_steps = 250
updates = 300
K = 5
access_bw_decision_interval = 5
sat_decision_interval = 1
```

不要默认使用这些 legacy / historical 入口：

```text
scripts/train_structured.py
scripts/evaluate_structured.py
scripts/train_sat_mcgae.py
scripts/train_stage_mcgae.py
scripts/train.py
scripts/evaluate.py
```

例外：如果任务明确要求维护/兼容 legacy，才进入这些脚本。

## 3. 训练流程简述

一次 PPO update 的主流程：

```text
1. 当前 actor 与 structured env 交互，收集 rollout。
2. 构造每个 stage 的 return target。
3. critic 拟合该 target。
4. 用拟合后的 critic 计算 actor advantage。
5. 三个 actor head 分阶段更新：accel -> sat -> bw。
6. 写 metrics、phase trace、checkpoint。
```

三个 stage：

```text
accel: UAV mobility / acceleration control
sat: satellite association / routing choice
bw: access/backhaul bandwidth allocation
```

关键概念：

```text
MC return: finite-horizon non-bootstrap return
bootstrap-GAE: TD/bootstrap-based GAE return/advantage
n-step: truncated MC + bootstrap tail
mixed target: alpha * MC + (1-alpha) * bootstrap-GAE
MC auxiliary: bootstrap-GAE main loss + beta * MC critic loss
```

## 4. 当前方案和结论

| code | method | 当前判断 |
|---|---|---|
| MC | finite-horizon MC return | final/update300 最好，稳定，无明显后期崩坏。 |
| bootstrap-GAE | bootstrap GAE return | update150 当前 reward 第一，但 final/update300 严重退化。 |
| A | MC cold-start -> bootstrap-GAE tracking | update250 很强，final/update300 有退化。 |
| B | n-step / truncated MC target | final/update300 最好，稳定但不突出。 |
| C | MC/bootstrap-GAE mixed target | update250 优于 final，但 SAT overlap 过高。 |
| D | bootstrap-GAE main + MC auxiliary critic loss | update200-250 很强，final/update300 严重退化。 |

当前 best-representative 排名，协议为 `5 eval seeds x 64 episodes`：

| rank | scheme | checkpoint | reward_sum |
|---:|---|---|---:|
| 1 | bootstrap-GAE | `checkpoint_update0150.pt` | `63.6101 +/- 1.4194` |
| 2 | D mc-aux | `checkpoint_update0250.pt` | `62.0025 +/- 1.8004` |
| 3 | A MC100->bootstrap200 | `checkpoint_update0250.pt` | `61.2884 +/- 2.1333` |
| 4 | MC finite-horizon | `final/update300` | `55.8661 +/- 1.8679` |
| 5 | C mixed alpha1to03 | `checkpoint_update0250.pt` | `55.8223 +/- 1.7223` |
| 6 | B nstep64 | `final/update300` | `50.6272 +/- 1.1249` |
| 7 | native rule baseline | rule | `45.7225 +/- 0.8670` |

Interpretation：

```text
1. bootstrap@150 当前 reward 第一，但后期崩坏最严重。
2. D@250 和 A@250 是强候选，A@250 的 processed/drop/backlog/D_sys/sat_overlap 更干净。
3. MC final 是稳定强基线，不是最高 reward，但最稳。
4. C@250 reward 接近 MC final，但 sat_overlap 接近 1，行为解释风险高。
5. B 稳定但弱，适合作为 n-step ablation。
```

## 5. 重要结果目录

s9 归档位置：

```text
D:\sagin_marl\runs\phase2\mc_formal_3uav20gu_t250_k5_300u_seed45210_20260601_112543
D:\sagin_marl\runs\phase2\bootstrap_formal_3uav20gu_t250_k5_300u_micro250_20260602_030753\bootstrap_gae
D:\sagin_marl\runs\phase2\hybrid_mc100_bootstrap200_3uav20gu_t250_k5_300u_seed45210_autodl_20260603_145523
D:\sagin_marl\runs\phase2\hybrid_mix_alpha1to03_3uav20gu_t250_k5_300u_seed45210_autodl_20260603_175023
D:\sagin_marl\runs\phase2\hybrid_bootstrap_mcaux_beta0p3_3uav20gu_t250_k5_300u_seed45210_autodl_20260603_195356
D:\sagin_marl\runs\phase2\hybrid_nstep64_3uav20gu_t250_k5_300u_seed45210_autodl_20260603_214045
```

s9 sweep / eval 结果：

```text
D:\sagin_marl\runs\phase2\s9_full_cross_eval_20260604
D:\sagin_marl\runs\phase2\s9_abc_checkpoint_sweep_20260604
D:\sagin_marl\runs\phase2\s9_mc_bootstrap_checkpoint_sweep_20260604
D:\sagin_marl\runs\phase2\s9_best_representative_cross_eval_20260604
```

关键 CSV / MD：

```text
D:\sagin_marl\runs\phase2\s9_best_representative_cross_eval_20260604\phase2_best_representative_cross_eval.csv
D:\sagin_marl\runs\phase2\s9_best_representative_cross_eval_20260604\phase2_best_representative_cross_eval.md
D:\sagin_marl\runs\phase2\s9_mc_bootstrap_checkpoint_sweep_20260604\best_by_scheme.csv
D:\sagin_marl\runs\phase2\s9_abc_checkpoint_sweep_20260604\best_by_scheme.csv
```

AutoDL 结果已转存到 s9：

```text
D:\sagin_marl\runs\autodl_archive\phase2_A_C_autodl_20260603.tgz
D:\sagin_marl\runs\autodl_archive\phase2_D_bootstrap_mcaux_20260603.tgz
D:\sagin_marl\runs\autodl_archive\phase2_B_nstep64_autodl_20260604.tgz
```

## 6. Bootstrap 正式训练时间线

目录：

```text
D:\sagin_marl\runs\phase2\bootstrap_formal_3uav20gu_t250_k5_300u_micro250_20260602_030753\bootstrap_gae
```

时间线：

```text
2026-06-02 03:08:44  启动 bootstrap-GAE 正式训练
2026-06-02 13:53     checkpoint_update0050.pt
2026-06-02 22:58     checkpoint_update0100.pt
2026-06-02 23:36:14  原进程在 update102 附近退出
2026-06-03 00:00     从 checkpoint_update0100.pt 续跑，microbatch 250 -> 500
2026-06-03 02:12     checkpoint_update0150.pt，当前全方案 best representative
2026-06-03 04:28     checkpoint_update0200.pt
2026-06-03 06:34     checkpoint_update0250.pt，已明显退化
2026-06-03 08:36:44  final/update300 完成，严重退化
```

续跑只改变 `critic_update_microbatch_size`，不改变 target、rollout size、优化目标。结果不是 bit-exact，但算法口径仍是 bootstrap-GAE。

## 7. s9 常用命令

查看 GPU：

```powershell
nvidia-smi
```

正式 eval 示例：

```powershell
D:\anaconda3\envs\sagin-rl\python.exe scripts\evaluate_structured_mixed_heads_native.py `
  --config configs\current\structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml `
  --base_checkpoint <checkpoint.pt> `
  --episodes 64 `
  --num_envs 64 `
  --episode_seed_base 900000 `
  --device cuda `
  --policy_mode deterministic `
  --exec_accel_source policy `
  --exec_sat_source policy `
  --exec_bw_source policy `
  --access_bw_decision_interval 5 `
  --sat_decision_interval 1 `
  --out_dir <eval_out_dir> `
  --label <label>
```

规则 baseline：

```powershell
--baseline_policy cluster_center_queue_aware
```

必须显式保留：

```text
--access_bw_decision_interval 5
--sat_decision_interval 1
```

## 8. 协作分工建议

多个 agent 窗口建议分工，避免互相踩：

| agent | 任务 | 主要文件/目录 | 注意事项 |
|---|---|---|---|
| Agent A | 训练稳定化实现 | `scripts/train_joint_mcgae.py`, `sagin_marl/rl/stage_mcgae.py` | 做 best checkpoint / early stopping / resume fine-tune，别改评估口径。 |
| Agent B | 实验调度与 s9 运行 | `runs/phase2/*`, s9 scripts | 新实验用新目录，不覆盖旧 run。 |
| Agent C | 结果聚合与表格 | sweep CSV/MD, `.local_guidance/*` | 保持 eval seed 协议一致，区分 validation/test。 |
| Agent D | 论文叙事与文档 | `docs/`, `.local_guidance/`, thesis notes | 不要把结果夸成算法万能，重点写训练稳定性和通信调度场景。 |
| Agent E | 代码结构/可维护性 | `scripts/`, `configs/`, package modules | 不要在实验高峰期大搬目录，先保证入口可运行。 |

多 agent 同时工作规则：

```text
1. 每个 agent 先读 agent.md 与 .local_guidance/phase2_hybrid_return_targets_plan_20260603.md。
2. 修改源码前先看 git status 和 git diff。
3. 不要 revert 别人的改动。
4. 不要覆盖 runs/phase2 既有目录。
5. 新实验目录必须带方案、seed、日期和关键超参。
6. 任何结论必须写清 eval protocol：episodes、num_envs、seed list、checkpoint。
7. 不要只报 final.pt；Phase 2 必须看 checkpoint sweep。
```

## 9. 下一步实验路线

最重要的是验证“强中期 checkpoint 能不能继续进化而不退化”。

优先级：

```text
1. bootstrap@150 conservative continuation
2. A@250 conservative continuation
3. D@250 conservative continuation
4. bootstrap@150 + KL anchor to best policy
5. bootstrap@150 -> MC auxiliary / n-step / mixed target stabilization
```

低风险 continuation 方案：

```text
resume from best checkpoint
continue 30-80 updates
actor_lr = 0.1x or 0.2x
critic_lr = 0.3x or 0.5x
save_every = 10
eval every 10 updates
```

PPO trust-region 稳定化：

```text
lower clip_range, e.g. 0.2 -> 0.1
stricter target_kl
reduce actor epochs
reduce entropy bonus if policy keeps drifting
skip/stop stage update if KL exceeds threshold
```

Policy anchor 方案：

```text
L = PPO loss + beta_kl * KL(pi_current || pi_best)
```

适合 bootstrap@150，因为它很强但后期崩坏最严重。

评估种子必须分层：

```text
training rollout seeds: 正常训练使用
validation eval seeds: 用来选 best checkpoint / early stopping
test eval seeds: 最后只评一次，用于论文报告
```

## 10. 当前不要做的事

```text
不要只用 final.pt 比较所有方案。
不要再说 bootstrap-GAE 不可用。
不要把 bootstrap@150 的强结果解释成 final 也强。
不要原样从 best checkpoint 继续跑到 300 期待自然变好。
不要把 C@250 当主线而忽略 sat_overlap ~= 1 的行为风险。
不要在没有不同训练 seed 复现前把任何方案写成最终胜利。
不要把 smoke test 或单 seed 结果写成论文结论。
```

## 11. 代码联动规则

改 return target / critic 训练：

```text
sagin_marl/rl/stage_mcgae.py
scripts/train_joint_mcgae.py
scripts/train_stage_mcgae.py
```

改 actor/distribution/log-prob：

```text
sagin_marl/rl/distributions.py
sagin_marl/rl/structured_actor.py
sagin_marl/rl/native_actor_cuda.py
sagin_marl/env/native_cuda/
```

改 structured env / native execution：

```text
sagin_marl/env/structured_batch_env_core.py
sagin_marl/env/native_cuda/
```

改 config：

```text
configs/current/structured_joint_mcgae_3uav_20gu_t250_positive_relcritic.yaml
sagin_marl/env/config.py
```

改文档：

```text
docs/README.md
.local_guidance/phase2_hybrid_return_targets_plan_20260603.md
agent.md
```

## 12. 论文口径

投稿定位更接近通信网络 / SAGIN resource scheduling，而不是纯 RL 算法论文。

推荐叙事：

```text
1. SAGIN 分阶段资源调度具有长时域、多动作头耦合、credit assignment 难点。
2. 多种 return target 都可能形成强中期策略。
3. 关键挑战是训练后期 policy degradation，而不是某个 target 天然失败。
4. MC final 稳定，bootstrap@150/D@250/A@250 强但需要 checkpoint selection。
5. 论文贡献应围绕训练稳定性、分阶段调度建模、可复现实验协议和通信指标改善。
```

不要写成：

```text
我们提出了一个复杂 RL 算法并全面优于所有方法。
```

更稳的写法：

```text
We identify and evaluate return-target choices for staged SAGIN resource scheduling, showing that strong policies may emerge before final convergence and that checkpoint selection / anti-drift stabilization is essential for reliable training.
```

## 13. 常用检查

语法检查：

```powershell
D:\anaconda3\envs\sagin-rl\python.exe -m py_compile scripts\train_joint_mcgae.py scripts\evaluate_structured_mixed_heads_native.py
```

本地也可用：

```bash
python3 -m py_compile scripts/train_joint_mcgae.py scripts/evaluate_structured_mixed_heads_native.py
```

查看 git 状态：

```bash
git status --short
git diff --stat
```

如果训练卡住：

```text
1. 看 phase_trace.jsonl 定位 collect / critic / actor 阶段。
2. 看 train.log / resume stdout/stderr。
3. 看 nvidia-smi 是否 GPU 还在跑。
4. 不要直接 kill，先确认 checkpoint 是否已写。
```
