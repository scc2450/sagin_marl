# BW Clean 下一阶段 8 个实验矩阵（2026-04-16）

## 1. 这份文档的目的

这份文档只做一件事：

- 把当前最值得做的 **8 个 BW clean fixed-bank 实验** 记成一个清楚、可回看的实验矩阵；
- 避免后面把几件不同的问题混在一起：
  - `train_scope` 分诊；
  - `readout/loss` 几何；
  - `fresh readout` 和旧 `loc_head` 的区别；
  - grouped PCGrad 的近似版本；
  - comparative readout / encoder 是否真的有必要上。

这份文档默认讨论的是：

- `score_only_softmax` clean 路线；
- `fixed bank + fixed teacher target + frozen follow policy`；
- 重点看 **shared student 为什么连 fixed bank 都吃不稳**。

---

## 2. 当前已经确认、不能再混淆的前提

### 2.1 已知现象

- teacher target 不是纯噪声，`lookup` 几乎可以把 fixed target 拟合到位。
- shared student 连 fixed bank 上的 target 都吃不稳。
- 梯度冲突最强暴露在 `loc_head`，其次是 `user_fusion`。
- 现有结构里 user-user 交互主要靠 pooled summary，不是强 comparative set model。

### 2.2 这几件事要特别注意

1. `train_scope` 扫描不是“选最好 scope”这么简单。
   它真正回答的是：
   - 是最后读出层问题？
   - 是 upstream 表示问题？
   - 还是连当前监督接口都不顺？

2. `masked_kl` 不是“外部意见里那套几何方案”的完整替身。
   当前 `masked_kl` 只对应：
   - `KL(target || softmax(score))`
   它 **不包含**：
   - `score-space regression`
   - 也不等于“完整 readout/loss geometry 修正”。

3. `pcgrad, task_group_size=1` 不进入主实验线。
   原因很简单：
   - 近似 full per-sample PCGrad 太慢。
   - 当前主线只考虑 **可承受的 grouped PCGrad**。

4. grouped PCGrad 不能再用“随机打乱后相邻切块”的思路当默认。
   更合理的近似是：
   - 按 `target` 形状或 regime 做 **非随机分组**。

---

## 3. 统一实验设置

除非某个实验特别注明，否则尽量统一：

- 脚本主入口：
  - [scripts/diagnose_structured_bw_fixed_teacher_fit.py](/d:/研三上/毕设/sagin_marl/scripts/diagnose_structured_bw_fixed_teacher_fit.py:377)
- checkpoint：
  - [runs/structured_clean/h5_exactgate_trust_kl_lr1e3_u60/actor_best.pt](/d:/研三上/毕设/sagin_marl/runs/structured_clean/h5_exactgate_trust_kl_lr1e3_u60/actor_best.pt:1)
- config：
  - [runs/structured_clean/h5_exactgate_trust_kl_lr1e3_u60/config_source.yaml](/d:/研三上/毕设/sagin_marl/runs/structured_clean/h5_exactgate_trust_kl_lr1e3_u60/config_source.yaml:1)
- `panel_states = 32`
- `panel_seed = 35000`
- `teacher_horizon = 5`
- `offline_steps = 20`
- `eval_steps = 0 1 2 5 10 20`
- 先固定 `loc_readout = fused`

主指标：

- `history[*].current_target_gap.mean`
  - 越低越好。
- `history[*].panel_eval.current_return.mean`
  - 越高越好。
- `history[*].panel_eval.target_minus_current.mean`
  - 越大越好。
- `fixed_target_beats_current_frac`
  - 越高越好。

---

## 4. 八个实验

| 编号 | 实验名 | 改什么 | 目的 | 现成支持 | 主要看什么 |
| --- | --- | --- | --- | --- | --- |
| E1 | 旧 `loc_head` 基线 | `train_scope=loc_head`, `clean_loss=huber`, `grad_aggregation=mean` | 作为当前标准基线 | 已支持 | gap/return 基线 |
| E2 | 旧 `user_fusion` 分诊 | `train_scope=user_fusion` | 判断最终融合层单独更新是否更像主战场 | 已支持 | 相对 E1 的 fit 稳定性 |
| E3 | 旧 `all` 分诊 | `train_scope=all` | 判断放开全网后是变好还是污染更强 | 已支持 | 相对 E1/E2 的 fit 稳定性 |
| E4 | fresh `loc_head` + Huber | 冻结 backbone，重置 `loc_head`，`clean_loss=huber` | 判断旧 `loc_head` 是否已经卡死 | 需小改动 | 新头能否稳定降 gap |
| E5 | fresh `loc_head` + `score_kl` | 在 E4 基础上改成 `clean_loss=masked_kl` | 只测试“纯 KL 替代 action Huber”是否有帮助 | 需 E4 小改动 | 相对 E4 是否更稳 |
| E6 | fresh `loc_head` + `score_kl + score_reg` | 在 E5 基础上增加 centered score regression | 更贴近“readout/loss geometry 修正” | 需代码改动 | 是否明显优于 E4/E5 |
| E7 | non-random grouped PCGrad | 在当前最好 loss 上，加 grouped PCGrad，非随机分组 | 判断便宜版 conflict handling 值不值得保留 | 需代码改动 | 相对 `mean` 是否更稳 |
| E8 | frozen-backbone comparative readout | 冻结 backbone，换更强 comparative readout | 判断问题是现有读出形式弱，还是表示本身不够 | 需代码改动 | 是否优于 fresh unary `loc_head` |

---

## 5. 每个实验的具体解释

### E1. 旧 `loc_head` 基线

设置：

- `train_scope=loc_head`
- `clean_loss=huber`
- `grad_aggregation=mean`
- 旧 checkpoint 里的 `loc_head`，不重置

对应现有参考结果：

- [runs/tmp/fixed_teacher_fit_scope_loc_head.json](/d:/研三上/毕设/sagin_marl/runs/tmp/fixed_teacher_fit_scope_loc_head.json:1)

作用：

- 给后面所有实验一个共同参照物。
- 如果连这一步都不稳定复现，后面所有对比都不可信。

---

### E2. 旧 `user_fusion` 分诊

设置：

- `train_scope=user_fusion`
- 其他和 E1 相同

对应现有参考结果：

- [runs/tmp/fixed_teacher_fit_scope_user_fusion.json](/d:/研三上/毕设/sagin_marl/runs/tmp/fixed_teacher_fit_scope_user_fusion.json:1)

作用：

- 看“最终融合层单独更新”有没有比旧 `loc_head` 更稳。
- 如果 `user_fusion` 比 `loc_head` 明显更好，说明问题不只是最后一个小 MLP。

---

### E3. 旧 `all` 分诊

设置：

- `train_scope=all`
- 其他和 E1 相同

对应现有参考结果：

- [runs/tmp/fixed_teacher_fit_scope_all.json](/d:/研三上/毕设/sagin_marl/runs/tmp/fixed_teacher_fit_scope_all.json:1)

作用：

- 判断“放开全网”到底是帮忙还是加剧污染。

这一组 E1-E3 的判读规则：

- `E1` 好，`E3` 也好：
  - 主因更像读出层，upstream 不是硬瓶颈。
- `E1` 差，`E3` 好：
  - upstream 表示参与较大。
- `E1` 好，`E3` 差：
  - 一放开全网就互相污染，优化/共享冲突更强。
- **E1/E2/E3 都差**：
  - 不再继续纠结 scope。
  - 直接进入 E4-E6，优先查：
    - 旧 `loc_head` 是否已卡死；
    - 当前监督接口是否不顺。

---

### E4. fresh `loc_head` + Huber

设置：

- 冻结 backbone
- **重新初始化 `loc_head`**
- `clean_loss=huber`
- `grad_aggregation=mean`

这一步需要小改动，当前脚本还不能直接做。

作用：

- 区分：
  - 是旧 `loc_head` 已经卡死；
  - 还是 backbone 表示本身就不够。

判读：

- fresh `loc_head` 明显能 fit：
  - 说明“旧 readout 已经拧坏/卡住”是强嫌疑。
- fresh `loc_head` 也 fit 不动：
  - 不能只怪旧 `loc_head`，要继续查 loss/interface 或表示本身。

---

### E5. fresh `loc_head` + `score_kl`

设置：

- 基于 E4
- 把 clean loss 改成当前已有的 `masked_kl`

注意：

- 这一步 **只是在测**：
  - “把 action-space Huber 换成 `KL(target || softmax(score))` 有没有帮助”
- 它 **不是**“完整几何方案”的充分替身。

作用：

- 给 E6 做一个必要的中间对照。

判读：

- E5 比 E4 好：
  - 说明至少“直接在 `score` 上做分布匹配”有帮助。
- E5 不如 E4：
  - 只能说明“纯 KL 单独替换不够”；
  - **不能**直接推出“几何主因不成立”。

---

### E6. fresh `loc_head` + `score_kl + score_reg`

设置：

- 基于 E5
- 在 `KL(target || softmax(score))` 之外，再加一个 score-space regression

推荐的 target score 形式：

```text
z* = log(target + eps) - masked_mean(log(target + eps))
```

然后对 `pred_score` 和 `z*` 做：

- centered Huber，或
- centered MSE

作用：

- 这一步才更接近“外部意见里真正想测的几何修正”。

判读：

- 如果 E6 明显优于 E4/E5：
  - 很强地支持“readout/loss geometry 是主因之一”。
- 如果 E6 也不行：
  - 说明问题不能靠最后一层 loss 改法单独解决。

---

### E7. non-random grouped PCGrad

设置：

- 不做 `task_group_size=1`
- 只考虑 **可承受的 grouped PCGrad**
- 关键不是随机分组，而是 **非随机分组**

推荐分组特征：

- `valid_count`
- `target_entropy`
- `target_top1_mass`
- `l1(target, ref)` 或 `rho`

推荐做法：

1. 先按 `valid_count` 分桶
2. 桶内按 `target_entropy`
3. 再按 `target_top1_mass`
4. 每 `group_size` 个样本切一组

推荐先试：

- `group_size = 8`
- `group_size = 16`
- `group_size = 32`

作用：

- 判断“分组版 conflict handling”到底值不值得留在线里。

注意：

- 当前代码里的 grouped PCGrad 本质上是随机打乱后相邻切块，不够理想。
- full per-sample PCGrad 太慢，不进主线。

判读：

- 如果非随机 grouped PCGrad 明显优于 `mean`：
  - 说明 cheap conflict handling 仍值得保留。
- 如果帮助不明显：
  - 直接降级这条线，不继续投入。

---

### E8. frozen-backbone comparative readout

设置：

- 冻结 backbone
- 不改 encoder
- 只把 `loc_head` 换成更强的 comparative readout

优先考虑的形式：

```text
score_i = a(u_i, c) + mean_{j != i} b(u_i, u_j, c)
```

或一层轻量 user self-attention readout。

作用：

- 区分：
  - 是现有 unary `loc_head` 形式太弱；
  - 还是 backbone 表示本身已经不够。

判读：

- E8 比 E4-E6 明显更好：
  - 说明现有 readout 形式太弱，问题不一定在 backbone。
- E8 也不行：
  - 更像表示空间本身就没把 comparative 信息组织好。

---

## 6. 推荐执行顺序

最省信息浪费的顺序是：

1. `E1-E3`
2. `E4`
3. `E5`
4. `E6`
5. `E7`
6. `E8`

更具体地说：

- 如果 `E1-E3` 已经说明 scope 差异很小，立刻转 `E4-E6`
- 如果 `E4` 就明显好，先把重点放在旧 `loc_head` / readout interface
- 如果 `E6` 明显最好，优先走几何修正
- 如果 `E6` 也不够，再看 `E7`
- 如果 `E7` 还是不够，再上 `E8`

---

## 7. 当前明确不进入主线的东西

### 7.1 full per-sample PCGrad

不进主线，原因：

- 太慢；
- 对当前阶段的信息增益不成比例。

### 7.2 直接把 `masked_kl` 的好坏当成“几何主因”的总判据

不能这样做，原因：

- `masked_kl` 只是完整几何修正的一部分；
- 它没有 score-space regression；
- 它失败并不等于完整几何思路失败。

### 7.3 还没做 fresh readout probe 就直接改大结构

暂时不这么做，原因：

- 还没有把“旧 readout 卡死”和“表示本身不够”彻底分开；
- 太早上大结构，解释会变得很乱。

---

## 8. 一句话总结

这 8 个实验的核心顺序不是“多试几个 head”，而是：

**先分清 old readout 是否卡死，再分清 loss/readout 几何是否是主因，最后才决定要不要上 grouped conflict handling 或更强 comparative readout。**

