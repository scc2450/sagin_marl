# BW Clean 训练平台期诊断记录（2026-04-16）

## 1. 诊断目标

本文整理当前 `BW clean per-user` 方案在正式训练后期出现平台期时的诊断结论，重点回答下面几个问题：

1. `target` 本身是不是好信号。
2. 为什么很多 state 都有更好的 `target`，但训练后期评估结果还是平台。
3. 问题更像出在：
   - teacher 信号本身；
   - 输入状态不完整；
   - actor 网络结构/规模；
   - 还是把很多 `per-state target` 合成为一次共享参数更新的这一步。

本文只整理**当前已经做过的检查**和**当前最稳妥的结论**，不把还没证实的猜测写成结论。

---

## 2. 当前最稳妥的结论

### 2.1 已经基本可以成立的结论

1. **`target` 不是纯噪声，也不是后期整体失效。**
   - 在固定 panel 上，后期 checkpoint 附近，`target` 相对 `ref` 仍然大多数情况下更好。
   - 也就是说，teacher 仍然能在很多单个 snapshot 上给出有价值的 per-state 改进建议。

2. **问题不主要像是“明显缺输入，导致相同 LocalBwState 对应完全不同 target”。**
   - 已经做过一次直接检查：没有发现大量“`LocalBwState` 很近，但 teacher target 差很多”的样本对。
   - 所以当前证据**不支持**“actor 因为明显缺输入，所以根本没法区分这些 target”这个解释。

3. **问题最像出在 student 侧：很多 per-state 的好 target，一起变成一次共享参数更新时，会互相拉扯。**
   - 也就是：单个 state 上 target 往往是好的，但很多 state 放在一起学时，梯度会冲突。
   - 冲突平均后的更新方向，在训练后期并没有稳定对应到更好的整体评估结果。

4. **固定一批样本反复训练时，也不能稳定把 actor 拉近这批 fixed target。**
   - 这说明当前训练并不是在动作空间里对 target 做稳定投影。
   - 这和“共享参数下很多 state 的 target 互相拉扯”是对得上的。

### 2.2 当前不能直接下结论的地方

1. **不能直接下结论说“就是网络容量不够”。**
   - 目前更强的证据是“梯度兼容性/更新聚合有问题”，不是“容量一定不够”。

2. **不能直接下结论说“就是输入缺失”。**
   - 当前的 `LocalBwState` 歧义检查并没有支持这条解释。

3. **不能直接下结论说“teacher 后期更接近全局最优”。**
   - 当前能说的只是：在固定 panel 上，后期 target 没有变差，仍然常常比 ref 好。

---

## 3. 目前的证据链

### 3.1 `target` 本身大多是好的

训练里已经加了 exact gate：

- 只有 `target_return > ref_return` 的样本才参与 clean loss。
- 所以真正进入训练的样本，对应的 target 在该 snapshot 上是比当前动作更好的。

另外，在固定 panel 的离线诊断里，后期 checkpoint 上仍能看到：

- `full_beats_ref_frac` 很高；
- `target_minus_current` 仍为正；
- `target` 对 full-episode return 仍然有正向改进。

所以当前更像是：

- **teacher 仍然能提出很多“单样本上更好”的动作建议；**
- **问题出在 actor 怎么把这些建议同时学进去。**

### 3.2 固定 bank 反复训练时，不能稳定拉近 fixed target

对 `u60 actor_best` 做过固定 bank 诊断：

- 冻结 32 个 snapshot；
- 冻结这批 snapshot 上的 exact teacher target；
- 然后 student 在这同一批样本上重复训练。

关键结果：

- `current_target_gap.mean`
  - step 0: `0.119`
  - step 1: `0.169`
  - step 5: `0.133`
  - step 10: `0.141`
  - step 20: `0.103`

这说明：

- 它**不是**稳定地把 actor 拉近 fixed target；
- 中间很多步其实是在变远；
- 到 step 20 只是比 step 0 略近一点，不是“已经基本学完 target”。

这件事很关键，因为它说明：

- 当前更新并不是“朝 target 的稳定投影”；
- 即使固定同一批 target，训练过程也会来回摆。

### 3.3 一次真实 online update 后，固定 panel 4/4 次变差

对平台附近的 `u60 actor_best` 做过真实 online update 方向检查：

- 先用当前 actor 采一批 fresh rollout；
- 做一次真实 clean update；
- 然后回到同一批固定 panel 上看 pre/post 变化。

结果：

- 4/4 次 `delta_return_mean` 都是负的：
  - `-111.5`
  - `-106.2`
  - `-75.3`
  - `-105.0`

这说明：

- 平台不是因为“没有更好的 target 了”；
- 而是因为：**当前这一步 update，把 batch 上的 per-state target 合成成一次共享参数更新之后，这步更新本身对固定评估是负的。**

### 3.4 梯度冲突是当前最强的直接证据

已经做过 clean 梯度冲突诊断。

最关键的结果是 `loc_head`：

- `batch_vs_panel_cos ≈ -0.9977`
- 换一批 seed 后仍然 `≈ -0.9928`

同时，batch 内部也有明显冲突：

- `neg_cos_frac ≈ 0.50`

这说明：

1. batch 内部不同样本的监督梯度确实会互相打架；
2. 更关键的是，这些冲突样本平均之后形成的更新方向，在 `loc_head` 上和另一批 held-out states 想要的平均方向几乎相反。

这比“teacher 可能坏了”或“输入可能缺了”更像当前主问题。

### 3.5 暂时没有看到强的 “LocalBwState 很像，但 target 差很多” 证据

已经做过一次直接检查：

- 收一批 improving samples；
- 把 `LocalBwState` 全字段展平；
- 比较 state 距离和 target 距离的关系。

结果：

- `state_target_corr = 0.533`
- `near_state_far_target_frac = 0.0`

这说明：

- 当前没有发现大量“同样的 LocalBwState，却要求完全不同 target”的样本对；
- 所以现在并不支持“输入状态明显不完整，所以 actor 根本没法学”的强结论。

注意：

- 这里只是否定了“明显的原始输入歧义”；
- 还没有直接否定“内部表示空间仍然不够好”。

### 3.6 按模块看，问题最强地落在 `loc_head`，其次是 `user_fusion`

又做了一轮按模块的梯度冲突定位，使用的是同一个 `u60 actor_best`，但为了不和其他训练抢资源，这轮采用了较小样本：

- `batch_states = 8`
- `panel_states = 8`
- `rollout_env_steps = 40`
- `num_envs = 4`

检查的模块包括：

- `user_encoder`
- `user_refine`
- `user_fusion`
- `loc_head`
- `bw_policy` 整体

结果见：

- `runs/tmp/grad_scope_user_encoder.json`
- `runs/tmp/grad_scope_user_refine.json`
- `runs/tmp/grad_scope_user_fusion.json`
- `runs/tmp/grad_scope_loc_head.json`
- `runs/tmp/grad_scope_bw_policy.json`

关键结论：

- `loc_head`
  - `batch_vs_panel_cos = -0.9816`
  - 这和之前较大样本下 `≈ -0.99` 的结果一致，说明最强的不对齐就在最后读出层。

- `user_fusion`
  - `batch_vs_panel_cos = -0.4266`
  - 说明最终融合层也存在明显不对齐，但没有 `loc_head` 那么极端。

- `user_encoder`
  - `batch_vs_panel_cos = -0.0311`

- `user_refine`
  - `batch_vs_panel_cos = -0.0807`

- `bw_policy` 整体
  - `batch_vs_panel_cos = -0.0792`

这组结果最自然的解释是：

- **最明显的问题不在前面编码层，而在最后读出层 `loc_head`；**
- **`user_fusion` 可能也有一定贡献；**
- **更前面的表示层没有显示出同等强度的系统性反向。**

这并不等于“前面层完全没问题”，但当前最强的定位点已经比较清楚了。

### 3.7 在 actor 内部表征空间里，也没看到强的 target 歧义

为了进一步区分“前面表征有问题”还是“最后读出有问题”，又做了一次**内部表征 vs target** 的歧义检查。

这次不再只比较原始 `LocalBwState`，而是直接比较 actor 前向中的几层表征：

- `user_0`
- `user_1`
- `fused`
- `loc_src`

脚本：

- `scripts/diagnostics/diagnose/diagnose_structured_bw_local_state_target_ambiguity.py`

新结果：

- `runs/tmp/local_rep_target_ambiguity_h5_u60_64.json`

关键结果：

- `raw_local_state`
  - `state_target_corr = 0.533`
  - `near_state_far_target_frac = 0.0`

- `user_0`
  - `state_target_corr = 0.815`
  - `near_state_far_target_frac = 0.0`

- `user_1`
  - `state_target_corr = 0.810`
  - `near_state_far_target_frac = 0.0`

- `fused`
  - `state_target_corr = 0.726`
  - `near_state_far_target_frac = 0.0`

- `loc_src`
  - `state_target_corr = 0.726`
  - `near_state_far_target_frac = 0.0`

这说明：

- 前面表征层并没有把“相似 state -> 不同 target”这件事明显放大；
- `user_0 / user_1 / fused / loc_src` 这些内部表征里，都没有出现大量“很近但 target 差很大”的样本对；
- 因此当前仍然**不支持**“前面表示空间已经把这些 target 混坏了”这个解释。

把 3.6 和 3.7 放在一起看，当前更像：

- **前面表征仍然保留了相当多和 target 对齐的结构；**
- **问题更强地集中在最后读出层 `loc_head`，以及把很多 per-state 目标折成一次共享更新的这一步。**

---

## 4. 当前最像的问题机制

当前最像的机制是下面这条：

1. 每个训练样本都有自己的 `(state_i, target_i)`；
2. 对很多样本来说，`target_i` 相对当前动作是更好的；
3. 但 actor 不是给每个样本单独一套参数，而是用一套共享参数同时学习很多 `(state_i, target_i)`；
4. 不同样本对同一组参数的更新要求会冲突；
5. 冲突平均后得到的一次参数更新，并不能让所有 state 都更接近各自 target；
6. 到训练后期，这种冲突平均后的更新，已经不足以继续稳定提升整体评估结果。

换句话说：

- **问题不是 teacher 没有给出 per-state 的好建议；**
- **问题是很多 per-state 的好建议在共享参数空间里不够兼容。**

---

## 5. 对“输入不完整”和“网络有问题”的当前判断

### 5.1 输入状态不完整：当前是“怀疑过，但证据不强”

之所以怀疑过，是因为：

- teacher 生成 target 时依赖的是完整 replay 语义；
- actor 真正看到的是压缩后的 `LocalBwState`；
- 理论上存在“teacher 用到了 actor 看不到的信息”的可能性。

但到目前为止：

- 没找到很多“`LocalBwState` 很像，但 target 差很多”的样本；
- 所以当前不支持把这个解释当主结论。

更准确的说法是：

- **不能完全排除输入表达不够；**
- **但它目前不是证据最强的解释。**

### 5.2 网络结构/规模有问题：当前是“不能排除，但也不能直接定性”

如果只是看当前现象，确实会让人怀疑：

- 网络太小；
- 表征能力不够；
- 共享结构不适合这个学习目标。

但当前还不能直接说：

- “就是网络容量不够”；
- 或者“换大网络就一定好”。

因为目前更强的证据是：

- target 大多是好的；
- 输入歧义没有被强力支持；
- actor 内部表征空间也没有显示出强的 target 歧义；
- 真实问题更像是**更新聚合后的梯度兼容性差**，并且最强地落在 `loc_head`。

所以当前更准确的表达应该是：

- **网络结构/规模可能是放大问题的因素；**
- **但目前最强的直接问题仍然是 shared-parameter update 的梯度冲突。**

---

## 6. 涉及到的主要代码文件

下面按角色列出相关代码文件。

### 6.1 完整状态、snapshot、replay 相关

- `sagin_marl/env/structured_driver.py`
  - `export_bw_stage_state()`
  - `load_bw_stage_state()`
  - `build_bw_stage_snapshot()`
  - `_export_stage_world_cache()`
  - 作用：
    - 导出/恢复 `bw_stage_state`
    - 构造 BW 阶段 snapshot
    - 支撑 exact replay

- `sagin_marl/env/sagin_env.py`
  - `export_runtime_state()`
  - 作用：
    - 导出环境 runtime state
    - `bw_stage_state["env_state"]` 的主要来源

- `sagin_marl/rl/structured_types.py`
  - `LocalBwState`
  - `BwStageSnapshot`
  - 作用：
    - 定义 actor 使用的局部状态结构
    - 定义 snapshot 的结构化表示

- `sagin_marl/rl/structured_stage_builders.py`
  - `build_local_bw_states_from_snapshot()`
  - 作用：
    - 把 snapshot 转成 actor 真正吃的 `LocalBwState`

### 6.2 Actor 结构与前向

- `sagin_marl/rl/structured_actor.py`
  - `BwPolicy`
  - `act_bw()`
  - `evaluate_bw()`
  - 作用：
    - 定义 BW actor 的主要模块：
      - `ego_encoder`
      - `sat_encoder`
      - `sat_refine`
      - `query_proj_1`
      - `query_proj_2`
      - `user_encoder`
      - `user_refine`
      - `user_fusion`
      - `loc_head`
    - 当前 clean 训练用的 `score_only_softmax` 也在这里

### 6.3 Clean teacher / target 生成 / 更新逻辑

- `sagin_marl/rl/structured_mappo.py`
  - clean teacher 生成
  - clean exact gate
  - clean trust region
  - clean actor supervised loss
  - 作用：
    - 生成 per-state target
    - 比较 `target_return` 与 `ref_return`
    - 计算 clean Huber loss
    - 执行一次 clean update

- `sagin_marl/rl/structured_buffer.py`
  - 作用：
    - rollout buffer
    - 保存训练 batch 中的 structured states / snapshot state 等

- `sagin_marl/rl/structured_train.py`
  - 作用：
    - 训练过程中的 rollout、buffer 组织、metrics 汇总

- `scripts/train_structured.py`
  - 作用：
    - 训练入口
    - 配置加载
    - metrics.csv / checkpoint_eval.csv 写出

### 6.4 与当前诊断直接相关的脚本

- `scripts/diagnostics/diagnose/diagnose_structured_bw_clean_horizon_sensitivity.py`
  - 检查不同 horizon 下 teacher target / gain 的变化

- `scripts/diagnostics/diagnose/diagnose_structured_bw_clean_plateau.py`
  - 固定 panel 上比较不同 checkpoint 的 local/full gain

- `scripts/diagnostics/diagnose/diagnose_structured_bw_fixed_teacher_fit.py`
  - 固定 bank、固定 target 的反复训练诊断

- `scripts/diagnostics/diagnose/diagnose_structured_bw_online_update_direction.py`
  - 检查一次真实 online update 对固定 panel 的 pre/post 影响

- `scripts/diagnostics/diagnose/diagnose_structured_bw_clean_gradient_conflict.py`
  - 检查 clean 监督梯度冲突、`batch_vs_panel_cos`

- `scripts/diagnostics/diagnose/diagnose_structured_bw_local_state_target_ambiguity.py`
  - 检查 `LocalBwState` 以及 actor 内部表征相似但 target 相差很大的样本是否很多

---

## 7. 已生成的重要诊断结果文件

下面这些结果文件是本文结论的主要数据来源：

- `runs/tmp/horizon_sensitivity_h5best_64.json`
- `runs/tmp/horizon_sensitivity_h5best_20vs30_64.json`
- `runs/tmp/clean_plateau_diagnosis_h5_panel32_fixedmask.json`
- `runs/tmp/fixed_teacher_fit_h5_u60_panel32.json`
- `runs/tmp/online_update_direction_h5_u60_t4.json`
- `runs/tmp/clean_grad_conflict_h5_u60_lochead.json`
- `runs/tmp/clean_grad_conflict_h5_u60_lochead_seed2.json`
- `runs/tmp/clean_grad_conflict_h5_u60_bwpolicy.json`
- `runs/tmp/local_state_target_ambiguity_h5_u60_64.json`

---

## 8. 当前最简洁的结论

如果只保留一句话，当前最接近事实的表述是：

**现在的主要问题不是 teacher 没信号，也不是已经找到很强的输入缺失证据，而是很多 per-state 的好 target 一起训练时，在共享参数空间里互相拉扯，导致 clean update 不能稳定把 actor 同时拉向这些 target。**

这也是为什么会同时看到：

- per-state 的 target 大多是好的；
- 但固定 bank 反复训练时不能稳定更接近 target；
- 真实 online update 在平台附近会伤到整体评估。
