# BW Clean `group_mode=target_stats` 最小改动方案（2026-04-16）

## 1. 目标

这份方案只解决一个很具体的问题：

- 当前 clean update 里的 grouped PCGrad 是：
  - 先 `shuffle`
  - 再按相邻样本硬切组
- 这会把不同 regime 的样本先在组内平均掉，弱化“真正该被 conflict handling 处理的组间冲突”。

所以这里要加一个 **最小改动版** 的：

- `group_mode = target_stats`

让 PCGrad 在 `task_group_size > 1` 时，不再完全随机分组，而是按一些**便宜、稳定、与 target 几何直接相关**的统计量做分组。

这份方案刻意追求：

- 改动小
- 可回退
- 不引入新的重计算
- 不改变 `mean` 路径
- 不要求 full per-sample PCGrad

---

## 2. 当前代码里分组发生在哪里

当前 clean update 主路径在：

- [sagin_marl/rl/structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3253)

当前 grouped PCGrad 的关键逻辑是：

1. 先构造 `all_indices`
2. `np.random.shuffle(all_indices)`
3. 取出一个 minibatch
4. 计算 `sample_losses`
5. 传给 `_bw_clean_pcgrad_backward(...)`
6. 在 `_bw_clean_pcgrad_backward(...)` 里，把 `sample_losses` 按相邻顺序每 `task_group_size` 个切一组求平均

对应位置：

- shuffle 发生在：
  - [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3357)
- 分组切块发生在：
  - [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3192)

所以最小改动思路是：

**不改 `_bw_clean_pcgrad_backward()` 的主体逻辑，只改传给它的样本顺序。**

也就是：

- 先按 `target_stats` 算一个“更合理的样本顺序”
- 再把 `sample_losses`、`target_mb`、`valid_mb`、`local_batch_mb` 等按这个顺序重排
- 然后仍然让 `_bw_clean_pcgrad_backward()` 按 contiguous chunk 切组

这样改动最小。

---

## 3. 推荐新增的 config 项

在 [sagin_marl/env/config.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/config.py:330) 这一段附近新增：

```python
bw_clean_pcgrad_group_mode: str = "random"  # "random" | "target_stats"
bw_clean_pcgrad_group_entropy_bins: int = 3
bw_clean_pcgrad_group_gap_bins: int = 3
```

最小版其实只需要第一项：

- `bw_clean_pcgrad_group_mode`

后两项只是为了后面更方便调桶数量；如果想更极简，也可以先不加，直接在代码里写死为 `3`。

对应在 `StructuredMAPPO.__init__` 里新增读取：

```python
self.bw_clean_pcgrad_group_mode = (
    "random"
    if cfg is None
    else str(getattr(cfg, "bw_clean_pcgrad_group_mode", "random") or "random").strip().lower()
)
if self.bw_clean_pcgrad_group_mode not in {"random", "target_stats"}:
    raise ValueError(...)
```

---

## 4. `target_stats` 最小版到底算哪些特征

最小改动版我建议只算 4 个特征，全部都已经在当前 clean update 里有足够信息得到，不需要额外 rollout。

### 4.1 `valid_count`

定义：

- 有效 user 数量

实现：

```python
valid_count = valid_mask.sum(axis=-1)
```

为什么必须要有：

- 不同 `valid_count` 的 simplex 维度本来就不同；
- 这往往对应不同的 loss geometry 和不同的冲突模式；
- 先按这个分桶最稳。

### 4.2 `target_entropy`

定义：

```python
entropy = -sum_i target_i * log(target_i + eps)
```

只在 valid slots 上算。

为什么要有：

- 它能区分：
  - 尖峰型 target
  - 平滑型 target
- 对 `score -> softmax` 这条路来说，这个量直接对应“当前样本的分布几何有多尖”。

### 4.3 `target_top1_mass`

定义：

- valid slots 上 `max(target_i)`

为什么要有：

- 它比 entropy 更直观地区分：
  - “几乎 winner-take-most”
  - “没有特别强的 winner”
- 对分组来说非常便宜，而且和 entropy 互补。

### 4.4 `target_gap_l1`

定义：

```python
target_gap_l1 = sum_i |target_i - ref_i|
```

只在 valid slots 上算。

为什么要有：

- 它区分：
  - teacher 只是做小修正
  - teacher 想大幅重排当前动作
- 这通常也对应不同梯度幅度和不同冲突强度。

---

## 5. 为什么先不把 `rho / utility_l1 / teacher_gain` 放进最小版

它们不是没价值，而是为了保持改动最小，先不作为第一版核心键。

### `rho`

优点：

- 和 teacher target 的 step size 很直接相关。

缺点：

- 虽然当前 `_bw_clean_target_actions_parallel()` 已经返回了 `rho`，
  但在 clean update 主路径里现在是 `_rho_values_np`，没有继续往下用；
- 第一版若只靠 `target_gap_l1`，已经能近似区分“小修正 vs 大修正”。

### `utility_l1`

优点：

- 很接近 teacher utility direction 的强度。

缺点：

- 和 `target_gap_l1`、`rho` 信息有一定重叠；
- 第一版不是最必要。

### `teacher_gain`

优点：

- 很像“这个样本值不值得认真对齐”。

缺点：

- 需要依赖 target vs ref return；
- 当前虽然已经算了 `target_beats_flags`，但每个样本的 gain 值在 clean update 主路径里没有作为后续特征直接保留；
- 第一版不如先用更纯粹的 target 几何特征。

结论：

**第一版最小特征集就用：**

- `valid_count`
- `target_entropy`
- `target_top1_mass`
- `target_gap_l1`

如果第一版有信号，再加：

- `rho`
- `teacher_gain`

---

## 6. 最小分组规则

最小版不做聚类，只做 **排序 + 切块**。

推荐规则：

1. 先按 `valid_count` 升序
2. 再按 `target_entropy` 升序
3. 再按 `target_top1_mass` 降序
4. 再按 `target_gap_l1` 升序

然后：

- 排好序之后
- 每 `task_group_size` 个样本切成一组

这样得到的效果大概是：

- 同一组里，样本的 simplex 维度相近
- target 的尖峰程度相近
- top-1 winner 的强度相近
- 相对 ref 的改动幅度也相近

这比随机切块更有语义，而且实现几乎最简单。

---

## 7. 推荐新增的 helper

建议在 `StructuredMAPPO` 里新增一个 helper，例如：

```python
def _bw_clean_target_stats_order(
    self,
    *,
    target_actions_np: np.ndarray,
    ref_actions_np: np.ndarray,
    valid_masks_np: np.ndarray,
) -> np.ndarray:
    ...
```

返回：

- 一个 `order_np`
- 表示样本应该被重排成什么顺序

最小实现逻辑：

```python
mask_f = valid_masks_np.astype(np.float32, copy=False)
valid_count = valid_masks_np.sum(axis=-1).astype(np.int64, copy=False)

target_safe = np.where(valid_masks_np, np.clip(target_actions_np, 1.0e-8, None), 0.0)
target_norm = target_safe / np.clip(target_safe.sum(axis=-1, keepdims=True), 1.0e-8, None)

target_entropy = -np.sum(
    np.where(valid_masks_np, target_norm * np.log(target_norm), 0.0),
    axis=-1,
    dtype=np.float64,
)
target_top1_mass = np.max(np.where(valid_masks_np, target_norm, 0.0), axis=-1)
target_gap_l1 = np.sum(np.abs(target_actions_np - ref_actions_np) * mask_f, axis=-1, dtype=np.float64)

order_np = np.lexsort((
    target_gap_l1,
    -target_top1_mass,
    target_entropy,
    valid_count,
))
return order_np.astype(np.int64, copy=False)
```

说明：

- `np.lexsort` 最后一个 key 优先级最高，所以这里把 `valid_count` 放最后。
- 这已经足够做第一版。

---

## 8. 在 clean update 里具体插在哪

主改动点就在 [structured_mappo.py](/d:/研三上/毕设/sagin_marl/sagin_marl/rl/structured_mappo.py:3321) 之后。

当前代码拿到训练样本后，会直接：

```python
all_indices = np.arange(num_train_samples, dtype=np.int64)
np.random.shuffle(all_indices)
...
mb_rel = torch.as_tensor(all_indices[start : start + minibatch_size], ...)
```

最小改法：

### 第一步：在构造训练样本后，先准备 `train_target_stats_order_np`

也就是在拿到：

- `target_actions`
- `old_actions`
- `valid_masks_train`

之后，额外准备一份 NumPy 版输入，用于算顺序：

```python
if num_train_samples > 0 and self.bw_clean_pcgrad_group_mode == "target_stats":
    train_target_stats_order_np = self._bw_clean_target_stats_order(
        target_actions_np=np.asarray(target_actions_np[train_idx_np], dtype=np.float32),
        ref_actions_np=np.asarray(ref_actions[train_idx_np], dtype=np.float32),
        valid_masks_np=np.asarray(valid_masks_np[train_idx_np], dtype=bool),
    )
else:
    train_target_stats_order_np = None
```

### 第二步：每个 epoch 里仍然保留 shuffle，但只在组级别 shuffle

这是最小改动里最值得保留的一点：

- 不是完全固定顺序
- 而是先按 `target_stats` 排好
- 再以 `task_group_size` 为单位 shuffle 这些块

理由：

- 保留一定随机性，避免每轮都完全一样
- 又不破坏组内语义

最小实现可以写成：

```python
if train_target_stats_order_np is None:
    np.random.shuffle(all_indices)
else:
    ordered = train_target_stats_order_np.copy()
    group_size = max(int(self.bw_clean_pcgrad_task_group_size), 1)
    if group_size > 1 and ordered.size > group_size:
        blocks = [ordered[i : i + group_size] for i in range(0, ordered.size, group_size)]
        np.random.shuffle(blocks)
        all_indices = np.concatenate(blocks, axis=0)
    else:
        all_indices = ordered
```

这样就不需要改 `_bw_clean_pcgrad_backward()`。

### 第三步：只有 `grad_aggregation == "pcgrad"` 时才启用这个顺序

也就是说：

- `mean` 路径完全不动
- 只有 grouped PCGrad 才吃这个新顺序

这样风险最小。

---

## 9. 我不建议的“最小改动”

### 9.1 直接把 grouping 逻辑塞进 `_bw_clean_pcgrad_backward()`

不建议，原因：

- 那个函数现在只知道 `sample_losses`
- 不知道 `target/ref/valid_mask`
- 强行塞进去会把依赖关系搞乱

### 9.2 第一版就做聚类

不建议，原因：

- 改动大
- 解释复杂
- 第一版先看简单的排序切块有没有信号更值

### 9.3 第一版就把 `rho/gain/utility_l1` 全塞进去

不建议，原因：

- 特征过多，分不清到底是谁有用
- 第一版应该只测最核心、最几何直接的几项

---

## 10. 最小可落地改动清单

如果按最小实现走，实际只需要这几步：

1. 在 [config.py](/d:/研三上/毕设/sagin_marl/sagin_marl/env/config.py:330) 新增：
   - `bw_clean_pcgrad_group_mode`

2. 在 `StructuredMAPPO.__init__` 里新增：
   - `self.bw_clean_pcgrad_group_mode`

3. 在 `StructuredMAPPO` 里新增一个 helper：
   - `_bw_clean_target_stats_order(...)`

4. 在 `_update_bw_clean_per_user(...)` 里：
   - 构造 `train_target_stats_order_np`
   - 在 `pcgrad` 路径下，用它替代纯随机顺序

5. 先不改 `_bw_clean_pcgrad_backward(...)`

这就是最小改动版。

---

## 11. 一句话总结

最小版 `group_mode=target_stats` 的核心不是“发明新的 PCGrad”，而是：

**保留现有 grouped PCGrad 的切块实现，只把“样本顺序”从随机打乱改成“按 `valid_count + target_entropy + top1_mass + target_gap_l1` 排序后再切块”。**

这已经足够把“随机近似”升级成“有任务语义的近似”，同时保持改动很小。

