# 测试文件说明

本文档概述 `tests/` 目录下的项目测试文件（不含依赖/虚拟环境中的测试）。

- `tests/conftest.py`: 将项目根目录加入 `sys.path`，保证测试可直接导入 `sagin_marl` 及脚本模块。
- `tests/test_action_masking.py`: 在 `fixed_satellite_strategy=True` 情况下执行一次 `env.step`，验证动作字典结构可用、环境能推进并返回 `num_uav` 个观测。
- `tests/test_baselines.py`: 验证 `zero_accel_policy` 输出的形状为 `(n, 2)`、dtype 为 `float32`，且全为 0。
- `tests/test_config_parsing.py`: 校验 `load_config` 能将 YAML 中的科学计数法数值解析为 `float`，并正确解析布尔字段；同时验证 `ablation`/`ablation_flags` 的嵌套配置解析。
- `tests/test_early_stopping.py`: 训练流程在早停参数配置下能够提前终止，`metrics.csv` 的更新次数小于 `total_updates`。
- `tests/test_env_reset_shapes.py`: `env.reset` 返回的观测结构与形状一致性（`own/users/sats/nbrs` 及对应 `*_mask` 的维度）。
- `tests/test_env_step_invariants.py`: 连续 `step` 后 `gu_queue/uav_queue/sat_queue` 始终非负且为有限值。
- `tests/test_run_dir_paths.py`: `scripts/train.py`、`scripts/evaluate.py`、`scripts/render_episode.py` 的运行目录与输出路径解析逻辑（`run_dir`/`run_id`/默认文件名/基线输出）。
- `tests/test_sat_queue_tracking.py`: 卫星队列更新的一致性，验证 `last_sat_incoming`、`last_sat_processed` 形状与非负，处理量不超过容量且队列更新符合公式。
- `tests/test_smoke_train.py`: 最小配置下训练流程可跑通的 smoke test。
- `tests/test_smoke_train_vec.py`: 多环境并行训练 smoke test（`num_envs=2, backend=sync`），验证向量化 rollout 流程可执行。
- `tests/test_traffic_curriculum.py`: 验证 `traffic_level` 在 `reset` 后能正确映射到有效到达率（Level 0/1/2），并在后续 `step` 中生效。
