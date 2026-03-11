已写回这三份主线配置：
[phase1_actions_curriculum_stage1_accel_setpool_prealert_top1gain_close_risk_danger_nbr.yaml](/d:/研三上/毕设/sagin_marl/configs/phase1_actions_curriculum_stage1_accel_setpool_prealert_top1gain_close_risk_danger_nbr.yaml)
[phase1_actions_curriculum_stage1_accel_setpool_prealert_top1gain_close_risk.yaml](/d:/研三上/毕设/sagin_marl/configs/phase1_actions_curriculum_stage1_accel_setpool_prealert_top1gain_close_risk.yaml)
[s1_safe_static_v3_prealert_close_risk.yaml](/d:/研三上/毕设/sagin_marl/configs/stage1_safety_followup/s1_safe_static_v3_prealert_close_risk.yaml)

回写后的参数是：
- `task_arrival_rate = 1.3e5`
- `b_acc = 2.2e6`
- `queue_init_gu_steps = 16`
- `queue_max_gu = 3.0e7`
- `b_sat_total = 8.75e5`
- `queue_init_uav_steps = 40`
- `queue_max_uav = 1.0e9`
- `sat_cpu_freq = 1.8e9`
- `queue_init_sat_steps = 32`
- `queue_max_sat = 2.0e8`

我还直接用写回后的主配置复跑了 10-episode 验证，结果在 [stagewise_writtenback_n10_report.md](/d:/研三上/毕设/sagin_marl/runs/queue_diag/stagewise_writtenback_n10_report.md)：
- `passes_thresholds = 1`
- `failed_thresholds = none`
- `total_drop_fraction_mean = 0.007601`
- `active_queue_empty_step_fraction_mean = 0.062`
- `all_layers_nonempty_step_fraction_mean = 0.63475`
- `outflow_arrival_ratio_mean = 0.955`

分阶段定参流程我建议就固定成这版：

1. 先定 `task_arrival_rate`，阶段内不动它。
2. 阶段1只看 `GU`，调 `b_acc + queue_init_gu_steps + queue_max_gu`。
3. 先看 `gu_queue_drift_ratio_mean`。
4. 如果 `drift < 0`，先加 `queue_init_gu_steps`，再检查加完后 `drift` 还是否过线。
5. 如果 `drift > 0`，先加 `queue_max_gu`，再检查加完后 `drift` 还是否过线。
6. `drift` 过线后，再用 `gu_empty_step_fraction_mean` 和 `gu_drop_ratio_mean` 收口。
7. 只有 `GU` 过线，才进入阶段2。
8. 阶段2只看 `UAV`，调 `b_sat_total + queue_init_uav_steps + queue_max_uav`。
9. 阶段2按你的要求，不用 `sat_empty_step_fraction` 和 `sat_drop_fraction` 卡门槛。
10. 同样先看 `uav_queue_drift_ratio_mean`，再按负漂移补 `init`、正漂移补 `max`，最后用 `uav_empty_step_fraction_mean`、`uav_drop_ratio_mean`、`uav_buffer_fill_fraction_p95_mean` 收口。
11. 只有 `UAV` 过线，才进入阶段3。
12. 阶段3看 `SAT`，调 `sat_cpu_freq + queue_init_sat_steps + queue_max_sat`，流程同样是先 `drift`，后 `empty/drop/fill`。
13. 最后再看系统级护栏：`total_drop`、`active_queue_empty`、`all_layers_nonempty`、`outflow_arrival_ratio`。

这次没有改脚本逻辑，只回写了配置并做了配置级复核。