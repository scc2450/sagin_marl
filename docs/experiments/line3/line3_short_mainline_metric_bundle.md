# Line3 Mainline Metric Bundle

Mainline run:

- `runs/line3_short/setpool30_prealert`

Raw source files:

- `runs/line3_short/setpool30_prealert/metrics.csv`
- `runs/line3_short/setpool30_prealert/eval_trained_seed43000.csv`
- `runs/line3_short/setpool30_prealert/eval_fixed_seed43000.csv`

Generated focus files:

- `docs/line3_short_mainline_metrics_focus.csv`
- `docs/line3_short_mainline_eval_trained_focus.csv`
- `docs/line3_short_mainline_eval_fixed_focus.csv`

Columns kept from `metrics.csv`:

- `step`
- `episode_reward`
- `collision_rate`
- `queue_total_active`
- `queue_total_active_p95`
- `queue_total_active_p99`
- `r_queue_pen`
- `r_queue_delta`
- `r_collision_penalty`
- `r_term_accel`
- `action_std_mean`
- `log_std_mean`

Columns kept from `eval_trained_seed43000.csv` and `eval_fixed_seed43000.csv`:

- `episode`
- `reward_sum`
- `collision`
- `queue_total_active`
- `queue_total_active_max`
- `queue_total_active_p95_step`
- `termination_reason`
- `min_inter_uav_dist`
- `near_collision_steps`

Notes:

- The eval files do not have `episode_reward`; the closest field is `reward_sum`.
- The eval files do not have `collision_rate`; they only have per-episode `collision`.
- The eval files do not have `queue_total_active_p99`.
- The eval files do not have `r_queue_pen`, `r_queue_delta`, `r_collision_penalty`, or `r_term_accel`.
- The eval files do not have `action_std_mean` or `log_std_mean`.

Recommended send-out bundle:

- `docs/line3_short_mainline_metric_bundle.md`
- `docs/line3_short_mainline_metrics_focus.csv`
- `docs/line3_short_mainline_eval_trained_focus.csv`
- `docs/line3_short_mainline_eval_fixed_focus.csv`
