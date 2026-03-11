# shared43000 Pair Summary (queue_total_active_excl_step0): baseline vs danger_nbr_5d

- baseline: `runs/line3_short/setpool30_prealert_top1gain_eta35_close_risk/eval_trained_shared43000_qex0.csv`
- danger5d: `runs/line3_short/setpool30_prealert_top1gain_eta35_close_risk_danger_nbr_min5/eval_trained_shared43000_qex0.csv`
- paired csv: `runs/line3_short/pair_compare_shared43000_baseline_vs_danger5d_qex0.csv`
- metric: `queue_total_active_excl_step0`
- seed rule: `seed = 43000 + episode`
- queue_much_worse threshold: `delta_queue_total_active_excl_step0 > 100`

## Group Counts

- group1_safety_improved: 5
- group2_queue_much_worse: 1
- group3_overall_worse: 2

## Top Queue Delta Episodes

- episode 18 seed 43018: group=group2_queue_much_worse, delta_queue_excl_step0=116.372, delta_min_dist=8.535, delta_near=0.0125
- episode 6 seed 43006: group=other, delta_queue_excl_step0=98.006, delta_min_dist=4.136, delta_near=0.0025
- episode 8 seed 43008: group=group1_safety_improved, delta_queue_excl_step0=9.515, delta_min_dist=11.445, delta_near=-0.012652
- episode 17 seed 43017: group=other, delta_queue_excl_step0=5.741, delta_min_dist=41.306, delta_near=0
- episode 12 seed 43012: group=other, delta_queue_excl_step0=0, delta_min_dist=-18.409, delta_near=0.0025
