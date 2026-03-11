# shared43000 Pair Summary: baseline vs danger_nbr_5d

- baseline: `runs/line3_short/setpool30_prealert_top1gain_eta35_close_risk/eval_trained_shared43000.csv`
- danger5d: `runs/line3_short/setpool30_prealert_top1gain_eta35_close_risk_danger_nbr_min5/eval_trained_shared43000.csv`
- paired csv: `runs/line3_short/pair_compare_shared43000_baseline_vs_danger5d.csv`
- seed rule: `seed = 43000 + episode`
- queue_much_worse threshold: `delta_queue_total_active > 100`

## Group Counts

- group1_safety_improved: 5
- group2_queue_much_worse: 2
- group3_overall_worse: 2

## Group 1: safety improved

- episode 1 seed 43001: collision 1->0, queue 0->0, min_dist 4.26->84.35
- episode 2 seed 43002: collision 1->0, queue 9201.871->9030.885, min_dist 12.205->25.735
- episode 8 seed 43008: collision 1->0, queue 1221.197->1228.158, min_dist 16.928->28.373
- episode 9 seed 43009: collision 1->0, queue 0->0, min_dist 9.485->169.433
- episode 10 seed 43010: collision 1->0, queue 0->0, min_dist 10.36->270.478

## Group 2: safety similar but queue much worse

- episode 6 seed 43006: collision 0->0, delta_queue 449.4, delta_min_dist 4.136, delta_near 0.0025
- episode 18 seed 43018: collision 0->0, delta_queue 149.116, delta_min_dist 8.535, delta_near 0.0125

## Group 3: overall worse

- episode 14 seed 43014: collision 0->1, delta_queue 75133.008, min_dist 24.392->11.57
- episode 16 seed 43016: collision 0->1, delta_queue 0, min_dist 153.008->9.684

## Top Queue Delta Episodes

- episode 14 seed 43014: group=group3_overall_worse, delta_queue=75133.008, delta_min_dist=-12.822, delta_near=0.071923
- episode 6 seed 43006: group=group2_queue_much_worse, delta_queue=449.4, delta_min_dist=4.136, delta_near=0.0025
- episode 18 seed 43018: group=group2_queue_much_worse, delta_queue=149.116, delta_min_dist=8.535, delta_near=0.0125
- episode 17 seed 43017: group=other, delta_queue=17.214, delta_min_dist=41.306, delta_near=0
- episode 8 seed 43008: group=group1_safety_improved, delta_queue=6.96, delta_min_dist=11.445, delta_near=-0.012652
