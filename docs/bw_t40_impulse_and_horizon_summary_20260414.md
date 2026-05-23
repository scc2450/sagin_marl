# BW T40 Impulse And Horizon Summary (2026-04-14)

## Scope

- Diagnostics use `heuristic_action` vs `heuristic_action + relative perturbation`.
- Perturbation rule: move `20%` of the source user's current BW weight to the minimum-weight user.
- Follow policy after the first step: deterministic `queue_aware_bw`.
- Reward mode: `weighted_workload_level`.
- Each case uses `1600` BW snapshots, `T=40`, `seed_base=3042`.
- Training results below are the post-bugfix `8GU/subset4, T=40` runs with direct `branch_delta` teacher, `num_envs=8`, `rollout_env_steps=80`, `updates=60`.

## Files

- `2GU/subset1` config: [structured_bw_sanity_1uav_2gu_t40_stronggap_diagnose.yaml](../configs/tmp/structured_bw_sanity_1uav_2gu_t40_stronggap_diagnose.yaml)
- `4GU/subset2` config: [structured_bw_sanity_1uav_4gu_t40_pairhot_diagnose.yaml](../configs/tmp/structured_bw_sanity_1uav_4gu_t40_pairhot_diagnose.yaml)
- `8GU/subset4` config: [structured_bw_sanity_1uav_8gu_t40_quadhot_current_recipe_noes.yaml](../configs/tmp/structured_bw_sanity_1uav_8gu_t40_quadhot_current_recipe_noes.yaml)
- `8GU` Beijing complex-channel config: [structured_bw_sanity_1uav_8gu_t40_quadhot_beijing_channel_sanity.yaml](../configs/tmp/structured_bw_sanity_1uav_8gu_t40_quadhot_beijing_channel_sanity.yaml)
- `8GU` Beijing complex-channel + `2.0x` resource config: [structured_bw_sanity_1uav_8gu_t40_quadhot_beijing_channel_sanity_res200.yaml](../configs/tmp/structured_bw_sanity_1uav_8gu_t40_quadhot_beijing_channel_sanity_res200.yaml)
- `2GU` impulse JSON: [heuristic_perturb_impulse_2gu_t40_relative_k40_seed3042_n1600.json](../runs/tmp/heuristic_perturb_impulse_2gu_t40_relative_k40_seed3042_n1600.json)
- `4GU` impulse JSON: [heuristic_perturb_impulse_4gu_t40_relative_k40_seed3042_n1600.json](../runs/tmp/heuristic_perturb_impulse_4gu_t40_relative_k40_seed3042_n1600.json)
- `8GU` impulse JSON: [heuristic_perturb_impulse_8gu_subset4_t40_relative_k40_seed3042_n1600.json](../runs/tmp/heuristic_perturb_impulse_8gu_subset4_t40_relative_k40_seed3042_n1600.json)
- `8GU` Beijing complex-channel impulse JSON: [heuristic_perturb_impulse_8gu_beijing_t40_relative_k40_seed3042_n1600.json](../runs/tmp/heuristic_perturb_impulse_8gu_beijing_t40_relative_k40_seed3042_n1600.json)
- `8GU` Beijing complex-channel + `2.0x` resource impulse JSON: [heuristic_perturb_impulse_8gu_beijing_res200_t40_relative_k40_seed3042_n1600.json](../runs/tmp/heuristic_perturb_impulse_8gu_beijing_res200_t40_relative_k40_seed3042_n1600.json)
- `8GU` training runs:
  - [h3 checkpoint_eval.csv](../runs/structured/eight_gu_t40_env8_roll80_h3_u60_bugfix/checkpoint_eval.csv)
  - [h5 checkpoint_eval.csv](../runs/structured/eight_gu_t40_env8_roll80_h5_u60_bugfix/checkpoint_eval.csv)
  - [h10 checkpoint_eval.csv](../runs/structured/eight_gu_t40_env8_roll80_h10_u60_bugfix/checkpoint_eval.csv)
  - [h20 checkpoint_eval.csv](../runs/structured/eight_gu_t40_env8_roll80_h20_u60_bugfix/checkpoint_eval.csv)
  - [h30 checkpoint_eval.csv](../runs/structured/eight_gu_t40_env8_roll80_h30_u60_bugfix/checkpoint_eval.csv)

## Impulse Diagnostics

`arrival_ref = num_gu * task_arrival_rate`, so the normalized queue numbers below are comparable across `2GU/4GU/8GU`.

### Snapshot

| Case | first action L1 | k=1 median/arrival_ref | k=3 | k=10 | k=20 | k=30 | k=40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2GU/subset1` | `0.3462` | `0.3084` | `0.0476` | `0.0293` | `0.0000` | `0.0000` | `0.0000` |
| `4GU/subset2` | `0.2653` | `0.1641` | `0.0733` | `0.0272` | `0.0069` | `0.0013` | `0.00015` |
| `8GU/subset4` | `0.1571` | `0.0957` | `0.0550` | `0.0250` | `0.0135` | `0.0051` | `0.00042` |

### Recovery Thresholds

| Case | first `k` with median <= `5% arrival_ref` | first `k` with median <= `1% arrival_ref` | first `k` with median <= `0.5% arrival_ref` | first `k` with median <= `5%` of initial impulse | first `k` with median <= `1%` of initial impulse |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2GU/subset1` | `3` | `14` | `16` | `12` | `16` |
| `4GU/subset2` | `6` | `19` | `22` | `20` | `30` |
| `8GU/subset4` | `4` | `26` | `31` | `31` | `37` |

### Tail At `k=40`

| Case | `k=40` count | `k=40 mean/arrival_ref` | `k=40 p90/arrival_ref` | `k=40 frac(queue_l1 > 1e5)` |
| --- | ---: | ---: | ---: | ---: |
| `2GU/subset1` | `40` | `0.0317` | `0.0024` | `0.050` |
| `4GU/subset2` | `40` | `0.1240` | `0.6655` | `0.200` |
| `8GU/subset4` | `40` | `0.1190` | `0.4785` | `0.275` |

### Reading These Three Cases

- `2GU/subset1` is still the shortest-memory case in typical behavior. Its median queue gap falls below `5% arrival_ref` by `k=3`, below `1% arrival_ref` by `k=14`, and reaches `0` by `k=19`.
- `4GU/subset2` is clearly longer-memory than `2GU`. Its median queue gap does not get below `1% arrival_ref` until `k=19`, and below `5%` of the initial impulse until `k=20`.
- `8GU/subset4` is the longest-memory case in typical behavior. Its median queue gap does not get below `1% arrival_ref` until `k=26`, and below `5%` of the initial impulse until `k=31`.
- Tail behavior stays hard for both `4GU` and `8GU` even at `k=40`. `4GU` has the larger `p90` queue tail at `k=40`, while `8GU` has the larger fraction of snapshots still above `1e5`.
- The first-step perturbation is not the reason `8GU` looks longer. Under relative perturbation, the first-step action change is actually smallest for `8GU` (`0.1571`) and largest for `2GU` (`0.3462`).

## 8GU Channel-vs-Resource Ablation

This ablation isolates a mistake in an earlier verbal summary. Comparing the old simple `8GU` case directly against the Beijing complex-channel + `res200` case mixes two changes at once:

- channel/traffic realism changed
- resources were also increased to `2.0x`

After adding the missing middle case, the cleaner reading is:

- complex-channel realism already shortens the impulse a lot
- the `2.0x` resource bump is mainly useful for restoring the fixed-policy BW sanity regime, not for causing the impulse shortening by itself

### 8GU `k=40` Comparison

| Case | first action L1 | k=1 median/arrival_ref | k=3 | k=10 | k=20 | k=30 | k=40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8GU` old simple | `0.1571` | `0.0957` | `0.0550` | `0.0250` | `0.0135` | `0.0051` | `0.00042` |
| `8GU` Beijing complex-channel | `0.0954` | `0.0346` | `0.0198` | `0.0069` | `0.0023` | `0.0011` | `0.00081` |
| `8GU` Beijing complex-channel + `res200` | `0.1019` | `0.0439` | `0.0213` | `0.0089` | `0.0039` | `0.0023` | `0.00125` |

### 8GU Recovery Thresholds

| Case | first `k` with median <= `2% arrival_ref` | first `k` with median <= `1% arrival_ref` | first `k` with median <= `0.5% arrival_ref` | first `k` with median <= `25%` of initial impulse | first `k` with median <= `10%` of initial impulse |
| --- | ---: | ---: | ---: | ---: | ---: |
| `8GU` old simple | `14` | `26` | `31` | `11` | `27` |
| `8GU` Beijing complex-channel | `3` | `7` | `14` | `8` | `16` |
| `8GU` Beijing complex-channel + `res200` | `4` | `9` | `17` | `8` | `19` |

### 8GU Tail At `k=40`

| Case | `k=40 mean/arrival_ref` | `k=40 p90/arrival_ref` | `k=40 frac(queue_l1 > 1e5)` |
| --- | ---: | ---: | ---: |
| `8GU` old simple | `0.1190` | `0.4785` | `0.275` |
| `8GU` Beijing complex-channel | `0.00135` | `0.00315` | `0.000` |
| `8GU` Beijing complex-channel + `res200` | `0.00177` | `0.00289` | `0.000` |

### Reading This Ablation

- The big jump happens when moving from the old simple `8GU` environment to the Beijing complex-channel environment. The median queue-gap threshold `<= 1% arrival_ref` moves from `k=26` down to `k=7`.
- Adding `res200` on top of the Beijing complex-channel case does not shorten the impulse further. If anything, the median thresholds move slightly later (`k=7 -> 9`, `k=14 -> 17`).
- The resource bump still matters for a different reason: it restores the fixed-policy sanity regime where hot users remain under-served while cold users become slightly over-served again under `uniform_bw`.
- So the earlier shortcut sentence "the resource bump mainly caused the impulse shortening" was too strong. The better statement is: the channel/propagation realism did most of the shortening, while `res200` mainly repaired the BW-regime sanity.

## 8GU Horizon Training

Reference fixed policy on the same `8GU/subset4, T=40` EMA reward:

- `reward = -1010.9303`
- `processed = 1.061178`
- `drop = 0.001818`
- `backlog = 2.486015`

### Best Checkpoint Per Horizon

| Horizon | Best update | Reward | Gap vs fixed heuristic | Weighted delta | Processed | Drop | Backlog |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h=3` | `55` | `-1048.2748` | `-37.3445` | `1651.0603` | `1.059839` | `0.003250` | `2.439516` |
| `h=5` | `60` | `-1036.4114` | `-25.4811` | `1661.1190` | `1.059394` | `0.003868` | `2.431423` |
| `h=10` | `55` | `-1018.3850` | `-7.4547` | `1674.0639` | `1.062096` | `0.002958` | `2.423158` |
| `h=20` | `55` | `-1015.8272` | `-4.8969` | `1674.0624` | `1.061676` | `0.003049` | `2.453630` |
| `h=30` | `60` | `-1012.0506` | `-1.1204` | `1676.4650` | `1.061417` | `0.003101` | `2.455987` |

### Reading The Horizon Sweep

- The horizon sweep is monotonic in the main reward: `h=3 < h=5 < h=10 < h=20 < h=30`.
- `h=30` is the best tested horizon so far and is only about `1.12` reward behind the fixed `queue_aware_bw` baseline.
- This lines up with the `8GU/subset4` impulse diagnosis: the median queue gap does not fall below `1% arrival_ref` until `k=26`, and does not fall below `5%` of the initial impulse until `k=31`.
- Put differently: `h=3/5/10` clearly truncate a large part of the typical downstream effect, `h=20` still truncates some of it, and `h=30` is the first tested horizon that roughly covers the bulk of the typical impulse.

## Takeaways

- The old `2GU` intuition does not transfer directly to `4GU/8GU` at `T=40`.
- For typical samples, the effective memory ordering is:
  - `2GU/subset1`: shortest
  - `4GU/subset2`: longer
  - `8GU/subset4`: longest
- For tail samples, both `4GU` and `8GU` still have meaningful hard cases at `k=40`, so a horizon that matches the median will still miss some tail mass.
- The post-bugfix `8GU` training results are consistent with the diagnostic: longer `branch_delta` horizons keep helping through at least `h=30`.
- Within the `8GU` ablation, the large impulse-shortening effect comes from turning on the more realistic channel/propagation stack, not from the later `res200` resource bump alone.
- The `res200` bump is still useful, but its main role is to move the environment back into a clearer BW-allocation regime for sanity checking and training, rather than to be the main source of the shorter impulse.
