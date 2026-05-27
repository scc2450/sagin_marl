# Clean Per-User Reward-Obs Configs

These configs are derived from:

- `configs/tmp/structured_bw_sanity_1uav_8gu_t40_beijing_res200_rewardobs_h30.yaml`

They keep the same environment and `Reward-Obs` setting as the current best branch-teacher baseline, and only switch the BW route to the clean per-user setup.

Common changes relative to the branch-delta baseline:

- `structured_bw_parameterization: score_only_softmax`
- `bw_clean_per_user_enabled: true`
- `bw_actor_advantage_override_mode: gae`
- `bw_clean_per_user_loss: huber`
- `bw_clean_trust_region_enabled: true`
- `bw_clean_trust_region_target_kl: 0.01`
- `bw_clean_trust_region_kl_coef_init: 0.1`
- `bw_clean_trust_region_backtrack_factor: 0.5`
- `bw_clean_trust_region_max_backtracks: 4`
- `bw_actor_branch_parallel_envs: 4`
- `bw_actor_branch_parallel_backend: sync`
- `update_direction_probe_enabled: false`

Files:

- `structured_bw_clean_per_user_beijing_res200_rewardobs_h5.yaml`
- `structured_bw_clean_per_user_beijing_res200_rewardobs_h10.yaml`
- `structured_bw_clean_per_user_beijing_res200_rewardobs_h20.yaml`
- `structured_bw_clean_per_user_beijing_res200_rewardobs_h30.yaml`

Recommended order:

1. Short-screen `h=5,10,20`.
2. If the best result keeps moving upward with longer horizon, also run `h=30`.
3. Use the best clean config for the formal comparison against `Reward-Obs + branch_delta h=30`.
4. Only after the best `h` is fixed should you do `Huber vs KL`.
5. Input ablation stays for later.

Example short-screen command:

```powershell
.venv\Scripts\python.exe scripts/train_structured.py `
  --config configs/clean_per_user/structured_bw_clean_per_user_beijing_res200_rewardobs_h10.yaml `
  --run_dir runs/structured_clean/h10_short `
  --updates 10 `
  --num_envs 8 `
  --vec_backend sync `
  --device cuda
```

Example formal-compare command:

```powershell
.venv\Scripts\python.exe scripts/train_structured.py `
  --config configs/clean_per_user/structured_bw_clean_per_user_beijing_res200_rewardobs_h10.yaml `
  --run_dir runs/structured_clean/h10_formal `
  --updates 60 `
  --num_envs 8 `
  --vec_backend sync `
  --device cuda
```
