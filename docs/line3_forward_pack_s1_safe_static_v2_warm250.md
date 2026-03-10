# Line 3 Forward Pack for `s1_safe_static_v2_warm250`

Forward-ready pack for external line-3 review.

This `md` is a forwarding excerpt, not a replacement for the original `.py` files.

Recommended files to send:

- this `md`
- `sagin_marl/env/sagin_env.py`
- `sagin_marl/rl/policy.py`
- `sagin_marl/rl/critic.py`
- `runs/ablation_followup/s1_safe_static_v2_warm250/config.yaml`
- `runs/ablation_followup/s1_safe_static_v2_warm250/obs_sample_eval_ep18_uav0.json`
- `runs/ablation_followup/s1_safe_static_v2_warm250/debug_eval_ep18_seed43000base.csv`
- `runs/ablation_followup/s1_safe_static_v2_warm250/debug_eval_ep09_seed43000base.csv`

Notes:

- `obs_sample_eval_ep18_uav0.json` and `debug_eval_ep18_seed43000base.csv` are not either/or.
- Best option: send both.
- If only one can be sent:
  - for observation representation, send `obs_sample_eval_ep18_uav0.json`
  - for failure-process diagnosis, send `debug_eval_ep18_seed43000base.csv`

## 1. `sagin_env.py`

Source: `sagin_marl/env/sagin_env.py`

### `_build_spaces()`

```python
    def _build_spaces(self) -> None:
        cfg = self.cfg
        self._obs_space = gym.spaces.Dict(
            {
                "own": gym.spaces.Box(-np.inf, np.inf, shape=(self.own_dim,), dtype=np.float32),
                "users": gym.spaces.Box(-np.inf, np.inf, shape=(cfg.users_obs_max, self.user_dim), dtype=np.float32),
                "users_mask": gym.spaces.Box(0.0, 1.0, shape=(cfg.users_obs_max,), dtype=np.float32),
                "sats": gym.spaces.Box(-np.inf, np.inf, shape=(cfg.sats_obs_max, self.sat_dim), dtype=np.float32),
                "sats_mask": gym.spaces.Box(0.0, 1.0, shape=(cfg.sats_obs_max,), dtype=np.float32),
                "nbrs": gym.spaces.Box(-np.inf, np.inf, shape=(cfg.nbrs_obs_max, self.nbr_dim), dtype=np.float32),
                "nbrs_mask": gym.spaces.Box(0.0, 1.0, shape=(cfg.nbrs_obs_max,), dtype=np.float32),
            }
        )

        self._act_space = gym.spaces.Dict(
            {
                "accel": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "bw_logits": gym.spaces.Box(
                    -cfg.bw_logit_scale, cfg.bw_logit_scale, shape=(cfg.users_obs_max,), dtype=np.float32
                ),
                "sat_logits": gym.spaces.Box(
                    -cfg.sat_logit_scale, cfg.sat_logit_scale, shape=(cfg.sats_obs_max,), dtype=np.float32
                ),
            }
        )
```

### `step()` with termination

```python
    def step(self, actions: Dict[str, Dict]):
        cfg = self.cfg
        step_start = time.perf_counter()
        step_profile = self._empty_step_profile()
        self.global_step = int(getattr(self, "global_step", 0)) + 1
        self.prev_queue_sum = float(
            np.sum(self.gu_queue) + np.sum(self.uav_queue) + np.sum(self.sat_queue)
        )
        self.prev_queue_sum_active = float(np.sum(self.gu_queue) + np.sum(self.uav_queue))
        prev_scale = self._queue_arrival_scale(float(getattr(self, "prev_arrival_sum", 0.0)))
        self.prev_q_norm_active = float(np.clip(self.prev_queue_sum_active / prev_scale, 0.0, 1.0))
        self.prev_centroid_dist_mean = self._compute_centroid_stats()[1]
        if cfg.num_gu > 0:
            d2d = np.linalg.norm(self.gu_pos - self.uav_pos[:, None, :], axis=2)
            self.prev_d_min = float(np.min(d2d))
        else:
            self.prev_d_min = 0.0
        profile_start = time.perf_counter()
        self._apply_uav_dynamics(actions)
        step_profile["dynamics_time_sec"] = time.perf_counter() - profile_start

        self.prev_association = self.last_association.copy()
        profile_start = time.perf_counter()
        sat_pos, sat_vel = self._get_orbit_states()
        visible = self._visible_sats_sorted(sat_pos)
        step_profile["orbit_visible_time_sec"] = time.perf_counter() - profile_start

        profile_start = time.perf_counter()
        assoc = self._associate_users()
        candidate_lists = self._build_candidate_users(assoc)
        access_rates, eta = self._compute_access_rates(assoc, candidate_lists, actions)
        step_profile["assoc_access_time_sec"] = time.perf_counter() - profile_start

        profile_start = time.perf_counter()
        gu_outflow = self._update_gu_queues(access_rates, assoc)
        sat_selection = self._select_satellites(sat_pos, sat_vel, actions, visible)
        self._update_energy(sat_selection)
        rate_matrix, sat_loads = self._compute_backhaul_rates(sat_pos, sat_vel, sat_selection)
        outflow_matrix = self._update_uav_queues(gu_outflow, rate_matrix)
        self._update_sat_queues(outflow_matrix)
        step_profile["backhaul_queue_time_sec"] = time.perf_counter() - profile_start

        profile_start = time.perf_counter()
        self._cached_candidates = candidate_lists
        self._cache_sat_obs(sat_pos, sat_vel, visible)
        self._cached_eta = eta
        step_profile["obs_time_sec"] = time.perf_counter() - profile_start

        profile_start = time.perf_counter()
        reward = self._compute_reward()
        collision = bool(getattr(self, "last_reward_parts", {}).get("collision_event", 0.0) > 0.5)
        if not collision:
            collision = self._check_collision()
        self._episode_step_count = int(getattr(self, "_episode_step_count", 0)) + 1
        if collision:
            self._episode_collision_count = int(getattr(self, "_episode_collision_count", 0)) + 1
        energy_depleted = cfg.energy_enabled and np.any(self.uav_energy <= 0.0)
        terminated = collision or energy_depleted
        truncated = self.t >= (cfg.T_steps - 1)

        self.t += 1
        step_profile["reward_time_sec"] = time.perf_counter() - profile_start

        profile_start = time.perf_counter()
        obs = {agent: self._get_obs(idx) for idx, agent in enumerate(self.agents)}
        step_profile["obs_time_sec"] += time.perf_counter() - profile_start
        profile_start = time.perf_counter()
        self._refresh_global_state_cache()
        step_profile["state_time_sec"] = time.perf_counter() - profile_start
        step_profile["step_total_time_sec"] = time.perf_counter() - step_start
        self.last_step_profile = step_profile
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, rewards, terminations, truncations, infos
```

### `_apply_uav_dynamics()`

```python
    def _apply_uav_dynamics(self, actions: Dict[str, Dict]) -> None:
        cfg = self.cfg
        accel = np.zeros((cfg.num_uav, 2), dtype=np.float32)
        use_avoidance = ablation_flag(
            cfg,
            "use_avoidance_layer",
            fallback_attr="avoidance_enabled",
            default=False,
        )
        use_energy_safety = ablation_flag(
            cfg,
            "use_energy_safety_layer",
            fallback_attr="energy_safety_enabled",
            default=False,
        )
        d_alert = cfg.avoidance_alert_factor * cfg.d_safe if use_avoidance else 0.0
        repulse_mode = str(getattr(cfg, "avoidance_repulse_mode", "inverse") or "inverse").strip().lower()
        eta_avoid = float(getattr(self, "avoidance_eta_eff", cfg.avoidance_eta))
        _, _, centroid_transfer_ratio = self._centroid_anneal_state()
        cross_enabled = bool(getattr(cfg, "centroid_cross_anneal_enabled", False))
        if cross_enabled:
            avoid_gain = float(getattr(cfg, "centroid_cross_avoidance_gain", 0.0) or 0.0)
            eta_avoid = eta_avoid * max(0.0, 1.0 + avoid_gain * centroid_transfer_ratio)
            eta_min = max(float(getattr(cfg, "avoidance_eta_min", 0.0) or 0.0), 0.0)
            eta_max_cfg = getattr(cfg, "avoidance_eta_max", None)
            eta_max = float(cfg.a_max) if eta_max_cfg is None else float(eta_max_cfg)
            eta_max = max(eta_min, eta_max)
            eta_avoid = float(np.clip(eta_avoid, eta_min, eta_max))
        self.last_avoidance_eta_exec = float(eta_avoid)
        for i, agent in enumerate(self.agents):
            a = np.array(actions[agent]["accel"], dtype=np.float32)
            a = np.clip(a, -1.0, 1.0) * cfg.a_max
            a_rep = np.zeros(2, dtype=np.float32)
            if use_avoidance and d_alert > 0.0:
                for j in range(cfg.num_uav):
                    if i == j:
                        continue
                    diff = self.uav_pos[i] - self.uav_pos[j]
                    dist = float(np.linalg.norm(diff))
                    if dist < d_alert and dist > 1e-6:
                        direction = diff / dist
                        if repulse_mode == "linear":
                            denom = max(d_alert - cfg.d_safe, 1e-6)
                            strength = float(np.clip((d_alert - dist) / denom, 0.0, 1.0))
                        elif repulse_mode == "quadratic":
                            denom = max(d_alert - cfg.d_safe, 1e-6)
                            base = float(np.clip((d_alert - dist) / denom, 0.0, 1.0))
                            strength = base * base
                        else:
                            strength = (1.0 / dist - 1.0 / d_alert)
                        a_rep += eta_avoid * strength * direction
                if bool(getattr(cfg, "avoidance_repulse_clip", True)):
                    a_rep = np.clip(a_rep, -cfg.a_max, cfg.a_max)
            if cfg.energy_enabled and use_energy_safety:
                v_next = self.uav_vel[i] + a * cfg.tau0
                speed_next = float(np.linalg.norm(v_next))
                est_energy = self.uav_energy[i] - float(self._fly_power(speed_next)) * cfg.tau0
                safe_threshold = cfg.energy_safe_threshold * cfg.uav_energy_init
                if est_energy < safe_threshold:
                    cur_speed = float(np.linalg.norm(self.uav_vel[i]))
                    if cur_speed > 1e-6:
                        direction = self.uav_vel[i] / cur_speed
                    else:
                        a_norm = float(np.linalg.norm(a))
                        if a_norm > 1e-6:
                            direction = a / a_norm
                        else:
                            direction = np.zeros(2, dtype=np.float32)
                    target_delta = cfg.uav_opt_speed - cur_speed
                    a = direction * np.clip(target_delta / max(cfg.tau0, 1e-6), -cfg.a_max, cfg.a_max)
            a = a + a_rep
            a = np.clip(a, -cfg.a_max, cfg.a_max)
            accel[i] = a
        self.last_exec_accel = accel.copy()
        self.uav_vel = np.clip(self.uav_vel + accel * cfg.tau0, -cfg.v_max, cfg.v_max)
        self.uav_pos = self.uav_pos + self.uav_vel * cfg.tau0
        if cfg.boundary_mode == "reflect":
            for i in range(cfg.num_uav):
                for axis in range(2):
                    if self.uav_pos[i, axis] < 0.0:
                        self.uav_pos[i, axis] = -self.uav_pos[i, axis]
                        self.uav_vel[i, axis] = -self.uav_vel[i, axis]
                    elif self.uav_pos[i, axis] > cfg.map_size:
                        self.uav_pos[i, axis] = 2 * cfg.map_size - self.uav_pos[i, axis]
                        self.uav_vel[i, axis] = -self.uav_vel[i, axis]
        self.uav_pos = np.clip(self.uav_pos, 0.0, cfg.map_size)
        self._refresh_uav_cache()
```

### `_compute_reward()`

```python
    def _compute_reward(self) -> float:
        cfg = self.cfg
        use_active_queue_delta = ablation_flag(
            cfg,
            "use_active_queue_delta",
            fallback_attr="queue_delta_use_active",
            default=False,
        )
        use_energy_reward = ablation_flag(cfg, "use_energy_reward", default=cfg.energy_enabled)
        use_reward_tanh = ablation_flag(
            cfg,
            "use_reward_tanh",
            fallback_attr="reward_tanh_enabled",
            default=False,
        )
        use_queue_log_smoothing = ablation_flag(
            cfg,
            "use_queue_log_smoothing",
            default=False,
        )
        queue_penalty_mode = str(getattr(cfg, "queue_penalty_mode", "quadratic") or "quadratic").lower()

        if cfg.energy_enabled:
            p_max = self._energy_scale()
            r_energy = -np.mean(self.last_energy_cost / (p_max + 1e-9))
        else:
            r_energy = 0.0

        q_gu = float(np.sum(self.gu_queue))
        q_uav = float(np.sum(self.uav_queue))
        q_sat = float(np.sum(self.sat_queue))
        q_total = q_gu + q_uav + q_sat
        q_total_active = q_gu + q_uav
        arrival_sum = float(np.sum(self.last_gu_arrival))
        outflow_sum = float(np.sum(self.last_gu_outflow))
        backhaul_sum = float(np.sum(getattr(self, "last_sat_incoming", 0.0)))
        drop_sum = float(np.sum(self.gu_drop)) + float(np.sum(self.uav_drop))
        service_ratio = outflow_sum / (arrival_sum + 1e-9)
        drop_ratio = drop_sum / (arrival_sum + 1e-9)
        service_ratio = float(np.clip(service_ratio, 0.0, 1.0))
        drop_ratio = float(np.clip(drop_ratio, 0.0, 1.0))
        if cfg.num_gu > 0:
            assoc_ratio = float(np.mean(self.last_association >= 0))
        else:
            assoc_ratio = 0.0

        arrival_ref = float(getattr(self, "effective_task_arrival_rate", cfg.task_arrival_rate))
        arrival_scale = max(arrival_ref * float(cfg.num_gu) * float(cfg.tau0), 1e-9)
        service_norm = outflow_sum / arrival_scale
        drop_norm = drop_sum / arrival_scale
        drop_event = 1.0 if drop_sum > 1e-9 else 0.0

        queue_norm_scale = self._queue_arrival_scale(arrival_sum)
        q_norm_active = float(np.clip(q_total_active / queue_norm_scale, 0.0, 1.0))
        prev_q_norm_active = float(getattr(self, "prev_q_norm_active", q_norm_active))
        q_norm_delta = float(prev_q_norm_active - q_norm_active)
        queue_weight = float(cfg.omega_q)
        q_delta_weight = float(cfg.eta_q_delta)
        crash_weight = float(cfg.eta_crash)

        if use_active_queue_delta:
            queue_term = q_norm_active
            queue_delta = float(np.clip(q_norm_delta, -1.0, 1.0))
        else:
            queue_term = q_total
            queue_delta = 0.0

        if cfg.a_max > 0:
            accel_norm2 = float(np.mean(np.sum(self.last_exec_accel**2, axis=1))) / (cfg.a_max**2 + 1e-9)
        else:
            accel_norm2 = 0.0

        centroid_reward, _ = self._compute_centroid_stats()
        _, centroid_eta, _ = self._centroid_anneal_state()
        term_service = cfg.eta_service * service_norm
        term_drop_step = -float(getattr(cfg, "eta_drop_step", 0.0) or 0.0) * drop_event
        term_drop = -cfg.eta_drop * drop_norm + term_drop_step
        term_queue = -queue_weight * queue_term
        term_q_delta = q_delta_weight * queue_delta
        term_centroid = centroid_eta * centroid_reward
        term_accel = -cfg.eta_accel * accel_norm2
        term_energy = cfg.omega_e * r_energy if use_energy_reward else 0.0
        raw_reward = (
            term_service
            + term_drop
            + term_queue
            + term_q_delta
            + term_centroid
            + term_accel
            + term_energy
        )

        collision_now = self._check_collision()
        collision_penalty = -crash_weight if collision_now else 0.0
        battery_penalty = -cfg.eta_batt if (cfg.energy_enabled and np.any(self.uav_energy <= 0.0)) else 0.0
        fail_penalty = collision_penalty + battery_penalty

        reward = raw_reward
        if use_reward_tanh:
            reward = math.tanh(raw_reward)
        reward = reward + fail_penalty

        self.last_reward_parts = {
            "service_ratio": service_ratio,
            "drop_ratio": drop_ratio,
            "drop_sum": drop_sum,
            "queue_total": q_total,
            "queue_total_active": q_total_active,
            "assoc_ratio": assoc_ratio,
            "queue_delta": queue_delta,
            "q_norm_active": q_norm_active,
            "collision_event": 1.0 if collision_now else 0.0,
            "collision_penalty": collision_penalty,
            "fail_penalty": fail_penalty,
            "term_service": term_service,
            "term_drop": term_drop,
            "term_queue": term_queue,
            "term_q_delta": term_q_delta,
            "term_centroid": term_centroid,
            "term_energy": float(term_energy),
            "term_accel": term_accel,
            "reward_raw": raw_reward,
        }
        return float(reward)
```

### `_cache_sat_obs()`

```python
    def _cache_sat_obs(
        self,
        sat_pos: np.ndarray,
        sat_vel: np.ndarray,
        visible: List[List[int]],
    ) -> None:
        cfg = self.cfg
        sat_obs = np.zeros((cfg.num_uav, cfg.sats_obs_max, self.sat_dim), dtype=np.float32)
        sat_mask = np.zeros((cfg.num_uav, cfg.sats_obs_max), dtype=np.float32)
        for u in range(cfg.num_uav):
            for i, l in enumerate(visible[u][: cfg.sats_obs_max]):
                rel_pos = sat_pos[l] - self._uav_ecef(u)
                rel_vel = sat_vel[l] - self._uav_vel_ecef(u)
                nu = self._doppler(u, l, sat_pos, sat_vel)
                d = np.linalg.norm(rel_pos) + 1e-9
                gain = (cfg.speed_of_light / (4.0 * math.pi * cfg.carrier_freq * d)) ** 2
                if cfg.atm_loss_enabled:
                    theta = self._elevation_angle(u, l, sat_pos)
                    atm_loss = channel.atmospheric_loss_db(theta, cfg.atm_loss_db)
                    gain *= 10 ** (-atm_loss / 10.0)
                gain *= cfg.uav_tx_gain * cfg.sat_rx_gain
                snr = channel.snr_linear(cfg.uav_tx_power, gain, cfg.noise_density, cfg.b_sat_total)
                if cfg.doppler_observed and cfg.doppler_atten_enabled:
                    chi = channel.doppler_attenuation(np.array([nu]), cfg.subcarrier_spacing)[0]
                    snr = snr * chi
                sat_obs[u, i, 0:3] = rel_pos / (cfg.r_earth + cfg.sat_height)
                sat_obs[u, i, 3:6] = rel_vel / (cfg.r_earth + cfg.sat_height)
                sat_obs[u, i, 6] = nu / max(cfg.nu_max, 1.0)
                sat_obs[u, i, 7] = channel.spectral_efficiency(snr)
                sat_obs[u, i, 8] = self.sat_queue[l] / cfg.queue_max_sat
                sat_mask[u, i] = 1.0
        self._cached_sat_obs = sat_obs
        self._cached_sat_mask = sat_mask
```

### `_get_obs()`

```python
    def _get_obs(self, u: int) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        own = np.array(
            [
                self.uav_pos[u, 0] / cfg.map_size,
                self.uav_pos[u, 1] / cfg.map_size,
                self.uav_vel[u, 0] / cfg.v_max,
                self.uav_vel[u, 1] / cfg.v_max,
                self.uav_energy[u] / max(cfg.uav_energy_init, 1e-9),
                self.uav_queue[u] / cfg.queue_max_uav,
                self.t / max(cfg.T_steps, 1),
            ],
            dtype=np.float32,
        )

        users = np.zeros((cfg.users_obs_max, self.user_dim), dtype=np.float32)
        users_mask = np.zeros((cfg.users_obs_max,), dtype=np.float32)
        cand = self._cached_candidates[u] if self._cached_candidates else []
        for i, k in enumerate(cand[: cfg.users_obs_max]):
            rel = self.gu_pos[k] - self.uav_pos[u]
            users[i, 0:2] = rel / cfg.map_size
            users[i, 2] = self.gu_queue[k] / cfg.queue_max_gu
            users[i, 3] = self._cached_eta[u, i]
            users[i, 4] = 1.0 if self.prev_association[k] == u else 0.0
            users_mask[i] = 1.0

        sats = self._cached_sat_obs[u].copy()
        sats_mask = self._cached_sat_mask[u].copy()

        nbrs = np.zeros((cfg.nbrs_obs_max, self.nbr_dim), dtype=np.float32)
        nbrs_mask = np.zeros((cfg.nbrs_obs_max,), dtype=np.float32)
        self._ensure_neighbor_cache()
        order = self._cached_uav_neighbor_order[u]
        count = 0
        for idx in order:
            if idx == u:
                continue
            rel_pos = self.uav_pos[idx] - self.uav_pos[u]
            rel_vel = self.uav_vel[idx] - self.uav_vel[u]
            nbrs[count, 0:2] = rel_pos / cfg.map_size
            nbrs[count, 2:4] = rel_vel / cfg.v_max
            nbrs_mask[count] = 1.0
            count += 1
            if count >= cfg.nbrs_obs_max:
                break

        return {
            "own": own,
            "users": users,
            "users_mask": users_mask,
            "sats": sats,
            "sats_mask": sats_mask,
            "nbrs": nbrs,
            "nbrs_mask": nbrs_mask,
        }
```

## 2. `policy.py`

Source: `sagin_marl/rl/policy.py`

### `flatten_obs()`

```python
def flatten_obs(obs: Dict[str, np.ndarray], cfg) -> np.ndarray:
    parts = [
        obs["own"].ravel(),
        obs["users"].ravel(),
        obs["users_mask"].ravel(),
        obs["sats"].ravel(),
        obs["sats_mask"].ravel(),
        obs["nbrs"].ravel(),
        obs["nbrs_mask"].ravel(),
    ]
    return np.concatenate(parts).astype(np.float32)
```

### `ActorNet`

```python
class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.enable_bw = cfg.enable_bw_action
        self.enable_sat = not cfg.fixed_satellite_strategy
        self.bw_scale = float(cfg.bw_logit_scale)
        self.sat_scale = float(cfg.sat_logit_scale)

        self.obs_norm = nn.LayerNorm(obs_dim) if getattr(cfg, "input_norm_enabled", False) else nn.Identity()
        self.fc1 = nn.Linear(obs_dim, cfg.actor_hidden)
        self.fc2 = nn.Linear(cfg.actor_hidden, cfg.actor_hidden)

        self.mu_head = nn.Linear(cfg.actor_hidden, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

        if self.enable_bw:
            self.bw_head = nn.Linear(cfg.actor_hidden, cfg.users_obs_max)
            self.bw_log_std = nn.Parameter(torch.zeros(cfg.users_obs_max))
        else:
            self.bw_head = None
            self.bw_log_std = None

        if self.enable_sat:
            self.sat_head = nn.Linear(cfg.actor_hidden, cfg.sats_obs_max)
            self.sat_log_std = nn.Parameter(torch.zeros(cfg.sats_obs_max))
        else:
            self.sat_head = None
            self.sat_log_std = None

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.obs_norm(obs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        out = {"mu": mu}
        if self.bw_head is not None:
            out["bw_mu"] = self.bw_head(x)
        if self.sat_head is not None:
            out["sat_mu"] = self.sat_head(x)
        return out

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
        out = self.forward(obs)
        mu = out["mu"]
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if deterministic:
            z = mu
        else:
            z = dist.rsample()
        accel = _squash_action(z, scale=1.0)
        logprob = _logprob_from_squashed(dist, accel, scale=1.0)

        bw_logits = None
        sat_logits = None

        if self.enable_bw:
            bw_mu = out["bw_mu"]
            bw_log_std = torch.clamp(self.bw_log_std, -5.0, 2.0)
            bw_std = torch.exp(bw_log_std)
            bw_dist = Normal(bw_mu, bw_std)
            z_bw = bw_mu if deterministic else bw_dist.rsample()
            bw_logits = _squash_action(z_bw, scale=self.bw_scale)
            logprob = logprob + _logprob_from_squashed(bw_dist, bw_logits, scale=self.bw_scale)

        if self.enable_sat:
            sat_mu = out["sat_mu"]
            sat_log_std = torch.clamp(self.sat_log_std, -5.0, 2.0)
            sat_std = torch.exp(sat_log_std)
            sat_dist = Normal(sat_mu, sat_std)
            z_sat = sat_mu if deterministic else sat_dist.rsample()
            sat_logits = _squash_action(z_sat, scale=self.sat_scale)
            logprob = logprob + _logprob_from_squashed(sat_dist, sat_logits, scale=self.sat_scale)

        action = self._concat_actions(accel, bw_logits, sat_logits)
        return PolicyOutput(
            action=action,
            logprob=logprob,
            accel=accel,
            bw_logits=bw_logits,
            sat_logits=sat_logits,
            dist_out=out,
        )

    def evaluate_actions_parts(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        out: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not torch.isfinite(obs).all():
            print("NaN/Inf detected in obs passed to evaluate_actions_parts")
            raise ValueError("obs contains NaN/Inf")
        if out is None:
            out = self.forward(obs)
        accel_action, bw_action, sat_action = self._split_actions(action)

        mu = out["mu"]
        if not torch.isfinite(mu).all():
            print("NaN/Inf detected in actor mu inside evaluate_actions_parts")
            raise ValueError("actor mu contains NaN/Inf")
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        if not torch.isfinite(std).all():
            print("NaN/Inf detected in actor std inside evaluate_actions_parts")
            raise ValueError("actor std contains NaN/Inf")
        dist = Normal(mu, std)
        logprob_parts: Dict[str, torch.Tensor] = {"accel": _logprob_from_squashed(dist, accel_action, scale=1.0)}
        entropy_parts: Dict[str, torch.Tensor] = {"accel": dist.entropy().sum(-1)}

        if self.enable_bw and bw_action is not None:
            bw_mu = out["bw_mu"]
            bw_log_std = torch.clamp(self.bw_log_std, -5.0, 2.0)
            bw_std = torch.exp(bw_log_std)
            bw_dist = Normal(bw_mu, bw_std)
            logprob_parts["bw"] = _logprob_from_squashed(bw_dist, bw_action, scale=self.bw_scale)
            entropy_parts["bw"] = bw_dist.entropy().sum(-1)

        if self.enable_sat and sat_action is not None:
            sat_mu = out["sat_mu"]
            sat_log_std = torch.clamp(self.sat_log_std, -5.0, 2.0)
            sat_std = torch.exp(sat_log_std)
            sat_dist = Normal(sat_mu, sat_std)
            logprob_parts["sat"] = _logprob_from_squashed(sat_dist, sat_action, scale=self.sat_scale)
            entropy_parts["sat"] = sat_dist.entropy().sum(-1)

        return logprob_parts, entropy_parts
```

## 3. `critic.py`

Source: `sagin_marl/rl/critic.py`

```python
class CriticNet(nn.Module):
    def __init__(self, state_dim: int, cfg):
        super().__init__()
        self.state_norm = nn.LayerNorm(state_dim) if getattr(cfg, "input_norm_enabled", False) else nn.Identity()
        self.fc1 = nn.Linear(state_dim, cfg.critic_hidden)
        self.fc2 = nn.Linear(cfg.critic_hidden, cfg.critic_hidden)
        self.v = nn.Linear(cfg.critic_hidden, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.state_norm(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.v(x).squeeze(-1)
```

## 4. Sample files to attach directly

These two files should be sent as real files, not only summarized text:

- `runs/ablation_followup/s1_safe_static_v2_warm250/obs_sample_eval_ep18_uav0.json`
- `runs/ablation_followup/s1_safe_static_v2_warm250/debug_eval_ep18_seed43000base.csv`
- `runs/ablation_followup/s1_safe_static_v2_warm250/debug_eval_ep09_seed43000base.csv` as a normal-episode comparison

If the receiver only wants key windows from the debug CSV, point them to:

- first takeover window: steps `228-232`
- collision window: steps `376-381`
