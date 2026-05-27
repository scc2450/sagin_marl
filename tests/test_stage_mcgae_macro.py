from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from scripts.train_stage_mcgae import _stage_gae_from_mc_targets


def _batch(*, durations, terminated=None, truncated=None):
    durations_np = np.asarray(durations, dtype=np.int64)
    n = int(durations_np.size)
    if terminated is None:
        terminated = [False] * n
    if truncated is None:
        truncated = [False] * n
    # One env, BW stage.  Transition gaps match duration in primitive env steps.
    transition_indices = []
    cur = 2
    for d in durations_np.tolist():
        transition_indices.append(cur)
        cur += int(d) * 3
    return SimpleNamespace(
        num_samples=n,
        transition_indices=np.asarray(transition_indices, dtype=np.int64),
        env_indices=np.zeros((n,), dtype=np.int64),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
        duration=torch.as_tensor(durations_np, dtype=torch.long),
    )


def _manual_macro_gae(mc, values, durations, terminated, truncated, gamma, lam):
    mc_t = torch.as_tensor(mc, dtype=torch.float32)
    value_t = torch.as_tensor(values, dtype=torch.float32)
    adv = torch.zeros_like(mc_t)
    ret = torch.zeros_like(mc_t)
    next_adv = torch.tensor(0.0)
    next_value = torch.tensor(0.0)
    next_mc = torch.tensor(0.0)
    have_next = False
    for pos in range(len(mc) - 1, -1, -1):
        ended = bool(terminated[pos] or truncated[pos])
        gamma_d = float(gamma) ** int(durations[pos])
        if have_next and not ended:
            collapsed_reward = mc_t[pos] - gamma_d * next_mc
            bootstrap_value = next_value
            bootstrap_adv = next_adv
        else:
            collapsed_reward = mc_t[pos]
            bootstrap_value = torch.tensor(0.0)
            bootstrap_adv = torch.tensor(0.0)
        delta = collapsed_reward + gamma_d * bootstrap_value - value_t[pos]
        adv[pos] = delta + gamma_d * float(lam) * bootstrap_adv
        ret[pos] = adv[pos] + value_t[pos]
        next_adv = adv[pos]
        next_value = value_t[pos]
        next_mc = mc_t[pos]
        have_next = True
    return ret, adv


def test_macro_gae_lambda_one_matches_mc_residual_with_variable_duration():
    gamma = 0.9
    learner = SimpleNamespace(gamma=gamma, gae_lambda=1.0, cfg=SimpleNamespace(stage_mcgae_macro_advantage_mode="macro_gae"))
    durations = [5, 2, 4]
    terminated = [False, False, True]
    values = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float32)
    mc = torch.tensor([5.0, 3.0, 1.25], dtype=torch.float32)

    ret, adv, out_values = _stage_gae_from_mc_targets(
        learner,
        stage_id=2,
        stage_batch=_batch(durations=durations, terminated=terminated),
        mc_target=mc,
        device=torch.device("cpu"),
        stage_values=values,
    )

    torch.testing.assert_close(out_values, values)
    torch.testing.assert_close(ret, mc)
    torch.testing.assert_close(adv, mc - values)


def test_macro_gae_lambda_less_than_one_uses_duration_aware_bootstrap():
    gamma = 0.9
    lam = 0.5
    learner = SimpleNamespace(gamma=gamma, gae_lambda=lam, cfg=SimpleNamespace(stage_mcgae_macro_advantage_mode="macro_gae"))
    durations = [5, 2, 4]
    terminated = [False, False, True]
    truncated = [False, False, False]
    values = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float32)
    mc = torch.tensor([5.0, 3.0, 1.25], dtype=torch.float32)

    ret, adv, _ = _stage_gae_from_mc_targets(
        learner,
        stage_id=2,
        stage_batch=_batch(durations=durations, terminated=terminated, truncated=truncated),
        mc_target=mc,
        device=torch.device("cpu"),
        stage_values=values,
    )
    expected_ret, expected_adv = _manual_macro_gae(mc, values, durations, terminated, truncated, gamma, lam)

    torch.testing.assert_close(ret, expected_ret)
    torch.testing.assert_close(adv, expected_adv)
    assert not torch.allclose(adv, mc - values)


def test_macro_mc_residual_mode_uses_direct_mc_minus_value():
    learner = SimpleNamespace(gamma=0.9, gae_lambda=0.5, cfg=SimpleNamespace(stage_mcgae_macro_advantage_mode="mc_residual"))
    values = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float32)
    mc = torch.tensor([5.0, 3.0, 1.25], dtype=torch.float32)

    ret, adv, _ = _stage_gae_from_mc_targets(
        learner,
        stage_id=2,
        stage_batch=_batch(durations=[5, 2, 4], terminated=[False, False, True]),
        mc_target=mc,
        device=torch.device("cpu"),
        stage_values=values,
    )

    torch.testing.assert_close(ret, mc)
    torch.testing.assert_close(adv, mc - values)


def test_dense_k1_lambda_one_still_matches_mc_residual():
    learner = SimpleNamespace(gamma=0.95, gae_lambda=1.0)
    values = torch.tensor([0.0, 0.5, -0.25], dtype=torch.float32)
    mc = torch.tensor([2.0, 1.0, 0.25], dtype=torch.float32)

    ret, adv, _ = _stage_gae_from_mc_targets(
        learner,
        stage_id=2,
        stage_batch=_batch(durations=[1, 1, 1], terminated=[False, False, True]),
        mc_target=mc,
        device=torch.device("cpu"),
        stage_values=values,
    )

    torch.testing.assert_close(ret, mc)
    torch.testing.assert_close(adv, mc - values)


def test_dense_k1_mc_residual_mode_bypasses_gae():
    learner = SimpleNamespace(gamma=0.95, gae_lambda=0.5, cfg=SimpleNamespace(stage_mcgae_bw_advantage_mode="mc_residual"))
    values = torch.tensor([0.0, 0.5, -0.25], dtype=torch.float32)
    mc = torch.tensor([2.0, 1.0, 0.25], dtype=torch.float32)

    ret, adv, _ = _stage_gae_from_mc_targets(
        learner,
        stage_id=2,
        stage_batch=_batch(durations=[1, 1, 1], terminated=[False, False, True]),
        mc_target=mc,
        device=torch.device("cpu"),
        stage_values=values,
    )

    torch.testing.assert_close(ret, mc)
    torch.testing.assert_close(adv, mc - values)


def test_stage_specific_mode_overrides_macro_fallback():
    learner = SimpleNamespace(
        gamma=0.9,
        gae_lambda=0.5,
        cfg=SimpleNamespace(stage_mcgae_macro_advantage_mode="mc_residual", stage_mcgae_bw_advantage_mode="macro_gae"),
    )
    values = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float32)
    mc = torch.tensor([5.0, 3.0, 1.25], dtype=torch.float32)
    durations = [5, 2, 4]
    terminated = [False, False, True]

    ret, adv, _ = _stage_gae_from_mc_targets(
        learner,
        stage_id=2,
        stage_batch=_batch(durations=durations, terminated=terminated),
        mc_target=mc,
        device=torch.device("cpu"),
        stage_values=values,
    )
    expected_ret, expected_adv = _manual_macro_gae(
        mc,
        values,
        durations,
        terminated,
        [False, False, False],
        0.9,
        0.5,
    )

    torch.testing.assert_close(ret, expected_ret)
    torch.testing.assert_close(adv, expected_adv)
