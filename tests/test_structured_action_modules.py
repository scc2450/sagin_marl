from __future__ import annotations

import torch
import torch.nn as nn

from sagin_marl.rl import (
    structured_accel_actor_schema as accel_schema,
    structured_bw_actor_schema as bw_schema,
    structured_sat_actor_schema as sat_schema,
)
from sagin_marl.rl.structured_actor import (
    AccelPolicy,
    BwPolicy,
    SatSubsetPolicy,
    StructuredActor,
    _subset_member_tensor,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_types import LocalAccelState, LocalBwState, LocalSatState
from tests.structured_test_utils import make_structured_test_cfg


def _sat_state(row_count: int = 2, sat_count: int = 4) -> LocalSatState:
    members = torch.tensor([[-1], [0], [1], [2], [3]], dtype=torch.long)
    return LocalSatState(
        ego_features=torch.randn(row_count, sat_schema.SAT_EGO_DIM),
        demand_features=torch.randn(row_count, sat_schema.SAT_DEMAND_DIM),
        role_features=torch.randn(row_count, sat_schema.SAT_ROLE_DIM),
        sat_tokens=torch.randn(row_count, sat_count, sat_schema.SAT_TOKEN_DIM),
        sat_mask=torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.bool),
        sat_valid_mask=torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool),
        subset_members=members,
        subset_mask=torch.tensor([[0, 1, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=torch.bool),
        candidate_sat_ids=torch.arange(sat_count, dtype=torch.long).view(1, sat_count).expand(row_count, -1),
    )


def _bw_state(row_count: int = 3, gu_count: int = 4) -> LocalBwState:
    return LocalBwState(
        ego_features=torch.randn(row_count, bw_schema.BW_EGO_DIM),
        selected_sat_tokens=torch.randn(row_count, 2, bw_schema.BW_SAT_TOKEN_DIM),
        selected_sat_mask=torch.tensor([[1, 1], [1, 0], [0, 0]], dtype=torch.bool),
        gu_tokens=torch.randn(row_count, gu_count, bw_schema.BW_GU_TOKEN_DIM),
        gu_mask=torch.ones(row_count, gu_count, dtype=torch.bool),
        bw_valid_mask=torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            dtype=torch.bool,
        ),
    )


def _accel_state(row_count: int = 2, gu_count: int = 4, peer_count: int = 1, sat_count: int = 3) -> LocalAccelState:
    return LocalAccelState(
        ego_features=torch.randn(row_count, accel_schema.ACCEL_EGO_DIM),
        ego_cell=torch.randn(row_count, accel_schema.ACCEL_CELL_DIM),
        gu_tokens=torch.randn(row_count, gu_count, accel_schema.ACCEL_GU_TOKEN_DIM),
        gu_mask=torch.ones(row_count, gu_count, dtype=torch.bool),
        peer_tokens=torch.randn(row_count, peer_count, accel_schema.ACCEL_PEER_TOKEN_DIM),
        peer_mask=torch.ones(row_count, peer_count, dtype=torch.bool),
        sat_tokens=torch.randn(row_count, sat_count, accel_schema.ACCEL_SAT_TOKEN_DIM),
        sat_mask=torch.ones(row_count, sat_count, dtype=torch.bool),
    )


def test_sat_subset_members_are_canonical():
    members, sizes = _subset_member_tensor(4, 2, torch.device("cpu"))

    assert tuple(members[0].tolist()) == (-1, -1)
    assert int(sizes[0].item()) == 0
    assert tuple(members[1].tolist()) == (0, -1)
    assert tuple(members[-1].tolist()) == (2, 3)


def test_structured_raw_input_layernorm_is_explicitly_configured() -> None:
    accel = AccelPolicy(
        ego_dim=accel_schema.ACCEL_EGO_DIM,
        cell_dim=accel_schema.ACCEL_CELL_DIM,
        gu_token_dim=accel_schema.ACCEL_GU_TOKEN_DIM,
        peer_token_dim=accel_schema.ACCEL_PEER_TOKEN_DIM,
        sat_token_dim=accel_schema.ACCEL_SAT_TOKEN_DIM,
        hidden_dim=32,
        embed_dim=16,
    )
    sat = SatSubsetPolicy(
        hidden_dim=32,
        embed_dim=16,
        sat_action_select_k=1,
        per_uav_visible_sat_token_max=4,
    )
    bw = BwPolicy(hidden_dim=32, embed_dim=16, down_query_count=2, num_competition_layers=1, num_heads=4)

    assert isinstance(accel.ego_norm, nn.Identity)
    assert isinstance(accel.gu_norm, nn.Identity)
    assert isinstance(sat.ego_input_norm, nn.Identity)
    assert isinstance(sat.sat_input_norm, nn.Identity)
    assert isinstance(bw.ego_input_norm, nn.Identity)
    assert isinstance(bw.gu_input_norm, nn.Identity)
    assert isinstance(bw.competition_blocks[0].norm_attn, nn.LayerNorm)
    assert isinstance(bw.competition_blocks[0].norm_ffn, nn.LayerNorm)

    bw_norm = BwPolicy(
        hidden_dim=32,
        embed_dim=16,
        down_query_count=2,
        num_competition_layers=1,
        num_heads=4,
        input_norm_enabled=True,
    )
    assert isinstance(bw_norm.ego_input_norm, nn.LayerNorm)
    assert isinstance(bw_norm.gu_input_norm, nn.LayerNorm)


def test_structured_factory_does_not_reuse_legacy_input_norm_switch_for_schema_networks() -> None:
    cfg = make_structured_test_cfg()
    cfg.input_norm_enabled = True
    cfg.structured_actor_input_norm_enabled = False
    cfg.structured_critic_input_norm_enabled = False

    bundle = build_structured_modules_from_config(cfg, hidden_dim=32, embed_dim=16)

    assert isinstance(bundle.actor.accel_policy.ego_norm, nn.Identity)
    assert isinstance(bundle.actor.sat_subset_policy.ego_input_norm, nn.Identity)
    assert isinstance(bundle.actor.bw_policy.ego_input_norm, nn.Identity)
    assert bundle.critic is not None
    assert isinstance(bundle.critic.uav_input_norm, nn.Identity)


def test_sat_subset_policy_emits_legal_subset():
    torch.manual_seed(0)
    state = _sat_state()
    policy = SatSubsetPolicy(
        hidden_dim=32,
        embed_dim=16,
        sat_action_select_k=1,
        per_uav_visible_sat_token_max=4,
    )

    out = policy(state, deterministic=True)

    assert out.subset_index.shape == (2,)
    assert out.selected_sat_indices.shape == (2, 1)
    assert torch.all(state.subset_mask.gather(1, out.subset_index.view(-1, 1)).squeeze(1))
    assert torch.isfinite(out.logprob).all()
    assert torch.isfinite(out.entropy).all()


def test_bw_policy_outputs_full_g_masked_dirichlet_simplex():
    torch.manual_seed(1)
    state = _bw_state()
    policy = BwPolicy(hidden_dim=32, embed_dim=16, down_query_count=2, num_competition_layers=1, num_heads=4)

    det = policy(state, deterministic=True)
    sample = policy(state, deterministic=False)

    for out in (det, sample):
        assert out.action.shape == (3, 4)
        assert out.logprob.shape == (3,)
        assert out.entropy.shape == (3,)
        torch.testing.assert_close(out.action[0, state.bw_valid_mask[0]].sum(), torch.tensor(1.0), atol=1.0e-5, rtol=1.0e-5)
        torch.testing.assert_close(out.action[1], state.bw_valid_mask[1].float())
        torch.testing.assert_close(out.action[2], torch.zeros(4))
        torch.testing.assert_close(out.action.masked_select(~state.bw_valid_mask), torch.zeros(9))
        assert torch.equal(out.latent_count, torch.tensor([1, 0, 0]))
        assert torch.isfinite(out.logprob).all()
        assert torch.isfinite(out.entropy).all()


def test_bw_policy_evaluate_actions_uses_per_latent_objective_scalar():
    torch.manual_seed(2)
    state = _bw_state()
    policy = BwPolicy(hidden_dim=32, embed_dim=16, down_query_count=2, num_competition_layers=1, num_heads=4)
    det = policy(state, deterministic=True)

    evaluated = policy.evaluate_actions(state, det.action)

    torch.testing.assert_close(evaluated.action, det.action)
    denom = torch.clamp(evaluated.valid_count.float() - 1.0, min=1.0)
    torch.testing.assert_close(evaluated.logprob, evaluated.logprob_raw / denom)
    torch.testing.assert_close(evaluated.entropy, evaluated.entropy_raw / denom)


def test_structured_actor_wires_redesigned_subpolicies():
    torch.manual_seed(3)
    actor = StructuredActor(
        AccelPolicy(
            ego_dim=accel_schema.ACCEL_EGO_DIM,
            cell_dim=accel_schema.ACCEL_CELL_DIM,
            gu_token_dim=accel_schema.ACCEL_GU_TOKEN_DIM,
            peer_token_dim=accel_schema.ACCEL_PEER_TOKEN_DIM,
            sat_token_dim=accel_schema.ACCEL_SAT_TOKEN_DIM,
            hidden_dim=32,
            embed_dim=16,
            gu_query_count=2,
            peer_query_count=1,
            sat_query_count=1,
        ),
        SatSubsetPolicy(
            hidden_dim=32,
            embed_dim=16,
            sat_action_select_k=1,
            per_uav_visible_sat_token_max=4,
            sat_competition_layers=1,
            sat_attention_heads=4,
        ),
        BwPolicy(
            hidden_dim=32,
            embed_dim=16,
            down_query_count=2,
            num_competition_layers=1,
            num_heads=4,
        ),
    )

    accel_out = actor.act_accel(_accel_state(), deterministic=True)
    sat_out = actor.act_sat(_sat_state(), deterministic=True)
    bw_out = actor.act_bw(_bw_state(), deterministic=True)

    assert accel_out.action.shape == (2, 2)
    assert sat_out.subset_index.shape == (2,)
    assert bw_out.action.shape == (3, 4)
