from __future__ import annotations

import torch

from sagin_marl.rl import structured_sat_actor_schema as schema
from sagin_marl.rl.structured_actor import SatSubsetPolicy, _subset_member_tensor
from sagin_marl.rl.structured_types import LocalSatState


def _synthetic_state(*, row_count: int = 3, sat_count: int = 4, select_k: int = 2) -> LocalSatState:
    members, _sizes = _subset_member_tensor(sat_count, select_k, torch.device("cpu"))
    candidate_ids = torch.arange(row_count * sat_count, dtype=torch.long).reshape(row_count, sat_count) + 10
    sat_valid = torch.tensor(
        [
            [True, True, False, False],
            [False, False, False, False],
            [True, False, True, True],
        ],
        dtype=torch.bool,
    )[:row_count, :sat_count]
    return LocalSatState(
        ego_features=torch.randn(row_count, schema.SAT_EGO_DIM),
        demand_features=torch.randn(row_count, schema.SAT_DEMAND_DIM),
        role_features=torch.linspace(0.0, 1.0, row_count).view(row_count, 1),
        sat_tokens=torch.randn(row_count, sat_count, schema.SAT_TOKEN_DIM),
        sat_mask=torch.ones(row_count, sat_count, dtype=torch.bool),
        sat_valid_mask=sat_valid,
        subset_members=members,
        subset_mask=torch.zeros(row_count, members.shape[0], dtype=torch.bool),
        candidate_sat_ids=candidate_ids,
    )


def test_sat_actor_schema_dims_and_local_state_fields_are_final_names():
    assert schema.SAT_EGO_DIM == len(schema.SAT_EGO_FIELDS) == 12
    assert schema.SAT_DEMAND_DIM == len(schema.SAT_DEMAND_FIELDS) == 8
    assert schema.SAT_ROLE_DIM == 1
    assert schema.SAT_TOKEN_DIM == len(schema.SAT_TOKEN_FIELDS) == 26

    state = _synthetic_state()
    assert state.ego_features.shape[-1] == schema.SAT_EGO_DIM
    assert state.demand_features.shape[-1] == schema.SAT_DEMAND_DIM
    assert state.role_features.shape[-1] == schema.SAT_ROLE_DIM
    assert state.sat_tokens.shape[-1] == schema.SAT_TOKEN_DIM
    assert not hasattr(state, "pair_mask")
    assert not hasattr(state, "pair_members")


def test_sat_subset_table_is_canonical_unique_and_empty_first():
    members, sizes = _subset_member_tensor(4, 2, torch.device("cpu"))
    assert tuple(members[0].tolist()) == (-1, -1)
    assert int(sizes[0].item()) == 0
    seen: set[tuple[int, ...]] = set()
    previous_size = 0
    for row, size_t in zip(members.tolist(), sizes.tolist()):
        size = int(size_t)
        subset = tuple(slot for slot in row if slot >= 0)
        assert size == len(subset)
        assert size >= previous_size
        assert subset == tuple(sorted(subset))
        assert subset not in seen
        seen.add(subset)
        previous_size = size


def test_sat_subset_policy_uses_valid_mask_and_decodes_global_ids():
    torch.manual_seed(7)
    state = _synthetic_state()
    policy = SatSubsetPolicy(
        hidden_dim=16,
        embed_dim=8,
        sat_competition_layers=1,
        sat_attention_heads=2,
        sat_action_select_k=2,
        per_uav_visible_sat_token_max=4,
    )
    out = policy(state, deterministic=True)

    assert out.subset_index.shape == (3,)
    assert out.selected_sat_indices.shape == (3, 2)
    assert torch.equal(out.selected_sat_indices[1], torch.full((2,), -1, dtype=torch.long))
    assert float(out.logprob[1].item()) == 0.0
    assert float(out.entropy[1].item()) == 0.0
    legal_mask = policy._legal_subset_mask(state, out.logits)
    assert bool(legal_mask[1, 0].item())
    assert int(legal_mask[1].sum().item()) == 1
    for row in (0, 2):
        selected = [int(v) for v in out.selected_sat_indices[row].tolist() if int(v) >= 0]
        valid_ids = set(state.candidate_sat_ids[row, state.sat_valid_mask[row]].tolist())
        assert set(selected) <= valid_ids


def test_invalid_sat_token_payload_does_not_change_logits():
    torch.manual_seed(11)
    state = _synthetic_state(row_count=1)
    policy = SatSubsetPolicy(
        hidden_dim=16,
        embed_dim=8,
        sat_competition_layers=1,
        sat_attention_heads=2,
        sat_action_select_k=2,
        per_uav_visible_sat_token_max=4,
    )
    logits_a = policy._compute_logits(state)
    mutated = _synthetic_state(row_count=1)
    mutated.ego_features = state.ego_features.clone()
    mutated.demand_features = state.demand_features.clone()
    mutated.role_features = state.role_features.clone()
    mutated.sat_tokens = state.sat_tokens.clone()
    mutated.sat_tokens[:, 2:, :] = torch.randn_like(mutated.sat_tokens[:, 2:, :]) * 100.0
    mutated.sat_mask = state.sat_mask.clone()
    mutated.sat_valid_mask = state.sat_valid_mask.clone()
    mutated.subset_members = state.subset_members
    mutated.subset_mask = state.subset_mask.clone()
    mutated.candidate_sat_ids = state.candidate_sat_ids.clone()
    logits_b = policy._compute_logits(mutated)
    torch.testing.assert_close(logits_a, logits_b)
