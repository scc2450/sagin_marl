from __future__ import annotations

import torch

from sagin_marl.rl import structured_critic_schema as schema
from sagin_marl.rl.structured_critic import StructuredCritic
from sagin_marl.rl.structured_types import StructuredWorldState


def _world(batch: int = 2) -> StructuredWorldState:
    torch.manual_seed(123)
    num_uav = 3
    num_gu = 4
    num_sat = 3
    gu_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)[:batch]
    sat_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)[:batch]
    uav_gu_mask = gu_mask[:, None, :].expand(batch, num_uav, num_gu).clone()
    uav_sat_mask = sat_mask[:, None, :].expand(batch, num_uav, num_sat).clone()
    uav_uav_mask = ~torch.eye(num_uav, dtype=torch.bool).view(1, num_uav, num_uav).expand(batch, -1, -1)
    return StructuredWorldState(
        uav_nodes=torch.randn(batch, num_uav, schema.CRITIC_UAV_NODE_DIM),
        gu_nodes=torch.randn(batch, num_gu, schema.CRITIC_GU_NODE_DIM),
        sat_nodes=torch.randn(batch, num_sat, schema.CRITIC_SAT_NODE_DIM),
        sat_ids=torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.long)[:batch],
        uav_gu_edges=torch.randn(batch, num_uav, num_gu, schema.CRITIC_UAV_GU_EDGE_DIM),
        uav_sat_edges=torch.randn(batch, num_uav, num_sat, schema.CRITIC_UAV_SAT_EDGE_DIM),
        uav_uav_edges=torch.randn(batch, num_uav, num_uav, schema.CRITIC_UAV_UAV_EDGE_DIM),
        global_scalars=torch.randn(batch, schema.CRITIC_GLOBAL_SCALAR_DIM),
        gu_mask=gu_mask,
        sat_mask=sat_mask,
        uav_gu_mask=uav_gu_mask,
        uav_sat_mask=uav_sat_mask,
        uav_uav_mask=uav_uav_mask,
        stage_id=torch.tensor([schema.CRITIC_STAGE_ACCEL, schema.CRITIC_STAGE_BW], dtype=torch.long)[:batch],
    )


def _critic() -> StructuredCritic:
    torch.manual_seed(321)
    return StructuredCritic(
        uav_dim=schema.CRITIC_UAV_NODE_DIM,
        gu_dim=schema.CRITIC_GU_NODE_DIM,
        sat_dim=schema.CRITIC_SAT_NODE_DIM,
        uav_gu_edge_dim=schema.CRITIC_UAV_GU_EDGE_DIM,
        uav_sat_edge_dim=schema.CRITIC_UAV_SAT_EDGE_DIM,
        uav_uav_edge_dim=schema.CRITIC_UAV_UAV_EDGE_DIM,
        hidden_dim=64,
        embed_dim=schema.CRITIC_EMBED_DIM,
        edge_embed_dim=schema.CRITIC_EDGE_EMBED_DIM,
        global_embed_dim=schema.CRITIC_GLOBAL_EMBED_DIM,
        system_token_dim=schema.CRITIC_SYSTEM_TOKEN_DIM,
        message_layers=1,
        value_head_hidden_dim=64,
    )


def _global_only_critic() -> StructuredCritic:
    torch.manual_seed(321)
    return StructuredCritic(
        uav_dim=schema.CRITIC_UAV_NODE_DIM,
        gu_dim=schema.CRITIC_GU_NODE_DIM,
        sat_dim=schema.CRITIC_SAT_NODE_DIM,
        uav_gu_edge_dim=schema.CRITIC_UAV_GU_EDGE_DIM,
        uav_sat_edge_dim=schema.CRITIC_UAV_SAT_EDGE_DIM,
        uav_uav_edge_dim=schema.CRITIC_UAV_UAV_EDGE_DIM,
        hidden_dim=64,
        embed_dim=schema.CRITIC_EMBED_DIM,
        edge_embed_dim=schema.CRITIC_EDGE_EMBED_DIM,
        global_embed_dim=schema.CRITIC_GLOBAL_EMBED_DIM,
        system_token_dim=schema.CRITIC_SYSTEM_TOKEN_DIM,
        message_layers=1,
        value_head_hidden_dim=64,
        value_mode="global_only",
    )


def _global_linear_critic() -> StructuredCritic:
    torch.manual_seed(321)
    return StructuredCritic(
        uav_dim=schema.CRITIC_UAV_NODE_DIM,
        gu_dim=schema.CRITIC_GU_NODE_DIM,
        sat_dim=schema.CRITIC_SAT_NODE_DIM,
        uav_gu_edge_dim=schema.CRITIC_UAV_GU_EDGE_DIM,
        uav_sat_edge_dim=schema.CRITIC_UAV_SAT_EDGE_DIM,
        uav_uav_edge_dim=schema.CRITIC_UAV_UAV_EDGE_DIM,
        hidden_dim=64,
        embed_dim=schema.CRITIC_EMBED_DIM,
        edge_embed_dim=schema.CRITIC_EDGE_EMBED_DIM,
        global_embed_dim=schema.CRITIC_GLOBAL_EMBED_DIM,
        system_token_dim=schema.CRITIC_SYSTEM_TOKEN_DIM,
        message_layers=1,
        value_head_hidden_dim=64,
        value_mode="global_linear",
    )


def test_structured_critic_system_readout_shapes_and_head_inputs() -> None:
    critic = _critic()
    world = _world()
    out = critic(world)
    assert set(out) == {"accel", "sat", "bw"}
    for value in out.values():
        assert value.shape == (2,)
        assert torch.isfinite(value).all()
    assert critic.value_accel_head[0].in_features == schema.CRITIC_SYSTEM_TOKEN_DIM
    assert critic.value_sat_head[0].in_features == schema.CRITIC_SYSTEM_TOKEN_DIM
    assert critic.value_bw_head[0].in_features == schema.CRITIC_SYSTEM_TOKEN_DIM


def test_masked_padded_sat_token_does_not_affect_system_value() -> None:
    critic = _critic()
    critic.eval()
    world = _world()
    baseline = critic(world)
    perturbed = StructuredWorldState(
        uav_nodes=world.uav_nodes,
        gu_nodes=world.gu_nodes,
        sat_nodes=world.sat_nodes.clone(),
        sat_ids=world.sat_ids,
        uav_gu_edges=world.uav_gu_edges,
        uav_sat_edges=world.uav_sat_edges.clone(),
        uav_uav_edges=world.uav_uav_edges,
        global_scalars=world.global_scalars,
        gu_mask=world.gu_mask,
        sat_mask=world.sat_mask,
        uav_gu_mask=world.uav_gu_mask,
        uav_sat_mask=world.uav_sat_mask,
        uav_uav_mask=world.uav_uav_mask,
        stage_id=world.stage_id,
    )
    perturbed.sat_nodes[~world.sat_mask] += 1000.0
    perturbed.uav_sat_edges[~world.uav_sat_mask] -= 1000.0
    changed = critic(perturbed)
    for key in baseline:
        torch.testing.assert_close(changed[key], baseline[key], atol=1.0e-6, rtol=1.0e-6)


def test_system_token_is_sensitive_to_global_scalars() -> None:
    critic = _critic()
    critic.eval()
    world = _world()
    shifted = StructuredWorldState(
        uav_nodes=world.uav_nodes,
        gu_nodes=world.gu_nodes,
        sat_nodes=world.sat_nodes,
        sat_ids=world.sat_ids,
        uav_gu_edges=world.uav_gu_edges,
        uav_sat_edges=world.uav_sat_edges,
        uav_uav_edges=world.uav_uav_edges,
        global_scalars=world.global_scalars.clone(),
        gu_mask=world.gu_mask,
        sat_mask=world.sat_mask,
        uav_gu_mask=world.uav_gu_mask,
        uav_sat_mask=world.uav_sat_mask,
        uav_uav_mask=world.uav_uav_mask,
        stage_id=world.stage_id,
    )
    shifted.global_scalars[:, schema.GLOBAL_TOTAL_GU_QUEUE_STEPS] += 10.0
    context_delta = (critic._system_context(shifted) - critic._system_context(world)).abs().max()
    assert context_delta.item() > 1.0e-6


def test_global_only_critic_ignores_tokens_edges_and_uses_global_scalars() -> None:
    critic = _global_only_critic()
    critic.eval()
    world = _world()
    baseline = critic(world)
    perturbed = StructuredWorldState(
        uav_nodes=world.uav_nodes + 1000.0,
        gu_nodes=world.gu_nodes - 1000.0,
        sat_nodes=world.sat_nodes + 500.0,
        sat_ids=world.sat_ids,
        uav_gu_edges=world.uav_gu_edges + 250.0,
        uav_sat_edges=world.uav_sat_edges - 250.0,
        uav_uav_edges=world.uav_uav_edges + 125.0,
        global_scalars=world.global_scalars,
        gu_mask=~world.gu_mask,
        sat_mask=~world.sat_mask,
        uav_gu_mask=~world.uav_gu_mask,
        uav_sat_mask=~world.uav_sat_mask,
        uav_uav_mask=~world.uav_uav_mask,
        stage_id=world.stage_id,
    )
    changed_tokens = critic(perturbed)
    for key in baseline:
        torch.testing.assert_close(changed_tokens[key], baseline[key], atol=1.0e-6, rtol=1.0e-6)

    shifted_global = StructuredWorldState(
        uav_nodes=world.uav_nodes,
        gu_nodes=world.gu_nodes,
        sat_nodes=world.sat_nodes,
        sat_ids=world.sat_ids,
        uav_gu_edges=world.uav_gu_edges,
        uav_sat_edges=world.uav_sat_edges,
        uav_uav_edges=world.uav_uav_edges,
        global_scalars=world.global_scalars.clone(),
        gu_mask=world.gu_mask,
        sat_mask=world.sat_mask,
        uav_gu_mask=world.uav_gu_mask,
        uav_sat_mask=world.uav_sat_mask,
        uav_uav_mask=world.uav_uav_mask,
        stage_id=world.stage_id,
    )
    shifted_global.global_scalars[:, schema.GLOBAL_TOTAL_PREFIX_WEIGHTED_WORKLOAD_STEPS] += 10.0
    shifted_output = critic(shifted_global)
    assert any((shifted_output[key] - baseline[key]).abs().max().item() > 1.0e-6 for key in baseline)


def test_global_linear_critic_closed_form_fit_and_running_decay() -> None:
    critic = _global_linear_critic()
    world = _world()
    target = -3.0 + 2.0 * world.global_scalars[:, schema.GLOBAL_TOTAL_GU_QUEUE_STEPS]
    summary = critic.fit_global_linear("bw", world, target, ridge=1.0e-8, decay=0.0)
    pred = critic.value_bw(world)
    torch.testing.assert_close(pred, target, atol=1.0e-4, rtol=1.0e-4)
    assert summary["ev"] > 0.999

    old_coeff = critic.global_linear_coeff[2].clone()
    new_target = target + 10.0
    critic.fit_global_linear("bw", world, new_target, ridge=1.0e-8, decay=0.9)
    assert not torch.allclose(critic.global_linear_coeff[2], old_coeff)
    decayed_pred = critic.value_bw(world)
    assert (decayed_pred - pred).abs().mean().item() < 10.0
