from __future__ import annotations

from dataclasses import dataclass

from sagin_marl.env.config import ensure_structured_accel_actor_config
from sagin_marl.env.structured_batch_env_core import native_module_shape_spec_from_config
from sagin_marl.rl import structured_accel_actor_schema as accel_schema
from sagin_marl.rl import structured_bw_actor_schema as bw_schema
from sagin_marl.rl import structured_critic_schema as critic_schema
from sagin_marl.rl import structured_sat_actor_schema as sat_schema
from sagin_marl.rl.structured_actor import AccelPolicy, BwPolicy, SatSubsetPolicy, StructuredActor
from sagin_marl.rl.structured_critic import StructuredCritic


@dataclass
class StructuredModuleBundle:
    actor: StructuredActor
    critic: StructuredCritic | None


def build_structured_modules_from_config(
    cfg,
    *,
    hidden_dim: int | None = None,
    embed_dim: int | None = None,
    actor_hidden_dim: int | None = None,
    actor_embed_dim: int | None = None,
    critic_hidden_dim: int | None = None,
    critic_embed_dim: int | None = None,
    build_critic: bool = True,
) -> StructuredModuleBundle:
    ensure_structured_accel_actor_config(cfg)
    shared_hidden = 256 if hidden_dim is None else int(hidden_dim)
    shared_embed = 128 if embed_dim is None else int(embed_dim)
    actor_hidden = int(
        getattr(cfg, "actor_hidden", shared_hidden)
        if actor_hidden_dim is None
        else actor_hidden_dim
    )
    actor_embed = int(
        getattr(cfg, "actor_set_embed_dim", shared_embed)
        if actor_embed_dim is None
        else actor_embed_dim
    )
    accel_hidden = int(getattr(cfg, "accel_hidden", 0) or actor_hidden)
    accel_embed = int(getattr(cfg, "accel_embed_dim", 0) or actor_embed)
    sat_hidden = int(getattr(cfg, "sat_hidden", 0) or actor_hidden)
    sat_embed = int(getattr(cfg, "sat_embed_dim", 0) or actor_embed)
    bw_hidden = int(getattr(cfg, "bw_hidden", 0) or actor_hidden)
    bw_embed = int(getattr(cfg, "bw_embed_dim", 0) or actor_embed)
    critic_hidden = int(
        getattr(cfg, "critic_hidden", shared_hidden)
        if critic_hidden_dim is None
        else critic_hidden_dim
    )
    critic_embed = int(
        getattr(cfg, "critic_embed_dim", critic_schema.CRITIC_EMBED_DIM)
        if critic_embed_dim is None
        else critic_embed_dim
    )
    use_actor_input_norm = bool(getattr(cfg, "structured_actor_input_norm_enabled", False))
    use_critic_input_norm = bool(getattr(cfg, "structured_critic_input_norm_enabled", False))
    actor_encoder_layers = int(getattr(cfg, "actor_encoder_mlp_layers", 2) or 2)
    actor_context_layers = int(getattr(cfg, "actor_context_mlp_layers", 2) or 2)
    actor_head_layers = int(getattr(cfg, "actor_head_mlp_layers", 2) or 2)
    accel_encoder_layers = int(getattr(cfg, "accel_encoder_mlp_layers", actor_encoder_layers) or actor_encoder_layers)
    accel_context_layers = int(getattr(cfg, "accel_context_mlp_layers", actor_context_layers) or actor_context_layers)
    accel_head_layers = int(getattr(cfg, "accel_head_mlp_layers", 1) or 1)
    sat_encoder_layers = int(getattr(cfg, "sat_encoder_mlp_layers", actor_encoder_layers) or actor_encoder_layers)
    sat_context_layers = int(getattr(cfg, "sat_context_mlp_layers", actor_context_layers) or actor_context_layers)
    sat_head_layers = int(getattr(cfg, "sat_head_mlp_layers", actor_head_layers) or actor_head_layers)
    bw_encoder_layers = int(getattr(cfg, "bw_encoder_mlp_layers", actor_encoder_layers) or actor_encoder_layers)
    bw_context_layers = int(getattr(cfg, "bw_context_mlp_layers", actor_context_layers) or actor_context_layers)
    bw_head_layers = int(getattr(cfg, "bw_head_mlp_layers", actor_head_layers) or actor_head_layers)
    critic_encoder_layers = int(getattr(cfg, "critic_encoder_mlp_layers", 2) or 2)
    critic_message_mlp_layers = int(getattr(cfg, "critic_message_mlp_layers", 2) or 2)
    critic_value_head_layers = int(getattr(cfg, "critic_value_head_layers", 2) or 2)
    shape = native_module_shape_spec_from_config(cfg)

    accel_policy = AccelPolicy(
        ego_dim=accel_schema.ACCEL_EGO_DIM,
        cell_dim=accel_schema.ACCEL_CELL_DIM,
        gu_token_dim=accel_schema.ACCEL_GU_TOKEN_DIM,
        peer_token_dim=accel_schema.ACCEL_PEER_TOKEN_DIM,
        sat_token_dim=accel_schema.ACCEL_SAT_TOKEN_DIM,
        gu_query_count=int(getattr(cfg, "accel_gu_query_count", accel_schema.ACCEL_GU_QUERY_COUNT)),
        peer_query_count=int(getattr(cfg, "accel_peer_query_count", accel_schema.ACCEL_PEER_QUERY_COUNT)),
        sat_query_count=int(getattr(cfg, "accel_sat_query_count", accel_schema.ACCEL_SAT_QUERY_COUNT)),
        hidden_dim=accel_hidden,
        embed_dim=accel_embed,
        encoder_mlp_layers=accel_encoder_layers,
        context_mlp_layers=accel_context_layers,
        head_mlp_layers=accel_head_layers,
        interaction_layers=int(getattr(cfg, "accel_interaction_layers", 0) or 0),
        attention_heads=int(getattr(cfg, "accel_attention_heads", 4) or 4),
        action_scale=1.0,
        input_norm_enabled=use_actor_input_norm,
    )
    accel_log_std_init = float(getattr(cfg, "accel_log_std_init", 0.0) or 0.0)
    accel_policy.log_std.data.fill_(accel_log_std_init)
    accel_policy.log_std.requires_grad_(bool(getattr(cfg, "accel_log_std_trainable", True)))
    sat_policy = SatSubsetPolicy(
        ego_dim=sat_schema.SAT_EGO_DIM,
        demand_dim=sat_schema.SAT_DEMAND_DIM,
        role_dim=sat_schema.SAT_ROLE_DIM,
        sat_token_dim=sat_schema.SAT_TOKEN_DIM,
        hidden_dim=sat_hidden,
        embed_dim=sat_embed,
        encoder_mlp_layers=sat_encoder_layers,
        context_mlp_layers=sat_context_layers,
        head_mlp_layers=sat_head_layers,
        sat_competition_layers=int(getattr(cfg, "sat_competition_layers", 2) or 2),
        sat_attention_heads=int(getattr(cfg, "sat_attention_heads", 4) or 4),
        sat_action_select_k=int(getattr(cfg, "sat_action_select_k", shape.sat_num_select)),
        per_uav_visible_sat_token_max=int(getattr(cfg, "per_uav_visible_sat_token_max", shape.visible_sats_max)),
        input_norm_enabled=use_actor_input_norm,
    )
    bw_policy = BwPolicy(
        ego_dim=bw_schema.BW_EGO_DIM,
        sat_token_dim=bw_schema.BW_SAT_TOKEN_DIM,
        gu_token_dim=bw_schema.BW_GU_TOKEN_DIM,
        hidden_dim=bw_hidden,
        embed_dim=bw_embed,
        down_query_count=int(getattr(cfg, "bw_down_query_count", 2) or 2),
        num_competition_layers=int(getattr(cfg, "bw_competition_layers", 2) or 2),
        num_heads=int(getattr(cfg, "bw_attention_heads", 4) or 4),
        encoder_mlp_layers=bw_encoder_layers,
        context_mlp_layers=bw_context_layers,
        head_mlp_layers=bw_head_layers,
        tau_min=float(getattr(cfg, "bw_tau_min", 0.5) or 0.5),
        tau_max=float(getattr(cfg, "bw_tau_max", 2.0) or 2.0),
        kappa_min=float(getattr(cfg, "bw_kappa_min", 0.5) or 0.5),
        kappa_max=float(getattr(cfg, "bw_kappa_max", 32.0) or 32.0),
        fixed_tau=getattr(cfg, "bw_fixed_tau", None),
        fixed_kappa=getattr(cfg, "bw_fixed_kappa", None),
        native_dirichlet_diagnostic_mode=str(
            getattr(cfg, "bw_native_dirichlet_diagnostic_mode", "current") or "current"
        ),
        manual_competition_attention_enabled=bool(
            getattr(cfg, "bw_manual_competition_attention_enabled", False)
        ),
        input_norm_enabled=use_actor_input_norm,
    )
    actor = StructuredActor(
        accel_policy=accel_policy,
        sat_subset_policy=sat_policy,
        bw_policy=bw_policy,
    )
    critic = None
    if build_critic:
        critic = StructuredCritic(
            uav_dim=critic_schema.CRITIC_UAV_NODE_DIM,
            gu_dim=critic_schema.CRITIC_GU_NODE_DIM,
            sat_dim=critic_schema.CRITIC_SAT_NODE_DIM,
            uav_gu_edge_dim=critic_schema.CRITIC_UAV_GU_EDGE_DIM,
            uav_sat_edge_dim=critic_schema.CRITIC_UAV_SAT_EDGE_DIM,
            uav_uav_edge_dim=critic_schema.CRITIC_UAV_UAV_EDGE_DIM,
            hidden_dim=critic_hidden,
            embed_dim=critic_embed,
            edge_embed_dim=int(getattr(cfg, "critic_edge_embed_dim", critic_schema.CRITIC_EDGE_EMBED_DIM)),
            global_embed_dim=int(getattr(cfg, "critic_global_embed_dim", critic_schema.CRITIC_GLOBAL_EMBED_DIM)),
            system_token_dim=int(getattr(cfg, "critic_system_token_dim", critic_schema.CRITIC_SYSTEM_TOKEN_DIM)),
            message_layers=int(getattr(cfg, "critic_message_layers", critic_schema.CRITIC_MESSAGE_LAYERS)),
            encoder_mlp_layers=critic_encoder_layers,
            message_mlp_layers=critic_message_mlp_layers,
            fixed_bounded_relations_enabled=bool(getattr(cfg, "critic_fixed_bounded_relations_enabled", True)),
            value_head_hidden_dim=int(getattr(cfg, "critic_value_head_hidden", critic_schema.CRITIC_VALUE_HEAD_HIDDEN)),
            value_head_layers=critic_value_head_layers,
            value_mode=str(getattr(cfg, "critic_value_mode", "relational") or "relational"),
            input_norm_enabled=use_critic_input_norm,
            global_feature_enabled=bool(getattr(cfg, "critic_global_feature_enabled", False)),
            popart_enabled=bool(getattr(cfg, "critic_popart_enabled", False)),
            popart_beta=float(getattr(cfg, "critic_popart_beta", 0.999)),
            stage_specific_paths_enabled=bool(getattr(cfg, "critic_stage_specific_paths_enabled", False)),
            sat_hidden_dim=int(getattr(cfg, "critic_sat_hidden", 0) or critic_hidden),
            sat_embed_dim=int(getattr(cfg, "critic_sat_embed_dim", 0) or critic_embed),
            sat_message_layers=int(getattr(cfg, "critic_sat_message_layers", 0) or getattr(cfg, "critic_message_layers", critic_schema.CRITIC_MESSAGE_LAYERS)),
            sat_encoder_mlp_layers=int(getattr(cfg, "critic_sat_encoder_mlp_layers", 0) or critic_encoder_layers),
            sat_message_mlp_layers=int(getattr(cfg, "critic_sat_message_mlp_layers", 0) or critic_message_mlp_layers),
            sat_value_head_hidden_dim=int(getattr(cfg, "critic_sat_value_head_hidden", 0) or getattr(cfg, "critic_value_head_hidden", critic_schema.CRITIC_VALUE_HEAD_HIDDEN)),
            sat_value_head_layers=int(getattr(cfg, "critic_sat_value_head_layers", 0) or critic_value_head_layers),
            bw_local_ego_dim=bw_schema.BW_EGO_DIM,
            bw_local_sat_node_dim=bw_schema.BW_SAT_TOKEN_DIM,
            bw_local_sat_edge_dim=0,
            bw_local_user_node_dim=bw_schema.BW_GU_TOKEN_DIM,
            bw_local_user_edge_dim=0,
            bw_action_dim=shape.bw_action_dim,
        )
    return StructuredModuleBundle(actor=actor, critic=critic)
