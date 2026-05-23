from __future__ import annotations

import argparse
import copy
import csv
import os
import shutil
import sys
import time
from datetime import datetime
from dataclasses import asdict
from typing import Callable, Dict, List

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_checkpoint import (
    load_structured_train_state,
    save_structured_train_state,
)
from sagin_marl.rl.structured_bw_update_direction import (
    append_bw_update_direction_probe_row,
    build_bw_advantage_probe_context,
    compute_bw_branch_advantage_override,
    compute_bw_true_advantage_override,
    collect_bw_snapshot_panel,
    evaluate_bw_advantage_alignment,
    evaluate_bw_snapshot_panel,
    write_bw_update_direction_probe_payload,
)
from sagin_marl.rl.structured_critic import ZeroStructuredCritic
from sagin_marl.rl.structured_eval import (
    append_structured_checkpoint_eval_row,
    evaluate_structured_actor,
    evaluate_structured_actor_exec_sources,
    evaluate_structured_fixed_policy,
    update_structured_checkpoint_eval_state,
)
from sagin_marl.rl.structured_factory import build_structured_modules_from_config
from sagin_marl.rl.structured_mappo import (
    StructuredMAPPO,
    bw_actor_only_signal_critic_free_enabled,
    sat_clean_joint_critic_free_enabled,
)
from sagin_marl.rl.structured_train import (
    _make_episode_stats_state,
    close_structured_env_group,
    make_structured_env_group,
    run_structured_training,
)
from sagin_marl.utils.checkpoint import load_checkpoint_forgiving
from sagin_marl.utils.progress import Progress
from sagin_marl.utils.runtime_state_bank import (
    extract_runtime_state_bank_entries,
    load_runtime_state_bank_payload,
)
from sagin_marl.utils.seeding import set_seed
from sagin_marl.utils.torch_compile_cache import report_torch_compile_cache


def _resolve_torch_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested via --device, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def _parse_int_list(value: str | None) -> list[int]:
    if value is None:
        return []
    items: list[int] = []
    for part in str(value).replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        items.append(int(text))
    return items


def _resolve_run_dir(log_dir: str, run_dir: str | None, run_id: str | None) -> str:
    if run_dir:
        return run_dir
    if run_id:
        if run_id == "auto":
            run_id = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(log_dir, run_id)
    return log_dir


def _configure_structured_actor_trainability(actor, cfg) -> None:
    mode = str(getattr(cfg, "structured_bw_trainable_mode", "full") or "full").strip().lower()
    if mode not in {"full", "heads_only"}:
        raise ValueError(f"Unsupported structured_bw_trainable_mode: {mode!r}")

    bw_policy = getattr(actor, "bw_policy", None)
    if bw_policy is None and (mode == "heads_only" or bool(getattr(cfg, "structured_bw_freeze_kappa", False))):
        raise RuntimeError("Structured actor is missing bw_policy for heads-only BW fine-tuning.")

    if mode == "heads_only":
        trainable_prefixes = (
            "loc_head.",
            "alpha_head.",
            "kappa_head.",
            "log_scale_head.",
            "log_scale_scalar_head.",
        )
        for name, param in bw_policy.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)

    if bool(getattr(cfg, "structured_bw_freeze_kappa", False)):
        kappa_head = getattr(bw_policy, "kappa_head", None)
        if kappa_head is not None:
            for param in kappa_head.parameters():
                param.requires_grad_(False)
        lowdim_kappa_raw = getattr(bw_policy, "lowdim_kappa_raw", None)
        if isinstance(lowdim_kappa_raw, torch.nn.Parameter):
            lowdim_kappa_raw.requires_grad_(False)


def _make_structured_actor_stage_optimizers(actor, actor_lr: float) -> Dict[int, torch.optim.Optimizer]:
    """Build independent optimizers for accel/sat/BW policy submodules."""

    stage_modules = {
        0: getattr(actor, "accel_policy", None),
        1: getattr(actor, "sat_subset_policy", None),
        2: getattr(actor, "bw_policy", None),
    }
    optimizers: Dict[int, torch.optim.Optimizer] = {}
    for stage_id, module in stage_modules.items():
        if module is None:
            continue
        params = [param for param in module.parameters() if param.requires_grad]
        if params:
            optimizers[int(stage_id)] = torch.optim.Adam(params, lr=float(actor_lr))
    return optimizers


def _save_config(run_dir: str, cfg, config_path: str) -> None:
    data = asdict(cfg)
    data["_config_source"] = config_path
    out_path = os.path.join(run_dir, "config.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    if os.path.isfile(config_path):
        shutil.copy2(config_path, os.path.join(run_dir, "config_source.yaml"))


def _write_metrics_csv(run_dir: str, history_rows: List[Dict[str, float]]) -> None:
    path = os.path.join(run_dir, "metrics.csv")
    fieldnames = [
        "update",
        "env_reward_mean",
        "rollout_reward_per_step",
        "bw_access_reward_mean",
        "bw_train_reward_mean",
        "reward_part_x_acc_mean",
        "reward_part_x_rel_mean",
        "reward_part_processed_ratio_eval_mean",
        "reward_part_drop_ratio_eval_mean",
        "reward_part_d_pre_mean",
        "reward_part_sat_overlap_eval_mean",
        "reward_part_g_pre_mean",
        "reward_part_service_gap_risk_mean",
        "episode_reward",
        "episode_reward_std",
        "episode_reward_p25",
        "episode_reward_p75",
        "episode_length_mean",
        "completed_episode_count",
        "episodes_finished",
        "env_steps",
        "transition_samples",
        "rollout_reset_time_sec",
        "native_rollout_prepare_time_sec",
        "rollout_collect_time_sec",
        "rollout_total_time_sec",
        "rollout_view_build_time_sec",
        "update_prepare_time_sec",
        "update_optimize_time_sec",
        "update_total_time_sec",
        "update_profile_old_logprob_sec",
        "update_profile_value_override_sec",
        "update_profile_returns_sec",
        "update_profile_stage_cache_sec",
        "update_profile_critic_train_sec",
        "update_profile_actor_train_sec",
        "update_profile_explained_variance_sec",
        "iteration_time_sec",
        "env_steps_per_sec",
        "samples_per_sec",
        "policy_loss",
        "value_loss",
        "value_loss_accel",
        "value_loss_sat",
        "value_loss_bw",
        "critic_popart_mean_bw",
        "critic_popart_std_bw",
        "explained_variance_accel",
        "explained_variance_sat",
        "explained_variance_bw",
        "entropy",
        "entropy_accel",
        "entropy_sat",
        "entropy_bw",
        "approx_kl",
        "approx_kl_accel",
        "approx_kl_sat",
        "approx_kl_bw",
        "clip_frac",
        "clip_frac_accel",
        "clip_frac_sat",
        "clip_frac_bw",
        "danger_imitation_loss",
        "danger_imitation_active_rate",
        "bw_counterfactual_credit_active_rate",
        "bw_counterfactual_credit_agent_active_rate",
        "bw_counterfactual_credit_mean",
        "bw_counterfactual_credit_abs_mean",
        "bw_counterfactual_credit_positive_frac",
        "bw_flow_proxy_aux_loss",
        "bw_flow_proxy_regression_loss",
        "bw_flow_proxy_pairwise_acc",
        "bw_flow_proxy_pair_count",
        "bw_grad_norm_policy",
        "bw_grad_norm_aux_scaled",
        "bw_grad_ratio_aux_to_policy",
        "bw_score_head_grad_norm_policy",
        "bw_score_head_grad_norm_aux_scaled",
        "bw_score_head_grad_ratio_aux_to_policy",
        "bw_abs_log_ratio_corr_valid_count",
        "bw_abs_log_ratio_corr_latent_count",
        "bw_kappa_mean",
        "bw_kappa_p10",
        "bw_kappa_p90",
        "bw_kappa_hi_frac",
        "bw_delta_student_teacher_corr",
        "bw_delta_student_true_corr",
        "bw_delta_actor_mix_alpha",
        "bw_delta_teacher_used",
        "bw_delta_teacher_observed",
        "bw_delta_teacher_probe_only",
        "bw_branch_gate_snr",
        "bw_branch_gate_triggered",
        "bw_actor_update_skipped",
        "clean_mean_target_gap",
        "clean_mean_update_shift",
        "clean_update_to_target_ratio",
        "clean_target_beats_ref_frac",
        "clean_target_beats_sampled_frac",
        "clean_measured_kl",
        "clean_kl_coef",
        "clean_row_sample_eligible",
        "clean_row_sample_count",
        "clean_row_sample_frac",
        "clean_ref_branch_count",
        "clean_perturb_branch_count",
        "clean_gate_branch_count",
        "clean_row_filter_count",
        "clean_row_filter_frac",
        "clean_candidate_positive_branch_count",
        "clean_candidate_selected_rows",
        "vs_ref_rows",
        "vs_ref_adv_mean",
        "vs_ref_adv_std",
        "vs_ref_positive_frac",
        "vs_ref_rows_accel",
        "vs_ref_adv_mean_accel",
        "vs_ref_adv_std_accel",
        "vs_ref_positive_frac_accel",
        "vs_ref_rows_sat",
        "vs_ref_adv_mean_sat",
        "vs_ref_adv_std_sat",
        "vs_ref_positive_frac_sat",
        "vs_ref_rows_bw",
        "vs_ref_adv_mean_bw",
        "vs_ref_adv_std_bw",
        "vs_ref_positive_frac_bw",
        "grad_norm_accel",
        "grad_norm_sat",
        "grad_norm_bw",
        "vs_ref_sampling_random_count_accel",
        "vs_ref_sampling_time_count_accel",
        "vs_ref_sampling_leverage_count_accel",
        "vs_ref_sampling_uncertainty_count_accel",
        "vs_ref_sampling_horizon_mean_accel",
        "vs_ref_sampling_horizon_min_accel",
        "vs_ref_sampling_horizon_max_accel",
        "vs_ref_sampling_priority_mean_accel",
        "vs_ref_sampling_priority_max_accel",
        "vs_ref_sampling_random_count_sat",
        "vs_ref_sampling_time_count_sat",
        "vs_ref_sampling_leverage_count_sat",
        "vs_ref_sampling_uncertainty_count_sat",
        "vs_ref_sampling_horizon_mean_sat",
        "vs_ref_sampling_horizon_min_sat",
        "vs_ref_sampling_horizon_max_sat",
        "vs_ref_sampling_priority_mean_sat",
        "vs_ref_sampling_priority_max_sat",
        "vs_ref_sampling_random_count_bw",
        "vs_ref_sampling_time_count_bw",
        "vs_ref_sampling_leverage_count_bw",
        "vs_ref_sampling_uncertainty_count_bw",
        "vs_ref_sampling_horizon_mean_bw",
        "vs_ref_sampling_horizon_min_bw",
        "vs_ref_sampling_horizon_max_bw",
        "vs_ref_sampling_priority_mean_bw",
        "vs_ref_sampling_priority_max_bw",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)


class _TraceLogger:
    def __init__(self, run_dir: str, *, echo: bool = False) -> None:
        self._path = os.path.join(run_dir, "bootstrap.log")
        self._echo = bool(echo)
        self._handle = None

    def __call__(self, message: str) -> None:
        if self._handle is None or self._handle.closed:
            self._handle = open(self._path, "a", encoding="utf-8", buffering=1)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        self._handle.write(line + "\n")
        if self._echo:
            print(line, flush=True)

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()


def _make_trace_fn(run_dir: str, *, enabled: bool, echo: bool = False) -> _TraceLogger | None:
    if not enabled:
        return None
    return _TraceLogger(run_dir, echo=echo)


def _emit_console_status(message: str) -> None:
    print(message, flush=True)


class _RuntimeStateBankResetController:
    def __init__(self, bank_path: str, *, bank_name: str, selection_seed: int) -> None:
        self.bank_path = os.path.abspath(bank_path)
        self.bank_name = str(bank_name)
        self.selection_seed = int(selection_seed)
        self._rng = np.random.default_rng(self.selection_seed)
        self._entries = None
        self.bank_size = 0

    def bind(self, env_group) -> None:
        if hasattr(env_group, "load_runtime_state_bank_file_many"):
            infos = env_group.load_runtime_state_bank_file_many(self.bank_path, bank_name=self.bank_name)
            sizes = [int(info["size"]) for info in infos]
            if not sizes:
                raise RuntimeError("Runtime-state bank bind returned no worker infos.")
            if any(size != sizes[0] for size in sizes):
                raise RuntimeError(f"Inconsistent runtime-state bank sizes across workers: {sizes}")
            self.bank_size = int(sizes[0])
            self._entries = None
        else:
            payload = load_runtime_state_bank_payload(self.bank_path)
            self._entries = extract_runtime_state_bank_entries(payload)
            self.bank_size = int(len(self._entries))
        if self.bank_size <= 0:
            raise RuntimeError(f"Runtime-state bank '{self.bank_path}' is empty.")

    def _sample_entry_indices(self, count: int) -> list[int]:
        if self.bank_size <= 0:
            raise RuntimeError("Runtime-state bank has not been bound or is empty.")
        return self._rng.integers(0, self.bank_size, size=max(int(count), 0), endpoint=False).astype(np.int64).tolist()

    def reset_many(self, *, structured_group, envs, drivers, indices, seeds) -> None:
        del seeds
        selected_indices = [int(index) for index in indices]
        entry_indices = self._sample_entry_indices(len(selected_indices))
        if structured_group is not None:
            structured_group.load_runtime_state_bank_entry_many(
                entry_indices,
                bank_name=self.bank_name,
                indices=selected_indices,
            )
            return
        if self._entries is None:
            raise RuntimeError("Local runtime-state bank entries were not loaded.")
        for env_index, entry_index in zip(selected_indices, entry_indices):
            envs[int(env_index)].load_runtime_state(self._entries[int(entry_index)])
            clear_step = getattr(drivers[int(env_index)], "_clear_step", None)
            if callable(clear_step):
                clear_step()


def _init_structured_tb_layout(writer: SummaryWriter) -> None:
    layout = {
        "Structured/Train": {
            "Reward": [
                "Multiline",
                [
                    "episode_reward",
                    "episode_reward_p25",
                    "episode_reward_p75",
                    "rollout_reward_per_step",
                    "env_reward_mean",
                ],
            ],
            "RewardStd": ["Multiline", ["episode_reward_std"]],
            "PPO": ["Multiline", ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]],
            "PPOByStage": [
                "Multiline",
                [
                    "value_loss_accel",
                    "value_loss_sat",
                    "value_loss_bw",
                    "entropy_accel",
                    "entropy_sat",
                    "entropy_bw",
                    "approx_kl_accel",
                    "approx_kl_sat",
                    "approx_kl_bw",
                    "clip_frac_accel",
                    "clip_frac_sat",
                    "clip_frac_bw",
                ],
            ],
            "ExplainedVariance": [
                "Multiline",
                [
                    "explained_variance_accel",
                    "explained_variance_sat",
                    "explained_variance_bw",
                ],
            ],
            "CriticPopArtBW": [
                "Multiline",
                ["critic_popart_mean_bw", "critic_popart_std_bw"],
            ],
            "DangerImitation": [
                "Multiline",
                ["danger_imitation_loss", "danger_imitation_active_rate"],
            ],
            "BwCounterfactual": [
                "Multiline",
                [
                    "bw_counterfactual_credit_active_rate",
                    "bw_counterfactual_credit_agent_active_rate",
                    "bw_counterfactual_credit_mean",
                    "bw_counterfactual_credit_abs_mean",
                    "bw_counterfactual_credit_positive_frac",
                ],
            ],
            "BwFlowProxyAux": [
                "Multiline",
                [
                    "bw_flow_proxy_aux_loss",
                    "bw_flow_proxy_regression_loss",
                    "bw_flow_proxy_pairwise_acc",
                    "bw_flow_proxy_pair_count",
                    "bw_grad_norm_policy",
                    "bw_grad_norm_aux_scaled",
                    "bw_grad_ratio_aux_to_policy",
                    "bw_score_head_grad_norm_policy",
                    "bw_score_head_grad_norm_aux_scaled",
                    "bw_score_head_grad_ratio_aux_to_policy",
                ],
            ],
            "BwGeometry": [
                "Multiline",
                [
                    "bw_abs_log_ratio_corr_valid_count",
                    "bw_abs_log_ratio_corr_latent_count",
                ],
            ],
            "BwDirichlet": [
                "Multiline",
                [
                    "bw_kappa_mean",
                    "bw_kappa_p10",
                    "bw_kappa_p90",
                    "bw_kappa_hi_frac",
                ],
            ],
            "Episodes": ["Multiline", ["episodes_finished", "completed_episode_count", "episode_length_mean"]],
            "Throughput": ["Multiline", ["env_steps_per_sec", "samples_per_sec"]],
            "Timing": [
                "Multiline",
                [
                    "rollout_reset_time_sec",
                    "native_rollout_prepare_time_sec",
                    "rollout_collect_time_sec",
                    "rollout_total_time_sec",
                    "rollout_view_build_time_sec",
                    "update_prepare_time_sec",
                    "update_optimize_time_sec",
                    "update_total_time_sec",
                    "iteration_time_sec",
                ],
            ],
            "CleanTeacher": [
                "Multiline",
                [
                    "clean_mean_target_gap",
                    "clean_mean_update_shift",
                    "clean_update_to_target_ratio",
                    "clean_target_beats_ref_frac",
                    "clean_measured_kl",
                    "clean_kl_coef",
                ],
            ],
        },
        "Structured/Eval": {
            "CheckpointEval": [
                "Multiline",
                [
                    "checkpoint_eval/reward_sum",
                    "checkpoint_eval/bw_weighted_workload_delta_sum",
                    "checkpoint_eval/bw_weighted_workload_level_sum",
                    "checkpoint_eval/processed_ratio_eval",
                    "checkpoint_eval/drop_ratio_eval",
                    "checkpoint_eval/pre_backlog_steps_eval",
                    "checkpoint_eval/sat_overlap_eval",
                    "checkpoint_eval/collision_episode_fraction",
                ],
            ],
            "FixedReference": [
                "Multiline",
                [
                    "checkpoint_eval/fixed_reward_sum",
                    "checkpoint_eval/fixed_bw_weighted_workload_delta_sum",
                    "checkpoint_eval/fixed_bw_weighted_workload_level_sum",
                    "checkpoint_eval/fixed_processed_ratio_eval",
                    "checkpoint_eval/fixed_drop_ratio_eval",
                    "checkpoint_eval/fixed_pre_backlog_steps_eval",
                    "checkpoint_eval/fixed_sat_overlap_eval",
                    "checkpoint_eval/fixed_collision_episode_fraction",
                ],
            ],
            "UpdateDirection": [
                "Multiline",
                [
                    "update_direction_probe/delta_actor_reward_mean",
                    "update_direction_probe/actor_reward_improved_state_frac",
                    "update_direction_probe/delta_actor_weighted_mean",
                    "update_direction_probe/delta_heuristic_gap_mean",
                    "update_direction_probe/delta_l1_to_heur_mean",
                    "update_direction_probe/delta_heuristic_beats_actor_frac",
                    "update_direction_probe/corr_advantage_vs_action_gap_k",
                    "update_direction_probe/corr_advantage_vs_true_adv_mc",
                    "update_direction_probe/corr_raw_advantage_vs_true_adv_mc",
                    "update_direction_probe/corr_return_target_vs_sampled_q_mc",
                    "update_direction_probe/corr_value_vs_policy_q_mc",
                    "update_direction_probe/corr_true_adv_mc_vs_delta_logprob",
                    "update_direction_probe/sign_agree_advantage_true_adv_mc",
                    "update_direction_probe/corr_advantage_vs_delta_logprob",
                    "update_direction_probe/old_logprob_replay_abs_diff_mean",
                ],
            ],
        },
    }
    writer.add_custom_scalars(layout)


def _log_structured_train_scalars(
    writer: SummaryWriter,
    *,
    update_idx: int,
    total_env_steps: int,
    row_payload: Dict[str, float],
    env_steps_per_sec: float,
) -> None:
    for key in (
        "env_reward_mean",
        "rollout_reward_per_step",
        "bw_access_reward_mean",
        "bw_train_reward_mean",
        "reward_part_x_acc_mean",
        "reward_part_x_rel_mean",
        "reward_part_processed_ratio_eval_mean",
        "reward_part_drop_ratio_eval_mean",
        "reward_part_d_pre_mean",
        "reward_part_sat_overlap_eval_mean",
        "reward_part_g_pre_mean",
        "reward_part_service_gap_risk_mean",
        "episode_reward",
        "episode_reward_std",
        "episode_reward_p25",
        "episode_reward_p75",
        "episode_length_mean",
        "completed_episode_count",
        "episodes_finished",
        "env_steps",
        "transition_samples",
        "rollout_reset_time_sec",
        "native_rollout_prepare_time_sec",
        "rollout_collect_time_sec",
        "rollout_total_time_sec",
        "rollout_view_build_time_sec",
        "update_prepare_time_sec",
        "update_optimize_time_sec",
        "update_total_time_sec",
        "iteration_time_sec",
        "samples_per_sec",
        "policy_loss",
        "value_loss",
        "value_loss_accel",
        "value_loss_sat",
        "value_loss_bw",
        "explained_variance_accel",
        "explained_variance_sat",
        "explained_variance_bw",
        "entropy",
        "entropy_accel",
        "entropy_sat",
        "entropy_bw",
        "approx_kl",
        "approx_kl_accel",
        "approx_kl_sat",
        "approx_kl_bw",
        "clip_frac",
        "clip_frac_accel",
        "clip_frac_sat",
        "clip_frac_bw",
        "danger_imitation_loss",
        "danger_imitation_active_rate",
        "bw_counterfactual_credit_active_rate",
        "bw_counterfactual_credit_agent_active_rate",
        "bw_counterfactual_credit_mean",
        "bw_counterfactual_credit_abs_mean",
        "bw_counterfactual_credit_positive_frac",
        "bw_flow_proxy_aux_loss",
        "bw_flow_proxy_regression_loss",
        "bw_flow_proxy_pairwise_acc",
        "bw_flow_proxy_pair_count",
        "bw_grad_norm_policy",
        "bw_grad_norm_aux_scaled",
        "bw_grad_ratio_aux_to_policy",
        "bw_score_head_grad_norm_policy",
        "bw_score_head_grad_norm_aux_scaled",
        "bw_score_head_grad_ratio_aux_to_policy",
        "bw_abs_log_ratio_corr_valid_count",
        "bw_abs_log_ratio_corr_latent_count",
        "bw_kappa_mean",
        "bw_kappa_p10",
        "bw_kappa_p90",
        "bw_kappa_hi_frac",
        "clean_mean_target_gap",
        "clean_mean_update_shift",
        "clean_update_to_target_ratio",
        "clean_target_beats_ref_frac",
        "clean_measured_kl",
        "clean_kl_coef",
    ):
        if key in row_payload:
            writer.add_scalar(key, float(row_payload[key]), int(update_idx))
    writer.add_scalar("total_env_steps", float(total_env_steps), int(update_idx))
    writer.add_scalar("env_steps_per_sec", float(env_steps_per_sec), int(update_idx))
    if "samples_per_sec" in row_payload:
        writer.add_scalar("samples_per_sec", float(row_payload["samples_per_sec"]), int(update_idx))


def _log_structured_eval_scalars(
    writer: SummaryWriter,
    *,
    update_idx: int,
    summary: Dict[str, float],
    fixed_summary: Dict[str, float],
    ) -> None:
    for key in (
        "reward_sum",
        "bw_weighted_workload_delta_sum",
        "bw_weighted_workload_level_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "sat_overlap_eval",
        "collision_episode_fraction",
    ):
        if key in summary:
            writer.add_scalar(f"checkpoint_eval/{key}", float(summary[key]), int(update_idx))
        if key in fixed_summary:
            writer.add_scalar(f"checkpoint_eval/fixed_{key}", float(fixed_summary[key]), int(update_idx))


def _log_update_direction_probe_scalars(
    writer: SummaryWriter,
    *,
    update_idx: int,
    pre_eval: Dict[str, object],
    post_eval: Dict[str, object],
    alignment: Dict[str, object] | None = None,
) -> None:
    pre_actor_reward = float(pre_eval["actor_reward"]["mean"])
    post_actor_reward = float(post_eval["actor_reward"]["mean"])
    pre_actor_weighted = float(pre_eval["actor_weighted"]["mean"])
    post_actor_weighted = float(post_eval["actor_weighted"]["mean"])
    pre_heuristic_reward = float(pre_eval["heuristic_reward"]["mean"])
    post_heuristic_reward = float(post_eval["heuristic_reward"]["mean"])
    pre_gap = pre_heuristic_reward - pre_actor_reward
    post_gap = post_heuristic_reward - post_actor_reward
    pre_l1 = float(pre_eval["l1_to_heur"]["mean"])
    post_l1 = float(post_eval["l1_to_heur"]["mean"])
    pre_heuristic_beats = float(pre_eval["heuristic_beats_actor_frac"])
    post_heuristic_beats = float(post_eval["heuristic_beats_actor_frac"])
    pre_state_actor_reward = np.asarray(pre_eval.get("state_actor_reward_means", []), dtype=np.float64)
    post_state_actor_reward = np.asarray(post_eval.get("state_actor_reward_means", []), dtype=np.float64)
    improved_state_frac = 0.0
    if pre_state_actor_reward.size > 0 and pre_state_actor_reward.size == post_state_actor_reward.size:
        improved_state_frac = float(np.mean(post_state_actor_reward > pre_state_actor_reward + 1.0e-9))
    scalars = {
        "pre_actor_reward_mean": pre_actor_reward,
        "post_actor_reward_mean": post_actor_reward,
        "delta_actor_reward_mean": post_actor_reward - pre_actor_reward,
        "actor_reward_improved_state_frac": improved_state_frac,
        "pre_actor_weighted_mean": pre_actor_weighted,
        "post_actor_weighted_mean": post_actor_weighted,
        "delta_actor_weighted_mean": post_actor_weighted - pre_actor_weighted,
        "pre_heuristic_gap_mean": pre_gap,
        "post_heuristic_gap_mean": post_gap,
        "delta_heuristic_gap_mean": post_gap - pre_gap,
        "pre_l1_to_heur_mean": pre_l1,
        "post_l1_to_heur_mean": post_l1,
        "delta_l1_to_heur_mean": post_l1 - pre_l1,
        "pre_heuristic_beats_actor_frac": pre_heuristic_beats,
        "post_heuristic_beats_actor_frac": post_heuristic_beats,
        "delta_heuristic_beats_actor_frac": post_heuristic_beats - pre_heuristic_beats,
    }
    if alignment is not None:
        scalars.update(
            {
                "adv_positive_frac": float(alignment["advantage"]["positive_frac"]),
                "old_logprob_replay_abs_diff_mean": float(
                    alignment["logprob_alignment"]["old_logprob_replay_abs_diff"]["mean"]
                ),
                "old_logprob_exact_replay_frac_1e-5": float(
                    alignment["logprob_alignment"]["old_logprob_exact_replay_frac_1e-5"]
                ),
                "corr_advantage_vs_delta_logprob": float(
                    alignment["logprob_alignment"]["corr_advantage_vs_delta_logprob"]
                ),
                "mean_delta_logprob_pos_adv": float(
                    alignment["logprob_alignment"]["mean_delta_logprob_pos_adv"]
                ),
                "mean_delta_logprob_neg_adv": float(
                    alignment["logprob_alignment"]["mean_delta_logprob_neg_adv"]
                ),
                "pos_adv_logprob_up_frac": float(
                    alignment["logprob_alignment"]["pos_adv_logprob_up_frac"]
                ),
                "neg_adv_logprob_down_frac": float(
                    alignment["logprob_alignment"]["neg_adv_logprob_down_frac"]
                ),
                "corr_advantage_vs_action_gap_k": float(
                    alignment["critic_alignment"]["corr_advantage_vs_action_gap_k"]
                ),
                "mean_action_gap_k_pos_adv": float(
                    alignment["critic_alignment"]["mean_action_gap_k_pos_adv"]
                ),
                "mean_action_gap_k_neg_adv": float(
                    alignment["critic_alignment"]["mean_action_gap_k_neg_adv"]
                ),
                "pos_adv_action_gap_positive_frac": float(
                    alignment["critic_alignment"]["pos_adv_action_gap_positive_frac"]
                ),
                "neg_adv_action_gap_negative_frac": float(
                    alignment["critic_alignment"]["neg_adv_action_gap_negative_frac"]
                ),
            }
        )
        branch_alignment = alignment.get("branch_alignment", {})
        if branch_alignment:
            branch_primary = branch_alignment.get("primary", {})
            scalars.update(
                {
                    "branch_primary_horizon": float(branch_alignment.get("primary_horizon", 0)),
                    "branch_sample_count": float(branch_primary.get("sample_count", 0.0)),
                    "corr_advantage_vs_branch_delta": float(
                        branch_primary.get("corr_advantage_vs_branch_delta", 0.0)
                    ),
                    "corr_raw_advantage_vs_branch_delta": float(
                        branch_primary.get("corr_raw_advantage_vs_branch_delta", 0.0)
                    ),
                    "corr_branch_delta_vs_delta_logprob": float(
                        branch_primary.get("corr_branch_delta_vs_delta_logprob", 0.0)
                    ),
                    "branch_delta_abs_mean": float(branch_primary.get("branch_delta_abs_mean", 0.0)),
                    "sign_agree_raw_advantage_branch_delta": float(
                        branch_primary.get("sign_agree_raw_advantage_branch_delta", 0.0)
                    ),
                }
            )
            for horizon_summary in branch_alignment.get("by_horizon", []):
                horizon = int(horizon_summary.get("horizon", 0))
                if horizon <= 0:
                    continue
                scalars[f"corr_raw_advantage_vs_branch_delta_h{horizon}"] = float(
                    horizon_summary.get("corr_raw_advantage_vs_branch_delta", 0.0)
                )
                scalars[f"corr_branch_delta_vs_delta_logprob_h{horizon}"] = float(
                    horizon_summary.get("corr_branch_delta_vs_delta_logprob", 0.0)
                )
        true_alignment = alignment.get("true_action_alignment", {})
        if true_alignment:
            scalars.update(
                {
                    "corr_advantage_vs_true_adv_mc": float(
                        true_alignment["corr_advantage_vs_true_adv_mc"]
                    ),
                    "sign_agree_advantage_true_adv_mc": float(
                        true_alignment["sign_agree_advantage_true_adv_mc"]
                    ),
                    "corr_true_adv_mc_vs_delta_logprob": float(
                        true_alignment["corr_true_adv_mc_vs_delta_logprob"]
                    ),
                    "true_adv_mc_abs_mean": float(true_alignment["true_adv_mc_abs_mean"]),
                    "true_adv_mc_std": float(true_alignment["true_adv_mc_std"]),
                    "true_adv_mc_snr": float(true_alignment["true_adv_mc_snr"]),
                    "ppo_pos_true_adv_positive_frac": float(
                        true_alignment["ppo_pos_true_adv_positive_frac"]
                    ),
                    "ppo_neg_true_adv_negative_frac": float(
                        true_alignment["ppo_neg_true_adv_negative_frac"]
                    ),
                    "mean_delta_logprob_true_adv_pos": float(
                        true_alignment["mean_delta_logprob_true_adv_pos"]
                    ),
                    "mean_delta_logprob_true_adv_neg": float(
                        true_alignment["mean_delta_logprob_true_adv_neg"]
                    ),
                }
            )
            decomposition = true_alignment.get("advantage_decomposition", {})
            if decomposition:
                scalars.update(
                    {
                        "corr_raw_advantage_vs_true_adv_mc": float(
                            decomposition["corr_raw_advantage_vs_true_adv_mc"]
                        ),
                        "sign_agree_raw_advantage_true_adv_mc": float(
                            decomposition["sign_agree_raw_advantage_true_adv_mc"]
                        ),
                        "corr_return_target_vs_sampled_q_mc": float(
                            decomposition["corr_return_target_vs_sampled_q_mc"]
                        ),
                        "return_target_sampled_q_error_abs_mean": float(
                            decomposition["return_target_sampled_q_error_abs_mean"]
                        ),
                        "corr_value_vs_policy_q_mc": float(
                            decomposition["corr_value_vs_policy_q_mc"]
                        ),
                        "value_policy_q_error_abs_mean": float(
                            decomposition["value_policy_q_error_abs_mean"]
                        ),
                    }
                )
    for key, value in scalars.items():
        writer.add_scalar(f"update_direction_probe/{key}", float(value), int(update_idx))


def _save_periodic_structured_checkpoint(
    run_dir: str,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    *,
    actor_stage_optimizers=None,
    update_idx: int,
    total_updates: int,
    total_env_steps: int,
    history_rows: List[Dict[str, float]],
    total_time_sec: float,
    checkpoint_eval_state: Dict[str, float] | None = None,
    checkpoint_eval_fixed_summary: Dict[str, float] | None = None,
) -> None:
    suffix = f"u{int(update_idx):04d}"
    _save_named_structured_checkpoint(
        run_dir,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        actor_stage_optimizers=actor_stage_optimizers,
        suffix=suffix,
        update_idx=update_idx,
        total_updates=total_updates,
        total_env_steps=total_env_steps,
        history_rows=history_rows,
        total_time_sec=total_time_sec,
        checkpoint_eval_state=checkpoint_eval_state,
        checkpoint_eval_fixed_summary=checkpoint_eval_fixed_summary,
    )


def _save_named_structured_checkpoint(
    run_dir: str,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    *,
    actor_stage_optimizers=None,
    suffix: str,
    update_idx: int,
    total_updates: int,
    total_env_steps: int,
    history_rows: List[Dict[str, float]],
    total_time_sec: float,
    checkpoint_eval_state: Dict[str, float] | None = None,
    checkpoint_eval_fixed_summary: Dict[str, float] | None = None,
) -> None:
    torch.save(actor.state_dict(), os.path.join(run_dir, f"actor_{suffix}.pt"))
    if critic is not None:
        torch.save(critic.state_dict(), os.path.join(run_dir, f"critic_{suffix}.pt"))
    save_structured_train_state(
        run_dir,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        actor_stage_optimizers=actor_stage_optimizers,
        update=int(update_idx),
        planned_total_updates=int(total_updates),
        total_env_steps=int(total_env_steps),
        history_rows=history_rows,
        total_time_sec=float(total_time_sec),
        checkpoint_eval_state=checkpoint_eval_state,
        checkpoint_eval_fixed_summary=checkpoint_eval_fixed_summary,
        suffix=suffix,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--log_dir", type=str, default="runs/structured")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--rollout_env_steps", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--actor_hidden_dim", type=int, default=None)
    parser.add_argument("--actor_embed_dim", type=int, default=None)
    parser.add_argument("--critic_hidden_dim", type=int, default=None)
    parser.add_argument("--critic_embed_dim", type=int, default=None)
    parser.add_argument("--critic_global_feature_enabled", action="store_true")
    parser.add_argument("--actor_lr", type=float, default=None)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--clip_ratio", type=float, default=None)
    parser.add_argument("--value_coef", type=float, default=None)
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--ppo_epochs", type=int, default=None)
    parser.add_argument("--num_mini_batch", type=int, default=None)
    parser.add_argument("--critic_warmup_before_actor_epochs", type=int, default=None)
    parser.add_argument("--critic_warmup_recompute_advantages", action="store_true")
    parser.add_argument("--critic_warmup_recompute_mode", type=str, default=None)
    parser.add_argument("--critic_loss_target_standardize", action="store_true")
    parser.add_argument("--critic_loss_running_standardize", action="store_true")
    parser.add_argument("--critic_replay_bank_enabled", action="store_true")
    parser.add_argument("--critic_replay_bank_capacity", type=int, default=None)
    parser.add_argument("--critic_popart_enabled", action="store_true")
    parser.add_argument("--critic_popart_beta", type=float, default=None)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--reward_mode", type=str, default=None)
    parser.add_argument("--accel_log_std_init", type=float, default=None)
    parser.add_argument("--bw_clean_trust_region_enabled", action="store_true")
    parser.add_argument("--bw_clean_trust_region_target_kl", type=float, default=None)
    parser.add_argument("--bw_clean_trust_region_kl_coef_init", type=float, default=None)
    parser.add_argument("--bw_clean_trust_region_kl_coef_min", type=float, default=None)
    parser.add_argument("--bw_clean_trust_region_kl_coef_max", type=float, default=None)
    parser.add_argument("--bw_clean_trust_region_backtrack_factor", type=float, default=None)
    parser.add_argument("--bw_clean_trust_region_max_backtracks", type=int, default=None)
    parser.add_argument("--bw_clean_per_user_enabled", action="store_true")
    parser.add_argument("--bw_clean_per_user_horizon", type=int, default=None)
    parser.add_argument("--bw_clean_row_sample_enabled", action="store_true")
    parser.add_argument("--bw_clean_row_sample_budget", type=int, default=None)
    parser.add_argument("--bw_clean_row_sample_seed", type=int, default=None)
    parser.add_argument("--bw_clean_candidate_select_enabled", action="store_true")
    parser.add_argument("--bw_clean_candidate_select_deltas", type=str, default=None)
    parser.add_argument("--bw_clean_candidate_select_include_onehot", action="store_true")
    parser.add_argument("--bw_clean_candidate_select_no_onehot", action="store_true")
    parser.add_argument("--bw_clean_candidate_select_include_uniform", action="store_true")
    parser.add_argument("--bw_clean_candidate_select_no_uniform", action="store_true")
    parser.add_argument("--bw_clean_candidate_select_row_filter_mode", type=str, default=None)
    parser.add_argument("--bw_clean_candidate_select_ref_return_eps", type=float, default=None)
    parser.add_argument("--bw_clean_candidate_select_gate_eps", type=float, default=None)
    parser.add_argument("--bw_return_mode", type=str, default=None)
    parser.add_argument("--bw_train_target_mode", type=str, default=None)
    parser.add_argument("--exec_bw_source", type=str, default=None)
    parser.add_argument("--bw_single_uav_policy_uav_id", type=int, default=None)
    parser.add_argument("--disable_interference", action="store_true")
    parser.add_argument("--bw_actor_advantage_override_mode", type=str, default=None)
    parser.add_argument("--bw_actor_branch_horizon", type=int, default=None)
    parser.add_argument("--bw_actor_branch_samples", type=int, default=None)
    parser.add_argument("--bw_actor_branch_ref_mode", type=str, default=None)
    parser.add_argument("--bw_actor_branch_follow_policy_mode", type=str, default=None)
    parser.add_argument("--bw_actor_branch_normalize", action="store_true")
    parser.add_argument("--bw_actor_branch_gate_snr_threshold", type=float, default=None)
    parser.add_argument("--bw_actor_branch_worker_fused_rollout_enabled", action="store_true")
    parser.add_argument("--bw_delta_critic_horizon", type=int, default=None)
    parser.add_argument("--bw_delta_critic_samples", type=int, default=None)
    parser.add_argument("--bw_delta_critic_ref_mode", type=str, default=None)
    parser.add_argument("--bw_delta_critic_follow_policy_mode", type=str, default=None)
    parser.add_argument("--bw_delta_critic_loss_coef", type=float, default=None)
    parser.add_argument("--bw_delta_critic_warmup_epochs", type=int, default=None)
    parser.add_argument("--bw_delta_teacher_student_corr_low", type=float, default=None)
    parser.add_argument("--bw_delta_teacher_student_corr_high", type=float, default=None)
    parser.add_argument("--bw_delta_teacher_student_mix_power", type=float, default=None)
    parser.add_argument("--bw_delta_teacher_student_disable_dense_teacher_when_ready", action="store_true")
    parser.add_argument("--bw_delta_teacher_student_ready_patience", type=int, default=None)
    parser.add_argument("--bw_delta_teacher_student_probe_interval_updates", type=int, default=None)
    parser.add_argument("--bw_delta_teacher_student_probe_reenable_corr", type=float, default=None)
    parser.add_argument("--structured_bw_freeze_kappa", action="store_true")
    parser.add_argument("--structured_bw_parameterization", type=str, default=None)
    parser.add_argument("--bw_flow_proxy_aux_enabled", action="store_true")
    parser.add_argument("--bw_flow_proxy_aux_coef", type=float, default=None)
    parser.add_argument("--bw_flow_proxy_aux_regression_coef", type=float, default=None)
    parser.add_argument("--update_direction_probe_interval_updates", type=int, default=None)
    parser.add_argument("--update_direction_probe_start_update", type=int, default=None)
    parser.add_argument("--update_direction_probe_bw_sample_limit", type=int, default=None)
    parser.add_argument("--update_direction_probe_branch_enabled", action="store_true")
    parser.add_argument("--update_direction_probe_branch_horizons", type=str, default=None)
    parser.add_argument("--update_direction_probe_branch_samples", type=int, default=None)
    parser.add_argument("--update_direction_probe_branch_ref_mode", type=str, default=None)
    parser.add_argument("--update_direction_probe_branch_follow_policy_mode", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--vec_backend", type=str, default="sync", choices=["sync", "subproc"])
    parser.add_argument("--init_actor", type=str, default=None)
    parser.add_argument("--init_critic", type=str, default=None)
    parser.add_argument("--resume_state", type=str, default=None)
    parser.add_argument("--runtime_state_bank", type=str, default=None)
    parser.add_argument("--runtime_state_bank_name", type=str, default="runtime_bank")
    parser.add_argument("--runtime_state_bank_seed", type=int, default=None)
    parser.add_argument("--checkpoint_eval_interval_updates", type=int, default=None)
    parser.add_argument("--checkpoint_eval_start_update", type=int, default=None)
    parser.add_argument("--checkpoint_eval_episodes", type=int, default=None)
    parser.add_argument("--checkpoint_eval_episode_seed_base", type=int, default=None)
    parser.add_argument("--checkpoint_eval_fixed_policy", type=str, default=None)
    parser.add_argument("--checkpoint_eval_policy_mode", type=str, default=None)
    parser.add_argument("--checkpoint_eval_reward_early_stop_enabled", action="store_true")
    parser.add_argument("--checkpoint_eval_reward_patience", type=int, default=None)
    parser.add_argument("--checkpoint_eval_reward_min_delta_rel", type=float, default=None)
    parser.add_argument("--checkpoint_eval_worsen_patience", type=int, default=None)
    parser.add_argument("--checkpoint_eval_stop_on_early_stop", action="store_true")
    parser.add_argument("--checkpoint_eval_save_best_models", action="store_true")
    parser.add_argument("--checkpoint_eval_use_best_as_final", action="store_true")
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--structured_env_tensor_backend",
        type=str,
        default=None,
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--verbose_console", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.critic_warmup_before_actor_epochs is not None:
        cfg.critic_warmup_before_actor_epochs = max(int(args.critic_warmup_before_actor_epochs), 0)
    if args.critic_warmup_recompute_advantages:
        cfg.critic_warmup_recompute_advantages = True
    if args.critic_warmup_recompute_mode is not None:
        cfg.critic_warmup_recompute_mode = str(args.critic_warmup_recompute_mode)
    if args.critic_loss_target_standardize:
        cfg.critic_loss_target_standardize = True
    if args.critic_loss_running_standardize:
        cfg.critic_loss_running_standardize = True
    if args.critic_replay_bank_enabled:
        cfg.critic_replay_bank_enabled = True
    if args.critic_replay_bank_capacity is not None:
        cfg.critic_replay_bank_capacity = max(int(args.critic_replay_bank_capacity), 0)
    if args.critic_popart_enabled:
        cfg.critic_popart_enabled = True
    if args.critic_popart_beta is not None:
        cfg.critic_popart_beta = float(args.critic_popart_beta)
    if args.critic_global_feature_enabled:
        cfg.critic_global_feature_enabled = True
    if args.gae_lambda is not None:
        cfg.gae_lambda = float(args.gae_lambda)
    if args.reward_mode is not None:
        cfg.reward_mode = str(args.reward_mode)
    if args.accel_log_std_init is not None:
        cfg.accel_log_std_init = float(args.accel_log_std_init)
    if args.bw_clean_trust_region_enabled:
        cfg.bw_clean_trust_region_enabled = True
    if args.bw_clean_trust_region_target_kl is not None:
        cfg.bw_clean_trust_region_target_kl = max(float(args.bw_clean_trust_region_target_kl), 1.0e-8)
    if args.bw_clean_trust_region_kl_coef_init is not None:
        cfg.bw_clean_trust_region_kl_coef_init = max(float(args.bw_clean_trust_region_kl_coef_init), 0.0)
    if args.bw_clean_trust_region_kl_coef_min is not None:
        cfg.bw_clean_trust_region_kl_coef_min = max(float(args.bw_clean_trust_region_kl_coef_min), 0.0)
    if args.bw_clean_trust_region_kl_coef_max is not None:
        cfg.bw_clean_trust_region_kl_coef_max = max(float(args.bw_clean_trust_region_kl_coef_max), 0.0)
    if args.bw_clean_trust_region_backtrack_factor is not None:
        cfg.bw_clean_trust_region_backtrack_factor = min(
            max(float(args.bw_clean_trust_region_backtrack_factor), 1.0e-3),
            0.999,
        )
    if args.bw_clean_trust_region_max_backtracks is not None:
        cfg.bw_clean_trust_region_max_backtracks = max(int(args.bw_clean_trust_region_max_backtracks), 0)
    if args.bw_clean_per_user_enabled:
        cfg.bw_clean_per_user_enabled = True
    if args.bw_clean_per_user_horizon is not None:
        cfg.bw_clean_per_user_horizon = max(int(args.bw_clean_per_user_horizon), 1)
    if args.bw_clean_row_sample_enabled:
        cfg.bw_clean_row_sample_enabled = True
    if args.bw_clean_row_sample_budget is not None:
        cfg.bw_clean_row_sample_budget = max(int(args.bw_clean_row_sample_budget), 0)
    if args.bw_clean_row_sample_seed is not None:
        cfg.bw_clean_row_sample_seed = int(args.bw_clean_row_sample_seed)
    if args.bw_clean_candidate_select_enabled:
        cfg.bw_clean_candidate_select_enabled = True
    if args.bw_clean_candidate_select_deltas is not None:
        cfg.bw_clean_candidate_select_deltas = str(args.bw_clean_candidate_select_deltas)
    if args.bw_clean_candidate_select_include_onehot:
        cfg.bw_clean_candidate_select_include_onehot = True
    if args.bw_clean_candidate_select_no_onehot:
        cfg.bw_clean_candidate_select_include_onehot = False
    if args.bw_clean_candidate_select_include_uniform:
        cfg.bw_clean_candidate_select_include_uniform = True
    if args.bw_clean_candidate_select_no_uniform:
        cfg.bw_clean_candidate_select_include_uniform = False
    if args.bw_clean_candidate_select_row_filter_mode is not None:
        cfg.bw_clean_candidate_select_row_filter_mode = str(args.bw_clean_candidate_select_row_filter_mode)
    if args.bw_clean_candidate_select_ref_return_eps is not None:
        cfg.bw_clean_candidate_select_ref_return_eps = max(float(args.bw_clean_candidate_select_ref_return_eps), 0.0)
    if args.bw_clean_candidate_select_gate_eps is not None:
        cfg.bw_clean_candidate_select_gate_eps = max(float(args.bw_clean_candidate_select_gate_eps), 0.0)
    if args.bw_return_mode is not None:
        cfg.bw_return_mode = str(args.bw_return_mode)
    if args.bw_train_target_mode is not None:
        cfg.bw_train_target_mode = str(args.bw_train_target_mode)
    if args.exec_bw_source is not None:
        cfg.exec_bw_source = str(args.exec_bw_source)
    if args.bw_single_uav_policy_uav_id is not None:
        cfg.bw_single_uav_policy_uav_id = max(int(args.bw_single_uav_policy_uav_id), 0)
    if args.disable_interference:
        cfg.interference_enabled = False
    if args.bw_actor_advantage_override_mode is not None:
        cfg.bw_actor_advantage_override_mode = str(args.bw_actor_advantage_override_mode)
    if args.bw_actor_branch_horizon is not None:
        cfg.bw_actor_branch_horizon = max(int(args.bw_actor_branch_horizon), 1)
    if args.bw_actor_branch_samples is not None:
        cfg.bw_actor_branch_samples = max(int(args.bw_actor_branch_samples), 1)
    if args.bw_actor_branch_ref_mode is not None:
        cfg.bw_actor_branch_ref_mode = str(args.bw_actor_branch_ref_mode)
    if args.bw_actor_branch_follow_policy_mode is not None:
        cfg.bw_actor_branch_follow_policy_mode = str(args.bw_actor_branch_follow_policy_mode)
    if args.bw_actor_branch_normalize:
        cfg.bw_actor_branch_normalize = True
    if args.bw_actor_branch_gate_snr_threshold is not None:
        cfg.bw_actor_branch_gate_snr_threshold = max(float(args.bw_actor_branch_gate_snr_threshold), 0.0)
    if args.bw_actor_branch_worker_fused_rollout_enabled:
        cfg.bw_actor_branch_worker_fused_rollout_enabled = True
    if args.bw_delta_critic_horizon is not None:
        cfg.bw_delta_critic_horizon = max(int(args.bw_delta_critic_horizon), 1)
    if args.bw_delta_critic_samples is not None:
        cfg.bw_delta_critic_samples = max(int(args.bw_delta_critic_samples), 1)
    if args.bw_delta_critic_ref_mode is not None:
        cfg.bw_delta_critic_ref_mode = str(args.bw_delta_critic_ref_mode)
    if args.bw_delta_critic_follow_policy_mode is not None:
        cfg.bw_delta_critic_follow_policy_mode = str(args.bw_delta_critic_follow_policy_mode)
    if args.bw_delta_critic_loss_coef is not None:
        cfg.bw_delta_critic_loss_coef = max(float(args.bw_delta_critic_loss_coef), 0.0)
    if args.bw_delta_critic_warmup_epochs is not None:
        cfg.bw_delta_critic_warmup_epochs = max(int(args.bw_delta_critic_warmup_epochs), 0)
    if args.bw_delta_teacher_student_corr_low is not None:
        cfg.bw_delta_teacher_student_corr_low = float(args.bw_delta_teacher_student_corr_low)
    if args.bw_delta_teacher_student_corr_high is not None:
        cfg.bw_delta_teacher_student_corr_high = float(args.bw_delta_teacher_student_corr_high)
    if args.bw_delta_teacher_student_mix_power is not None:
        cfg.bw_delta_teacher_student_mix_power = max(float(args.bw_delta_teacher_student_mix_power), 1.0e-6)
    if args.bw_delta_teacher_student_disable_dense_teacher_when_ready:
        cfg.bw_delta_teacher_student_disable_dense_teacher_when_ready = True
    if args.bw_delta_teacher_student_ready_patience is not None:
        cfg.bw_delta_teacher_student_ready_patience = max(int(args.bw_delta_teacher_student_ready_patience), 1)
    if args.structured_env_tensor_backend is not None:
        cfg.structured_env_tensor_backend = str(args.structured_env_tensor_backend)
    if args.bw_delta_teacher_student_probe_interval_updates is not None:
        cfg.bw_delta_teacher_student_probe_interval_updates = max(
            int(args.bw_delta_teacher_student_probe_interval_updates),
            0,
        )
    if args.bw_delta_teacher_student_probe_reenable_corr is not None:
        cfg.bw_delta_teacher_student_probe_reenable_corr = float(args.bw_delta_teacher_student_probe_reenable_corr)
    if args.structured_bw_freeze_kappa:
        cfg.structured_bw_freeze_kappa = True
    if args.structured_bw_parameterization is not None:
        cfg.structured_bw_parameterization = str(args.structured_bw_parameterization)
    if args.bw_flow_proxy_aux_enabled:
        cfg.bw_flow_proxy_aux_enabled = True
    if args.bw_flow_proxy_aux_coef is not None:
        cfg.bw_flow_proxy_aux_coef = max(float(args.bw_flow_proxy_aux_coef), 0.0)
    if args.bw_flow_proxy_aux_regression_coef is not None:
        cfg.bw_flow_proxy_aux_regression_coef = max(float(args.bw_flow_proxy_aux_regression_coef), 0.0)
    if args.update_direction_probe_interval_updates is not None:
        cfg.update_direction_probe_interval_updates = max(int(args.update_direction_probe_interval_updates), 0)
    if args.update_direction_probe_start_update is not None:
        cfg.update_direction_probe_start_update = max(int(args.update_direction_probe_start_update), 0)
    if args.update_direction_probe_bw_sample_limit is not None:
        cfg.update_direction_probe_bw_sample_limit = max(int(args.update_direction_probe_bw_sample_limit), 1)
    if args.update_direction_probe_branch_enabled:
        cfg.update_direction_probe_branch_enabled = True
    if args.update_direction_probe_branch_horizons is not None:
        cfg.update_direction_probe_branch_horizons = [
            value for value in _parse_int_list(args.update_direction_probe_branch_horizons) if int(value) > 0
        ]
    if args.update_direction_probe_branch_samples is not None:
        cfg.update_direction_probe_branch_samples = max(int(args.update_direction_probe_branch_samples), 0)
    if args.update_direction_probe_branch_ref_mode is not None:
        cfg.update_direction_probe_branch_ref_mode = str(args.update_direction_probe_branch_ref_mode)
    if args.update_direction_probe_branch_follow_policy_mode is not None:
        cfg.update_direction_probe_branch_follow_policy_mode = str(args.update_direction_probe_branch_follow_policy_mode)
    if args.checkpoint_eval_interval_updates is not None:
        cfg.checkpoint_eval_enabled = int(args.checkpoint_eval_interval_updates) > 0
        cfg.checkpoint_eval_interval_updates = max(int(args.checkpoint_eval_interval_updates), 0)
    if args.checkpoint_eval_start_update is not None:
        cfg.checkpoint_eval_start_update = max(int(args.checkpoint_eval_start_update), 0)
    if args.checkpoint_eval_episodes is not None:
        cfg.checkpoint_eval_enabled = int(args.checkpoint_eval_episodes) > 0
        cfg.checkpoint_eval_episodes = max(int(args.checkpoint_eval_episodes), 0)
    if args.checkpoint_eval_episode_seed_base is not None:
        cfg.checkpoint_eval_episode_seed_base = int(args.checkpoint_eval_episode_seed_base)
    if args.checkpoint_eval_fixed_policy is not None:
        cfg.checkpoint_eval_fixed_policy = str(args.checkpoint_eval_fixed_policy)
    if args.checkpoint_eval_policy_mode is not None:
        cfg.checkpoint_eval_policy_mode = str(args.checkpoint_eval_policy_mode)
    if args.checkpoint_eval_reward_early_stop_enabled:
        cfg.checkpoint_eval_reward_early_stop_enabled = True
    if args.checkpoint_eval_reward_patience is not None:
        cfg.checkpoint_eval_reward_patience = max(int(args.checkpoint_eval_reward_patience), 1)
    if args.checkpoint_eval_reward_min_delta_rel is not None:
        cfg.checkpoint_eval_reward_min_delta_rel = max(float(args.checkpoint_eval_reward_min_delta_rel), 0.0)
    if args.checkpoint_eval_worsen_patience is not None:
        cfg.checkpoint_eval_worsen_patience = max(int(args.checkpoint_eval_worsen_patience), 1)
    if args.checkpoint_eval_stop_on_early_stop:
        cfg.checkpoint_eval_stop_on_early_stop = True
    if args.checkpoint_eval_save_best_models:
        cfg.checkpoint_eval_save_best_models = True
    if args.checkpoint_eval_use_best_as_final:
        cfg.checkpoint_eval_use_best_as_final = True
    if args.num_envs < 1:
        raise ValueError("--num_envs must be >= 1")
    if max(int(getattr(cfg, "bw_actor_branch_parallel_envs", 0) or 0), 0) <= 0 and int(args.num_envs) > 1:
        cfg.bw_actor_branch_parallel_envs = int(args.num_envs)
    if not str(getattr(cfg, "bw_actor_branch_parallel_backend", "") or "").strip():
        cfg.bw_actor_branch_parallel_backend = str(args.vec_backend)
    if max(int(getattr(cfg, "sat_clean_parallel_envs", 0) or 0), 0) <= 0 and int(args.num_envs) > 1:
        cfg.sat_clean_parallel_envs = int(args.num_envs)
    if not str(getattr(cfg, "sat_clean_parallel_backend", "") or "").strip():
        cfg.sat_clean_parallel_backend = str(args.vec_backend)
    if args.resume_state and (args.init_actor or args.init_critic):
        raise ValueError("--resume_state cannot be combined with --init_actor/--init_critic.")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    set_seed(cfg.seed)

    rollout_env_steps = int(args.rollout_env_steps) if args.rollout_env_steps is not None else int(cfg.buffer_size)
    actor_lr = float(args.actor_lr) if args.actor_lr is not None else float(cfg.actor_lr)
    critic_lr = float(args.critic_lr) if args.critic_lr is not None else float(cfg.critic_lr)
    clip_ratio = float(args.clip_ratio) if args.clip_ratio is not None else float(cfg.clip_ratio)
    value_coef = float(args.value_coef) if args.value_coef is not None else float(cfg.value_coef)
    entropy_coef = float(args.entropy_coef) if args.entropy_coef is not None else float(cfg.entropy_coef)
    max_grad_norm = float(args.max_grad_norm) if args.max_grad_norm is not None else float(cfg.max_grad_norm)
    ppo_epochs = int(args.ppo_epochs) if args.ppo_epochs is not None else int(cfg.ppo_epochs)
    num_mini_batch = int(args.num_mini_batch) if args.num_mini_batch is not None else int(cfg.num_mini_batch)

    run_dir = _resolve_run_dir(args.log_dir, args.run_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    _save_config(run_dir, cfg, os.path.abspath(args.config))
    trace = _make_trace_fn(
        run_dir,
        enabled=bool(getattr(cfg, "train_trace_enabled", True) or args.verbose_console),
        echo=bool(args.verbose_console),
    )
    _trace: Callable[[str], None] = trace if trace is not None else (lambda message: None)
    tb_writer = SummaryWriter(run_dir)
    _init_structured_tb_layout(tb_writer)
    _trace(f"run_dir_ready path={run_dir}")
    _trace(f"structured_trainer envs={int(args.num_envs)} backend={args.vec_backend}")
    device = _resolve_torch_device(args.device)
    _trace(f"structured_trainer device={device}")
    report_torch_compile_cache(context="train_structured", device=device, cfg=cfg)
    _emit_console_status(
        "Run "
        f"{os.path.basename(run_dir)} | device={device} | backend={args.vec_backend} | "
        f"envs={int(args.num_envs)} | T={int(cfg.T_steps)} | rollout={int(rollout_env_steps)}"
    )

    train_accel_enabled = bool(True if getattr(cfg, "train_accel", None) is None else getattr(cfg, "train_accel"))
    train_sat_enabled = bool(True if getattr(cfg, "train_sat", None) is None else getattr(cfg, "train_sat"))
    train_bw_enabled = bool(True if getattr(cfg, "train_bw", None) is None else getattr(cfg, "train_bw"))
    bw_actor_only_signal_critic_free = bw_actor_only_signal_critic_free_enabled(
        cfg,
        train_accel=train_accel_enabled,
        train_sat=train_sat_enabled,
        train_bw=train_bw_enabled,
    )
    sat_clean_joint_critic_free = sat_clean_joint_critic_free_enabled(
        cfg,
        train_accel=train_accel_enabled,
        train_sat=train_sat_enabled,
        train_bw=train_bw_enabled,
    )
    critic_free_build = bool(bw_actor_only_signal_critic_free or sat_clean_joint_critic_free)

    _trace("build_modules:start")
    bundle = build_structured_modules_from_config(
        cfg,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_embed_dim=args.actor_embed_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_embed_dim=args.critic_embed_dim,
        build_critic=not critic_free_build,
    )
    actor = bundle.actor
    critic_model = bundle.critic
    _configure_structured_actor_trainability(actor, cfg)
    actor.to(device)
    critic = ZeroStructuredCritic(device) if critic_model is None else critic_model.to(device)
    actor_optimizer = torch.optim.Adam([param for param in actor.parameters() if param.requires_grad], lr=actor_lr)
    actor_stage_optimizers = _make_structured_actor_stage_optimizers(actor, actor_lr)
    critic_optimizer = None if critic_model is None else torch.optim.Adam(critic_model.parameters(), lr=critic_lr)
    _trace(f"build_modules:done critic={'off' if critic_model is None else 'on'}")
    learner = StructuredMAPPO(
        actor=actor,
        critic=critic,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_ratio=clip_ratio,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        ppo_epochs=ppo_epochs,
        num_mini_batch=num_mini_batch,
        target_mode="step_level",
        danger_imitation_enabled=bool(getattr(cfg, "danger_imitation_enabled", False)),
        danger_imitation_coef=float(getattr(cfg, "danger_imitation_coef", 0.0) or 0.0),
        actor_optimizer=actor_optimizer,
        actor_stage_optimizers=actor_stage_optimizers,
        critic_optimizer=critic_optimizer,
        device=device,
        cfg=cfg,
        train_accel=train_accel_enabled,
        train_sat=train_sat_enabled,
        train_bw=train_bw_enabled,
        exec_accel_source=str(getattr(cfg, "exec_accel_source", "policy") or "policy"),
        exec_sat_source=str(getattr(cfg, "exec_sat_source", "policy") or "policy"),
        exec_bw_source=str(getattr(cfg, "exec_bw_source", "policy") or "policy"),
    )
    _trace(
        "structured_exec "
        f"train=(accel={learner.train_actor_stage[0]}, sat={learner.train_actor_stage[1]}, bw={learner.train_actor_stage[2]}) "
        f"exec=(accel={learner.exec_source_by_stage[0]}, sat={learner.exec_source_by_stage[1]}, bw={learner.exec_source_by_stage[2]})"
    )
    _emit_console_status(
        "Heads "
        f"train[a={learner.train_actor_stage[0]}, s={learner.train_actor_stage[1]}, b={learner.train_actor_stage[2]}] | "
        f"exec[a={learner.exec_source_by_stage[0]}, s={learner.exec_source_by_stage[1]}, b={learner.exec_source_by_stage[2]}]"
    )
    if critic_model is None:
        critic_reason = "BW actor-only signal path" if bw_actor_only_signal_critic_free else "SAT clean joint path"
        _emit_console_status(f"Critic off | {critic_reason}")

    device = learner.device
    if args.init_actor:
        info = load_checkpoint_forgiving(actor, args.init_actor, map_location=device)
        if info.get("adapted_keys"):
            _trace(f"loaded_actor path={args.init_actor} adapted={len(info['adapted_keys'])}")
    if args.init_critic:
        if critic_model is None:
            raise ValueError(
                "--init_critic is unsupported when critic-free structured training disables critic construction."
            )
        info = load_checkpoint_forgiving(critic_model, args.init_critic, map_location=device)
        if info.get("adapted_keys"):
            _trace(f"loaded_critic path={args.init_critic} adapted={len(info['adapted_keys'])}")

    start_update = 0
    total_env_steps = 0
    total_time_sec = 0.0
    history_rows: List[Dict[str, float]] = []
    checkpoint_eval_state: Dict[str, float] = {}
    checkpoint_eval_fixed_summary: Dict[str, float] | None = None
    if args.resume_state:
        resume_meta = load_structured_train_state(
            args.resume_state,
            actor,
            critic_model,
            actor_optimizer,
            critic_optimizer,
            actor_stage_optimizers=actor_stage_optimizers,
            device=device,
        )
        start_update = int(resume_meta["update"])
        total_env_steps = int(resume_meta["total_env_steps"])
        total_time_sec = float(resume_meta["total_time_sec"])
        history_rows = [dict(row) for row in resume_meta.get("history_rows", [])]
        checkpoint_eval_state = dict(resume_meta.get("checkpoint_eval_state", {}) or {})
        checkpoint_eval_fixed_summary = (
            dict(resume_meta["checkpoint_eval_fixed_summary"])
            if resume_meta.get("checkpoint_eval_fixed_summary") is not None
            else None
        )
        _trace(
            f"resume_state path={args.resume_state} update={start_update} total_env_steps={total_env_steps}"
        )
        _emit_console_status(
            f"Resume from update {start_update} | total_env_steps={total_env_steps}"
        )

    _trace("make_env_group:start")
    env_group = make_structured_env_group(cfg, args.num_envs, backend=args.vec_backend, mode="train")
    _trace("make_env_group:done")
    reset_controller = None
    if args.runtime_state_bank:
        runtime_state_bank_seed = (
            int(args.runtime_state_bank_seed)
            if args.runtime_state_bank_seed is not None
            else int(cfg.seed) + 99173
        )
        reset_controller = _RuntimeStateBankResetController(
            args.runtime_state_bank,
            bank_name=args.runtime_state_bank_name,
            selection_seed=runtime_state_bank_seed,
        )
        reset_controller.bind(env_group)
        _trace(
            "runtime_state_bank:bound "
            f"path={reset_controller.bank_path} size={int(reset_controller.bank_size)} "
            f"seed={int(runtime_state_bank_seed)} name={reset_controller.bank_name}"
        )
        _emit_console_status(
            "Runtime-state bank "
            f"{os.path.basename(reset_controller.bank_path)} | size={int(reset_controller.bank_size)}"
        )
    checkpoint_eval_enabled = (
        bool(getattr(cfg, "checkpoint_eval_enabled", False))
        and int(getattr(cfg, "checkpoint_eval_interval_updates", 0) or 0) > 0
        and int(getattr(cfg, "checkpoint_eval_episodes", 0) or 0) > 0
    )
    checkpoint_eval_interval = max(int(getattr(cfg, "checkpoint_eval_interval_updates", 0) or 0), 0)
    checkpoint_eval_start_update = max(int(getattr(cfg, "checkpoint_eval_start_update", 0) or 0), 0)
    if checkpoint_eval_enabled and checkpoint_eval_start_update <= 0:
        checkpoint_eval_start_update = checkpoint_eval_interval
    checkpoint_eval_csv_path = os.path.join(run_dir, "checkpoint_eval.csv")
    if checkpoint_eval_enabled and start_update <= 0 and os.path.exists(checkpoint_eval_csv_path):
        os.remove(checkpoint_eval_csv_path)
    bw_actor_advantage_override_mode = str(
        getattr(cfg, "bw_actor_advantage_override_mode", "gae") or "gae"
    ).strip().lower()
    if bw_actor_advantage_override_mode not in {"gae", "branch_delta", "true_adv_mc", "delta_critic", "delta_teacher_student"}:
        raise ValueError(
            "bw_actor_advantage_override_mode must be one of {'gae', 'branch_delta', 'true_adv_mc', 'delta_critic', 'delta_teacher_student'}."
        )
    bw_actor_override_enabled = bw_actor_advantage_override_mode in {"branch_delta", "true_adv_mc"}
    bw_actor_branch_override_enabled = bw_actor_advantage_override_mode == "branch_delta"
    bw_actor_true_mc_override_enabled = bw_actor_advantage_override_mode == "true_adv_mc"
    bw_actor_branch_horizon = max(int(getattr(cfg, "bw_actor_branch_horizon", 10) or 10), 1)
    bw_actor_branch_samples = max(int(getattr(cfg, "bw_actor_branch_samples", 1) or 1), 1)
    bw_actor_branch_ref_mode = str(
        getattr(cfg, "bw_actor_branch_ref_mode", "deterministic") or "deterministic"
    ).strip().lower()
    bw_actor_branch_follow_policy_mode = str(
        getattr(cfg, "bw_actor_branch_follow_policy_mode", "stochastic") or "stochastic"
    ).strip().lower()
    bw_actor_branch_normalize = bool(getattr(cfg, "bw_actor_branch_normalize", False))
    bw_actor_branch_gate_snr_threshold = max(float(getattr(cfg, "bw_actor_branch_gate_snr_threshold", 0.0) or 0.0), 0.0)
    bw_actor_true_mc_horizon = max(int(getattr(cfg, "bw_actor_true_mc_horizon", 10) or 10), 1)
    bw_actor_true_mc_samples = max(int(getattr(cfg, "bw_actor_true_mc_samples", 4) or 4), 1)
    update_direction_probe_enabled = (
        bool(getattr(cfg, "update_direction_probe_enabled", False))
        and int(getattr(cfg, "update_direction_probe_interval_updates", 0) or 0) > 0
        and int(getattr(cfg, "update_direction_probe_panel_states", 0) or 0) > 0
        and int(getattr(cfg, "update_direction_probe_k_steps", 0) or 0) > 0
    )
    update_direction_probe_interval = max(int(getattr(cfg, "update_direction_probe_interval_updates", 0) or 0), 0)
    update_direction_probe_start_update = max(int(getattr(cfg, "update_direction_probe_start_update", 0) or 0), 0)
    update_direction_probe_actor_policy_mode = str(
        getattr(cfg, "update_direction_probe_actor_policy_mode", "deterministic") or "deterministic"
    ).strip().lower()
    if update_direction_probe_actor_policy_mode not in {"deterministic", "stochastic"}:
        raise ValueError("update_direction_probe_actor_policy_mode must be one of {'deterministic', 'stochastic'}.")
    update_direction_probe_actor_policy_samples = max(
        int(getattr(cfg, "update_direction_probe_actor_policy_samples", 1) or 1),
        1,
    )
    update_direction_probe_actor_deterministic = update_direction_probe_actor_policy_mode != "stochastic"
    update_direction_probe_true_mc_enabled = bool(getattr(cfg, "update_direction_probe_true_mc_enabled", False))
    update_direction_probe_true_mc_samples = max(
        int(getattr(cfg, "update_direction_probe_true_mc_samples", 0) or 0),
        0,
    )
    if not update_direction_probe_true_mc_enabled:
        update_direction_probe_true_mc_samples = 0
    if update_direction_probe_enabled and update_direction_probe_start_update <= 0:
        update_direction_probe_start_update = update_direction_probe_interval
    update_direction_probe_csv_path = os.path.join(run_dir, "update_direction_probe.csv")
    update_direction_probe_dir = os.path.join(run_dir, "update_direction_probe")
    update_direction_probe_panel = None
    if bw_actor_override_enabled:
        if not (learner.train_actor_stage[2] and not learner.train_actor_stage[0] and not learner.train_actor_stage[1]):
            raise ValueError(
                f"bw_actor_advantage_override_mode={bw_actor_advantage_override_mode} currently requires BW-only actor training "
                "(train_accel=False, train_sat=False, train_bw=True)."
            )
        if int(cfg.num_uav) != 1:
            raise ValueError(
                f"bw_actor_advantage_override_mode={bw_actor_advantage_override_mode} currently requires num_uav == 1."
            )
    if update_direction_probe_enabled:
        if not (learner.train_actor_stage[2] and not learner.train_actor_stage[0] and not learner.train_actor_stage[1]):
            raise ValueError(
                "update_direction_probe_enabled currently requires BW-only actor training "
                "(train_accel=False, train_sat=False, train_bw=True)."
            )
        if int(cfg.num_uav) != 1:
            raise ValueError("update_direction_probe_enabled currently requires num_uav == 1.")
        if start_update <= 0:
            if os.path.exists(update_direction_probe_csv_path):
                os.remove(update_direction_probe_csv_path)
            if os.path.isdir(update_direction_probe_dir):
                shutil.rmtree(update_direction_probe_dir)

    resume_time_offset = float(total_time_sec)
    wall_start = time.perf_counter()
    progress = Progress(max(int(args.updates), 0), desc="Train")
    checkpoint_eval_stop_on_early_stop = bool(getattr(cfg, "checkpoint_eval_stop_on_early_stop", False))
    checkpoint_eval_save_best_models = bool(getattr(cfg, "checkpoint_eval_save_best_models", False))
    checkpoint_eval_use_best_as_final = bool(getattr(cfg, "checkpoint_eval_use_best_as_final", False))
    best_checkpoint_saved = False
    best_checkpoint_update = 0
    final_update_completed = int(start_update)
    history = []
    episode_stats_state = _make_episode_stats_state(
        num_envs=int(args.num_envs),
        episode_stat_window=max(int(getattr(cfg, "train_episode_stat_window", 100) or 100), 1),
    )
    try:
        if checkpoint_eval_enabled and checkpoint_eval_fixed_summary is None:
            fixed_policy = str(getattr(cfg, "checkpoint_eval_fixed_policy", "zero") or "zero")
            _trace(f"checkpoint_eval_reference:start policy={fixed_policy}")
            checkpoint_eval_fixed_summary = evaluate_structured_fixed_policy(
                cfg,
                baseline_policy=fixed_policy,
                episodes=max(int(getattr(cfg, "checkpoint_eval_episodes", 20) or 0), 1),
                episode_seed_base=getattr(cfg, "checkpoint_eval_episode_seed_base", None),
                num_envs=max(int(getattr(cfg, "checkpoint_eval_num_envs", args.num_envs) or args.num_envs), 1),
            )
            _trace(
                f"checkpoint_eval_reference:done policy={fixed_policy} "
                f"reward={checkpoint_eval_fixed_summary['reward_sum']:.4f}, "
                f"weighted={checkpoint_eval_fixed_summary.get('bw_weighted_workload_delta_sum', 0.0):.4f}, "
                f"weighted_level={checkpoint_eval_fixed_summary.get('bw_weighted_workload_level_sum', 0.0):.4f}, "
                f"processed={checkpoint_eval_fixed_summary['processed_ratio_eval']:.4f}, "
                f"drop={checkpoint_eval_fixed_summary['drop_ratio_eval']:.4f}, "
                f"pre_backlog={checkpoint_eval_fixed_summary['pre_backlog_steps_eval']:.4f}, "
                f"sat_overlap={checkpoint_eval_fixed_summary['sat_overlap_eval']:.4f}, "
                f"collision={checkpoint_eval_fixed_summary['collision_episode_fraction']:.4f}"
            )
            _emit_console_status(
                "Fixed reference "
                f"{fixed_policy}: reward={checkpoint_eval_fixed_summary['reward_sum']:.3f} | "
                f"weighted={checkpoint_eval_fixed_summary.get('bw_weighted_workload_delta_sum', 0.0):.3f} | "
                f"weighted_level={checkpoint_eval_fixed_summary.get('bw_weighted_workload_level_sum', 0.0):.3f} | "
                f"processed={checkpoint_eval_fixed_summary['processed_ratio_eval']:.3f} | "
                f"drop={checkpoint_eval_fixed_summary['drop_ratio_eval']:.4f} | "
                f"backlog={checkpoint_eval_fixed_summary['pre_backlog_steps_eval']:.2f}"
            )
            _log_structured_eval_scalars(
                tb_writer,
                update_idx=0,
                summary=checkpoint_eval_fixed_summary,
                fixed_summary=checkpoint_eval_fixed_summary,
            )
        if update_direction_probe_enabled:
            _trace("update_direction_probe:init_panel:start")
            update_direction_probe_panel = collect_bw_snapshot_panel(
                cfg,
                episodes=max(int(getattr(cfg, "update_direction_probe_panel_episodes", 6) or 0), 1),
                states=max(int(getattr(cfg, "update_direction_probe_panel_states", 16) or 0), 1),
                seed=int(getattr(cfg, "update_direction_probe_panel_seed", 35000) or 35000),
            )
            write_bw_update_direction_probe_payload(
                os.path.join(update_direction_probe_dir, "panel_meta.json"),
                {
                    "panel_states": int(len(update_direction_probe_panel)),
                    "panel_episodes": int(getattr(cfg, "update_direction_probe_panel_episodes", 6) or 6),
                    "panel_seed": int(getattr(cfg, "update_direction_probe_panel_seed", 35000) or 35000),
                    "k_steps": int(getattr(cfg, "update_direction_probe_k_steps", 20) or 20),
                    "bw_sample_limit": int(getattr(cfg, "update_direction_probe_bw_sample_limit", 16) or 16),
                    "train_bw_only": True,
                    "actor_policy_mode": str(update_direction_probe_actor_policy_mode),
                    "actor_policy_samples": int(update_direction_probe_actor_policy_samples),
                    "true_mc_enabled": bool(update_direction_probe_true_mc_enabled),
                    "true_mc_samples": int(update_direction_probe_true_mc_samples),
                    "branch_enabled": bool(getattr(cfg, "update_direction_probe_branch_enabled", False)),
                    "branch_horizons": list(getattr(cfg, "update_direction_probe_branch_horizons", [2, 5, 10]) or []),
                    "branch_samples": int(getattr(cfg, "update_direction_probe_branch_samples", 0) or 0),
                    "branch_ref_mode": str(getattr(cfg, "update_direction_probe_branch_ref_mode", "deterministic") or "deterministic"),
                    "branch_follow_policy_mode": str(
                        getattr(cfg, "update_direction_probe_branch_follow_policy_mode", "stochastic") or "stochastic"
                    ),
                },
            )
            _trace(
                "update_direction_probe:init_panel:done "
                f"states={int(len(update_direction_probe_panel))} "
                f"k_steps={int(getattr(cfg, 'update_direction_probe_k_steps', 20) or 20)}"
            )
            _emit_console_status(
                "Update-direction probe "
                f"states={int(len(update_direction_probe_panel))} | "
                f"k={int(getattr(cfg, 'update_direction_probe_k_steps', 20) or 20)} | "
                f"mode={update_direction_probe_actor_policy_mode} | "
                f"samples={int(update_direction_probe_actor_policy_samples)} | "
                f"true_mc={int(update_direction_probe_true_mc_samples)}"
            )
        for update_idx in range(start_update + 1, int(args.updates) + 1):
            iter_start = time.perf_counter()
            _trace(f"update:start idx={int(update_idx)} rollout_env_steps={int(rollout_env_steps)}")
            update_direction_probe_due = (
                update_direction_probe_enabled
                and update_idx >= update_direction_probe_start_update
                and ((update_idx - update_direction_probe_start_update) % update_direction_probe_interval == 0)
            )
            pre_probe_eval = None
            pre_probe_actor = None
            probe_context_holder = {"value": None}
            if update_direction_probe_due:
                pre_probe_actor = copy.deepcopy(actor).to(device).eval()
                panel_eval_seed_base = int(cfg.seed) + 2_000_000 + int(update_idx) * 10_000
                _trace(f"update_direction_probe:pre:start idx={int(update_idx)}")
                pre_probe_eval = evaluate_bw_snapshot_panel(
                    pre_probe_actor,
                    update_direction_probe_panel or [],
                    cfg,
                    device,
                    k_steps=int(getattr(cfg, "update_direction_probe_k_steps", 20) or 20),
                    deterministic=update_direction_probe_actor_deterministic,
                    actor_policy_samples=int(update_direction_probe_actor_policy_samples),
                    seed_base=int(panel_eval_seed_base),
                )
                _trace(
                    f"update_direction_probe:pre:done idx={int(update_idx)} "
                    f"actor_reward={float(pre_probe_eval['actor_reward']['mean']):.6f}"
                )
            def _capture_probe_context(buffer, bootstrap_world_state) -> None:
                if bw_actor_override_enabled:
                    learner.clear_actor_advantage_override()
                    if bw_actor_branch_override_enabled:
                        override_summary = compute_bw_branch_advantage_override(
                            learner=learner,
                            buffer=buffer,
                            bootstrap_world_state=bootstrap_world_state,
                            cfg=cfg,
                            device=device,
                            horizon=int(bw_actor_branch_horizon),
                            branch_samples=int(bw_actor_branch_samples),
                            branch_seed=int(cfg.seed) + 3_000_000 + int(update_idx) * 10_000,
                            ref_mode=str(bw_actor_branch_ref_mode),
                            follow_policy_mode=str(bw_actor_branch_follow_policy_mode),
                            normalize=bool(bw_actor_branch_normalize),
                        )
                        if override_summary is None:
                            raise RuntimeError("branch_delta actor override produced no BW samples.")
                        learner.set_actor_advantage_override(
                            int(override_summary["stage_id"]),
                            override_summary["advantages"],
                        )
                        branch_std = max(float(override_summary.get("branch_std", 0.0) or 0.0), 1e-8)
                        branch_abs_mean = float(override_summary.get("branch_abs_mean", 0.0) or 0.0)
                        branch_gate_snr = branch_abs_mean / branch_std
                        branch_gate_triggered = (
                            bw_actor_branch_gate_snr_threshold > 0.0
                            and branch_gate_snr < bw_actor_branch_gate_snr_threshold
                        )
                        learner.set_bw_branch_gate_state(
                            snr=branch_gate_snr,
                            triggered=branch_gate_triggered,
                        )
                        if branch_gate_triggered:
                            learner.request_skip_actor_update_once(int(override_summary["stage_id"]))
                        _trace(
                            "branch_actor_override:"
                            f" idx={int(update_idx)}"
                            f" samples={int(override_summary['sample_count'])}"
                            f" h={int(override_summary['horizon'])}"
                            f" branch_abs={float(override_summary['branch_abs_mean']):.6f}"
                            f" branch_std={float(override_summary['branch_std']):.6f}"
                            f" branch_snr={branch_gate_snr:.6f}"
                            f" gate={int(bool(branch_gate_triggered))}"
                            f" corr_def={float(override_summary['corr_default_vs_branch']):+.3f}"
                            f" sign={float(override_summary['sign_agree_default_vs_branch']):.3f}"
                            f" norm={int(bool(override_summary['normalize']))}"
                        )
                    elif bw_actor_true_mc_override_enabled:
                        override_summary = compute_bw_true_advantage_override(
                            learner=learner,
                            buffer=buffer,
                            bootstrap_world_state=bootstrap_world_state,
                            cfg=cfg,
                            device=device,
                            k_steps=int(bw_actor_true_mc_horizon),
                            true_mc_samples=int(bw_actor_true_mc_samples),
                            true_mc_seed=int(cfg.seed) + 4_000_000 + int(update_idx) * 10_000,
                        )
                        if override_summary is None:
                            raise RuntimeError("true_adv_mc actor override produced no BW samples.")
                        learner.set_actor_advantage_override(
                            int(override_summary["stage_id"]),
                            override_summary["advantages"],
                        )
                        _trace(
                            "true_mc_actor_override:"
                            f" idx={int(update_idx)}"
                            f" samples={int(override_summary['sample_count'])}"
                            f" h={int(override_summary['horizon'])}"
                            f" mc={int(override_summary['true_mc_samples'])}"
                            f" abs={float(override_summary['true_adv_abs_mean']):.6f}"
                            f" corr_def={float(override_summary['corr_default_vs_true_adv']):+.3f}"
                            f" sign={float(override_summary['sign_agree_default_vs_true_adv']):.3f}"
                        )
                if update_direction_probe_due:
                    probe_context_holder["buffer"] = buffer
                    probe_context_holder["bootstrap_world_state"] = bootstrap_world_state
            learner.current_update_index = int(update_idx)
            rows = run_structured_training(
                env_group,
                learner,
                num_updates=1,
                rollout_env_steps=rollout_env_steps,
                reset_seed=int(cfg.seed) + (update_idx - 1) * max(args.num_envs, 1) * 1000,
                reset_on_start=bool(update_idx == start_update + 1),
                reset_controller=reset_controller,
                episode_stats_state=episode_stats_state,
                episode_stat_window=max(int(getattr(cfg, "train_episode_stat_window", 100) or 100), 1),
                trace_fn=trace,
                trace_interval=max(int(getattr(cfg, "train_trace_rollout_interval", 0) or 0), 0),
                before_update_callback=_capture_probe_context
                if (update_direction_probe_due or bw_actor_override_enabled)
                else None,
            )
            history.extend(rows)
            row = rows[-1]
            final_update_completed = int(update_idx)
            total_env_steps += int(rollout_env_steps) * int(args.num_envs)
            iter_wall = max(time.perf_counter() - iter_start, 1e-9)
            env_steps_per_sec = float(row.env_steps_per_sec) if float(row.env_steps_per_sec) > 0.0 else float(
                (int(rollout_env_steps) * int(args.num_envs)) / iter_wall
            )
            row_payload = asdict(row)
            row_payload["update"] = int(update_idx)
            history_rows.append(row_payload)
            total_time_sec = resume_time_offset + float(time.perf_counter() - wall_start)
            _log_structured_train_scalars(
                tb_writer,
                update_idx=int(update_idx),
                total_env_steps=int(total_env_steps),
                row_payload=row_payload,
                env_steps_per_sec=float(env_steps_per_sec),
            )
            value_progress = (
                "v=off "
                if bool(getattr(learner, "disable_critic_training_for_bw_actor_only_signal", False))
                else f"v={row.value_loss:.2f} "
            )
            progress.desc = (
                f"Train r={row.env_reward_mean:.3f} "
                f"{value_progress}"
                f"{env_steps_per_sec:.0f} env/s "
                f"{float(row.samples_per_sec):.0f} samples/s"
            )
            progress.update(update_idx)
            value_trace = (
                "value=off"
                if bool(getattr(learner, "disable_critic_training_for_bw_actor_only_signal", False))
                else f"value={row.value_loss:.6f}"
            )
            _trace(
                f"update:done idx={int(update_idx)} reward={row.env_reward_mean:.6f} "
                f"env_s={env_steps_per_sec:.2f} samples_s={float(row.samples_per_sec):.2f} "
                f"policy={row.policy_loss:.6f} {value_trace}"
            )
            if update_direction_probe_due and pre_probe_eval is not None:
                if probe_context_holder.get("value") is None and probe_context_holder.get("buffer") is not None:
                    probe_context_holder["value"] = build_bw_advantage_probe_context(
                        learner,
                        probe_context_holder["buffer"],
                        probe_context_holder.get("bootstrap_world_state"),
                        sample_limit=max(int(getattr(cfg, "update_direction_probe_bw_sample_limit", 16) or 16), 1),
                        sample_seed=int(cfg.seed) + 100000 + int(update_idx),
                    )
                _trace(f"update_direction_probe:post:start idx={int(update_idx)}")
                post_probe_eval = evaluate_bw_snapshot_panel(
                    actor,
                    update_direction_probe_panel or [],
                    cfg,
                    device,
                    k_steps=int(getattr(cfg, "update_direction_probe_k_steps", 20) or 20),
                    deterministic=update_direction_probe_actor_deterministic,
                    actor_policy_samples=int(update_direction_probe_actor_policy_samples),
                    seed_base=int(panel_eval_seed_base),
                )
                _trace(
                    f"update_direction_probe:post:done idx={int(update_idx)} "
                    f"actor_reward={float(post_probe_eval['actor_reward']['mean']):.6f}"
                )
                pre_actor_reward = float(pre_probe_eval["actor_reward"]["mean"])
                post_actor_reward = float(post_probe_eval["actor_reward"]["mean"])
                pre_actor_weighted = float(pre_probe_eval["actor_weighted"]["mean"])
                post_actor_weighted = float(post_probe_eval["actor_weighted"]["mean"])
                pre_heuristic_reward = float(pre_probe_eval["heuristic_reward"]["mean"])
                post_heuristic_reward = float(post_probe_eval["heuristic_reward"]["mean"])
                pre_l1 = float(pre_probe_eval["l1_to_heur"]["mean"])
                post_l1 = float(post_probe_eval["l1_to_heur"]["mean"])
                pre_gap = pre_heuristic_reward - pre_actor_reward
                post_gap = post_heuristic_reward - post_actor_reward
                pre_heuristic_beats = float(pre_probe_eval["heuristic_beats_actor_frac"])
                post_heuristic_beats = float(post_probe_eval["heuristic_beats_actor_frac"])
                pre_state_actor_reward = np.asarray(pre_probe_eval.get("state_actor_reward_means", []), dtype=np.float64)
                post_state_actor_reward = np.asarray(post_probe_eval.get("state_actor_reward_means", []), dtype=np.float64)
                actor_reward_improved_state_frac = 0.0
                if pre_state_actor_reward.size > 0 and pre_state_actor_reward.size == post_state_actor_reward.size:
                    actor_reward_improved_state_frac = float(
                        np.mean(post_state_actor_reward > pre_state_actor_reward + 1.0e-9)
                    )
                alignment_eval = None
                if pre_probe_actor is not None and probe_context_holder["value"] is not None:
                    alignment_eval = evaluate_bw_advantage_alignment(
                        pre_actor=pre_probe_actor,
                        post_actor=actor,
                        probe_context=probe_context_holder["value"],
                        cfg=cfg,
                        device=device,
                        k_steps=int(getattr(cfg, "update_direction_probe_k_steps", 20) or 20),
                        true_mc_enabled=bool(update_direction_probe_true_mc_enabled),
                        true_mc_samples=int(update_direction_probe_true_mc_samples),
                        true_mc_seed=int(cfg.seed) + 3_000_000 + int(update_idx) * 10_000,
                        branch_enabled=bool(getattr(cfg, "update_direction_probe_branch_enabled", False)),
                        branch_horizons=list(getattr(cfg, "update_direction_probe_branch_horizons", [2, 5, 10]) or []),
                        branch_samples=max(int(getattr(cfg, "update_direction_probe_branch_samples", 0) or 0), 0),
                        branch_seed=int(cfg.seed) + 4_000_000 + int(update_idx) * 10_000,
                        branch_ref_mode=str(getattr(cfg, "update_direction_probe_branch_ref_mode", "deterministic") or "deterministic"),
                        branch_follow_policy_mode=str(
                            getattr(cfg, "update_direction_probe_branch_follow_policy_mode", "stochastic")
                            or "stochastic"
                        ),
                    )
                branch_alignment = {} if alignment_eval is None else alignment_eval.get("branch_alignment", {})
                branch_primary = branch_alignment.get("primary", {}) if branch_alignment else {}
                branch_by_horizon = {
                    int(item.get("horizon", 0)): item
                    for item in branch_alignment.get("by_horizon", [])
                    if int(item.get("horizon", 0)) > 0
                } if branch_alignment else {}
                def _branch_primary_float(name: str, default: float = 0.0) -> float:
                    return float(branch_primary.get(name, default)) if branch_primary else float(default)
                def _branch_horizon_float(horizon: int, name: str, default: float = 0.0) -> float:
                    return float(branch_by_horizon.get(int(horizon), {}).get(name, default))
                probe_row = {
                    "update": int(update_idx),
                    "panel_states": int(len(update_direction_probe_panel or [])),
                    "k_steps": int(getattr(cfg, "update_direction_probe_k_steps", 20) or 20),
                    "actor_policy_mode": str(update_direction_probe_actor_policy_mode),
                    "actor_policy_samples": int(update_direction_probe_actor_policy_samples),
                    "pre_actor_reward_mean": pre_actor_reward,
                    "post_actor_reward_mean": post_actor_reward,
                    "delta_actor_reward_mean": post_actor_reward - pre_actor_reward,
                    "actor_reward_improved_state_frac": float(actor_reward_improved_state_frac),
                    "pre_actor_weighted_mean": pre_actor_weighted,
                    "post_actor_weighted_mean": post_actor_weighted,
                    "delta_actor_weighted_mean": post_actor_weighted - pre_actor_weighted,
                    "pre_heuristic_reward_mean": pre_heuristic_reward,
                    "post_heuristic_reward_mean": post_heuristic_reward,
                    "pre_heuristic_gap_mean": pre_gap,
                    "post_heuristic_gap_mean": post_gap,
                    "delta_heuristic_gap_mean": post_gap - pre_gap,
                    "pre_l1_to_heur_mean": pre_l1,
                    "post_l1_to_heur_mean": post_l1,
                    "delta_l1_to_heur_mean": post_l1 - pre_l1,
                    "pre_heuristic_beats_actor_frac": pre_heuristic_beats,
                    "post_heuristic_beats_actor_frac": post_heuristic_beats,
                    "delta_heuristic_beats_actor_frac": post_heuristic_beats - pre_heuristic_beats,
                    "probe_batch_samples": (
                        0 if alignment_eval is None else int(alignment_eval["logprob_alignment"]["sample_count"])
                    ),
                    "probe_action_gap_samples": (
                        0 if alignment_eval is None else int(alignment_eval["critic_alignment"]["sample_count"])
                    ),
                    "adv_positive_frac": (
                        0.0 if alignment_eval is None else float(alignment_eval["advantage"]["positive_frac"])
                    ),
                    "old_logprob_replay_abs_diff_mean": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["logprob_alignment"]["old_logprob_replay_abs_diff"]["mean"])
                    ),
                    "old_logprob_exact_replay_frac_1e-5": (
                        0.0 if alignment_eval is None else float(alignment_eval["logprob_alignment"]["old_logprob_exact_replay_frac_1e-5"])
                    ),
                    "corr_advantage_vs_delta_logprob": (
                        0.0 if alignment_eval is None else float(alignment_eval["logprob_alignment"]["corr_advantage_vs_delta_logprob"])
                    ),
                    "mean_delta_logprob_pos_adv": (
                        0.0 if alignment_eval is None else float(alignment_eval["logprob_alignment"]["mean_delta_logprob_pos_adv"])
                    ),
                    "mean_delta_logprob_neg_adv": (
                        0.0 if alignment_eval is None else float(alignment_eval["logprob_alignment"]["mean_delta_logprob_neg_adv"])
                    ),
                    "pos_adv_logprob_up_frac": (
                        0.0 if alignment_eval is None else float(alignment_eval["logprob_alignment"]["pos_adv_logprob_up_frac"])
                    ),
                    "neg_adv_logprob_down_frac": (
                        0.0 if alignment_eval is None else float(alignment_eval["logprob_alignment"]["neg_adv_logprob_down_frac"])
                    ),
                    "corr_advantage_vs_action_gap_k": (
                        0.0 if alignment_eval is None else float(alignment_eval["critic_alignment"]["corr_advantage_vs_action_gap_k"])
                    ),
                    "mean_action_gap_k_pos_adv": (
                        0.0 if alignment_eval is None else float(alignment_eval["critic_alignment"]["mean_action_gap_k_pos_adv"])
                    ),
                    "mean_action_gap_k_neg_adv": (
                        0.0 if alignment_eval is None else float(alignment_eval["critic_alignment"]["mean_action_gap_k_neg_adv"])
                    ),
                    "pos_adv_action_gap_positive_frac": (
                        0.0 if alignment_eval is None else float(alignment_eval["critic_alignment"]["pos_adv_action_gap_positive_frac"])
                    ),
                    "neg_adv_action_gap_negative_frac": (
                        0.0 if alignment_eval is None else float(alignment_eval["critic_alignment"]["neg_adv_action_gap_negative_frac"])
                    ),
                    "branch_enabled": (
                        0 if not branch_alignment else int(bool(branch_alignment.get("enabled", False)))
                    ),
                    "branch_ref_mode": "" if not branch_alignment else str(branch_alignment.get("ref_mode", "")),
                    "branch_follow_policy_mode": (
                        "" if not branch_alignment else str(branch_alignment.get("follow_policy_mode", ""))
                    ),
                    "branch_samples_per_action": (
                        0 if not branch_alignment else int(branch_alignment.get("samples_per_action", 0))
                    ),
                    "branch_primary_horizon": (
                        0 if not branch_alignment else int(branch_alignment.get("primary_horizon", 0))
                    ),
                    "branch_sample_count": _branch_primary_float("sample_count"),
                    "corr_advantage_vs_branch_delta": _branch_primary_float("corr_advantage_vs_branch_delta"),
                    "corr_raw_advantage_vs_branch_delta": _branch_primary_float("corr_raw_advantage_vs_branch_delta"),
                    "sign_agree_raw_advantage_branch_delta": _branch_primary_float("sign_agree_raw_advantage_branch_delta"),
                    "corr_branch_delta_vs_delta_logprob": _branch_primary_float("corr_branch_delta_vs_delta_logprob"),
                    "branch_delta_abs_mean": _branch_primary_float("branch_delta_abs_mean"),
                    "branch_delta_std": _branch_primary_float("branch_delta_std"),
                    "branch_delta_snr": _branch_primary_float("branch_delta_snr"),
                    "mean_branch_delta_pos_adv": _branch_primary_float("mean_branch_delta_pos_adv"),
                    "mean_branch_delta_neg_adv": _branch_primary_float("mean_branch_delta_neg_adv"),
                    "mean_branch_delta_pos_raw_adv": _branch_primary_float("mean_branch_delta_pos_raw_adv"),
                    "mean_branch_delta_neg_raw_adv": _branch_primary_float("mean_branch_delta_neg_raw_adv"),
                    "raw_pos_branch_delta_positive_frac": _branch_primary_float("raw_pos_branch_delta_positive_frac"),
                    "raw_neg_branch_delta_negative_frac": _branch_primary_float("raw_neg_branch_delta_negative_frac"),
                    "branch_delta_pos_logprob_up_frac": _branch_primary_float("branch_delta_pos_logprob_up_frac"),
                    "branch_delta_neg_logprob_down_frac": _branch_primary_float("branch_delta_neg_logprob_down_frac"),
                    "corr_raw_advantage_vs_branch_delta_h2": _branch_horizon_float(2, "corr_raw_advantage_vs_branch_delta"),
                    "corr_branch_delta_vs_delta_logprob_h2": _branch_horizon_float(2, "corr_branch_delta_vs_delta_logprob"),
                    "branch_delta_abs_mean_h2": _branch_horizon_float(2, "branch_delta_abs_mean"),
                    "corr_raw_advantage_vs_branch_delta_h5": _branch_horizon_float(5, "corr_raw_advantage_vs_branch_delta"),
                    "corr_branch_delta_vs_delta_logprob_h5": _branch_horizon_float(5, "corr_branch_delta_vs_delta_logprob"),
                    "branch_delta_abs_mean_h5": _branch_horizon_float(5, "branch_delta_abs_mean"),
                    "corr_raw_advantage_vs_branch_delta_h10": _branch_horizon_float(10, "corr_raw_advantage_vs_branch_delta"),
                    "corr_branch_delta_vs_delta_logprob_h10": _branch_horizon_float(10, "corr_branch_delta_vs_delta_logprob"),
                    "branch_delta_abs_mean_h10": _branch_horizon_float(10, "branch_delta_abs_mean"),
                    "true_mc_sample_count": (
                        0 if alignment_eval is None else int(alignment_eval["true_action_alignment"]["sample_count"])
                    ),
                    "true_mc_policy_samples_per_action": (
                        0 if alignment_eval is None else int(alignment_eval["true_action_alignment"]["policy_samples_per_action"])
                    ),
                    "corr_advantage_vs_true_adv_mc": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["corr_advantage_vs_true_adv_mc"])
                    ),
                    "sign_agree_advantage_true_adv_mc": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["sign_agree_advantage_true_adv_mc"])
                    ),
                    "corr_true_adv_mc_vs_delta_logprob": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["corr_true_adv_mc_vs_delta_logprob"])
                    ),
                    "true_adv_mc_abs_mean": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["true_adv_mc_abs_mean"])
                    ),
                    "true_adv_mc_std": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["true_adv_mc_std"])
                    ),
                    "true_adv_mc_snr": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["true_adv_mc_snr"])
                    ),
                    "mean_true_adv_mc_pos_adv": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["mean_true_adv_mc_pos_adv"])
                    ),
                    "mean_true_adv_mc_neg_adv": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["mean_true_adv_mc_neg_adv"])
                    ),
                    "ppo_pos_true_adv_positive_frac": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["ppo_pos_true_adv_positive_frac"])
                    ),
                    "ppo_neg_true_adv_negative_frac": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["ppo_neg_true_adv_negative_frac"])
                    ),
                    "mean_delta_logprob_true_adv_pos": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["mean_delta_logprob_true_adv_pos"])
                    ),
                    "mean_delta_logprob_true_adv_neg": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["mean_delta_logprob_true_adv_neg"])
                    ),
                    "true_adv_pos_logprob_up_frac": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["true_adv_pos_logprob_up_frac"])
                    ),
                    "true_adv_neg_logprob_down_frac": (
                        0.0 if alignment_eval is None else float(alignment_eval["true_action_alignment"]["true_adv_neg_logprob_down_frac"])
                    ),
                    "corr_norm_advantage_vs_raw_advantage": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["corr_norm_advantage_vs_raw_advantage"])
                    ),
                    "sign_agree_norm_raw_advantage": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["sign_agree_norm_raw_advantage"])
                    ),
                    "corr_raw_advantage_vs_true_adv_mc": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["corr_raw_advantage_vs_true_adv_mc"])
                    ),
                    "sign_agree_raw_advantage_true_adv_mc": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["sign_agree_raw_advantage_true_adv_mc"])
                    ),
                    "corr_raw_advantage_vs_delta_logprob": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["corr_raw_advantage_vs_delta_logprob"])
                    ),
                    "raw_advantage_abs_mean": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["raw_advantage_abs_mean"])
                    ),
                    "raw_advantage_std": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["raw_advantage_std"])
                    ),
                    "raw_advantage_snr": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["raw_advantage_snr"])
                    ),
                    "corr_return_target_vs_sampled_q_mc": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["corr_return_target_vs_sampled_q_mc"])
                    ),
                    "return_target_sampled_q_error_mean": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["return_target_sampled_q_error_mean"])
                    ),
                    "return_target_sampled_q_error_abs_mean": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["return_target_sampled_q_error_abs_mean"])
                    ),
                    "corr_value_vs_policy_q_mc": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["corr_value_vs_policy_q_mc"])
                    ),
                    "value_policy_q_error_mean": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["value_policy_q_error_mean"])
                    ),
                    "value_policy_q_error_abs_mean": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["value_policy_q_error_abs_mean"])
                    ),
                    "return_target_std": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["return_target_std"])
                    ),
                    "sampled_q_mc_std": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["sampled_q_mc_std"])
                    ),
                    "value_pred_std": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["value_pred_std"])
                    ),
                    "policy_q_mc_std": (
                        0.0
                        if alignment_eval is None
                        else float(alignment_eval["true_action_alignment"]["advantage_decomposition"]["policy_q_mc_std"])
                    ),
                    "alignment_judgement": (
                        "" if alignment_eval is None else str(alignment_eval["judgement"])
                    ),
                    "env_reward_mean": float(row.env_reward_mean),
                    "bw_train_reward_mean": float(row.bw_train_reward_mean),
                    "policy_loss": float(row.policy_loss),
                    "value_loss_bw": float(row.value_loss_bw),
                    "critic_popart_mean_bw": float(row.critic_popart_mean_bw),
                    "critic_popart_std_bw": float(row.critic_popart_std_bw),
                    "approx_kl_bw": float(row.approx_kl_bw),
                    "clip_frac_bw": float(row.clip_frac_bw),
                    "entropy_bw": float(row.entropy_bw),
                }
                append_bw_update_direction_probe_row(update_direction_probe_csv_path, probe_row)
                write_bw_update_direction_probe_payload(
                    os.path.join(update_direction_probe_dir, f"u{int(update_idx):04d}.json"),
                    {
                        "update": int(update_idx),
                        "panel_states": int(len(update_direction_probe_panel or [])),
                        "k_steps": int(getattr(cfg, "update_direction_probe_k_steps", 20) or 20),
                        "actor_policy_mode": str(update_direction_probe_actor_policy_mode),
                        "actor_policy_samples": int(update_direction_probe_actor_policy_samples),
                        "true_mc_enabled": bool(update_direction_probe_true_mc_enabled),
                        "true_mc_samples": int(update_direction_probe_true_mc_samples),
                        "branch_enabled": bool(getattr(cfg, "update_direction_probe_branch_enabled", False)),
                        "branch_horizons": list(getattr(cfg, "update_direction_probe_branch_horizons", [2, 5, 10]) or []),
                        "branch_samples": int(getattr(cfg, "update_direction_probe_branch_samples", 0) or 0),
                        "branch_ref_mode": str(getattr(cfg, "update_direction_probe_branch_ref_mode", "deterministic") or "deterministic"),
                        "branch_follow_policy_mode": str(
                            getattr(cfg, "update_direction_probe_branch_follow_policy_mode", "stochastic") or "stochastic"
                        ),
                        "pre": pre_probe_eval,
                        "post": post_probe_eval,
                        "delta": {
                            "actor_reward_mean": float(probe_row["delta_actor_reward_mean"]),
                            "actor_reward_improved_state_frac": float(probe_row["actor_reward_improved_state_frac"]),
                            "actor_weighted_mean": float(probe_row["delta_actor_weighted_mean"]),
                            "heuristic_gap_mean": float(probe_row["delta_heuristic_gap_mean"]),
                            "l1_to_heur_mean": float(probe_row["delta_l1_to_heur_mean"]),
                            "heuristic_beats_actor_frac": float(probe_row["delta_heuristic_beats_actor_frac"]),
                        },
                        "alignment": alignment_eval,
                        "update_metrics": {
                            "env_reward_mean": float(row.env_reward_mean),
                            "bw_train_reward_mean": float(row.bw_train_reward_mean),
                            "policy_loss": float(row.policy_loss),
                            "value_loss_bw": float(row.value_loss_bw),
                            "critic_popart_mean_bw": float(row.critic_popart_mean_bw),
                            "critic_popart_std_bw": float(row.critic_popart_std_bw),
                            "approx_kl_bw": float(row.approx_kl_bw),
                            "clip_frac_bw": float(row.clip_frac_bw),
                            "entropy_bw": float(row.entropy_bw),
                        },
                    },
                )
                _log_update_direction_probe_scalars(
                    tb_writer,
                    update_idx=int(update_idx),
                    pre_eval=pre_probe_eval,
                    post_eval=post_probe_eval,
                    alignment=alignment_eval,
                )
                _emit_console_status(
                    f"DirProbe u{int(update_idx):04d} | "
                    f"dSelfReward={float(probe_row['delta_actor_reward_mean']):+.4f} | "
                    f"better={float(probe_row['actor_reward_improved_state_frac']):.3f} | "
                    f"dGap={float(probe_row['delta_heuristic_gap_mean']):+.4f} | "
                    f"adv->gap={float(probe_row['corr_advantage_vs_action_gap_k']):+.3f} | "
                    f"raw->branch={float(probe_row['corr_raw_advantage_vs_branch_delta']):+.3f} | "
                    f"branch->dlogp={float(probe_row['corr_branch_delta_vs_delta_logprob']):+.3f} | "
                    f"adv->true={float(probe_row['corr_advantage_vs_true_adv_mc']):+.3f} | "
                    f"true->dlogp={float(probe_row['corr_true_adv_mc_vs_delta_logprob']):+.3f} | "
                    f"adv->dlogp={float(probe_row['corr_advantage_vs_delta_logprob']):+.3f} | "
                    f"{probe_row['alignment_judgement']}"
                )
            if args.save_interval > 0 and update_idx % int(args.save_interval) == 0:
                _trace(f"checkpoint:save_start idx={int(update_idx)}")
                _save_periodic_structured_checkpoint(
                    run_dir,
                    actor,
                    critic_model,
                    actor_optimizer,
                    critic_optimizer,
                    actor_stage_optimizers=actor_stage_optimizers,
                    update_idx=update_idx,
                    total_updates=int(args.updates),
                    total_env_steps=total_env_steps,
                    history_rows=history_rows,
                    total_time_sec=total_time_sec,
                    checkpoint_eval_state=checkpoint_eval_state,
                    checkpoint_eval_fixed_summary=checkpoint_eval_fixed_summary,
                )
                _trace(f"checkpoint:save_done idx={int(update_idx)}")
            checkpoint_eval_due = (
                checkpoint_eval_enabled
                and update_idx >= checkpoint_eval_start_update
                and ((update_idx - checkpoint_eval_start_update) % checkpoint_eval_interval == 0)
            )
            if checkpoint_eval_due:
                _trace(f"checkpoint_eval:start idx={int(update_idx)}")
                exec_accel_source = str(getattr(cfg, "exec_accel_source", "policy") or "policy").strip().lower()
                exec_sat_source = str(getattr(cfg, "exec_sat_source", "policy") or "policy").strip().lower()
                exec_bw_source = str(getattr(cfg, "exec_bw_source", "policy") or "policy").strip().lower()
                checkpoint_eval_policy_mode = str(
                    getattr(cfg, "checkpoint_eval_policy_mode", "deterministic") or "deterministic"
                ).strip().lower()
                if checkpoint_eval_policy_mode not in {"deterministic", "stochastic"}:
                    raise ValueError(
                        f"Unsupported checkpoint_eval_policy_mode={checkpoint_eval_policy_mode!r}; "
                        "expected 'deterministic' or 'stochastic'."
                    )
                checkpoint_eval_deterministic = checkpoint_eval_policy_mode != "stochastic"
                if {exec_accel_source, exec_sat_source, exec_bw_source} == {"policy"}:
                    summary, _ = evaluate_structured_actor(
                        cfg,
                        actor,
                        device=device,
                        episodes=max(int(getattr(cfg, "checkpoint_eval_episodes", 20) or 0), 1),
                        episode_seed_base=getattr(cfg, "checkpoint_eval_episode_seed_base", None),
                        deterministic=checkpoint_eval_deterministic,
                    )
                else:
                    summary, _ = evaluate_structured_actor_exec_sources(
                        cfg,
                        actor,
                        device=device,
                        episodes=max(int(getattr(cfg, "checkpoint_eval_episodes", 20) or 0), 1),
                        episode_seed_base=getattr(cfg, "checkpoint_eval_episode_seed_base", None),
                        deterministic=checkpoint_eval_deterministic,
                        num_envs=max(int(getattr(cfg, "checkpoint_eval_num_envs", args.num_envs) or args.num_envs), 1),
                        vec_backend=str(getattr(cfg, "checkpoint_eval_vec_backend", args.vec_backend) or args.vec_backend),
                    )
                flags = update_structured_checkpoint_eval_state(checkpoint_eval_state, summary, cfg)
                row_payload = {
                    "update": int(update_idx),
                    "checkpoint_suffix": f"u{int(update_idx):04d}",
                    "episodes": summary["episodes"],
                    "reward_sum": summary["reward_sum"],
                    "bw_weighted_workload_delta_sum": summary.get("bw_weighted_workload_delta_sum", 0.0),
                    "bw_weighted_workload_level_sum": summary.get("bw_weighted_workload_level_sum", 0.0),
                    "processed_ratio_eval": summary["processed_ratio_eval"],
                    "drop_ratio_eval": summary["drop_ratio_eval"],
                    "pre_backlog_steps_eval": summary["pre_backlog_steps_eval"],
                    "D_sys_report": summary["D_sys_report"],
                    "x_acc_mean": summary["x_acc_mean"],
                    "x_rel_mean": summary["x_rel_mean"],
                    "g_pre_mean": summary["g_pre_mean"],
                    "d_pre_mean": summary["d_pre_mean"],
                    "sat_overlap_eval": summary["sat_overlap_eval"],
                    "collision_episode_fraction": summary["collision_episode_fraction"],
                }
                fixed_summary = checkpoint_eval_fixed_summary or summary
                row_payload.update(
                    {
                        "fixed_reward_sum": fixed_summary["reward_sum"],
                        "fixed_bw_weighted_workload_delta_sum": fixed_summary.get("bw_weighted_workload_delta_sum", 0.0),
                        "fixed_bw_weighted_workload_level_sum": fixed_summary.get("bw_weighted_workload_level_sum", 0.0),
                        "fixed_processed_ratio_eval": fixed_summary["processed_ratio_eval"],
                        "fixed_drop_ratio_eval": fixed_summary["drop_ratio_eval"],
                        "fixed_pre_backlog_steps_eval": fixed_summary["pre_backlog_steps_eval"],
                        "fixed_D_sys_report": fixed_summary["D_sys_report"],
                        "fixed_x_acc_mean": fixed_summary["x_acc_mean"],
                        "fixed_x_rel_mean": fixed_summary["x_rel_mean"],
                        "fixed_g_pre_mean": fixed_summary["g_pre_mean"],
                        "fixed_d_pre_mean": fixed_summary["d_pre_mean"],
                        "fixed_sat_overlap_eval": fixed_summary["sat_overlap_eval"],
                        "fixed_collision_episode_fraction": fixed_summary["collision_episode_fraction"],
                    }
                )
                row_payload.update(flags)
                append_structured_checkpoint_eval_row(checkpoint_eval_csv_path, row_payload)
                if checkpoint_eval_save_best_models and float(flags.get("model_improved", 0.0)) > 0.5:
                    _trace(f"checkpoint_eval:save_best idx={int(update_idx)}")
                    _save_named_structured_checkpoint(
                        run_dir,
                        actor,
                        critic_model,
                        actor_optimizer,
                        critic_optimizer,
                        actor_stage_optimizers=actor_stage_optimizers,
                        suffix="best",
                        update_idx=update_idx,
                        total_updates=int(args.updates),
                        total_env_steps=total_env_steps,
                        history_rows=history_rows,
                        total_time_sec=resume_time_offset + float(time.perf_counter() - wall_start),
                        checkpoint_eval_state=checkpoint_eval_state,
                        checkpoint_eval_fixed_summary=checkpoint_eval_fixed_summary,
                    )
                    best_checkpoint_saved = True
                    best_checkpoint_update = int(update_idx)
                _log_structured_eval_scalars(
                    tb_writer,
                    update_idx=int(update_idx),
                    summary=summary,
                    fixed_summary=fixed_summary,
                )
                _trace(
                    f"checkpoint_eval:done idx={int(update_idx)} "
                    f"reward={summary['reward_sum']:.4f} "
                    f"weighted={summary.get('bw_weighted_workload_delta_sum', 0.0):.4f} "
                    f"weighted_level={summary.get('bw_weighted_workload_level_sum', 0.0):.4f} "
                    f"processed={summary['processed_ratio_eval']:.4f} "
                    f"drop={summary['drop_ratio_eval']:.4f} "
                    f"pre_backlog={summary['pre_backlog_steps_eval']:.4f}"
                )
                _emit_console_status(
                    f"Eval u{int(update_idx):04d} | reward={summary['reward_sum']:.3f} | "
                    f"weighted={summary.get('bw_weighted_workload_delta_sum', 0.0):.3f} | "
                    f"weighted_level={summary.get('bw_weighted_workload_level_sum', 0.0):.3f} | "
                    f"processed={summary['processed_ratio_eval']:.3f} | "
                    f"drop={summary['drop_ratio_eval']:.4f} | "
                    f"backlog={summary['pre_backlog_steps_eval']:.2f}"
                )
                if checkpoint_eval_stop_on_early_stop and float(flags.get("early_stop_triggered", 0.0)) > 0.5:
                    _trace(f"checkpoint_eval:early_stop idx={int(update_idx)}")
                    _emit_console_status(
                        f"EarlyStop u{int(update_idx):04d} | "
                        f"reward={summary['reward_sum']:.3f} | "
                        f"processed={summary['processed_ratio_eval']:.3f} | "
                        f"backlog={summary['pre_backlog_steps_eval']:.2f}"
                    )
                    break
        _trace("final_save:start")
        _write_metrics_csv(run_dir, history_rows)
        torch.save(actor.state_dict(), os.path.join(run_dir, "actor_last.pt"))
        if critic_model is not None:
            torch.save(critic_model.state_dict(), os.path.join(run_dir, "critic_last.pt"))
        save_structured_train_state(
            run_dir,
            actor,
            critic_model,
            actor_optimizer,
            critic_optimizer,
            actor_stage_optimizers=actor_stage_optimizers,
            update=int(final_update_completed),
            planned_total_updates=int(args.updates),
            total_env_steps=total_env_steps,
            history_rows=history_rows,
            total_time_sec=resume_time_offset + float(time.perf_counter() - wall_start),
            checkpoint_eval_state=checkpoint_eval_state,
            checkpoint_eval_fixed_summary=checkpoint_eval_fixed_summary,
            suffix="last",
        )
        best_actor_path = os.path.join(run_dir, "actor_best.pt")
        best_critic_path = os.path.join(run_dir, "critic_best.pt")
        best_state_path = os.path.join(run_dir, "train_state_best.pt")
        if checkpoint_eval_use_best_as_final and best_checkpoint_saved and os.path.exists(best_actor_path):
            shutil.copyfile(best_actor_path, os.path.join(run_dir, "actor_final.pt"))
            if os.path.exists(best_critic_path):
                shutil.copyfile(best_critic_path, os.path.join(run_dir, "critic_final.pt"))
            if os.path.exists(best_state_path):
                shutil.copyfile(best_state_path, os.path.join(run_dir, "train_state.pt"))
            _trace(f"final_save:use_best update={int(best_checkpoint_update)}")
        else:
            torch.save(actor.state_dict(), os.path.join(run_dir, "actor_final.pt"))
            if critic_model is not None:
                torch.save(critic_model.state_dict(), os.path.join(run_dir, "critic_final.pt"))
            save_structured_train_state(
                run_dir,
                actor,
                critic_model,
                actor_optimizer,
                critic_optimizer,
                actor_stage_optimizers=actor_stage_optimizers,
                update=int(final_update_completed),
                planned_total_updates=int(args.updates),
                total_env_steps=total_env_steps,
                history_rows=history_rows,
                total_time_sec=resume_time_offset + float(time.perf_counter() - wall_start),
                checkpoint_eval_state=checkpoint_eval_state,
                checkpoint_eval_fixed_summary=checkpoint_eval_fixed_summary,
            )
        _trace("final_save:done")
        _emit_console_status(f"Done | run_dir={run_dir}")
    finally:
        _trace("trainer:close")
        if trace is not None:
            trace.close()
        tb_writer.close()
        progress.close()
        close_structured_env_group(env_group)


if __name__ == "__main__":
    main()
