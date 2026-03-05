from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Optional

from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    def __init__(self, log_dir: str, fieldnames: Optional[Iterable[str]] = None):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self._init_tb_layout()
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self.fieldnames: Optional[List[str]] = list(fieldnames) if fieldnames is not None else None
        self._header_written = False
        self._init_csv()

    def _init_csv(self) -> None:
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and header[0] == "step":
                    self.fieldnames = header[1:]
                    self._header_written = True
            return
        if self.fieldnames is not None:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step"] + list(self.fieldnames))
            self._header_written = True

    def _init_tb_layout(self) -> None:
        layout = {
            "Training/Reward": {
                "EpisodeReward": ["Multiline", ["episode_reward"]],
            },
            "Training/RewardParts": {
                "Ratios": [
                    "Multiline",
                    [
                        "r_service_ratio",
                        "r_drop_ratio",
                        "r_assoc_ratio",
                        "r_queue_delta",
                        "r_dist",
                        "r_dist_delta",
                    ],
                ],
                "Penalties": [
                    "Multiline",
                    [
                        "r_queue_pen",
                        "r_collision_penalty",
                        "r_battery_penalty",
                        "r_fail_penalty",
                    ],
                ],
                "Guidance": [
                    "Multiline",
                    [
                        "r_centroid",
                        "centroid_dist_mean",
                        "r_bw_align",
                        "r_sat_score",
                    ],
                ],
                "Terms": [
                    "Multiline",
                    [
                        "r_term_service",
                        "r_term_drop",
                        "r_term_queue",
                        "r_term_assoc",
                        "r_term_q_delta",
                        "r_term_dist",
                        "r_term_dist_delta",
                        "r_term_centroid",
                        "r_term_bw_align",
                        "r_term_sat_score",
                        "r_term_energy",
                        "r_term_accel",
                    ],
                ],
            },
            "Training/Losses": {
                "Losses": ["Multiline", ["policy_loss", "value_loss", "entropy", "imitation_loss"]],
            },
            "Training/Diagnostics": {
                "PPO": ["Multiline", ["approx_kl", "clip_frac"]],
                "Advantage": ["Multiline", ["adv_raw_mean", "adv_raw_std", "adv_norm_mean", "adv_norm_std"]],
                "RewardNorm": ["Multiline", ["reward_rms_sigma", "reward_clip_frac"]],
            },
            "Training/Queues": {
                "QueueMean": ["Multiline", ["gu_queue_mean", "uav_queue_mean", "sat_queue_mean"]],
                "QueueMax": ["Multiline", ["gu_queue_max", "uav_queue_max", "sat_queue_max"]],
                "QueueNorm": ["Multiline", ["q_norm_active", "prev_q_norm_active", "q_norm_delta"]],
            },
            "Training/Drops": {
                "Drops": ["Multiline", ["drop_sum", "gu_drop_sum", "uav_drop_sum"]],
            },
            "Training/Satellite": {
                "SatFlow": ["Multiline", ["sat_incoming_sum", "sat_processed_sum"]],
            },
            "Training/Performance": {
                "Throughput": ["Multiline", ["env_steps_per_sec", "update_steps_per_sec"]],
                "UpdateTime": ["Multiline", ["rollout_time_sec", "optim_time_sec", "update_time_sec"]],
            },
            "Training/Totals": {
                "Totals": ["Multiline", ["total_env_steps", "total_time_sec"]],
            },
            "Training/Energy": {
                "EnergyMean": ["Multiline", ["energy_mean"]],
            },
        }
        self.writer.add_custom_scalars(layout)

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        if not self._header_written:
            self.fieldnames = list(metrics.keys())
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step"] + list(self.fieldnames))
            self._header_written = True
        fieldnames = self.fieldnames or list(metrics.keys())
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([step] + [metrics.get(name, 0.0) for name in fieldnames])
        for key, val in metrics.items():
            self.writer.add_scalar(key, val, step)

    def close(self) -> None:
        self.writer.close()
