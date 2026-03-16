from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Optional

from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    def __init__(
        self,
        log_dir: str,
        fieldnames: Optional[Iterable[str]] = None,
        tb_fields: Optional[Iterable[str]] = None,
        env_step_fields: Optional[Iterable[str]] = None,
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self._init_tb_layout()
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self.fieldnames: Optional[List[str]] = list(fieldnames) if fieldnames is not None else None
        self.tb_fields = set(tb_fields) if tb_fields is not None else None
        self.env_step_fields = set(env_step_fields) if env_step_fields is not None else set()
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
            "Training/Main": {
                "EpisodeReturn": ["Multiline", ["episode_reward", "rollout_reward_per_step"]],
                "EpisodeThroughputTerms": [
                    "Multiline",
                    ["episode_term_throughput_access", "episode_term_throughput_backhaul"],
                ],
                "Throughput": ["Multiline", ["throughput_access_norm", "throughput_backhaul_norm"]],
                "Queues": ["Multiline", ["gu_queue_mean", "uav_queue_mean", "queue_total_active"]],
                "Safety": ["Multiline", ["collision_rate"]],
            },
            "Training/PPO": {
                "Losses": ["Multiline", ["policy_loss", "value_loss"]],
                "Stability": [
                    "Multiline",
                    [
                        "entropy",
                        "approx_kl",
                        "clip_frac",
                        "explained_variance",
                    ],
                ],
            },
            "Training/Imitation": {
                "DangerImitation": [
                    "Multiline",
                    [
                        "danger_imitation_loss",
                        "danger_imitation_coef",
                        "danger_imitation_active_rate",
                    ],
                ],
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
        tb_fields = self.tb_fields if self.tb_fields is not None else set(fieldnames)
        env_step = int(metrics.get("total_env_steps", step))
        for key, val in metrics.items():
            if key not in tb_fields:
                continue
            scalar_step = env_step if key in self.env_step_fields else step
            self.writer.add_scalar(key, val, scalar_step)

    def close(self) -> None:
        self.writer.close()
