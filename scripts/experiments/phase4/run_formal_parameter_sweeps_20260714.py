from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


BASE_CONFIG_RELCRITIC = (
    "configs/experiments/phase4_learning_ablation/"
    "structured_joint_mcgae_3uav20gu_t250_relational_critic.yaml"
)
BASE_CONFIG_GLOBALCRITIC = (
    "configs/experiments/phase4_learning_ablation/"
    "structured_joint_mcgae_3uav20gu_t250_global_only_critic.yaml"
)
BASE_CONFIG_MAPPO_LIKE = (
    "configs/experiments/phase4_learning_ablation/"
    "structured_joint_mcgae_3uav20gu_t250_mappo_like_flat_actor_critic_stabilized.yaml"
)

NOMINAL_TASK_ARRIVAL_RATE = 2.0e6
NOMINAL_B_ACC = 2.0e6
NOMINAL_B_BACKHAUL_PER_SAT = 1.0e7
NOMINAL_SAT_CPU_FREQ = 5.0e10

LOAD_SWEEP_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)
RESOURCE_SWEEP_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00)

DEFAULT_EVAL_SEED_BASES = (980000, 981000, 982000)
DEFAULT_EPISODES = 64
DEFAULT_NUM_ENVS = 64


@dataclass(frozen=True)
class LearnedSpec:
    method_id: str
    paper_label: str
    config: str
    checkpoints: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class FixedSpec:
    method_id: str
    paper_label: str
    baseline_policy: str
    config: str = BASE_CONFIG_RELCRITIC


LEARNED_SPECS = (
    LearnedSpec(
        method_id="relcritic",
        paper_label="STARS",
        config=BASE_CONFIG_RELCRITIC,
        checkpoints=(
            (
                45211,
                "runs/phase4_learning_ablation/3uav20gu_t250/relational_critic/"
                "seed45211_20260711_213639_bootstrapgae_nocompile/best_checkpoint.pt",
            ),
            (
                73129,
                "runs/phase4_learning_ablation/3uav20gu_t250/relational_critic/"
                "seed73129_20260711_213639_bootstrapgae_nocompile/best_checkpoint.pt",
            ),
            (
                91457,
                "runs/phase4_learning_ablation/3uav20gu_t250/relational_critic/"
                "seed91457_20260712_124615_resume_u0400_bootstrapgae_nocompile/best_checkpoint.pt",
            ),
        ),
    ),
    LearnedSpec(
        method_id="globalcritic",
        paper_label="STARS-GC",
        config=BASE_CONFIG_GLOBALCRITIC,
        checkpoints=(
            (
                45211,
                "runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/"
                "seed45211_20260712_124615_bootstrapgae_nocompile/best_checkpoint.pt",
            ),
            (
                73129,
                "runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/"
                "seed73129_20260712_124615_bootstrapgae_nocompile/best_checkpoint.pt",
            ),
            (
                91457,
                "runs/phase4_learning_ablation/3uav20gu_t250/global_only_critic/"
                "seed91457_20260712_124615_bootstrapgae_nocompile/best_checkpoint.pt",
            ),
        ),
    ),
    LearnedSpec(
        method_id="mappo_like",
        paper_label="HA-PPO",
        config=BASE_CONFIG_MAPPO_LIKE,
        checkpoints=(
            (
                45211,
                "runs/phase4_learning_ablation/3uav20gu_t250/"
                "mappo_like_flat_actor_critic_stabilized/"
                "seed45211_20260712_0016_resume_u0100_bootstrapgae_native_nocompile/"
                "best_checkpoint.pt",
            ),
            (
                73129,
                "runs/phase4_learning_ablation/3uav20gu_t250/"
                "mappo_like_flat_actor_critic_stabilized/"
                "seed73129_20260712_150723_bootstrapgae_native_nocompile/best_checkpoint.pt",
            ),
            (
                91457,
                "runs/phase4_learning_ablation/3uav20gu_t250/"
                "mappo_like_flat_actor_critic_stabilized/"
                "seed91457_20260712_202812_bootstrapgae_native_nocompile/best_checkpoint.pt",
            ),
        ),
    ),
)

FIXED_SPECS = (
    FixedSpec(
        method_id="cluster_center_queue_aware",
        paper_label="QCCS",
        baseline_policy="cluster_center_queue_aware",
    ),
    FixedSpec(
        method_id="maxweight_lyapunov",
        paper_label="Lyapunov",
        baseline_policy="maxweight_lyapunov",
    ),
    FixedSpec(
        method_id="queue_aware_bw",
        paper_label="QBS",
        baseline_policy="queue_aware_bw",
    ),
    FixedSpec(
        method_id="static_uniform",
        paper_label="Uniform",
        baseline_policy="static_uniform",
    ),
)

STATUS_FIELDNAMES = (
    "event",
    "timestamp",
    "sweep",
    "gpu",
    "point_index",
    "multiplier",
    "task_arrival_rate_bits_per_gu_slot",
    "task_arrival_rate_mbit_per_gu_slot",
    "b_acc_hz",
    "b_acc_mhz",
    "b_backhaul_per_sat_hz",
    "b_backhaul_per_sat_mhz",
    "sat_cpu_freq_hz",
    "sat_cpu_freq_gbps_equiv",
    "method_id",
    "paper_label",
    "method_family",
    "training_seed",
    "checkpoint_role",
    "eval_seed_base",
    "episodes",
    "num_envs",
    "config_path",
    "checkpoint_path",
    "baseline_policy",
    "out_dir",
    "label",
    "rc",
    "elapsed_sec",
    "reward_sum",
    "processed_ratio_eval",
    "drop_ratio_eval",
    "pre_backlog_steps_eval",
    "D_sys_report",
    "queue_total_mean",
    "collision_episode_fraction",
    "sat_overlap_eval",
    "episode_length",
    "arrival_step_mean",
    "outflow_arrival_ratio",
    "error",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_multiplier(value: float) -> str:
    return f"x{value:.2f}".replace(".", "p")


def _format_yaml_float(value: float) -> str:
    return f"{value:.10g}"


def _replace_scalar_keys(source: Path, dest: Path, replacements: dict[str, float | bool]) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    rendered: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if ":" not in stripped or stripped.startswith("#"):
            rendered.append(line)
            continue
        key = stripped.split(":", 1)[0].strip()
        if key not in replacements:
            rendered.append(line)
            continue
        value = replacements[key]
        if isinstance(value, bool):
            yaml_value = "true" if value else "false"
        else:
            yaml_value = _format_yaml_float(float(value))
        rendered.append(f"{indent}{key}: {yaml_value}")
        seen.add(key)
    missing = sorted(set(replacements) - seen)
    if missing:
        raise KeyError(f"{source} does not contain required keys: {', '.join(missing)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def _point_values(sweep: str, multiplier: float) -> dict[str, float | bool]:
    if sweep == "load":
        return {
            "resource_scale_enabled": False,
            "resource_scale_b_acc_multiplier": 1.0,
            "task_arrival_rate": NOMINAL_TASK_ARRIVAL_RATE * multiplier,
            "b_acc": NOMINAL_B_ACC,
            "b_backhaul_per_sat": NOMINAL_B_BACKHAUL_PER_SAT,
            "sat_cpu_freq": NOMINAL_SAT_CPU_FREQ,
        }
    if sweep == "resource":
        return {
            "resource_scale_enabled": False,
            "resource_scale_b_acc_multiplier": 1.0,
            "task_arrival_rate": NOMINAL_TASK_ARRIVAL_RATE,
            "b_acc": NOMINAL_B_ACC * multiplier,
            "b_backhaul_per_sat": NOMINAL_B_BACKHAUL_PER_SAT * multiplier,
            "sat_cpu_freq": NOMINAL_SAT_CPU_FREQ * multiplier,
        }
    raise ValueError(f"unknown sweep: {sweep}")


def _write_status(status_csv: Path, row: dict[str, object]) -> None:
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = status_csv.exists()
    with status_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDNAMES, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in STATUS_FIELDNAMES})


def _summary_metrics(summary_path: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    keys = (
        "reward_sum",
        "processed_ratio_eval",
        "drop_ratio_eval",
        "pre_backlog_steps_eval",
        "D_sys_report",
        "queue_total_mean",
        "collision_episode_fraction",
        "sat_overlap_eval",
        "episode_length",
        "arrival_step_mean",
        "outflow_arrival_ratio",
    )
    return {key: summary.get(key, "") for key in keys}


def _base_row(
    *,
    event: str,
    sweep: str,
    gpu: str,
    point_index: int,
    multiplier: float,
    values: dict[str, float | bool],
    method_id: str,
    paper_label: str,
    method_family: str,
    training_seed: int | None,
    eval_seed_base: int,
    episodes: int,
    num_envs: int,
    config_path: Path,
    checkpoint_path: Path | None,
    baseline_policy: str | None,
    out_dir: Path,
    label: str,
) -> dict[str, object]:
    task_arrival_rate = float(values["task_arrival_rate"])
    b_acc = float(values["b_acc"])
    b_backhaul = float(values["b_backhaul_per_sat"])
    sat_cpu = float(values["sat_cpu_freq"])
    return {
        "event": event,
        "timestamp": _now(),
        "sweep": sweep,
        "gpu": gpu,
        "point_index": point_index,
        "multiplier": f"{multiplier:.2f}",
        "task_arrival_rate_bits_per_gu_slot": task_arrival_rate,
        "task_arrival_rate_mbit_per_gu_slot": task_arrival_rate / 1.0e6,
        "b_acc_hz": b_acc,
        "b_acc_mhz": b_acc / 1.0e6,
        "b_backhaul_per_sat_hz": b_backhaul,
        "b_backhaul_per_sat_mhz": b_backhaul / 1.0e6,
        "sat_cpu_freq_hz": sat_cpu,
        "sat_cpu_freq_gbps_equiv": sat_cpu / 1.0e9,
        "method_id": method_id,
        "paper_label": paper_label,
        "method_family": method_family,
        "training_seed": "" if training_seed is None else training_seed,
        "checkpoint_role": "selected" if checkpoint_path else "fixed",
        "eval_seed_base": eval_seed_base,
        "episodes": episodes,
        "num_envs": num_envs,
        "config_path": str(config_path),
        "checkpoint_path": "" if checkpoint_path is None else str(checkpoint_path),
        "baseline_policy": "" if baseline_policy is None else baseline_policy,
        "out_dir": str(out_dir),
        "label": label,
    }


def _run_one(
    *,
    repo: Path,
    python: str,
    evaluator: Path,
    status_csv: Path,
    sweep: str,
    gpu: str,
    point_index: int,
    multiplier: float,
    values: dict[str, float | bool],
    method_id: str,
    paper_label: str,
    method_family: str,
    training_seed: int | None,
    eval_seed_base: int,
    episodes: int,
    num_envs: int,
    config_path: Path,
    checkpoint_path: Path | None,
    baseline_policy: str | None,
    out_dir: Path,
    label: str,
    skip_existing: bool,
) -> None:
    summary_path = out_dir / f"{label}_summary.json"
    base_row = _base_row(
        event="start",
        sweep=sweep,
        gpu=gpu,
        point_index=point_index,
        multiplier=multiplier,
        values=values,
        method_id=method_id,
        paper_label=paper_label,
        method_family=method_family,
        training_seed=training_seed,
        eval_seed_base=eval_seed_base,
        episodes=episodes,
        num_envs=num_envs,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        baseline_policy=baseline_policy,
        out_dir=out_dir,
        label=label,
    )
    if skip_existing and summary_path.exists():
        row = dict(base_row)
        row["event"] = "skip_existing"
        row["rc"] = 0
        row["elapsed_sec"] = 0
        row.update(_summary_metrics(summary_path))
        _write_status(status_csv, row)
        return

    _write_status(status_csv, base_row)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(evaluator),
        "--config",
        str(config_path),
        "--episodes",
        str(episodes),
        "--num_envs",
        str(num_envs),
        "--episode_seed_base",
        str(eval_seed_base),
        "--policy_mode",
        "deterministic",
        "--device",
        "cuda",
        "--out_dir",
        str(out_dir),
        "--label",
        label,
    ]
    if checkpoint_path is None:
        cmd.extend(["--baseline_policy", str(baseline_policy)])
    else:
        cmd.extend(["--base_checkpoint", str(checkpoint_path)])

    start = time.perf_counter()
    log_path = out_dir / "evaluate.log"
    proc = subprocess.run(
        cmd,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    log_path.write_text(proc.stdout, encoding="utf-8")

    row = dict(base_row)
    row["event"] = "end"
    row["timestamp"] = _now()
    row["rc"] = proc.returncode
    row["elapsed_sec"] = f"{elapsed:.3f}"
    if proc.returncode == 0 and summary_path.exists():
        row.update(_summary_metrics(summary_path))
    else:
        row["error"] = proc.stdout[-4000:]
    _write_status(status_csv, row)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with rc={proc.returncode}; see {log_path}")


def _iter_selected_methods(
    methods: set[str] | None,
) -> Iterable[LearnedSpec | FixedSpec]:
    for spec in LEARNED_SPECS:
        if methods is None or spec.method_id in methods or spec.paper_label in methods:
            yield spec
    for spec in FIXED_SPECS:
        if methods is None or spec.method_id in methods or spec.paper_label in methods:
            yield spec


def _validate_paths(repo: Path, python: str, methods: set[str] | None) -> None:
    missing: list[str] = []
    if not Path(python).exists():
        missing.append(python)
    evaluator = repo / "scripts/evaluate_structured_mixed_heads_native.py"
    if not evaluator.exists():
        missing.append(str(evaluator))
    for spec in _iter_selected_methods(methods):
        config = repo / spec.config
        if not config.exists():
            missing.append(str(config))
        if isinstance(spec, LearnedSpec):
            for _, checkpoint in spec.checkpoints:
                if not (repo / checkpoint).exists():
                    missing.append(str(repo / checkpoint))
    if missing:
        raise FileNotFoundError("missing required files:\n" + "\n".join(missing))


def _parse_methods(raw: str) -> set[str] | None:
    value = raw.strip()
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_point_indices(raw: str, max_index: int) -> set[int] | None:
    value = raw.strip()
    if not value or value.lower() == "all":
        return None
    indices: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            indices.update(range(start, end + 1))
        else:
            indices.add(int(item))
    invalid = sorted(index for index in indices if index < 1 or index > max_index)
    if invalid:
        raise ValueError(f"point indices out of range 1..{max_index}: {invalid}")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/home/sgy/workspace/sagin_marl_phase4_learning_ablation")
    parser.add_argument("--python", default="/home/sgy/workspace/sagin_marl/.venv/bin/python")
    parser.add_argument("--sweep", choices=["load", "resource"], required=True)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--status-csv", required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument(
        "--eval-seed-bases",
        default=",".join(str(item) for item in DEFAULT_EVAL_SEED_BASES),
    )
    parser.add_argument("--methods", default="all")
    parser.add_argument(
        "--point-indices",
        default="all",
        help="Comma-separated 1-based point indices, ranges, or all.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out_root = (repo / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)
    status_csv = (repo / args.status_csv).resolve() if not Path(args.status_csv).is_absolute() else Path(args.status_csv)
    methods = _parse_methods(str(args.methods))
    seed_bases = tuple(int(item.strip()) for item in str(args.eval_seed_bases).split(",") if item.strip())
    _validate_paths(repo, str(args.python), methods)

    multipliers = LOAD_SWEEP_MULTIPLIERS if args.sweep == "load" else RESOURCE_SWEEP_MULTIPLIERS
    point_indices = _parse_point_indices(str(args.point_indices), len(multipliers))
    evaluator = repo / "scripts/evaluate_structured_mixed_heads_native.py"
    planned = 0
    for point_index, multiplier in enumerate(multipliers, start=1):
        if point_indices is not None and point_index not in point_indices:
            continue
        point_label = _safe_multiplier(multiplier)
        values = _point_values(args.sweep, multiplier)
        config_root = out_root / "configs" / args.sweep / point_label
        for spec in _iter_selected_methods(methods):
            base_config = repo / spec.config
            config_path = config_root / Path(spec.config).name
            _replace_scalar_keys(base_config, config_path, values)
            if isinstance(spec, LearnedSpec):
                for training_seed, checkpoint in spec.checkpoints:
                    for eval_seed_base in seed_bases:
                        planned += 1
                        label = (
                            f"{args.sweep}_{point_label}_{spec.method_id}_seed{training_seed}"
                            f"_selected_seedbase{eval_seed_base}"
                        )
                        out_dir = (
                            out_root
                            / "results"
                            / args.sweep
                            / point_label
                            / "learned"
                            / spec.method_id
                            / f"seed{training_seed}"
                            / "selected"
                            / f"seedbase{eval_seed_base}"
                        )
                        if not args.dry_run:
                            _run_one(
                                repo=repo,
                                python=str(args.python),
                                evaluator=evaluator,
                                status_csv=status_csv,
                                sweep=args.sweep,
                                gpu=str(args.gpu),
                                point_index=point_index,
                                multiplier=multiplier,
                                values=values,
                                method_id=spec.method_id,
                                paper_label=spec.paper_label,
                                method_family="learned",
                                training_seed=training_seed,
                                eval_seed_base=eval_seed_base,
                                episodes=int(args.episodes),
                                num_envs=int(args.num_envs),
                                config_path=config_path,
                                checkpoint_path=repo / checkpoint,
                                baseline_policy=None,
                                out_dir=out_dir,
                                label=label,
                                skip_existing=not args.no_skip_existing,
                            )
            else:
                for eval_seed_base in seed_bases:
                    planned += 1
                    label = f"{args.sweep}_{point_label}_{spec.method_id}_fixed_seedbase{eval_seed_base}"
                    out_dir = (
                        out_root
                        / "results"
                        / args.sweep
                        / point_label
                        / "fixed"
                        / spec.method_id
                        / f"seedbase{eval_seed_base}"
                    )
                    if not args.dry_run:
                        _run_one(
                            repo=repo,
                            python=str(args.python),
                            evaluator=evaluator,
                            status_csv=status_csv,
                            sweep=args.sweep,
                            gpu=str(args.gpu),
                            point_index=point_index,
                            multiplier=multiplier,
                            values=values,
                            method_id=spec.method_id,
                            paper_label=spec.paper_label,
                            method_family="fixed",
                            training_seed=None,
                            eval_seed_base=eval_seed_base,
                            episodes=int(args.episodes),
                            num_envs=int(args.num_envs),
                            config_path=config_path,
                            checkpoint_path=None,
                            baseline_policy=spec.baseline_policy,
                            out_dir=out_dir,
                            label=label,
                            skip_existing=not args.no_skip_existing,
                        )

    print(
        json.dumps(
            {
                "sweep": args.sweep,
                "planned_jobs": planned,
                "out_root": str(out_root),
                "status_csv": str(status_csv),
                "seed_bases": seed_bases,
                "episodes": int(args.episodes),
                "num_envs": int(args.num_envs),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
