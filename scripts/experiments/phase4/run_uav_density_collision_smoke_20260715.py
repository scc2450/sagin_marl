from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
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

DEFAULT_NUM_UAVS = (2, 3, 4, 5, 6)
DEFAULT_EVAL_SEED_BASES = (980000, 981000, 982000)
DEFAULT_EPISODES = 64
DEFAULT_NUM_ENVS = 64
MAP_SIZE_M = 1500.0
NUM_GU = 20


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
)

STATUS_FIELDNAMES = (
    "event",
    "timestamp",
    "sweep",
    "gpu",
    "point_index",
    "num_uav",
    "num_gu",
    "map_size_m",
    "uav_density_per_km2",
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
    "near_collision_ratio",
    "near_collision_steps",
    "terminated_early",
    "sat_overlap_eval",
    "episode_length",
    "arrival_step_mean",
    "outflow_arrival_ratio",
    "error",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_yaml_value(value: int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.10g}"


def _replace_scalar_keys(source: Path, dest: Path, replacements: dict[str, int | float | bool]) -> None:
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
        rendered.append(f"{indent}{key}: {_format_yaml_value(replacements[key])}")
        seen.add(key)
    missing = sorted(set(replacements) - seen)
    if missing:
        raise KeyError(f"{source} does not contain required keys: {', '.join(missing)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def _iter_selected_methods(methods: set[str] | None) -> Iterable[LearnedSpec | FixedSpec]:
    for spec in LEARNED_SPECS:
        if methods is None or spec.method_id in methods or spec.paper_label in methods:
            yield spec
    for spec in FIXED_SPECS:
        if methods is None or spec.method_id in methods or spec.paper_label in methods:
            yield spec


def _parse_methods(raw: str) -> set[str] | None:
    value = raw.strip()
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_ints(raw: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = raw.strip()
    if not value or value.lower() == "default":
        return default
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


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
        "near_collision_ratio",
        "near_collision_steps",
        "terminated_early",
        "sat_overlap_eval",
        "episode_length",
        "arrival_step_mean",
        "outflow_arrival_ratio",
    )
    return {key: summary.get(key, "") for key in keys}


def _write_status(status_csv: Path, row: dict[str, object]) -> None:
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = status_csv.exists()
    with status_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDNAMES, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in STATUS_FIELDNAMES})


def _base_row(
    *,
    event: str,
    gpu: str,
    point_index: int,
    num_uav: int,
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
    area_km2 = (MAP_SIZE_M / 1000.0) ** 2
    return {
        "event": event,
        "timestamp": _now(),
        "sweep": "uav_density",
        "gpu": gpu,
        "point_index": point_index,
        "num_uav": num_uav,
        "num_gu": NUM_GU,
        "map_size_m": MAP_SIZE_M,
        "uav_density_per_km2": num_uav / area_km2,
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
    gpu: str,
    point_index: int,
    num_uav: int,
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
        gpu=gpu,
        point_index=point_index,
        num_uav=num_uav,
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
    proc = subprocess.run(
        cmd,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - start
    log_path = out_dir / "evaluate.log"
    log_path.write_text(proc.stdout, encoding="utf-8")

    row = dict(base_row)
    row["event"] = "end"
    row["timestamp"] = _now()
    row["rc"] = proc.returncode
    row["elapsed_sec"] = f"{elapsed:.3f}"
    if summary_path.exists():
        row.update(_summary_metrics(summary_path))
    else:
        row["error"] = proc.stdout[-4000:]
    _write_status(status_csv, row)
    if proc.returncode != 0 and not summary_path.exists():
        raise RuntimeError(f"{label} failed with rc={proc.returncode}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/home/sgy/workspace/sagin_marl_phase4_learning_ablation")
    parser.add_argument("--python", default="/home/sgy/workspace/sagin_marl/.venv/bin/python")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--status-csv", required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--eval-seed-bases", default=",".join(str(item) for item in DEFAULT_EVAL_SEED_BASES))
    parser.add_argument("--num-uavs", default=",".join(str(item) for item in DEFAULT_NUM_UAVS))
    parser.add_argument("--methods", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out_root = (repo / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)
    status_csv = (repo / args.status_csv).resolve() if not Path(args.status_csv).is_absolute() else Path(args.status_csv)
    methods = _parse_methods(str(args.methods))
    seed_bases = _parse_ints(str(args.eval_seed_bases), DEFAULT_EVAL_SEED_BASES)
    num_uavs = _parse_ints(str(args.num_uavs), DEFAULT_NUM_UAVS)
    _validate_paths(repo, str(args.python), methods)

    evaluator = repo / "scripts/evaluate_structured_mixed_heads_native.py"
    planned = 0
    for point_index, num_uav in enumerate(num_uavs, start=1):
        config_root = out_root / "configs" / f"{num_uav}uav{NUM_GU}gu"
        replacements = {"num_uav": int(num_uav)}
        for spec in _iter_selected_methods(methods):
            base_config = repo / spec.config
            config_path = config_root / Path(spec.config).name
            _replace_scalar_keys(base_config, config_path, replacements)
            if isinstance(spec, LearnedSpec):
                for training_seed, checkpoint in spec.checkpoints:
                    for eval_seed_base in seed_bases:
                        planned += 1
                        label = (
                            f"uav_density_{num_uav}uav_{spec.method_id}_seed{training_seed}"
                            f"_selected_seedbase{eval_seed_base}"
                        )
                        out_dir = (
                            out_root
                            / "results"
                            / f"{num_uav}uav{NUM_GU}gu"
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
                                gpu=str(args.gpu),
                                point_index=point_index,
                                num_uav=int(num_uav),
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
                    label = f"uav_density_{num_uav}uav_{spec.method_id}_fixed_seedbase{eval_seed_base}"
                    out_dir = (
                        out_root
                        / "results"
                        / f"{num_uav}uav{NUM_GU}gu"
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
                            gpu=str(args.gpu),
                            point_index=point_index,
                            num_uav=int(num_uav),
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
    print(f"planned_rows={planned} dry_run={args.dry_run} status={status_csv}")


if __name__ == "__main__":
    main()
