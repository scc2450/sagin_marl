from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "sagin_marl").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from sagin_marl.env.config import load_config
from sagin_marl.rl.structured_eval import (
    validate_structured_fixed_seed_long_rollout,
    validate_structured_fixed_seed_long_rollout_acceptance_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fixed-seed long-rollout parity between native and legacy structured backends.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--baseline-policy", type=str, default="cluster_center_queue_aware")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--episode-seed-base", type=int, default=None)
    parser.add_argument("--native-tensor-backend", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--atol", type=float, default=6.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--json-path", type=str, default=None)
    parser.add_argument(
        "--acceptance-matrix",
        action="store_true",
        help="Run the full document 14.3 matrix: source modes, flow proxy modes, and done/tail coverage.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    native_tensor_backend = (
        "cuda"
        if str(args.native_tensor_backend).strip().lower() == "auto" and torch.cuda.is_available()
        else "cpu"
        if str(args.native_tensor_backend).strip().lower() == "auto"
        else str(args.native_tensor_backend).strip().lower()
    )
    seed_base = int(cfg.seed) if args.episode_seed_base is None else int(args.episode_seed_base)
    validator = (
        validate_structured_fixed_seed_long_rollout_acceptance_matrix
        if bool(args.acceptance_matrix)
        else validate_structured_fixed_seed_long_rollout
    )
    report = validator(
        cfg,
        baseline_policy=str(args.baseline_policy),
        episodes=max(int(args.episodes), 1),
        episode_seed_base=seed_base,
        num_envs=max(int(args.num_envs), 1),
        native_tensor_backend=native_tensor_backend,
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        ("long_rollout_acceptance_matrix " if bool(args.acceptance_matrix) else "long_rollout_acceptance ")
        + f"baseline={report['baseline_policy']} episodes={report['episodes']} "
        f"native_backend={report['native_tensor_backend']} "
        f"passed={int(report['passed'])} exact_passed={int(report.get('exact_passed', report['passed']))}"
    )
    if bool(args.acceptance_matrix):
        print(
            "matrix_coverage "
            + " ".join(
                f"{key}={int(bool(value))}"
                for key, value in sorted(dict(report.get("coverage", {})).items())
            )
        )
        failed_cases = [
            case
            for case in report.get("cases", [])
            if not bool(case.get("passed", False))
        ]
        print(f"matrix_cases total={len(report.get('cases', []))} failed={len(failed_cases)}")
        if failed_cases:
            for case in failed_cases[:16]:
                print(f"matrix_case_failed: {case.get('name', '<unknown>')}")
        if not bool(report["passed"]):
            for case in failed_cases[:4]:
                for message in case.get("system_acceptance_errors", [])[:4]:
                    print(f"acceptance_mismatch[{case.get('name', '<unknown>')}]: {message}")
                for message in case.get("comparison_errors", [])[:4]:
                    print(f"exact_mismatch[{case.get('name', '<unknown>')}]: {message}")
            if report.get("missing_coverage"):
                print("missing_coverage: " + ",".join(str(x) for x in report.get("missing_coverage", [])))
            raise SystemExit(1)
        return
    print(
        "max_episode_diff "
        f"reward={float(report['episode_diff_max'].get('reward_sum', 0.0)):.8f} "
        f"steps={float(report['episode_diff_max'].get('step_count', 0.0)):.8f} "
        f"drop={float(report['episode_diff_max'].get('drop_ratio_total', 0.0)):.8f} "
        f"throughput={float(report['episode_diff_max'].get('processed_ratio_total', 0.0)):.8f} "
        f"backlog={float(report['episode_diff_max'].get('pre_backlog_total', 0.0)):.8f}"
    )
    print(
        "max_trace_diff "
        f"reward={float(report['trace_diff_max'].get('reward', 0.0)):.8f} "
        f"queue_total={float(report['trace_diff_max'].get('queue_total_sum', 0.0)):.8f} "
        f"backlog={float(report['trace_diff_max'].get('pre_backlog_steps_eval', 0.0)):.8f}"
    )
    if not bool(report["passed"]):
        for message in report.get("system_acceptance_errors", [])[:16]:
            print(f"acceptance_mismatch: {message}")
        for message in report.get("comparison_errors", [])[:16]:
            print(f"exact_mismatch: {message}")
        raise SystemExit(1)
    if not bool(report.get("exact_passed", True)):
        for message in report.get("comparison_errors", [])[:16]:
            print(f"exact_mismatch: {message}")


if __name__ == "__main__":
    main()
