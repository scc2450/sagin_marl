from __future__ import annotations

import argparse
from pathlib import Path

from sagin_marl.viz.plots import plot_learning_curve


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot reward learning curve from metrics.csv with an optional uncertainty band."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to metrics.csv")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path. Default: <csv_dir>/learning_curve.png",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv does not exist: {csv_path}")
    out_path = (
        Path(args.out).resolve()
        if args.out is not None
        else (csv_path.parent / "learning_curve.png").resolve()
    )
    plot_learning_curve(str(csv_path), str(out_path))
    print(f"Saved learning curve to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
