from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

"""
.venv/bin/python process_logs_for_report.py \
    --env cartpole-swingup-v0 \
    --algos sac td3 ppo \
    --output plots/report_cartpole.png \
    --logs-root ../_logs
"""


def _to_numpy_1d(values: object) -> np.ndarray:
    if hasattr(values, "detach") and hasattr(values, "cpu") and hasattr(values, "numpy"):
        return np.asarray(values.detach().cpu().numpy(), dtype=np.float64).ravel()
    return np.asarray(values, dtype=np.float64).ravel()


def _latest_eval_file(seed_dir: Path) -> Path | None:
    latest_path = None
    latest_mtime = -1.0

    for run_dir in seed_dir.iterdir():
        if not run_dir.is_dir():
            continue
        eval_file = run_dir / "eval_results.npy"
        if not eval_file.is_file():
            continue
        mtime = eval_file.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = eval_file

    return latest_path


def _load_seed_step_means(eval_file: Path) -> dict[int, float]:
    eval_dict = np.load(eval_file, allow_pickle=True).item()
    out: dict[int, float] = {}
    for step, rewards in eval_dict.items():
        out[int(step)] = float(_to_numpy_1d(rewards).mean())
    return out


def _aggregate_algorithm(env: str, algo: str, logs_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    algo_dir = logs_root / env / algo
    if not algo_dir.is_dir():
        raise FileNotFoundError(f"Algorithm log directory not found: {algo_dir}")

    per_seed_step_means: list[dict[int, float]] = []

    for seed_dir in sorted(algo_dir.glob("seed_*")):
        eval_file = _latest_eval_file(seed_dir)
        if eval_file is None:
            continue
        per_seed_step_means.append(_load_seed_step_means(eval_file))

    if not per_seed_step_means:
        raise FileNotFoundError(f"No eval_results.npy found under: {algo_dir}")

    all_steps = sorted({step for data in per_seed_step_means for step in data})

    steps = []
    means = []
    stds = []
    counts = []

    for step in all_steps:
        vals = [seed_data[step] for seed_data in per_seed_step_means if step in seed_data]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        steps.append(step)
        means.append(arr.mean())
        stds.append(arr.std(ddof=0))
        counts.append(arr.size)

    return (
        np.asarray(steps, dtype=np.int64),
        np.asarray(means, dtype=np.float64),
        np.asarray(stds, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        len(per_seed_step_means),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate eval logs across seeds and plot eval reward vs eval step.",
    )
    parser.add_argument("--env", required=True, help="Environment name (folder under _logs).")
    parser.add_argument("--algos", nargs="+", required=True, help="One or more algorithm names.")
    parser.add_argument("--logs-root", default="_logs", help="Root logs directory.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Default: ./report_<env>.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root)
    output_path = Path(args.output) if args.output else Path(f"report_{args.env}.png")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    for algo in args.algos:
        steps, means, stds, counts, n_seeds = _aggregate_algorithm(args.env, algo, logs_root)

        print(f"\nAlgorithm: {algo} | environment: {args.env} | seeds used: {n_seeds}")
        for s, m, st, c in zip(steps, means, stds, counts):
            print(
                f"step={int(s):d} "
                f"mean={m:.6f} "
                f"std={st:.6f} "
                f"n_seeds={int(c):d}"
            )
        overall_mean_across_steps = np.mean(means) if means.size > 0 else float("nan")
        print(f"Overall mean at across steps: {overall_mean_across_steps:.6f}")

        ax.errorbar(
            steps,
            means,
            yerr=stds,
            label=algo,
            linewidth=1.8,
            elinewidth=1.0,
            capsize=2,
        )

    ax.set_title(f"Eval rewards vs steps averaged for {n_seeds} seeds ({args.env})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Eval reward")
    x_formatter = ticker.ScalarFormatter(useOffset=False)
    x_formatter.set_scientific(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100000))  # tick every 100k
    ax.xaxis.set_major_formatter(x_formatter)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"\nSaved plot to: {output_path}")


if __name__ == "__main__":
    main()
