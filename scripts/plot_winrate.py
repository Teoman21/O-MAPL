"""Plot O-MAPL SMACv2 win-rate curves (mean +/- std over seeds), like the paper.

Reads the per-run CSVs written by omapl.train:
    runs/<exp_name>-<algo>-<scenario>-s<seed>/metrics.csv
filters evaluation rows (those with an `eval/win_rate`), aligns seeds by training
step, and draws one panel per scenario with the O-MAPL line in red and a shaded
std band — matching the figure's style (x-axis = evaluation step 0..100,
y-axis = win rate %).

    python scripts/plot_winrate.py
    python scripts/plot_winrate.py --metric eval/return_mean --out runs/return.png
    python scripts/plot_winrate.py --scenarios terran_5_vs_5 zerg_5_vs_5
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SCENARIOS = ["terran_5_vs_5", "zerg_5_vs_5", "terran_10_vs_10"]
OMAPL_RED = "#e8453c"


def load_runs(runs_dir, exp_name, algo, scenario, metric):
    """Return (steps, [series_per_seed]) aligned on the common step grid."""
    pattern = os.path.join(runs_dir, f"{exp_name}-{algo}-{scenario}-s*", "metrics.csv")
    series, step_index = [], None
    for csv_path in sorted(glob.glob(pattern)):
        df = pd.read_csv(csv_path)
        if metric not in df.columns:
            continue
        ev = df[["step", metric]].dropna(subset=[metric])
        if ev.empty:
            continue
        ev = ev.groupby("step", as_index=True)[metric].mean()
        series.append(ev)
        step_index = ev.index if step_index is None else step_index.union(ev.index)
    if not series:
        return None, None
    # Align every seed onto the union of steps (interpolate gaps, then ffill/bfill).
    aligned = []
    for s in series:
        s = s.reindex(step_index).interpolate().ffill().bfill()
        aligned.append(s.to_numpy())
    return np.asarray(step_index), np.vstack(aligned)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--exp_name", default="omapl")
    p.add_argument("--algo", default="omapl")
    p.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    p.add_argument("--metric", default="eval/win_rate")
    p.add_argument("--out", default="runs/omapl_smacv2_winrate.png")
    p.add_argument("--as_eval_step", action="store_true", default=True,
                   help="x-axis as evaluation-step index (0..N) instead of raw steps.")
    args = p.parse_args()

    n = len(args.scenarios)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
    ylabel = "Win Rate (%)" if "win" in args.metric else args.metric

    for ax, scenario in zip(axes[0], args.scenarios):
        steps, data = load_runs(args.runs_dir, args.exp_name, args.algo,
                                scenario, args.metric)
        if data is None:
            ax.set_title(f"{scenario}\n(no data)")
            ax.set_xlabel("Evaluation Step")
            continue
        x = np.arange(1, len(steps) + 1) if args.as_eval_step else steps
        mean, std = data.mean(axis=0), data.std(axis=0)
        ax.plot(x, mean, color=OMAPL_RED, lw=2, label="O-MAPL")
        ax.fill_between(x, mean - std, mean + std, color=OMAPL_RED, alpha=0.2)
        ax.set_title(scenario)
        ax.set_xlabel("Evaluation Step" if args.as_eval_step else "Train Step")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle("O-MAPL on SMACv2 (OG-MARL offline data)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {args.out}")
    # Also dump the final-step mean +/- std summary.
    print("\nFinal win rate (mean +/- std over seeds):")
    for scenario in args.scenarios:
        _, data = load_runs(args.runs_dir, args.exp_name, args.algo,
                            scenario, args.metric)
        if data is not None:
            print(f"  {scenario:18s}: {data[:, -1].mean():6.2f} +/- "
                  f"{data[:, -1].std():5.2f}  (n={data.shape[0]} seeds)")


if __name__ == "__main__":
    main()
