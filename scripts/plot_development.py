"""Compare population and expressed body sizes under matched V13 birth treatments."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def history(record):
    points = {r["time"]: r for r in [*record["checkpoints"], record["final"]]}
    return sorted(points.values(), key=lambda r: r["time"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="V13: expressing inherited bodies through growth")
    args = parser.parse_args()
    entries = json.loads(args.input.read_text())
    records = {(r["metadata"]["seed"], r["metadata"]["ablation"]): r for r in entries}
    seeds = sorted({seed for seed, _ in records})
    fig, axes = plt.subplots(len(seeds), 2, figsize=(10, 3 * len(seeds)), squeeze=False)
    horizon = max(r["final"]["time"] for r in entries) / 60
    for row, seed in enumerate(seeds):
        for mode, label, color in (
            ("none", "Juvenile offspring", "#278e6a"),
            ("adult_births", "Fully formed offspring", "#7965a8"),
        ):
            points = history(records[seed, mode])
            times = [r["time"] / 60 for r in points]
            axes[row, 0].plot(times, [r["population"] for r in points], color=color, label=label)
            axes[row, 1].plot(
                times, [sum(r["module_histogram"][1:]) for r in points], color=color, label=label
            )
            if mode == "none":
                axes[row, 1].plot(
                    times,
                    [sum(r["target_module_histogram"][1:]) for r in points],
                    color="#b8842a",
                    linestyle="--",
                    label="Encoded larger plans (juvenile treatment)",
                )
        axes[row, 0].set_ylabel(f"Environment {seed}\nPopulation")
        axes[row, 1].set_ylabel("Bodies with 2–3 modules")
        for ax in axes[row]:
            ax.set_xlim(0, horizon)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            if row == len(seeds) - 1:
                ax.set_xlabel("Simulated minutes")
        axes[row, 1].set_ylim(0, max(1, axes[row, 1].get_ylim()[1]))
        axes[row, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].set_title("All creatures")
    axes[0, 1].set_title("Expressed and encoded body size")
    for ax in axes[0]:
        ax.legend(fontsize=8)
    fig.suptitle(args.title, fontsize=14)
    fig.text(
        0.5,
        0.015,
        "Founders are identical one-module newborns in both treatments; "
        "their encoded plans may be larger.\n"
        "Genetic mutation remains active. Curves describe body sizes, not independent species.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=180)


if __name__ == "__main__":
    main()
