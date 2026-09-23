"""Plot V15 persistence and encoded rule variation without treating it as learning benefit."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="V15: evolving the local plasticity rule")
    parser.add_argument("--log-population", action="store_true")
    args = parser.parse_args()
    entries = json.loads(args.input.read_text())
    records = {(r["metadata"]["seed"], r["metadata"]["ablation"]): r for r in entries}
    if len(records) != len(entries):
        raise ValueError("Duplicate seed/treatment records")
    seeds = sorted({seed for seed, _ in records})
    horizon = max(r["final"]["time"] for r in entries) / 60
    fig, axes = plt.subplots(len(seeds), 2, figsize=(10, 3 * len(seeds)), squeeze=False)
    for row, seed in enumerate(seeds):
        for mode, label, color in (
            ("none", "Evolving rule", "#278e6a"),
            ("fixed_rule", "Fixed correlation rule", "#7965a8"),
            ("no_plasticity", "Acquired offsets disabled", "#b8842a"),
        ):
            if (seed, mode) not in records:
                continue
            r = records[seed, mode]
            points = {v["time"]: v for v in [*r["checkpoints"], r["final"]]}
            points = sorted(points.values(), key=lambda v: v["time"])
            times = [v["time"] / 60 for v in points]
            axes[row, 0].plot(times, [v["population"] for v in points], color=color, label=label)
            axes[row, 1].plot(
                times,
                [v["mean_noncorrelation_rule_weight"] for v in points],
                color=color,
                label=label,
            )
        axes[row, 0].set_ylabel(f"Environment {seed}\nPopulation")
        axes[row, 1].set_ylabel("Mean encoded |B| + |C| + |D|")
        if args.log_population:
            axes[row, 0].set_yscale("symlog", linthresh=1)
            axes[row, 0].set_yticks((0, 1, 10, 100), ("0", "1", "10", "100"))
        for ax in axes[row]:
            ax.set_xlim(0, horizon)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            if row == len(seeds) - 1:
                ax.set_xlabel("Simulated minutes")
    axes[0, 0].set_title("Living populations" + (" (log above 1)" if args.log_population else ""))
    axes[0, 1].set_title("Inherited rule variation")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(args.title)
    fig.text(
        0.5,
        0.025,
        "Genomes and habitats are matched at initialization; ordinary mutation remains active.\n"
        "Ignored rule genes can also change in controls. Their variation is not evidence of "
        "useful learning.\nExtinction ends a trial; an empty population has zero mean rule weight.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output.with_suffix(f".{extension}"), dpi=180)


if __name__ == "__main__":
    main()
