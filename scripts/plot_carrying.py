"""Plot paired V18 experiments from their committed audit."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    report = json.loads(Path("docs/results/v18-carrying.json").read_text())
    colors = ("#207b91", "#bb632b", "#7656a0")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, rate in zip(axes, (60, 300), strict=True):
        for trial in report["trials"]:
            if trial["handling_rate"] != rate:
                continue
            samples = sorted(
                {r["tick"]: r for r in [*trial["checkpoints"], trial["final"]]}.values(),
                key=lambda r: r["tick"],
            )
            ax.plot(
                [r["time"] / 60 for r in samples],
                [r["population"] for r in samples],
                color=colors[trial["seed"] - 1],
                linestyle="-" if trial["gut_capacity"] else "--",
                linewidth=2 if trial["gut_capacity"] else 1.2,
            )
        ax.set(
            title=f"Raw processing rate {rate} per tissue / second",
            xlabel="Simulated minutes",
            ylabel="Living creatures",
            ylim=(0, 205),
        )
        ax.grid(alpha=0.15)
    handles = [
        Line2D([0], [0], color=color, label=f"Seed {seed}") for seed, color in enumerate(colors, 1)
    ]
    handles += [
        Line2D([0], [0], color="#444444", label="Carrying", linewidth=2),
        Line2D([0], [0], color="#444444", label="Contact only", linestyle="--"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=8)
    fig.suptitle("V18: collect a meal, then digest while moving", fontsize=15)
    fig.savefig("docs/v18-carrying.png", dpi=150)
    svg = Path("docs/v18-carrying.svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
