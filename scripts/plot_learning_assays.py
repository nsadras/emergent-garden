"""Plot matched learning transplants, including shuffled returns when audited."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    report = json.loads(Path("docs/results/v19-learning-assays.json").read_text())
    groups = [(source, environment) for source in (1, 2, 3) for environment in (601, 602)]
    modes = [
        ("none", "Own returns", "#207b91"),
        ("no_motor_learning", "No motor updates", "#969d9e"),
    ]
    if report["shuffled_included"]:
        modes.append(("shuffled_motor_reward", "Another creature's returns", "#bb632b"))
    rows = {
        (r["source"], r["environment"], r["mode"]): r["final"]
        for r in report["trials"]
        if r["group"] == "descendants"
    }
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    width = 0.8 / len(modes)
    for ax, key, scale, ylabel in zip(
        axes,
        ("births", "fresh_absorbed"),
        (1, 1000),
        ("Births in 360 simulated seconds", "Fresh food absorbed (thousands of energy units)"),
        strict=True,
    ):
        for i, (mode, label, color) in enumerate(modes):
            ax.bar(
                np.arange(len(groups)) + (i - (len(modes) - 1) / 2) * width,
                [rows[source, environment, mode][key] / scale for source, environment in groups],
                width=width,
                label=label,
                color=color,
            )
        ax.set_xticks(range(len(groups)), [f"Source {s}\nEnv {e}" for s, e in groups], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.yaxis.grid(alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(modes), frameon=False)
    fig.suptitle("V19: paired tests of within-lifetime motor learning", fontsize=15)
    fig.savefig("docs/v19-learning-assays.png", dpi=150)
    svg = Path("docs/v19-learning-assays.svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
