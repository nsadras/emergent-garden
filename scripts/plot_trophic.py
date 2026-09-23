"""Plot mean food-transfer rates from completed, committed V9 trial records."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = json.loads(args.input.read_text())
    conditions = (
        ("v9-traced-mixed", "Attacks active"),
        ("v9-traced-no-attacks", "Attacks disabled"),
    )
    matrices = []
    for condition, _ in conditions:
        rows = [r for r in records if Path(r["path"]).parent.name == condition]
        if len(rows) != 3 or any(r["final"]["time"] != 3600 for r in rows):
            raise ValueError("Expected three completed 3,600-second trials per condition")
        matrices.append(
            [
                np.array([r["final"][key][:3] for r in rows]) / 3600
                for key in ("trophic_detritus_uptake", "trophic_predation_uptake")
            ]
        )
    maximum = max(float(m.mean(0).max()) for condition in matrices for m in condition)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    labels = ("Grazer", "Generalist", "Scavenger")
    for row, (_, condition) in enumerate(conditions):
        for column in range(2):
            values = matrices[row][column]
            mean, low, high = values.mean(0), values.min(0), values.max(0)
            ax = axes[row, column]
            layer = ax.imshow(mean, cmap="YlGnBu", vmin=0, vmax=maximum)
            ax.set_xticks(range(3), labels)
            ax.set_yticks(range(3), labels)
            ax.set_xlabel("Consumer" if column == 0 else "Predator")
            ax.set_ylabel(f"{condition}\n" + ("Detritus producer" if column == 0 else "Prey"))
            for i in range(3):
                for j in range(3):
                    label = f"{mean[i, j]:.2f}\n({low[i, j]:.2f}–{high[i, j]:.2f})"
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if mean[i, j] > maximum * 0.5 else "#152738",
                    )
            if row == 0:
                ax.set_title("Detritus uptake" if column == 0 else "Predation uptake")
    fig.colorbar(layer, ax=axes, shrink=0.75, label="Energy absorbed per simulated second")
    fig.suptitle(
        "Most scavenger detritus uptake comes from the scavenger guild\n"
        "Cell labels: mean (range) across environments 151–153",
        fontsize=13,
    )
    fig.supxlabel(
        "Guilds describe diet at each transfer, not ancestry or species.\n"
        "Rates average the full first hour. Energy can pass through more than one transfer.",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=180)


if __name__ == "__main__":
    main()
