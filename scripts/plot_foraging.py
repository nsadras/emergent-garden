"""Regenerate the foraging figure using only the committed audit JSON."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    report = json.loads(Path("docs/results/foraging-pilots.json").read_text())
    trials = report["trials"]
    colors = ("#207b91", "#bb632b", "#7656a0")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, rows, title in (
        (
            axes[0, 0],
            [r for r in trials if r["group"] == "v16-user-baseline"],
            "User's sparse-food settings: all three extinct",
        ),
        (
            axes[0, 1],
            report["fast_feeding_continuations"],
            "Same habitat, 5× processing rate: three persisted",
        ),
    ):
        for seed, trial in enumerate(rows, 1):
            samples = sorted(
                {r["tick"]: r for r in [*trial["checkpoints"], trial["final"]]}.values(),
                key=lambda r: r["tick"],
            )
            ax.plot(
                [r["time"] / 60 for r in samples],
                [r["population"] for r in samples],
                color=colors[seed - 1],
                label=f"Seed {seed}",
            )
        ax.set(title=title, xlabel="Simulated minutes", ylabel="Living creatures", ylim=(0, 220))
        ax.legend(frameon=False)
    groups = list(report["changes"])
    labels = (
        "Sparse\nbaseline",
        "More\nbackground",
        "Smaller\nhabitat",
        "Smaller\npellets",
        "Small habitat\n+ pellets",
        "Faster\nfeeding",
        "V17\nabsolute",
        "V17\ncontrast",
    )
    ax = axes[1, 0]
    for seed in (1, 2, 3):
        ax.scatter(
            [i + 0.13 * (seed - 2) for i in range(len(groups))],
            [
                next(r["final"]["births"] for r in trials if r["group"] == g and r["seed"] == seed)
                for g in groups
            ],
            color=colors[seed - 1],
            s=25,
        )
    ax.set_xticks(range(len(groups)), labels, fontsize=8)
    ax.set(
        title="Screening experiments: births within ten minutes", ylabel="Births", ylim=(0, None)
    )
    ax = axes[1, 1]
    for seed in (1, 2, 3):
        values = [
            next(r["final"]["population"] for r in trials if r["group"] == g and r["seed"] == seed)
            for g in ("v17-absolute", "v17-contrast")
        ]
        ax.plot([0, 1], values, "o-", color=colors[seed - 1], label=f"Seed {seed}")
    ax.set_xticks([0, 1], ["Absolute receptors", "Mean + spatial contrasts"])
    ax.set(
        title="V17 encoding: mixed results in the smaller habitat",
        ylabel="Living creatures at ten minutes",
        xlim=(-0.25, 1.25),
        ylim=(0, None),
    )
    ax.legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    fig.suptitle(
        "Seeking food: intake capacity mattered more than the tested sensor change", fontsize=15
    )
    fig.savefig("docs/foraging-pilots.png", dpi=150)
    fig.savefig("docs/foraging-pilots.svg")


if __name__ == "__main__":
    main()
