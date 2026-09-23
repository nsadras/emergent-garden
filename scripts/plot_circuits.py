"""Plot V19 screening outcomes from the audited, committed data."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    report = json.loads(Path("docs/results/v19-circuits.json").read_text())
    colors = ("#207b91", "#bb632b", "#7656a0")
    styles = {"baseline": ":", "varied": "-", "mutation": "--", "learning": "-", "noise-only": "--"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    groups = (("baseline", "varied", "mutation"), ("learning", "noise-only"))
    for ax, treatments in zip(axes, groups, strict=True):
        for trial in report["trials"]:
            group = trial["treatment"]
            if group not in treatments:
                continue
            samples = sorted(
                {r["tick"]: r for r in [*trial["checkpoints"], trial["final"]]}.values(),
                key=lambda r: r["tick"],
            )
            ax.plot(
                [r["time"] / 60 for r in samples],
                [r["population"] for r in samples],
                color=colors[trial["seed"] - 1],
                linestyle=styles[group],
                linewidth=1.8,
            )
        ax.set(xlabel="Simulated minutes", ylabel="Living creatures", ylim=(0, 215))
        ax.grid(alpha=0.15)
        ax.legend(
            handles=[
                Line2D([0], [0], color="#444444", linestyle=styles[g], label=g.capitalize())
                for g in treatments
            ],
            frameon=False,
            loc="upper right",
            fontsize=9,
        )
    axes[0].set_title("Founder circuits and mutation")
    axes[1].set_title("Motor learning vs. identical exploration noise")
    fig.legend(
        handles=[
            Line2D([0], [0], color=color, label=f"Seed {seed}")
            for seed, color in enumerate(colors, 1)
        ],
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    fig.suptitle("V19: neural variation has mixed ecological effects", fontsize=15)
    fig.savefig("docs/v19-circuits.png", dpi=150)
    svg = Path("docs/v19-circuits.svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
