"""Plot V21's ecological comparisons and its separate empty-plane diagnostic."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def save(fig, name):
    fig.savefig(f"docs/{name}.png", dpi=150)
    svg = Path(f"docs/{name}.svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    data = json.loads(Path("docs/results/v21-exploration.json").read_text())
    colors = ("#207b91", "#bb632b", "#7656a0")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True, sharey=True)
    for ax, suffix in zip(axes, ("learning", "noise-only"), strict=True):
        for row in data["trials"]:
            if row["treatment"] not in (f"iid-{suffix}", f"correlated-{suffix}"):
                continue
            points = sorted(
                {r["tick"]: r for r in [*row["checkpoints"], row["final"]]}.values(),
                key=lambda r: r["tick"],
            )
            ax.plot(
                [r["time"] / 60 for r in points],
                [r["population"] for r in points],
                color=colors[row["seed"] - 1],
                linestyle="-" if row["treatment"].startswith("correlated") else "--",
                linewidth=1.8,
            )
        ax.set(xlabel="Simulated minutes", ylabel="Living creatures", ylim=(0, None))
        ax.set_title("Motor learning enabled" if suffix == "learning" else "Motor updates disabled")
        ax.grid(alpha=0.15)
        ax.legend(
            handles=[
                Line2D([0], [0], color="#444444", linestyle=style, label=label)
                for style, label in (("--", "Independent noise"), ("-", "Two-second persistence"))
            ],
            frameon=False,
        )
    fig.legend(
        handles=[
            Line2D([0], [0], color=color, label=f"Seed {seed}")
            for seed, color in enumerate(colors, 1)
        ],
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    fig.suptitle("V21: persistent exploration changes ecological outcomes", fontsize=15)
    save(fig, "v21-exploration")

    movement = json.loads(Path("docs/results/v21-movement.json").read_text())
    # Use the first six predetermined sampled circuits, with no visual selection.
    group = movement["groups"][0]
    labels = {"none": "No exploration", "iid": "Independent", "correlated": "Persistent"}
    modes = {"none": "#9b9b9b", "iid": "#207b91", "correlated": "#c06a30"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        for row in group["trials"]:
            path = row["paths"][index]
            ax.plot(
                [p[0] for p in path], [p[1] for p in path], color=modes[row["mode"]], linewidth=1.4
            )
            ax.scatter(*path[-1], color=modes[row["mode"]], s=14)
        ax.scatter(0, 0, marker="+", color="black", s=35)
        ax.set(title=f"Sampled circuit {index + 1}", xlabel="World units", ylabel="World units")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15)
    fig.legend(
        handles=[Line2D([0], [0], color=color, label=labels[key]) for key, color in modes.items()],
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    fig.suptitle(
        "Same circuits and noise magnitude, different exploration timing\n"
        "120 s in an empty plane; fixed observations, standardized body, no learning",
        fontsize=13,
    )
    save(fig, "v21-movement")


if __name__ == "__main__":
    main()
