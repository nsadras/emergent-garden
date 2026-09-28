"""Plot the audited V24 ecological screen and separate circuit pulse probe."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def save(fig, name):
    fig.savefig(f"docs/{name}.png", dpi=150)
    svg = Path(f"docs/{name}.svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def populations(rows, name, title):
    colors = {"homogeneous": "#666666", "inherited": "#207b91", "evolving": "#b36025"}
    labels = {
        "homogeneous": "Same timing within a brain",
        "inherited": "Varied timing, no direct timing mutation",
        "evolving": "Varied timing with timing mutation",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)
    for row in rows:
        mode = row["treatment"]
        if mode not in colors:
            continue
        column = row["seed"] - 1
        points = sorted(
            {r["tick"]: r for r in [*row["checkpoints"], row["final"]]}.values(),
            key=lambda r: r["tick"],
        )
        for ax, field in zip(axes[:, column], ("population", "births"), strict=True):
            ax.plot(
                [r["time"] / 60 for r in points],
                [r[field] for r in points],
                color=colors[mode],
                linewidth=1.8,
            )
    for col in range(3):
        axes[0, col].set_title(f"Seed {col + 1}")
        axes[1, col].set_xlabel("Simulated minutes")
        for ax in axes[:, col]:
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.15)
    axes[0, 0].set_ylabel("Living creatures")
    axes[1, 0].set_ylabel("Cumulative births")
    fig.legend(
        handles=[
            Line2D([0], [0], color=color, label=labels[mode]) for mode, color in colors.items()
        ],
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(title, fontsize=14)
    save(fig, name)


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    data = json.loads(Path("docs/results/v24-neural-timing.json").read_text())
    populations(
        data["trials"],
        "v24-neural-timing",
        "V24: inherited neural response times — three matched evolutionary starts",
    )
    if data.get("continuations"):
        populations(
            data["continuations"],
            "v24-neural-timing-long",
            "V24: 30-minute continuations of the same three evolutionary starts",
        )

    probe = json.loads(Path("docs/results/v24-pulse-responses.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), constrained_layout=True)
    for mode, color, label in (
        ("homogeneous", "#666666", "Same timing within a brain"),
        ("heterogeneous", "#207b91", "Varied timing within a brain"),
    ):
        histograms = [t["modes"][mode]["time_histogram"] for t in probe["trials"]]
        edges = np.array(histograms[0]["edges"])
        counts = sum(np.array(hist["counts"]) for hist in histograms)
        axes[0].stairs(counts / counts.sum(), edges, color=color, linewidth=2, label=label)
        for trial, style in zip(probe["trials"], ("-", "--", ":"), strict=True):
            curve = trial["modes"][mode]["curve"]
            axes[1].plot(
                [p["time"] for p in curve],
                [p["activity_rms"][1] for p in curve],
                color=color,
                linestyle=style,
                linewidth=1.5,
            )
            records = trial["modes"][mode]["after_pulse"]
            points = [records[str(t)] for t in (0.0, 1.0, 5.0, 20.0)]
            axes[2].plot(
                [p["seconds_after_pulse"] for p in points],
                [p["mean_motor_rms"] for p in points],
                color=color,
                linestyle=style,
                marker="o",
                markersize=3,
            )
    axes[0].set(
        xscale="log",
        xlabel="Inherited response time (seconds)",
        ylabel="Fraction of active neurons",
        title="576 circuits across three seeds",
    )
    axes[1].axvspan(0, 1, color="#d9bd7c", alpha=0.25)
    axes[1].set(
        xlabel="Seconds since pulse began",
        ylabel="Median neural activity difference (RMS)",
        title="Pulse versus no-pulse branch",
    )
    axes[2].set(
        xlabel="Seconds after pulse ended",
        ylabel="Mean motor output difference (RMS)",
        title="Differences reaching the motor outputs",
    )
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.15)
    fig.legend(
        handles=axes[0].get_legend_handles_labels()[0],
        labels=axes[0].get_legend_handles_labels()[1],
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    fig.suptitle(
        "Paired circuit probe: altered responses do not establish useful memory\n"
        "Fixed weights, no learning or movement; solid / dashed / dotted = seeds 1 / 2 / 3",
        fontsize=13,
    )
    save(fig, "v24-pulse-responses")


if __name__ == "__main__":
    main()
