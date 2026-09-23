"""Show V11 mixed-community persistence and scavenger-dependency controls."""

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
    args = parser.parse_args()
    records = {row["path"]: row for row in json.loads(args.input.read_text())}
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)

    def line(ax, treatment, seed, group, **style):
        record = records[f"runs/{treatment}/seed-{seed}"]["ancestry"]
        points = {r["time"]: r for r in [*record["checkpoints"], record["final"]]}
        points = sorted(points.values(), key=lambda row: row["time"])
        ax.plot(
            [r["time"] / 60 for r in points],
            [r["groups"][group]["population"] for r in points],
            **style,
        )

    for row, seed in enumerate((181, 182, 183)):
        for column, treatment in enumerate(("v11-limited-mixed", "v11-unlimited-mixed")):
            for group, label, color in (
                (0, "Grazer ancestry", "#228b62"),
                (1, "Scavenger ancestry", "#c18425"),
            ):
                line(axes[row, column], treatment, seed, group, color=color, label=label)
            axes[row, column].set_ylim(0, 110)
        for treatment, label, color, style in (
            ("v11-limited-scavenger", "Alone, limited", "#b64830", "-"),
            ("v11-unlimited-scavenger", "Alone, unlimited", "#bd912a", "-"),
            ("v11-no-recycling", "Mixed, recycling off", "#62509b", "--"),
        ):
            line(axes[row, 2], treatment, seed, 1, color=color, label=label, linestyle=style)
        axes[row, 2].set_ylim(0, 200)
        axes[row, 0].set_ylabel(f"Environment {seed}\nPopulation")
        for ax in axes[row]:
            ax.set_xlim(0, 60)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 2:
                ax.set_xlabel("Simulated minutes")
    for ax, title in zip(
        axes[0],
        (
            "Mixed / limited processing",
            "Mixed / unlimited processing",
            "Scavenger ancestry controls",
        ),
        strict=True,
    ):
        ax.set_title(title, fontsize=11)
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Testing scavenger dependence under finite processing", fontsize=14)
    fig.text(
        0.5,
        0.025,
        "Mixtures start with 96 founders from each evolved source; "
        "alone treatments start with 192.\n"
        "Recycling-off mixtures retain fresh-food assimilation. "
        "All treatments feed at 5 Hz.\n"
        "Curves track source ancestry, not current diet or species; extinct trials end early.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=180)


if __name__ == "__main__":
    main()
