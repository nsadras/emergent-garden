"""Plot community assembly from committed compact records, without loading runs."""

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
    records = json.loads(args.input.read_text())
    treatments = (
        ("v8-assembly-grazers", "Grazer source alone"),
        ("v8-assembly-scavengers", "Scavenger source alone"),
        ("v8-assembly-mixed", "Mixed, all mechanisms"),
        ("v8-assembly-no-attacks", "Mixed, attacks disabled"),
        ("v8-assembly-no-recycling", "Mixed, recycling disabled"),
    )
    by_path = {row["path"]: row for row in records}
    fig, axes = plt.subplots(3, 5, figsize=(14, 8), sharex=True, sharey=True)
    for column, (path, title) in enumerate(treatments):
        for row, seed in enumerate((151, 152, 153)):
            record = by_path[f"runs/{path}/seed-{seed}"]["ancestry"]
            points = {r["time"]: r for r in [*record["checkpoints"], record["final"]]}
            points = sorted(points.values(), key=lambda r: r["time"])
            ax = axes[row, column]
            for group, label, color in (
                (0, "Grazer ancestry", "#228b62"),
                (1, "Scavenger ancestry", "#c18425"),
            ):
                ax.plot(
                    [r["time"] / 60 for r in points],
                    [r["groups"][group]["population"] for r in points],
                    color=color,
                    label=label,
                    lw=1.8,
                )
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 200)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if row == 2:
                ax.set_xlabel("Simulated minutes")
            if column == 0:
                ax.set_ylabel(f"Environment {seed}\nPopulation")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Community assembly depends on predation and recycling", fontsize=14)
    fig.text(
        0.5,
        0.025,
        "Each trial starts 192 bodies: one evolved source, or 96 from each. Colors track ancestry, "
        "not present diet.\nNormal evolution remains active; acquired neural state starts at zero. "
        "These are assembled communities, not spontaneous speciation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=180)


if __name__ == "__main__":
    main()
