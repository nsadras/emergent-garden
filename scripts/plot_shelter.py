"""Compare V10 shelter, protection ablation, and sensory ablation."""

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
    by_path = {r["path"]: r for r in json.loads(args.input.read_text())}
    treatments = (
        ("v10-shelter", "Cover and sensing"),
        ("v10-no-shelter", "Protection disabled"),
        ("v10-no-shelter-cue", "Cover sensing disabled"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
    for row, seed in enumerate((161, 162, 163)):
        for column, (path, title) in enumerate(treatments):
            record = by_path[f"runs/{path}/seed-{seed}"]["ancestry"]
            if record["final"]["time"] != 3600:
                raise ValueError("Expected completed one-hour trials")
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
            ax.set_ylim(0, 110)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(title)
            if row == 2:
                ax.set_xlabel("Simulated minutes")
            if column == 0:
                ax.set_ylabel(f"Environment {seed}\nPopulation")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Spatial shelter: matched protection and sensing controls", fontsize=14)
    fig.text(
        0.5,
        0.025,
        "Each community starts 96 bodies from each evolved source, with fresh neural states.\n"
        "Four of eight patches have cover; full cover weakens bites by 95% at either endpoint.\n"
        "Colors follow source ancestry, not current diet. "
        "All three treatments share initial genomes.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=180)


if __name__ == "__main__":
    main()
