"""Plot V8's four matched treatments from committed compact evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / "docs/results/v8.json").read_text())
    by_path = {row["path"]: row for row in rows}
    treatments = (
        ("evolving-plastic", "Evolving structure, plastic", "#158374", "-"),
        ("evolving-frozen", "Evolving structure, frozen", "#158374", "--"),
        ("fixed-plastic", "Fixed structure, plastic", "#667491", "-"),
        ("fixed-frozen", "Fixed structure, frozen", "#667491", "--"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey="row")
    for column, seed in enumerate((111, 112, 113)):
        for treatment, label, color, style in treatments:
            row = by_path[f"runs/v8-{treatment}/seed-{seed}"]
            points = {p["tick"]: p for p in [*row["checkpoints"], row["final"]]}
            points = sorted(points.values(), key=lambda p: p["time"])
            time = [p["time"] / 60 for p in points]
            axes[0, column].plot(
                time, [p["population"] for p in points], style, label=label, color=color, lw=1.7
            )
            axes[1, column].plot(
                time, [p["mean_neurons"] for p in points], style, color=color, lw=1.7
            )
            if treatment == "evolving-plastic":
                expressed = [
                    [n for n, count in enumerate(p["neuron_histogram"]) if count] for p in points
                ]
                axes[1, column].fill_between(
                    time,
                    [min(n) for n in expressed],
                    [max(n) for n in expressed],
                    color=color,
                    alpha=0.12,
                )
        axes[0, column].set_title(f"Environment {seed}")
        axes[0, column].set_ylim(0, 200)
        axes[1, column].set_xlabel("Simulated minutes")
        axes[1, column].set_ylim(12, 20)
        for ax in axes[:, column]:
            ax.set_xlim(0, 60)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Population")
    axes[1, 0].set_ylabel("Mean active recurrent neurons per module")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Evolving circuit structure: modest variation, mixed ecological outcomes", fontsize=14
    )
    fig.text(
        0.5,
        0.025,
        "One source ancestry, three new environments. All four treatments share initial genomes "
        "and cost coefficients.\nShading: living neuron-count range for evolving/plastic "
        "communities. Slots are bounded at 32; fixed graphs retain 16 active units. "
        "Metrics sampled every 120 seconds.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    for suffix in ("png", "svg"):
        fig.savefig(root / f"docs/v8-topology.{suffix}", dpi=180)


if __name__ == "__main__":
    main()
