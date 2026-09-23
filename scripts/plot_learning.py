"""Plot the paired V7 random-founder trials from committed compact evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root / "docs/results/v7.json").read_text())
    by_path = {row["path"]: row for row in rows}
    treatments = (
        ("v7-random-plastic", "Original plasticity", "#bd5d32", 1.0),
        ("v7-random-bounded", "Tighter plasticity", "#158374", 0.1),
        ("v7-random-frozen", "Frozen synapses", "#626b8a", 1.0),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey="row")
    for column, seed in enumerate((71, 72, 73)):
        for directory, label, color, bound in treatments:
            row = by_path[f"runs/{directory}/seed-{seed}"]
            checkpoints = {r["tick"]: r for r in [*row["checkpoints"], row["final"]]}
            points = sorted(checkpoints.values(), key=lambda p: p["time"])
            time = [p["time"] / 60 for p in points]
            population = [p["population"] for p in points]
            offsets = [p["mean_plastic_magnitude"] / bound for p in points]
            if population[-1] == 0:
                time.append(60)
                population.append(0)
                # No mean synaptic state exists after extinction.
                offsets.append(float("nan"))
            axes[0, column].plot(time, population, label=label, color=color, lw=1.8)
            axes[1, column].plot(time, offsets, color=color, lw=1.8)
        axes[0, column].set_title(f"Random founder seed {seed}")
        axes[0, column].set_yscale("symlog", linthresh=1)
        axes[0, column].set_yticks([0, 1, 5, 20, 100, 192], [0, 1, 5, 20, 100, 192])
        axes[0, column].set_ylim(-0.1, 220)
        axes[1, column].set_ylim(-0.02, 1.02)
        axes[1, column].set_xlabel("Simulated minutes")
        for ax in axes[:, column]:
            ax.set_xlim(0, 60)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Population (symmetric log scale)")
    axes[1, 0].set_ylabel("Mean |synaptic offset| / allowed limit")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Plasticity changes ecology; these trials do not demonstrate learning", fontsize=14
    )
    fig.text(
        0.5,
        0.025,
        "Matched initial genomes and environment per seed. Original: rate 0.2, limit 1.0; "
        "tighter: rate 0.02, limit 0.1. Both pay the same cost.\n"
        "Tighter parameters were selected after observing the original trials; "
        "they require validation on new seeds. Lines sample recorded metrics every 120 seconds.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    for suffix in ("png", "svg"):
        fig.savefig(root / f"docs/v7-learning.{suffix}", dpi=180)


if __name__ == "__main__":
    main()
