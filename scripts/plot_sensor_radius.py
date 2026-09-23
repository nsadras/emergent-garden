"""Compare the measured sensory signal and reproduction in V20."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    report = json.loads(Path("docs/results/v20-sensor-radius.json").read_text())
    colors = ("#207b91", "#bb632b", "#7656a0")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    for snapshot in report["snapshot_sensitivity"]:
        seed = snapshot["source_seed"]
        axes[0].plot(
            [r["radius_scale"] for r in snapshot["trials"]],
            [r["fresh_direction_rms"] for r in snapshot["trials"]],
            "o-",
            color=colors[seed - 1],
            label=f"Seed {seed}",
        )
    for seed in (1, 2, 3):
        rows = sorted(
            [r for r in report["trials"] if r["seed"] == seed], key=lambda r: r["radius_scale"]
        )
        axes[1].plot(
            [r["radius_scale"] for r in rows],
            [r["final"]["births"] for r in rows],
            "o-",
            color=colors[seed - 1],
        )
    axes[0].set(
        title="Same frozen worlds at 30 minutes",
        ylabel="Directional fresh-food input RMS",
        ylim=(0, 0.045),
    )
    axes[1].set(
        title="Matched fresh starts at 10 minutes", ylabel="Cumulative births", ylim=(0, 500)
    )
    for ax in axes:
        ax.set(xlabel="Receptor distance / body-module radius", xticks=[1, 2, 4])
        ax.grid(alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("V20: stronger directional signals, mixed reproduction", fontsize=15)
    fig.savefig("docs/v20-sensor-radius.png", dpi=150)
    svg = Path("docs/v20-sensor-radius.svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
