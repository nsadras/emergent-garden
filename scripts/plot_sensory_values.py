"""Plot V23's ecology and paired prediction diagnostics from audited records."""

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
    path = Path(f"docs/{name}.svg")
    fig.savefig(path)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main():
    report = json.loads(Path("docs/results/v23-sensory-values.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    modes = {
        "hidden": ("Hidden-only prediction", "#777e82", "--"),
        "sensory": ("Hidden + sensory inputs", "#207b91", "-"),
        "shuffled": ("Sensory + shuffled returns", "#bb632b", "-"),
    }
    ceiling = 1.05 * max(p["population"] for r in report["trials"]
                         for p in [*r["checkpoints"], r["final"]])
    for seed, ax in enumerate(axes, 1):
        for row in report["trials"]:
            if row["seed"] != seed:
                continue
            points = sorted({p["tick"]: p for p in [*row["checkpoints"], row["final"]]}.values(),
                            key=lambda p: p["tick"])
            _, color, style = modes[row["treatment"]]
            ax.plot([p["time"] / 60 for p in points], [p["population"] for p in points],
                    color=color, linestyle=style, linewidth=1.8)
        ax.set(title=f"Seed {seed}", xlabel="Simulated minutes", ylim=(0, ceiling))
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Living creatures")
    fig.legend(handles=[Line2D([0], [0], label=label, color=color, linestyle=style)
                        for label, color, style in modes.values()],
               loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("V23: direct sensory features for the energy predictor", fontsize=14)
    save(fig, "v23-sensory-values")

    paired = report.get("paired_forecasts")
    if paired:
        reports = [paired]
        if report.get("paired_fast_forecasts"):
            reports.append(report["paired_fast_forecasts"])
        fig, axes = plt.subplots(1, len(reports), figsize=(5 * len(reports), 4.5),
                                 constrained_layout=True, sharey=True, squeeze=False)
        ceilings = []
        for data, ax in zip(reports, axes[0], strict=True):
            for mode, offset, color, label in (
                ("hidden", -0.14, "#777e82", "Hidden only"),
                ("sensory", 0.14, "#207b91", "Hidden + sensations"),
            ):
                rows = [r["predictors"][mode]["all_adults"] for r in data["trials"]]
                ratios = [r["mse_observed"] / r["zero_mse_observed"] for r in rows]
                ceilings.extend(ratios)
                ax.bar([i + offset for i in (1, 2, 3)], ratios, width=0.25,
                       color=color, label=label)
            counts = [r["predictors"]["hidden"]["all_adults"]["count"] for r in data["trials"]]
            ax.set_xticks((1, 2, 3), [f"Seed {i}\nn={n}" for i, n in enumerate(counts, 1)])
            ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
            ax.set_title(f"Maximum critic rate {data['prediction_rate']:g}")
            ax.grid(axis="y", alpha=0.15)
        axes[0, 0].set_ylim(0, 1.12 * max(ceilings))
        axes[0, 0].set_ylabel("Prediction error / zero-prediction error\n(smaller is better)")
        axes[0, -1].legend(loc="upper right", frameon=False, fontsize=8)
        fig.suptitle(
            "Passive predictors share exactly the same creatures and outcomes", fontsize=13
        )
        save(fig, "v23-paired-forecasts")


if __name__ == "__main__":
    main()
