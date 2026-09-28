"""Plot the audited passive and native prediction-horizon comparisons."""

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
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    passive = json.loads(Path("docs/results/prediction-horizons.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True, sharey=True)
    for key, ax in zip(("long", "short"), axes, strict=True):
        trials = passive[key]["trials"]
        for mode, offset, label, color in (
            ("hidden", -0.14, "Hidden only", "#777e82"),
            ("sensory", 0.14, "Hidden + sensations", "#207b91"),
        ):
            rows = [r["predictors"][mode]["all_adults"] for r in trials]
            ratios = [r["mse_observed"] / r["zero_mse_observed"] for r in rows]
            ax.bar([i + offset for i in (1, 2, 3)], ratios, width=0.25, color=color, label=label)
        counts = [r["predictors"]["hidden"]["all_adults"]["count"] for r in trials]
        ax.set_xticks((1, 2, 3), [f"Seed {i}\nn={n}" for i, n in enumerate(counts, 1)])
        ax.set(
            title=f"{trials[0]['prediction_horizon']:g}-second prediction horizon", ylim=(0, 1.4)
        )
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Prediction error / zero-prediction error\n(smaller is better)")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Each forecast is scored against returns under its own horizon", fontsize=13)
    save(fig, "v23-prediction-horizons")

    report = json.loads(Path("docs/results/v23-short-pilots.json").read_text())
    modes = {
        "long": ("20 s / own returns", "#777e82", "-"),
        "long-shuffled": ("20 s / shuffled", "#777e82", "--"),
        "short": ("2 s / own returns", "#207b91", "-"),
        "short-shuffled": ("2 s / shuffled", "#207b91", "--"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True, sharey=True)
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
               loc="outside lower center", ncols=4, frameon=False)
    fig.suptitle("Native motor learning: short and long prediction horizons", fontsize=14)
    save(fig, "v23-short-pilots")


if __name__ == "__main__":
    main()
