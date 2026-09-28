"""Plot V22's matched pilots from their audited result record."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def forecasts():
    report = json.loads(Path("docs/results/v22-value-forecasts.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True, sharey=True)
    for centered, ax in zip((True, False), axes, strict=True):
        rows = [r for r in report["trials"] if r["centered"] == centered]
        for key, label, color, offset in (
            ("all_adults", "All initially mature creatures", "#207b91", -0.13),
            ("adults_at_least_30_seconds_old", "Initially mature and age ≥ 30 s", "#bb632b", 0.13),
        ):
            data = [r[key] for r in rows]
            ratios = [r["mse_observed"] / r["zero_mse_observed"] for r in data]
            positions = [seed + offset for seed in range(1, len(rows) + 1)]
            ax.bar(positions, ratios, width=0.24, color=color, label=label)
            for x, ratio, row in zip(positions, ratios, data, strict=True):
                ax.text(
                    x, ratio + 0.035, f"n={row['count']}", ha="center", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.3),
                )
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set(
            title="Centered target" if centered else "Direct energy target",
            xlabel="Community seed", xticks=(1, 2, 3), ylim=(0, 1.8),
        )
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Prediction error / zero-prediction error\n(smaller is better)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("V22: predictions precede 100 seconds of observed returns", fontsize=13)
    fig.savefig("docs/v22-value-forecasts.png", dpi=150)
    svg = Path("docs/v22-value-forecasts.svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main():
    report = json.loads(Path("docs/results/v22-value-prediction.json").read_text())
    modes = {
        "baseline": ("Running-mean baseline", "#777e82", "--"),
        "value": ("Value prediction", "#207b91", "-"),
        "shuffled": ("Value + shuffled returns", "#bb632b", "-"),
        "raw": ("Value prediction", "#207b91", "-"),
        "raw-shuffled": ("Value + shuffled returns", "#bb632b", "-"),
    }
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True, sharey=True)
    ceiling = 1.05 * max(
        point["population"] for row in report["trials"]
        for point in [*row["checkpoints"], row["final"]]
    )
    for index, (label, groups) in enumerate(
        (
            ("Centered target", ("baseline", "value", "shuffled")),
            ("Direct energy target", ("baseline", "raw", "raw-shuffled")),
        )
    ):
        for seed, ax in enumerate(axes[index], 1):
            for row in report["trials"]:
                if row["seed"] != seed or row["treatment"] not in groups:
                    continue
                points = sorted(
                    {p["tick"]: p for p in [*row["checkpoints"], row["final"]]}.values(),
                    key=lambda p: p["tick"],
                )
                _, color, style = modes[row["treatment"]]
                ax.plot(
                    [p["time"] / 60 for p in points],
                    [p["population"] for p in points],
                    color=color,
                    linestyle=style,
                    linewidth=1.8,
                )
            ax.set(title=f"{label} / seed {seed}", xlabel="Simulated minutes", ylim=(0, ceiling))
            ax.grid(alpha=0.15)
        axes[index, 0].set_ylabel("Living creatures")
    fig.legend(
        handles=[
            Line2D([0], [0], label=label, color=color, linestyle=style)
            for label, color, style in list(modes.values())[:3]
        ],
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    fig.suptitle("V22: matched ecological tests of energy prediction", fontsize=15)
    fig.savefig("docs/v22-value-prediction.png", dpi=150)
    svg = Path("docs/v22-value-prediction.svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    forecasts()


if __name__ == "__main__":
    main()
