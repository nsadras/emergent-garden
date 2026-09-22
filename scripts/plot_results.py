"""Regenerate standalone research plots from recorded simulation metrics."""

import json
import os
from functools import cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@cache
def compact_runs():
    return {
        entry["path"]: entry
        for path in Path("docs/results").glob("v[1-5].json")
        for entry in json.loads(path.read_text())
    }


def records(path):
    metrics = path / "metrics.jsonl"
    if metrics.exists():
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    else:
        entry = compact_runs()[str(path)]
        rows = [*entry["checkpoints"], entry["final"]]
    return sorted({r["tick"]: r for r in rows}.values(), key=lambda r: r["tick"])


def main():
    destination = Path("docs")
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    roots = ["v1-maturation", "v2-pilot", "v3-short-food", "v4-recovery", "v5-pilot"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for version, (root, ax) in enumerate(zip(roots, axes.flat, strict=False), 1):
        for seed in (11, 12, 13):
            path = Path("runs") / root / f"seed-{seed}"
            rows = records(path)
            ax.plot(
                [r["time"] / 60 for r in rows], [r["population"] for r in rows], label=path.name
            )
        ax.set(
            title=f"V{version}",
            xlabel="Simulated minutes",
            ylabel="Living creatures",
            ylim=(0, None),
        )
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.15)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0,
        0.85,
        "Independent random-founder runs\n\n"
        "World rules and parameters change between\n"
        "versions; these curves are viability checks,\n"
        "not paired causal comparisons.\n\n"
        "V3: shorter-lived food and forecast cues\n"
        "V4: developmental modules; food retuned\n"
        "V5: costly persistent secretions",
        va="top",
        linespacing=1.5,
    )
    fig.suptitle("Ecological persistence across five versions", fontsize=17)
    fig.savefig(destination / "evolution-versions.png", dpi=150)
    fig.savefig(destination / "evolution-versions.svg")
    plt.close(fig)

    fields = [
        ("population", "Living creatures"),
        ("generation_max", "Maximum living generation"),
        ("mean_diet", "Mean fresh-food allocation"),
        ("mean_modules", "Mean modules per body"),
        ("mean_secretion", "Mean secretion activation"),
        ("signal_mass", "Chemical quantity in dish"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for suffix, seed in zip(("a", "b", "c"), (31, 32, 33), strict=True):
        path = Path("runs") / f"v5-long-{suffix}" / f"seed-{seed}"
        rows = records(path)
        for ax, (field, title) in zip(axes.flat, fields, strict=True):
            ax.plot([r["time"] / 60 for r in rows], [r[field] for r in rows], label=path.name)
            ax.set(xlabel="Simulated minutes", title=title)
            ax.grid(alpha=0.15)
    for ax in axes.flat:
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("V5: longer evolution from inherited V4 populations", fontsize=17)
    fig.savefig(destination / "evolution-long.png", dpi=150)
    fig.savefig(destination / "evolution-long.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for column, seed in enumerate((11, 12, 13)):
        rows = records(Path("runs/v5-pilot") / f"seed-{seed}")
        rows += records(Path("runs") / f"v5-native-long-{seed}")
        rows = sorted({r["tick"]: r for r in rows}.values(), key=lambda r: r["tick"])
        times = [r["time"] / 60 for r in rows]
        axes[0, column].stackplot(
            times,
            *[[r["module_histogram"][i] for r in rows] for i in range(3)],
            labels=("1 module", "2 modules", "3 modules"),
            colors=("#40a56a", "#3098b7", "#db9b41"),
        )
        axes[0, column].set(title=f"Seed {seed}", ylabel="Living creatures", ylim=(0, 200))
        # Cumulative intake differences over two-minute windows, not instantaneous rates.
        sampled = [r for r in rows if r["time"] % 120 == 0]
        for field, label, color in (
            ("fresh_absorbed", "Fresh food", "#40a56a"),
            ("detritus_absorbed", "Detritus", "#db9b41"),
            ("predation_absorbed", "Predation", "#c95a66"),
        ):
            rates = [
                (b[field] - a[field]) / (b["time"] - a["time"])
                for a, b in zip(sampled[:-1], sampled[1:], strict=True)
            ]
            axes[1, column].plot(
                [r["time"] / 60 for r in sampled[1:]], rates, label=label, color=color
            )
        axes[1, column].set(ylabel="Received energy / second", ylim=(0, None))
    for ax in axes.flat:
        ax.set(xlabel="Simulated minutes", xlim=(0, 180))
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("V5: three hours from independent random founders", fontsize=17)
    fig.savefig(destination / "evolution-native.png", dpi=150)
    fig.savefig(destination / "evolution-native.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
