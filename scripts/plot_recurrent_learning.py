"""Plot audited V25 community outcomes and the separate supplied native task."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STYLES = {
    "baseline": ("#555555", "Mechanism off"),
    "noise-only": ("#b87925", "Hidden noise only"),
    "learning": ("#187b85", "Own-return learning"),
    "shuffled": ("#9d4678", "Shuffled recurrent feedback"),
}


def save(fig, name):
    fig.savefig(f"docs/{name}.png", dpi=150)
    svg = Path(f"docs/{name}.svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def probe():
    report = json.loads(Path("docs/results/v25-native-recurrent-probe.json").read_text())
    assert report["completed"] and report["exact_endpoint_validation_replay"]
    styles = {
        "no-updates": (STYLES["noise-only"][0], "No updates"),
        "learning": (STYLES["learning"][0], "Own-score learning"),
        "shuffled": (STYLES["shuffled"][0], "Shuffled scores"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True, sharey=True)
    for ax, seed in zip(axes, report["seeds"], strict=True):
        for trial in report["trials"]:
            if trial["seed"] != seed:
                continue
            color, _ = styles[trial["mode"]]
            curve = trial["curve"]
            initial = curve[0]["mse"]
            ax.plot(
                [p["episode"] * 2.5 / 60 for p in curve],
                [100 * p["mse"] / initial for p in curve],
                color=color,
                linewidth=2,
                marker="o",
                markersize=3,
            )
        ax.set(title=f"Seed {seed}: 64 separate circuits", xlabel="Minutes of neural training")
        ax.grid(alpha=0.15)
        ax.set_ylim(0, 115)
    axes[0].set_ylabel("Validation MSE (% of initial)")
    fig.legend(
        handles=[Line2D([0], [0], color=color, label=label) for color, label in styles.values()],
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    fig.suptitle(
        "Actual V25 controller: supplied delayed-cue diagnostic\n"
        "Stronger updates, episode resets, fixed noisy validation bank; no ecology",
        fontsize=12,
    )
    save(fig, "v25-native-recurrent-probe")


def ecology():
    report = json.loads(Path("docs/results/v25-recurrent-learning.json").read_text())
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for row in report["trials"]:
        color, _ = STYLES[row["treatment"]]
        points = sorted(
            {p["tick"]: p for p in [*row["checkpoints"], row["final"]]}.values(),
            key=lambda p: p["tick"],
        )
        for ax, key, scale in zip(
            axes[:, row["seed"] - 1],
            ("population", "births", "fresh_absorbed"),
            (1, 1, 1e3),
            strict=True,
        ):
            ax.plot(
                [p["time"] / 60 for p in points],
                [p[key] / scale for p in points],
                color=color,
                linewidth=1.8,
            )
    for col in range(3):
        axes[0, col].set_title(f"Matched founder seed {col + 1}")
        axes[-1, col].set_xlabel("Simulated minutes")
        for ax in axes[:, col]:
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.15)
    for ax, label in zip(
        axes[:, 0],
        ("Living creatures", "Cumulative births", "Fresh energy absorbed (thousands)"),
        strict=True,
    ):
        ax.set_ylabel(label)
    fig.legend(
        handles=[Line2D([0], [0], color=color, label=label) for color, label in STYLES.values()],
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    fig.suptitle(
        "V25: recurrent energetic learning in the unchanged sparse habitat\n"
        "Three matched evolutionary starts; no added charge for recurrent state",
        fontsize=12,
    )
    save(fig, "v25-recurrent-learning")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    probe()
    if not args.probe_only:
        ecology()


if __name__ == "__main__":
    main()
