"""Plot completed V12 community and paired lifetime outcomes without pooling replicates."""

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("communities", type=Path)
    parser.add_argument("lifetimes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = {r["path"]: r for r in json.loads(args.communities.read_text())}
    report = json.loads(args.lifetimes.read_text())
    assert report["completed"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    for row, seed in enumerate((201, 202, 203)):
        for col, treatment in enumerate(
            ("v12-bounded-mixed", "v12-no-motor-mixed", "v12-no-exploration")
        ):
            record = records[f"runs/{treatment}/seed-{seed}"]["ancestry"]
            points = {r["time"]: r for r in [*record["checkpoints"], record["final"]]}
            points = sorted(points.values(), key=lambda r: r["time"])
            ax = axes[row, col]
            for group, label, color in (
                (0, "Grazer ancestry", "#228b62"),
                (1, "Scavenger ancestry", "#c18425"),
            ):
                ax.plot(
                    [r["time"] / 60 for r in points],
                    [r["groups"][group]["population"] for r in points],
                    color=color,
                    label=label,
                )
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 110)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(f"Environment {seed}\nPopulation")
            if row == 2:
                ax.set_xlabel("Simulated minutes")
    for ax, title in zip(
        axes[0], ("Motor learning + exploration", "Exploration only", "Neither"), strict=True
    ):
        ax.set_title(title, fontsize=11)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("V12: motor adaptation did not consistently improve community outcomes")
    fig.text(
        0.5,
        0.025,
        "Matched founding genomes and environments; genetic mutation remains active.\n"
        "Ancestry is not species identity. Extinct trials end early; costs are matched.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            args.output.with_name(args.output.name + "-communities").with_suffix(f".{suffix}"),
            dpi=180,
        )
    plt.close(fig)

    trials = report["trials"]
    modes = ("none", "no_motor_learning", "no_exploration")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, metric, label in zip(
        axes, ("acquired", "offspring"), ("Energy acquired", "Offspring produced"), strict=True
    ):
        for genotype in sorted({r["genotype"] for r in trials}):
            values = [
                mean(r[metric] for r in trials if r["genotype"] == genotype and r["mode"] == mode)
                for mode in modes
            ]
            ax.plot(range(3), values, "o-", alpha=0.55, linewidth=1)
        ax.plot(
            range(3),
            [mean(r[metric] for r in trials if r["mode"] == mode) for mode in modes],
            "D--",
            color="black",
            linewidth=2,
            label="Mean of eight genotypes",
        )
        ax.set_xticks(range(3), ("Learning +\nexploration", "Exploration\nonly", "Neither"))
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("V12: matched physical lifetime assays")
    fig.text(
        0.5,
        0.02,
        "Each colored line: one genotype, averaged across three matched environments, 240 s each.\n"
        "No mutations; clonal offspring may share the dish. "
        "All eight genotypes come from one community.\n"
        "Outcomes belong to the original individual, including individuals that died early.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    for suffix in ("png", "svg"):
        fig.savefig(
            args.output.with_name(args.output.name + "-lifetimes").with_suffix(f".{suffix}"),
            dpi=180,
        )


if __name__ == "__main__":
    main()
