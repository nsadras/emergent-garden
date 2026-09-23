"""Plot completed V14 coordination comparisons, keeping source communities separate."""

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ("none", "no_internal", "no_body_sense", "no_coordination")


def save(fig, output, suffix):
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output.with_name(output.name + suffix).with_suffix(f".{extension}"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("communities", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lifetimes", type=Path, nargs="+")
    args = parser.parse_args()
    entries = json.loads(args.communities.read_text())
    records = {(r["metadata"]["seed"], r["metadata"]["ablation"]): r for r in entries}
    seeds = sorted({seed for seed, _ in records})
    if len(records) != len(entries) or len(records) != 4 * len(seeds):
        raise ValueError("Need all four treatments once per environment")
    fig, axes = plt.subplots(len(seeds), 4, figsize=(14, 3 * len(seeds)), squeeze=False)
    for row, seed in enumerate(seeds):
        for column, mode in enumerate(MODES):
            r = records[seed, mode]["ancestry"]
            if r["final"]["time"] != 3600:
                raise ValueError("Expected completed one-hour communities")
            points = {v["time"]: v for v in [*r["checkpoints"], r["final"]]}
            points = sorted(points.values(), key=lambda v: v["time"])
            ax = axes[row, column]
            for group, label, color in (
                (0, "V13 source ancestry", "#228b62"),
                (1, "V7 scavenger ancestry", "#c18425"),
            ):
                ax.plot(
                    [v["time"] / 60 for v in points],
                    [v["groups"][group]["population"] for v in points],
                    label=label,
                    color=color,
                )
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            if row == len(seeds) - 1:
                ax.set_xlabel("Simulated minutes")
            if column == 0:
                ax.set_ylabel(f"Environment {seed}\nPopulation")
    for ax, title in zip(
        axes[0],
        ("Position + neighbor signals", "Position only", "Neighbor signals only", "Neither"),
        strict=True,
    ):
        ax.set_title(title, fontsize=11)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("V14: private coordination within developing bodies")
    fig.text(
        0.5,
        0.025,
        "Each community starts with 96 newborns from each source; genotypes and habitats "
        "are matched within an environment.\n"
        "Mutation remains active. Colors track source ancestry, not current diet or species. "
        "Emission and cost rules remain active in every treatment.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    save(fig, args.output, "-communities")
    if not args.lifetimes:
        return
    reports = [json.loads(path.read_text()) for path in args.lifetimes]
    if not all(report["completed"] for report in reports):
        raise ValueError("Lifetime assays must be complete")
    mature = {report.get("start_mature", False) for report in reports}
    if len(mature) != 1:
        raise ValueError("Plot adult and juvenile starts separately")
    mature = mature.pop()
    fig, axes = plt.subplots(len(reports), 2, figsize=(11, 3.3 * len(reports)), squeeze=False)
    modes = ("none", "no_internal", "self_internal", "no_body_sense", "no_coordination")
    labels = ("Full", "No incoming\nsignals", "Self-signals", "No body\nposition", "Neither")
    for row, report in enumerate(reports):
        trials = report["trials"]
        for ax, key, label in zip(
            axes[row], ("acquired", "offspring"), ("Energy acquired", "Offspring"), strict=True
        ):
            for genotype in sorted({v["genotype"] for v in trials}):
                values = [
                    mean(v[key] for v in trials if v["genotype"] == genotype and v["mode"] == mode)
                    for mode in modes
                ]
                ax.plot(range(5), values, "o-", alpha=0.5, linewidth=1)
            ax.plot(
                range(5),
                [mean(v[key] for v in trials if v["mode"] == mode) for mode in modes],
                "D--",
                color="black",
                linewidth=2,
                label="Mean across sampled genotypes",
            )
            ax.set_xticks(range(5), labels)
            ax.set_ylabel(f"Source {Path(report['source']).parent.name}\n{label}")
            ax.grid(axis="y", alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "V14: fixed-genotype " + ("adult-start" if mature else "juvenile-start") + " assays"
    )
    starting = (
        "Original bodies start fully grown with fresh neural state; offspring start as juveniles."
        if mature
        else "Original bodies start as juveniles; outcomes include early death and failure to grow."
    )
    fig.text(
        0.5,
        0.02,
        "Each colored line: one two-module grazer genotype, averaged over three fresh "
        "environments, 240 seconds each.\n" + starting + "\n"
        "No mutations; clonal offspring may share the dish. Outcomes include early deaths.\n"
        "Genotypes from the same source community are related; their environment repeats "
        "are not independent evolution experiments.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    save(fig, args.output, "-lifetimes")


if __name__ == "__main__":
    main()
