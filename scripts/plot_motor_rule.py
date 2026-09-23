"""Plot the constructed V12 motor-readout assay, separately from dish evidence."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wide", type=Path)
    parser.add_argument("bounded", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    wide, bounded = (json.loads(path.read_text()) for path in (args.wide, args.bounded))
    fig, ax = plt.subplots(figsize=(10, 5))
    for report, learning, label, color in (
        (wide, True, "Per-weight bound (initial pilot)", "#a06a40"),
        (bounded, True, "Bound on total motor correction", "#228b62"),
        (bounded, False, "Learning disabled, same exploration", "#667789"),
    ):
        trials = [r for r in report["trials"] if r["learning"] == learning]
        if len(trials) != 3:
            raise ValueError("Expected three independent random streams per condition")
        values = np.array([[r["mean_loss"] for r in trial["curve"]] for trial in trials])
        times = [r["updates"] / report["config"]["controller_hz"] for r in trials[0]["curve"]]
        ax.plot(times, values.mean(0), color=color, label=label, lw=2)
        ax.fill_between(times, values.min(0), values.max(0), color=color, alpha=0.18)
    transition = bounded["updates_per_contingency"] / bounded["config"]["controller_hz"]
    ax.axvline(transition, color="#a34c62", linestyle="--", alpha=0.7)
    ax.text(transition + 2, 0.355, "Contingency reverses", color="#a34c62", fontsize=10)
    ax.set_xlabel("Assay time (seconds)")
    ax.set_ylabel("Mean squared motor error (lower is better)")
    ax.set_ylim(0.13, 0.375)
    ax.set_xlim(0, 2 * transition)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Motor readouts learn and relearn a supplied cue–action task")
    fig.text(
        0.5,
        0.03,
        "Constructed features and task rewards; unchanged genomes. "
        "64 circuits in each of three random streams.\n"
        "Lines show means; shading shows the range. "
        "This verifies the rule, not evolved learning or foraging.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=180)


if __name__ == "__main__":
    main()
