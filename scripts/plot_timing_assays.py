"""Show every matched timing/feedback contrast from the completed V24 assay audit."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    report = json.loads(Path("docs/results/v24-timing-assays.json").read_text())
    assert report["completed"] and len(report["trials"]) == 30
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "v24-timing-assays",
        }
    )
    labels = {
        "mean-tau": "Uniform timing\n(mean preserved)",
        "permuted": "Reassigned timing",
        "no-motor-learning": "Motor updates disabled",
        "shuffled-return": "Motor feedback shuffled",
    }
    colors = {1: "#207b91", 2: "#b36025", 3: "#744c91"}
    markers = {901: "o", 902: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True, constrained_layout=True)
    for ax, key, title in zip(
        axes,
        ("births", "fresh_absorbed"),
        ("Cumulative births", "Fresh-food absorption"),
        strict=True,
    ):
        for row in report["contrasts"]:
            effect = row["outcomes"][key]["relative_to_control"]
            assert effect is not None, "A zero reference needs an absolute-effect plot"
            index = list(labels).index(row["control"])
            pair = (row["source_seed"] - 1) * 2 + (row["environment"] - 901)
            y = index + (pair - 2.5) * 0.065
            ax.scatter(
                100 * effect,
                y,
                color=colors[row["source_seed"]],
                marker=markers[row["environment"]],
                s=48,
                zorder=3,
            )
        ax.axvline(0, color="#666666", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Original timing and learning: change vs control (%)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_yticks(range(len(labels)), list(labels.values()))
    axes[0].invert_yaxis()
    handles = [
        Line2D([0], [0], color=color, marker="s", linestyle="", label=f"Source {seed}")
        for seed, color in colors.items()
    ] + [
        Line2D([0], [0], color="#555555", marker=marker, linestyle="", label=f"World {env}")
        for env, marker in markers.items()
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=5, frameon=False)
    fig.suptitle(
        "V24: same evolved genotypes, fresh experience, frozen mutation\n"
        "All six paired community comparisons per control; positive favors the native treatment",
        fontsize=13,
    )
    fig.savefig("docs/v24-timing-assays.png", dpi=150)
    path = Path("docs/v24-timing-assays.svg")
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
