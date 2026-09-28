"""Plot the separate delayed-cue recurrent-learning diagnostic."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/emergent-garden-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    data = json.loads(Path("docs/results/recurrent-adaptation.json").read_text())
    assert data["completed"]
    styles = {
        "no-updates": ("#666666", "-", "No updates"),
        "full-trace": ("#207b91", "-", "Full trace"),
        "shuffled-full": ("#207b91", "--", "Full trace, shuffled scores"),
        "decayed-trace": ("#b36025", "-", "Two-second trace"),
        "shuffled-decayed": ("#b36025", "--", "Two-second trace, shuffled scores"),
    }
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "recurrent-adaptation",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    for ax, trial in zip(axes, data["trials"], strict=True):
        curve = trial["curve"]
        for mode, (color, style, label) in styles.items():
            ax.plot(
                [row["episode"] for row in curve],
                [row["modes"][mode]["mse"] for row in curve],
                color=color,
                linestyle=style,
                label=label,
                linewidth=1.8,
            )
        ax.set(title=f"Random stream {trial['seed']}", xlabel="Training episodes", ylim=(0, None))
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Mean squared error on separate noisy trials")
    fig.legend(
        handles=axes[0].get_legend_handles_labels()[0],
        labels=axes[0].get_legend_handles_labels()[1],
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Recurrent adaptation in a supplied delayed-cue task\n"
        "64 circuits per condition and stream; learned baseline; no bodies or ecology",
        fontsize=13,
    )
    fig.savefig("docs/recurrent-adaptation.png", dpi=150)
    path = Path("docs/recurrent-adaptation.svg")
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
