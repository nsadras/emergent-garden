"""Audit matched V19 founder, mutation, and motor-learning experiments."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from audit_carrying import checked

from emergent_garden.config import Config
from emergent_garden.history import PERFORMANCE_KEYS, read_history
from emergent_garden.storage import load_checkpoint

PRESETS = {
    "baseline": "v19-baseline",
    "varied": "v19",
    "mutation": "v19-mutation",
    "learning": "v19-learning",
    "noise-only": "v19-learning",
}
NEURAL_KEYS = (
    "mean_neurons",
    "mean_connections",
    "neuron_histogram",
    "mean_plastic_magnitude",
    "mean_motor_plastic_magnitude",
    "motor_rows_saturated_fraction",
    "mean_motor_learning_rate",
    "mean_exploration_sigma",
    "motor_learning_changes",
    "motor_learning_cost",
    "mean_encoded_learning_rule",
)


def circuit_run(path, duration, resumed=False):
    result = checked(path, duration, resumed)
    _, rows, _ = read_history(path, resumed)
    measured = load_checkpoint(path / "latest.pt").metrics()
    for key in NEURAL_KEYS:
        assert measured[key] == rows[-1][key], (path, key)
        result["final"][key] = measured[key]
    by_tick = {r["tick"]: r for r in rows}
    for row in result["checkpoints"]:
        row.update({key: by_tick[row["tick"]][key] for key in NEURAL_KEYS})
    return result


def physical(rows):
    return [
        {k: v for k, v in row.items() if k not in PERFORMANCE_KEYS | {"ecology_version"}}
        for row in rows
    ]


def same_state(left, right, location="root"):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), location
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), location
        for key in left:
            same_state(left[key], right[key], f"{location}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), location
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            same_state(a, b, f"{location}[{index}]")
    else:
        assert left == right, location


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long-treatments",
        nargs="*",
        choices=("varied", "mutation", "learning", "noise-only"),
        default=(),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    trials, parity, continuations = [], [], []
    configs = {
        group: asdict(Config.load(f"configs/{preset}.toml")) for group, preset in PRESETS.items()
    }
    for seed in (1, 2, 3):
        matched = None
        for group, config in configs.items():
            path = Path(f"runs/v19-{group}-pilot/seed-{seed}")
            assert asdict(Config.load(path / "config.toml")) == config, path
            founders = torch.load(path / "founders.pt", weights_only=True)["genomes"]
            if group == "varied":
                matched = founders
            elif group != "baseline":
                assert torch.equal(founders, matched), path
            result = circuit_run(path, 600)
            mode = "no_motor_learning" if group == "noise-only" else "none"
            assert result["segments"][0]["metadata"]["ablation"] == mode, path
            if mode == "no_motor_learning":
                assert result["final"]["motor_learning_changes"] == 0, path
            result.update(treatment=group, seed=seed, matched_varied_founders=group != "baseline")
            trials.append(result)
        old = Path(f"runs/v18-fast-carrying-pilot/seed-{seed}")
        new = Path(f"runs/v19-baseline-pilot/seed-{seed}")
        _, a, _ = read_history(old)
        _, b, _ = read_history(new)
        assert physical(a) == physical(b), seed
        a = torch.load(old / "latest.pt", weights_only=True)
        b = torch.load(new / "latest.pt", weights_only=True)
        a.pop("config")
        b.pop("config")
        b["rng"].pop("founder_structure")
        same_state(a, b)
        parity.append(
            dict(
                seed=seed, measurements=len(physical(read_history(new)[1])), exact_final_state=True
            )
        )
        if args.long_treatments:
            for group in args.long_treatments:
                result = circuit_run(Path(f"runs/v19-{group}-long-{seed}"), 1800, True)
                result.update(treatment=group, seed=seed)
                continuations.append(result)
    report = dict(
        interpretation="Three starts per treatment are screening evidence. Varied, mutation, "
        "learning, and noise-only groups have exactly matched inherited founders within each "
        "seed. The baseline changes founder architecture and its fan-in weight scaling. "
        "Learning and its control have identical exploration settings; the control suppresses "
        "motor updates while preserving recurrent plasticity and capacity costs. Continued "
        "runs are extensions, not independent replicates. Births and persistence alone do not "
        "establish useful within-lifetime learning or directional searching.",
        configurations=configs,
        trials=trials,
        neutral_v18_parity=parity,
        continuation_treatments=list(args.long_treatments),
        continuations=continuations,
    )
    output = Path("docs/results/v19-circuits.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {len(trials)} pilots and {len(continuations)} continuations: {output}")


if __name__ == "__main__":
    main()
