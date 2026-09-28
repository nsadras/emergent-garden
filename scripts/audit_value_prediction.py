"""Audit V22's learned-value, temporal-baseline, and shuffled-return pilots."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from audit_circuits import physical, same_state
from audit_exploration import checked as exploration_checked

from emergent_garden.config import Config
from emergent_garden.history import read_history
from emergent_garden.storage import load_checkpoint
from emergent_garden.value import value_shapes

VALUE_KEYS = (
    "mean_motor_value_norm",
    "motor_value_saturated_fraction",
    "mean_motor_predicted_return",
    "motor_value_error_rms",
)
GROUPS = {
    "baseline": ("v22-baseline", "none"),
    "value": ("v22", "none"),
    "shuffled": ("v22", "shuffled_motor_reward"),
    "raw": ("v22-raw", "none"),
    "raw-shuffled": ("v22-raw", "shuffled_motor_reward"),
}


def checked(path, duration, resumed=False):
    result = exploration_checked(path, duration, resumed)
    w = load_checkpoint(path / "latest.pt")
    _, rows, _ = read_history(path, resumed)
    metric, mask = w.metrics(), w.agents["module_mask"]
    lookup = {row["tick"]: row for row in rows}
    for key in VALUE_KEYS:
        assert metric[key] == rows[-1][key], (path, key)
        result["final"][key] = metric[key]
        for row in result["checkpoints"]:
            row[key] = lookup[row["tick"]][key]
    extra = w.config.input_size if w.config.motor_value_inputs else 0
    for key, shape in value_shapes(w.config.hidden_size, extra).items():
        value = w.agents[f"module_{key}"]
        assert value.shape == (*mask.shape, *shape), (path, key)
        assert torch.isfinite(value).all(), (path, key)
        assert not value[~mask].count_nonzero(), (path, key)
        if w.config.motor_value_rate == 0:
            assert not value.count_nonzero(), (path, key)
    if w.population:
        assert (
            w.agents["module_motor_value_weights"].norm(dim=-1).max()
            <= w.config.motor_value_limit + 1e-6
        ), path
        assert (
            w.agents["module_motor_value_prediction"].abs().max()
            <= w.config.motor_value_limit + 1e-6
        ), path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-treatments", choices=GROUPS, nargs="*", default=())
    args = parser.parse_args()
    torch.set_num_threads(1)
    configs = {
        group: asdict(Config.load(f"configs/{preset}.toml"))
        for group, (preset, _) in GROUPS.items()
    }
    a, b = configs["baseline"].copy(), configs["value"].copy()
    assert a.pop("motor_value_rate") == 0
    assert b.pop("motor_value_rate") == 0.02
    assert a == b
    a, b = configs["value"].copy(), configs["raw"].copy()
    assert a.pop("motor_value_centered") == 1
    assert b.pop("motor_value_centered") == 0
    assert a == b
    trials, parity, continuations = [], [], []
    for seed in (1, 2, 3):
        old = Path(f"runs/v21-iid-learning-pilot/seed-{seed}")
        reference = torch.load(old / "founders.pt", weights_only=True)["genomes"]
        for group, (_, mode) in GROUPS.items():
            path = Path(f"runs/v22-{group}-pilot/seed-{seed}")
            assert asdict(Config.load(path / "config.toml")) == configs[group], path
            founders = torch.load(path / "founders.pt", weights_only=True)["genomes"]
            assert torch.equal(founders, reference), path
            result = checked(path, 600)
            assert result["segments"][0]["metadata"]["ablation"] == mode, path
            result.update(treatment=group, seed=seed, matched_founders=True)
            trials.append(result)
        new = Path(f"runs/v22-baseline-pilot/seed-{seed}")
        old_config = asdict(Config.load(old / "config.toml"))
        new_config = asdict(Config.load(new / "config.toml"))
        old_config.pop("ecology_version")
        new_config.pop("ecology_version")
        assert old_config == new_config
        _, old_rows, _ = read_history(old)
        _, new_rows, _ = read_history(new)
        without_diagnostics = [
            {k: v for k, v in r.items() if k not in VALUE_KEYS} for r in new_rows
        ]
        assert physical(old_rows) == physical(without_diagnostics), seed
        a = torch.load(old / "latest.pt", weights_only=True)
        b = torch.load(new / "latest.pt", weights_only=True)
        a.pop("config")
        c = b.pop("config")
        for key in value_shapes(c["hidden_size"]):
            b["agents"].pop(f"module_{key}")
        same_state(a, b)
        parity.append(dict(seed=seed, measurements=len(new_rows), exact_common_final_state=True))
        for group in args.long_treatments:
            row = checked(Path(f"runs/v22-{group}-long-{seed}"), 1800, True)
            row.update(treatment=group, seed=seed)
            continuations.append(row)
    report = dict(
        interpretation="Three matched founder starts per treatment in the same sparse habitat. "
        "All retain independent motor exploration and the same energetic learning-capacity "
        "costs. The temporal-baseline controller is compared with a bounded learned value "
        "readout, with a second readout variant predicting direct net returns. Each predictor "
        "has a matched shuffled-return control. The centered head predicts returns minus "
        "the evolving temporal average; the direct head uses net energetic returns without "
        "that subtraction. The old temporal-baseline policy remains the shared reference. "
        "Shuffling changes actor, value, and baseline updates, not physical energy "
        "or sensory feedback. "
        "Singleton batches cannot be shuffled. Recurrent plasticity remains active. There is "
        "no extra head-specific energy charge in this experiment. These are evolutionary "
        "screening comparisons, not fixed-genotype evidence of improved credit assignment. "
        "Continuation branches are not additional replicates.",
        configurations=configs,
        trials=trials,
        neutral_v21_parity=parity,
        continuation_treatments=list(args.long_treatments),
        continuations=continuations,
    )
    output = Path("docs/results/v22-value-prediction.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {len(trials)} pilots and {len(continuations)} continuations: {output}")


if __name__ == "__main__":
    main()
