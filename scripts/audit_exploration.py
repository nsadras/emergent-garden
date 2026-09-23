"""Audit V21's matched exploration-persistence and motor-learning comparisons."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from audit_circuits import circuit_run, physical, same_state

from emergent_garden.config import Config
from emergent_garden.exploration import history_shapes
from emergent_garden.history import read_history
from emergent_garden.storage import load_checkpoint

GROUPS = {
    "iid-learning": ("v21-baseline", "none"),
    "iid-noise-only": ("v21-baseline", "no_motor_learning"),
    "correlated-learning": ("v21", "none"),
    "correlated-noise-only": ("v21", "no_motor_learning"),
}


def checked(path, duration, resumed=False):
    result = circuit_run(path, duration, resumed)
    world = load_checkpoint(path / "latest.pt")
    _, rows, _ = read_history(path, resumed)
    metric = world.metrics()
    assert metric["motor_noise_rms"] == rows[-1]["motor_noise_rms"], path
    result["final"]["motor_noise_rms"] = metric["motor_noise_rms"]
    lookup = {r["tick"]: r for r in rows}
    for row in result["checkpoints"]:
        row["motor_noise_rms"] = lookup[row["tick"]]["motor_noise_rms"]
    mask = world.agents["module_mask"]
    for key, shape in history_shapes(world.config.hidden_size).items():
        value = world.agents[f"module_{key}"]
        assert value.shape == (*mask.shape, *shape), (path, key)
        assert torch.isfinite(value).all(), (path, key)
        assert not value[~mask].count_nonzero(), (path, key)
    assert world.agents["module_motor_history_ready"].dtype == torch.bool
    if world.population:
        assert (
            world.agents["module_motor_plastic"].norm(dim=-1).max()
            <= world.config.motor_learning_limit + 1e-6
        ), path
    if world.ablation == "no_motor_learning":
        assert world.totals["motor_learning_changes"] == 0, path
        assert not world.agents["module_motor_plastic"].count_nonzero(), path
    return result


def neutral_parity(old, new):
    old_config, new_config = (asdict(Config.load(path / "config.toml")) for path in (old, new))
    assert new_config["motor_noise_tau"] == 0
    old_config.pop("ecology_version")
    new_config.pop("ecology_version")
    assert old_config == new_config
    _, old_rows, _ = read_history(old)
    _, new_rows, _ = read_history(new)
    without_diagnostic = [{k: v for k, v in r.items() if k != "motor_noise_rms"} for r in new_rows]
    assert physical(old_rows) == physical(without_diagnostic), (old, new)
    a = torch.load(old / "latest.pt", weights_only=True)
    b = torch.load(new / "latest.pt", weights_only=True)
    a.pop("config")
    c = b.pop("config")
    for key in history_shapes(c["hidden_size"]):
        b["agents"].pop(f"module_{key}")
    same_state(a, b)
    return dict(
        old=str(old), new=str(new), measurements=len(new_rows), exact_common_final_state=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-treatments", nargs="*", choices=GROUPS, default=())
    args = parser.parse_args()
    torch.set_num_threads(1)
    configurations = {
        group: asdict(Config.load(f"configs/{preset}.toml"))
        for group, (preset, _) in GROUPS.items()
    }
    a, b = (configurations[key].copy() for key in ("iid-learning", "correlated-learning"))
    assert a.pop("motor_noise_tau") == 0
    assert b.pop("motor_noise_tau") == 2
    assert a == b
    trials, parity, continuations = [], [], []
    for seed in (1, 2, 3):
        reference = torch.load(
            f"runs/v19-learning-pilot/seed-{seed}/founders.pt", weights_only=True
        )["genomes"]
        for group, (_, mode) in GROUPS.items():
            path = Path(f"runs/v21-{group}-pilot/seed-{seed}")
            assert asdict(Config.load(path / "config.toml")) == configurations[group], path
            founders = torch.load(path / "founders.pt", weights_only=True)["genomes"]
            assert torch.equal(founders, reference), path
            result = checked(path, 600)
            assert result["segments"][0]["metadata"]["ablation"] == mode, path
            result.update(treatment=group, seed=seed, matched_founders=True)
            trials.append(result)
            if group.startswith("iid-"):
                old = Path(f"runs/v19-{group.removeprefix('iid-')}-pilot/seed-{seed}")
                parity.append(neutral_parity(old, path))
        for group in args.long_treatments:
            result = checked(Path(f"runs/v21-{group}-long-{seed}"), 1800, True)
            result.update(treatment=group, seed=seed)
            continuations.append(result)
    report = dict(
        interpretation="A 2×2 comparison of independent versus two-second correlated motor "
        "exploration, crossed with enabled or disabled acquired motor updates. All four "
        "groups have exactly matching founders per seed, the same initial RNG streams, "
        "exploration magnitude, energetic capacity costs, and recurrent plasticity. "
        "Later genomes and observations diverge. Three starts per treatment are screening "
        "evidence. More births or persistent routes alone do not establish useful learning. "
        "Only configuration, new history tensors, and derived noise telemetry are excluded "
        "from the neutral legacy parity comparisons. Continuations are not new replicates.",
        configurations=configurations,
        trials=trials,
        neutral_v19_parity=parity,
        continuation_treatments=list(args.long_treatments),
        continuations=continuations,
    )
    output = Path("docs/results/v21-exploration.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {len(trials)} pilots and {len(continuations)} continuations: {output}")


if __name__ == "__main__":
    main()
