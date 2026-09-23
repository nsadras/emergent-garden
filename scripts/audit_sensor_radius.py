"""Audit V20's paired receptor-footprint experiments and snapshot diagnostics."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from audit_circuits import circuit_run, physical, same_state

from emergent_garden.config import Config
from emergent_garden.history import read_history
from emergent_garden.storage import load_checkpoint
from emergent_garden.world import World


def snapshot_sensitivity(path):
    source = load_checkpoint(path / "latest.pt")
    rows = []
    for scale in (1, 2, 4):
        w = World.from_state(source.state_dict())
        w.config = replace(w.config, ecology_version=20, sensor_radius_scale=scale)
        values = w.module_sensors(torch.arange(w.population))
        fresh = values[..., :4][w.agents["module_mask"]]
        mean = fresh.mean(-1, keepdim=True)
        rows.append(
            dict(
                radius_scale=scale,
                modules=len(fresh),
                fresh_mean_rms=mean.square().mean().sqrt().item(),
                fresh_direction_rms=(fresh - mean).square().mean().sqrt().item(),
                nonzero_fraction=(fresh.max(-1).values > 1e-6).float().mean().item(),
            )
        )
    return dict(
        source=str(path),
        source_time=source.time,
        source_seed=source.seed,
        interpretation="Receptor readings of the same living bodies in the same frozen fields. "
        "These paired observations isolate information availability, not successful behavior.",
        trials=rows,
    )


def main():
    torch.set_num_threads(1)
    treatments = (("v20-baseline", 1), ("v20-radius2", 2), ("v20", 4))
    trials, parity = [], []
    configs = {name: asdict(Config.load(f"configs/{name}.toml")) for name, _ in treatments}
    for seed in (1, 2, 3):
        reference = torch.load(f"runs/v19-varied-pilot/seed-{seed}/founders.pt", weights_only=True)[
            "genomes"
        ]
        for name, scale in treatments:
            path = Path(f"runs/{name}-pilot/seed-{seed}")
            assert asdict(Config.load(path / "config.toml")) == configs[name], path
            actual = torch.load(path / "founders.pt", weights_only=True)["genomes"]
            assert torch.equal(reference, actual), path
            result = circuit_run(path, 600)
            result.update(treatment=name, radius_scale=scale, seed=seed, matched_founders=True)
            trials.append(result)
        old, new = (
            Path(f"runs/v19-varied-pilot/seed-{seed}"),
            Path(f"runs/v20-baseline-pilot/seed-{seed}"),
        )
        _, old_rows, _ = read_history(old)
        _, new_rows, _ = read_history(new)
        assert physical(old_rows) == physical(new_rows), seed
        a = torch.load(old / "latest.pt", weights_only=True)
        b = torch.load(new / "latest.pt", weights_only=True)
        a.pop("config")
        b.pop("config")
        same_state(a, b)
        parity.append(dict(seed=seed, measurements=len(new_rows), exact_final_state=True))
    report = dict(
        interpretation="Three matched founder populations at receptor radii 1, 2, and 4 times "
        "each module's radius. All external fields use the wider footprint; body/gut inputs "
        "and movement, food, energy, and neural laws are unchanged. There is no extra sensory "
        "cost or inherited reach gene in this test. Three starts are screening evidence, "
        "and greater reproduction alone does not establish directional searching or learning.",
        configurations=configs,
        trials=trials,
        neutral_v19_parity=parity,
        snapshot_sensitivity=[
            snapshot_sensitivity(Path(f"runs/v19-varied-long-{seed}")) for seed in (1, 2, 3)
        ],
    )
    output = Path("docs/results/v20-sensor-radius.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {len(trials)} matched pilots and three frozen-world diagnostics: {output}")


if __name__ == "__main__":
    main()
