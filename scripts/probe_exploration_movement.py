"""Compare inherited circuits driving identical bodies in an empty plane.

This diagnostic holds fresh-food readings at 0.4 and energy at 0.5, with all
other observations zero. Each circuit drives one standardized module: speed
is max_speed times mean motor output, and angular speed is max_turn_degrees
times right-minus-left output. There are no walls, contacts, costs, deaths,
or reproduction. Both forms of acquired synaptic change are disabled.
Consequently this measures route changes from exploration, not food seeking
or an ecological benefit. The full-world pilots provide the separate test.
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch

from emergent_garden.brain import controller_step, initial_state
from emergent_garden.config import Config
from emergent_garden.senses import probe_features
from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256, atomic_save
from emergent_garden.world import genome_hash


@torch.no_grad()
def trial(config, genomes, seconds, noise_seed, mode):
    c = replace(config, motor_noise_tau=2.0 if mode == "correlated" else 0.0)
    n = len(genomes)
    inputs = genomes.new_zeros((n, c.input_size))
    inputs[:, c.input_names.index("energy")] = 0.5
    inputs[:, :4] = probe_features(c, genomes.new_full((n, 4), 0.4))
    tau = 0.2 + 4.8 * genomes[:, c.brain_parameter_count + 5].sigmoid()
    state = initial_state(c, genomes)
    for _ in range(25 * c.controller_hz):
        state, _ = controller_step(
            c, genomes, inputs, state, tau, plasticity=False, exploration_enabled=False
        )
    rng = torch.Generator().manual_seed(noise_seed)
    position = torch.zeros(n, 2, dtype=torch.float64)
    heading, distance, absolute_turn = (torch.zeros(n, dtype=torch.float64) for _ in range(3))
    paths, perturbations = [position[:6].clone()], []
    steps = seconds * c.controller_hz
    for tick in range(steps):
        state, actions = controller_step(
            c,
            genomes,
            inputs,
            state,
            tau,
            plasticity=False,
            noise=torch.randn(n, 2, generator=rng),
            exploration_enabled=mode != "none",
        )
        speed = c.max_speed * actions[:, :2].double().mean(-1)
        turn = math.radians(c.max_turn_degrees) * (actions[:, 1] - actions[:, 0]).double()
        for _ in range(c.physics_hz // c.controller_hz):
            middle = heading + turn * c.dt / 2
            position += speed[:, None] * torch.stack((middle.cos(), middle.sin()), -1) * c.dt
            heading += turn * c.dt
            distance += speed * c.dt
            absolute_turn += turn.abs() * c.dt
        perturbations.append(state["motor_applied_noise"].clone())
        if (tick + 1) % c.controller_hz == 0:
            paths.append(position[:6].clone())
    displacement = position.norm(dim=-1)
    assert (displacement <= distance + 1e-7).all()
    assert all(torch.isfinite(value).all() for value in state.values())
    perturbation = torch.stack(perturbations).double()
    centered = perturbation - perturbation.mean(0, keepdim=True)
    lag = c.controller_hz  # One simulated second, without mixing different circuits.
    denominator = (centered[:-lag].square().sum() * centered[lag:].square().sum()).sqrt()
    return dict(
        mode=mode,
        noise_tau=c.motor_noise_tau,
        noise_seed=noise_seed,
        seconds=seconds,
        displacement=displacement.tolist(),
        path_length=distance.tolist(),
        displacement_fraction=(displacement / distance).tolist(),
        net_rotations=(heading / (2 * math.pi)).tolist(),
        total_rotations=(absolute_turn / (2 * math.pi)).tolist(),
        noise_rms=perturbation.square().mean().sqrt().item(),
        noise_one_second_correlation=(centered[:-lag] * centered[lag:]).sum().item()
        / denominator.item()
        if denominator > 0
        else None,
        paths=torch.stack(paths).transpose(0, 1).tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--genomes", type=int, default=64)
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args()
    if args.genomes < 1 or args.seconds <= 1:
        parser.error("Use at least one genome and more than one simulated second")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    experiment = Path(__file__).read_bytes()
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    (args.output / "experiment.py").write_bytes(experiment)
    groups, selected = [], {}
    for index, source in enumerate(args.sources):
        c = Config.load(source / "config.toml")
        if c.ecology_version < 21:
            parser.error("This diagnostic requires V21 or later")
        pool = torch.load(source / "founders.pt", map_location="cpu", weights_only=True)["genomes"]
        rng = torch.Generator().manual_seed(982451653)
        genomes = pool[torch.randperm(len(pool), generator=rng)[: args.genomes]]
        selected[str(source)] = genomes
        group = dict(
            source=str(source),
            source_metadata=json.loads((source / "metadata.json").read_text()),
            config=asdict(c),
            genome_hashes=[genome_hash(g) for g in genomes],
            trials=[
                trial(c, genomes, args.seconds, 901 + index, mode)
                for mode in ("none", "iid", "correlated")
            ],
        )
        groups.append(group)
        print(source, flush=True)
    atomic_save(selected, args.output / "genomes.pt")
    report = dict(
        interpretation=__doc__,
        source_sha256=SOURCE_SHA256,
        experiment_sha256=hashlib.sha256(experiment).hexdigest(),
        groups=groups,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
