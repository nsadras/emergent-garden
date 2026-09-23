"""An isolated cue/action reversal check for V12's motor-learning rule.

The feature representation and task reward are supplied by this assay. These
are deliberately constructed circuits, never seeded into the ecological world.
Success verifies the local learning mechanism, not evolved learning or foraging.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from emergent_garden.config import Config
from emergent_garden.learning import motor_policy, motor_state
from emergent_garden.storage import SOURCE_SHA256
from emergent_garden.topology import initial_structure


def trial(config, count, seed, steps, learning):
    c = config
    genomes = torch.zeros(count, c.parameter_count)
    genomes[:, c.brain_parameter_count + c.trait_count :] = initial_structure(c, count, "cpu")
    state = motor_state(c, genomes)
    rng = torch.Generator().manual_seed(seed)
    reward = torch.zeros(count)
    interval = torch.full((count,), 1 / c.controller_hz)
    losses = []
    for step in range(2 * steps):
        cue = torch.where(torch.rand((count,), generator=rng) < 0.5, -1.0, 1.0)
        hidden = torch.zeros(count, c.hidden_size)
        hidden[:, 0] = cue
        state, actions = motor_policy(
            c,
            genomes,
            hidden,
            torch.zeros(count, c.output_size),
            state,
            torch.randn((count, 2), generator=rng),
            reward,
            interval,
            learning=learning,
        )
        target = (cue * (1 if step < steps else -1) + 1) / 2
        loss = (actions[:, 0] - target).square()
        reward = -loss * interval
        losses.append(loss.mean().item())
    window = min(100, steps)
    return dict(
        seed=seed,
        learning=learning,
        first_late_loss=sum(losses[steps - window : steps]) / window,
        reversed_early_loss=sum(losses[steps : steps + window]) / window,
        reversed_late_loss=sum(losses[-window:]) / window,
        curve=[
            dict(updates=start + window, mean_loss=sum(losses[start : start + window]) / window)
            for start in range(0, len(losses) - window + 1, window)
        ],
        maximum_readout_change=state["motor_plastic"].abs().max().item(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v12.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203])
    parser.add_argument("--circuits", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200, help="Updates per contingency")
    args = parser.parse_args()
    if args.circuits < 1 or args.steps < 1:
        parser.error("Circuits and updates must be positive")
    c = Config.load(args.config)
    if c.ecology_version < 12:
        parser.error("Motor learning requires V12+")
    torch.set_num_threads(1)
    report = dict(
        config=asdict(c),
        source_sha256=SOURCE_SHA256,
        experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        circuits=args.circuits,
        updates_per_contingency=args.steps,
        interpretation=__doc__,
        trials=[
            trial(c, args.circuits, seed, args.steps, learning)
            for seed in args.seeds
            for learning in (True, False)
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["trials"]:
        print({key: value for key, value in row.items() if key != "curve"})


if __name__ == "__main__":
    main()
