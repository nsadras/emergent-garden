"""Isolate delayed motor credit in a constructed, counterbalanced cue task.

An externally supplied feature encodes a random left/right cue. One noisy motor
choice receives a squared-error return after a delay. Intervening features,
noise, and rewards are zero. The required cue/action mapping reverses halfway.
These circuits are diagnostic fixtures, never introduced into the ecology.
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch

from emergent_garden.config import Config
from emergent_garden.learning import motor_policy, motor_state
from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256
from emergent_garden.topology import initial_structure


def trial(config, count, seed, episodes, delay, learning):
    c = config
    genomes = torch.zeros(count, c.parameter_count)
    genomes[:, c.brain_parameter_count + c.trait_count :] = initial_structure(
        c, count, "cpu", torch.Generator().manual_seed(seed + 32452843)
    )
    state = motor_state(c, genomes)
    rng = torch.Generator().manual_seed(seed)
    elapsed = torch.full((count,), 1 / c.controller_hz)
    neutral = torch.zeros(count, c.hidden_size)
    logits = torch.zeros(count, c.output_size)
    no_noise, no_reward = torch.zeros(count, 2), torch.zeros(count)
    losses, deterministic_losses = [], []
    wait_ticks = math.ceil(delay * c.controller_hz)
    for episode in range(2 * episodes):
        cue = torch.where(torch.rand((count,), generator=rng) < 0.5, -1.0, 1.0)
        hidden = neutral.clone()
        hidden[:, 0] = cue
        state, actions = motor_policy(
            c,
            genomes,
            hidden,
            logits,
            state,
            torch.randn((count, 2), generator=rng),
            no_reward,
            elapsed,
            learning=learning,
        )
        target = (cue * (1 if episode < episodes else -1) + 1) / 2
        loss = (actions[:, 0] - target).square()
        features = torch.cat((hidden, torch.ones(count, 1)), -1)
        if c.motor_normalized:
            features /= features.norm(dim=-1, keepdim=True).clamp_min(1)
        deterministic = (state["motor_plastic"][:, 0] * features).sum(-1).sigmoid()
        deterministic_losses.append((deterministic - target).square().mean().item())
        losses.append(loss.mean().item())
        for _ in range(wait_ticks):
            state, _ = motor_policy(
                c, genomes, neutral, logits, state, no_noise, no_reward, elapsed, learning=learning
            )
        state, _ = motor_policy(
            c, genomes, neutral, logits, state, no_noise, -loss, elapsed, learning=learning
        )
    window = min(50, episodes)
    return dict(
        seed=seed,
        learning=learning,
        delay=(wait_ticks + 1) / c.controller_hz,
        trace_tau=c.motor_trace_tau,
        first_late_loss=sum(losses[episodes - window : episodes]) / window,
        reversed_early_loss=sum(losses[episodes : episodes + window]) / window,
        reversed_late_loss=sum(losses[-window:]) / window,
        first_late_deterministic_loss=sum(deterministic_losses[episodes - window : episodes])
        / window,
        reversed_late_deterministic_loss=sum(deterministic_losses[-window:]) / window,
        curve=[
            dict(episodes=start + window, mean_loss=sum(losses[start : start + window]) / window)
            for start in range(0, len(losses) - window + 1, window)
        ],
        maximum_readout_norm=state["motor_plastic"].norm(dim=-1).max().item(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v19-learning.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[701, 702, 703])
    parser.add_argument("--circuits", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--delays", type=float, nargs="+", default=[0, 5, 20])
    parser.add_argument("--traces", type=float, nargs="+", default=[2, 10])
    args = parser.parse_args()
    if args.circuits < 1 or args.episodes < 1:
        parser.error("Circuits and episodes must be positive")
    if any(not math.isfinite(d) or d < 0 for d in args.delays):
        parser.error("Delays must be finite and nonnegative")
    if any(not math.isfinite(t) or t <= 0 for t in args.traces):
        parser.error("Trace durations must be finite and positive")
    c = Config.load(args.config)
    if c.ecology_version < 12:
        parser.error("Motor learning requires V12+")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    experiment = Path(__file__).read_bytes()
    (args.output / "experiment.py").write_bytes(experiment)
    report = dict(
        config=asdict(c),
        source_sha256=SOURCE_SHA256,
        experiment_sha256=hashlib.sha256(experiment).hexdigest(),
        circuits=args.circuits,
        episodes_per_contingency=args.episodes,
        interpretation=__doc__ + " A supplied representation and isolated choices make this a "
        "mechanism check, not evidence of evolved navigation. Temporal reward baselines, "
        "readout decay, and noisy credit can affect results in addition to trace decay.",
        completed=False,
        trials=[],
    )
    for seed in args.seeds:
        for delay in args.delays:
            for trace in args.traces:
                for learning in (True, False):
                    row = trial(
                        replace(c, motor_trace_tau=trace),
                        args.circuits,
                        seed,
                        args.episodes,
                        delay,
                        learning,
                    )
                    report["trials"].append(row)
                    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                    print({key: value for key, value in row.items() if key != "curve"}, flush=True)
    report["completed"] = True
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
