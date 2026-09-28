"""Exercise the actual V25 controller on a supplied delayed-cue learning task.

This is an implementation diagnostic, not a native ecological result. It uses
larger learning rates and bounds than the ecological preset, no forgetting,
and explicit episode resets. No desired movement weights enter live creatures.
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from emergent_garden.brain import advance, controller_step, initial_state
from emergent_garden.config import Config
from emergent_garden.inheritance import brain_parts
from emergent_garden.learning import shuffled_returns
from emergent_garden.neural_timing import timing_genes
from emergent_garden.runtime import StopFlag
from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256, atomic_save
from emergent_garden.topology import mask_parts

MODES = ("no-updates", "learning", "shuffled")


def fixture(circuits):
    c = Config(
        ecology_version=25,
        hidden_size=2,
        initial_neurons=2,
        min_neurons=1,
        initial_population=circuits,
        controller_hz=10,
        physics_hz=30,
        field_hz=5,
        neural_timing_range=4,
        recurrent_noise_sigma=0.15,
        recurrent_learning_rate=0.2,
        recurrent_learning_limit=0.5,
        recurrent_trace_tau=2,
        recurrent_baseline_tau=48,
        recurrent_half_life=1e30,
        motor_learning_rate=0,
        exploration_min=0,
        exploration_max=0,
        plasticity_rate=0,
    ).validate()
    g = torch.zeros(circuits, c.parameter_count, dtype=torch.float64)
    wi, wr, _, wo, _ = brain_parts(c, g)
    wi[:, :, 0] = torch.tensor([0.8, 0.2], dtype=torch.float64)
    wr[:] = torch.tensor([[0.4, -0.2], [0.1, 0.5]], dtype=torch.float64)
    wo[:, 0] = torch.tensor([0.7, -0.4], dtype=torch.float64)
    for part in mask_parts(c, g):
        part.fill_(1)
    g[:, c.brain_parameter_count + 5] = -math.log(5)  # Developed body time constant = 1 s.
    timing_genes(c, g)[:] = torch.atanh(
        torch.tensor([0.7, 2.0], dtype=torch.float64).log() / math.log(4)
    )
    assert g.abs().max() <= c.weight_limit
    return c, g


def evaluate(c, g, plastic, cue, noise):
    count, repeats = cue.shape
    genomes = g.repeat_interleave(repeats, dim=0)
    offsets = plastic.repeat_interleave(repeats, dim=0)
    tau = 0.2 + 4.8 * genomes[:, c.brain_parameter_count + 5].sigmoid()
    inputs = genomes.new_zeros((len(genomes), c.input_size))
    hidden = genomes.new_zeros((len(genomes), c.hidden_size))
    for step, epsilon in enumerate(noise):
        inputs[:, 0] = cue.flatten() if step < 6 else 0
        hidden, logits = advance(
            c,
            genomes,
            inputs,
            hidden,
            tau,
            offsets,
            True,
            hidden_noise=c.recurrent_noise_sigma * epsilon.reshape(-1, c.hidden_size),
        )
    return (logits[:, 0].reshape(count, repeats) - 0.12 * cue).square().mean(-1)


def train(c, g, cues, noise, validation_cues, validation_noise, seed, mode, stop):
    state = initial_state(c, g)
    tau = 0.2 + 4.8 * g[:, c.brain_parameter_count + 5].sigmoid()
    elapsed = g.new_full((len(g),), 1 / c.controller_hz)
    inputs = g.new_zeros((len(g), c.input_size))
    zero_reward = g.new_zeros(len(g))
    zero_noise = g.new_zeros((len(g), c.hidden_size))
    shuffle_rng = torch.Generator().manual_seed(seed + 449)
    curve, completed_episodes = [], 0

    def record():
        errors = evaluate(c, g, state["recurrent_plastic"], validation_cues, validation_noise)
        norms = state["recurrent_plastic"].norm(dim=-1)
        curve.append(
            dict(
                episode=completed_episodes,
                mse=errors.mean().item(),
                per_circuit_mse=errors.tolist(),
                mean_offset_norm=norms.mean().item(),
                saturated_row_fraction=(norms >= 0.999 * c.recurrent_learning_limit)
                .double()
                .mean()
                .item(),
            )
        )

    record()
    for episode, (cue, perturbations) in enumerate(zip(cues, noise, strict=True)):
        if stop.requested:
            break
        for step, epsilon in enumerate(perturbations):
            inputs[:, 0] = cue if step < 6 else 0
            state, actions = controller_step(
                c,
                g,
                inputs,
                state,
                tau,
                motor_learning=False,
                recurrent_learning=mode != "no-updates",
                recurrent_noise=epsilon,
                recurrent_reward=zero_reward,
                elapsed=elapsed,
            )
        # Read the actual noise-free motor logit before supplying terminal feedback.
        reward = -0.5 * (torch.logit(actions[:, 0]) - 0.12 * cue).square()
        if mode == "shuffled":
            reward = shuffled_returns(reward, elapsed, shuffle_rng)
        inputs.zero_()
        credited, _ = controller_step(
            c,
            g,
            inputs,
            state,
            tau,
            motor_learning=False,
            recurrent_learning=mode != "no-updates",
            recurrent_noise=zero_noise,
            recurrent_reward=reward,
            elapsed=elapsed,
        )
        # Episodic task: clear activity/eligibility, retain acquired weights and baseline.
        state = initial_state(c, g)
        for key in ("recurrent_plastic", "recurrent_baseline"):
            state[key] = credited[key]
        completed_episodes = episode + 1
        if completed_episodes % 64 == 0:
            record()
    if curve[-1]["episode"] != completed_episodes:
        record()
    if mode == "no-updates":
        assert not state["recurrent_plastic"].count_nonzero()
        assert curve[-1]["per_circuit_mse"] == curve[0]["per_circuit_mse"]
    assert state["recurrent_plastic"].norm(dim=-1).max() <= c.recurrent_learning_limit + 1e-12
    return dict(
        seed=seed,
        mode=mode,
        completed=completed_episodes == len(cues),
        curve=curve,
    ), state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--circuits", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=256)
    args = parser.parse_args()
    if args.circuits < 2 or args.episodes < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct seeds, at least two circuits, and positive episodes")
    torch.set_num_threads(1)
    c, g = fixture(args.circuits)
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).read_bytes()
    (args.output / Path(__file__).name).write_bytes(script)
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    c.save(args.output / "config.toml")
    atomic_save(g, args.output / "genomes.pt")
    report = dict(
        completed=False,
        native_config=asdict(c),
        seeds=args.seeds,
        circuits=args.circuits,
        episodes=args.episodes,
        script_sha256=hashlib.sha256(script).hexdigest(),
        source_sha256=SOURCE_SHA256,
        genomes_sha256=hashlib.sha256((args.output / "genomes.pt").read_bytes()).hexdigest(),
        interpretation="Supplied two-neuron delayed-cue task using the actual native advance "
        "and controller_step. Continuous updates receive zero return during 24 noisy steps "
        "and the terminal score at a separate feedback step. Activity and eligibility reset "
        "between episodes; recurrent offsets and learned baselines persist. Effective rate "
        ".1, row limit .5, baseline 48 seconds, and negligible forgetting differ from the "
        "ecological preset. No bodies, survival, evolution, or native energetic reward are "
        "tested. Three conditions share training cues/noise and a separate fixed validation "
        "bank. Validation never trains weights. Each episode spans 2.5 seconds including "
        "the feedback step; this does not establish adaptation within typical creature lives.",
        trials=[],
    )
    stop = StopFlag()
    try:
        for seed in args.seeds:
            rng = torch.Generator().manual_seed(seed)
            cues = torch.where(
                torch.rand(args.episodes, args.circuits, generator=rng) < 0.5, -1.0, 1.0
            ).double()
            noise = torch.randn(
                args.episodes, 24, args.circuits, 2, generator=rng, dtype=torch.float64
            )
            validation_cues = torch.tensor([-1.0, 1.0], dtype=torch.float64).repeat(16)
            validation_cues = validation_cues.expand(args.circuits, -1)
            validation_noise = torch.randn(
                24,
                args.circuits,
                32,
                2,
                generator=torch.Generator().manual_seed(seed + 1009),
                dtype=torch.float64,
            )
            for mode in MODES:
                if stop.requested:
                    return
                row, state = train(
                    c, g, cues, noise, validation_cues, validation_noise, seed, mode, stop
                )
                path = args.output / f"final-{seed}-{mode}.pt"
                atomic_save(state, path)
                row["final_state_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                report["trials"].append(row)
                (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"{seed}/{mode}: episodes={row['curve'][-1]['episode']} "
                    f"MSE={row['curve'][-1]['mse']:.6f}",
                    flush=True,
                )
        report["completed"] = all(row["completed"] for row in report["trials"])
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        stop.close()


if __name__ == "__main__":
    main()
