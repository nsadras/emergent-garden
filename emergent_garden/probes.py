"""Controlled neural history probes, separate from evolutionary fitness tests."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from .brain import controller_step, initial_state
from .config import Config
from .world import genome_hash


@torch.no_grad()
def cue_probe(config, genomes, delay, reset=False):
    """Paired left/right histories followed by exactly identical observations.

    This measures intrinsic controller history dependence and directional sign.
    It is not a navigation trial, fitness estimate, or evidence of learning.
    """
    if config.ecology_version < 3 or not math.isfinite(delay) or delay <= 0:
        raise ValueError("Cue probes require V3+ and a finite positive delay")
    g = genomes.repeat_interleave(2, dim=0)
    tau = 0.2 + 4.8 * g[:, config.brain_parameter_count + 5].sigmoid()
    state = initial_state(config, g)
    neutral = torch.zeros((len(g), config.input_size), device=g.device)
    neutral[:, -2] = 0.5
    for _ in range(2 * config.controller_hz):
        state, _ = controller_step(config, g, neutral, state, tau)
    cue = neutral.clone()
    cue[0::2, 12:16] = torch.tensor([0.2, 0.8, 0.0, 0.0], device=g.device)
    cue[1::2, 12:16] = torch.tensor([0.0, 0.0, 0.8, 0.2], device=g.device)
    for _ in range(config.controller_hz):
        state, _ = controller_step(config, g, cue, state, tau)
    steps = math.ceil(delay * config.controller_hz)
    for _ in range(steps):
        if reset:
            state["hidden"].zero_()
        state, actions = controller_step(config, g, neutral, state, tau)
    turns = (actions[:, 1] - actions[:, 0]).reshape(-1, 2)
    difference = turns[:, 1] - turns[:, 0]
    return dict(
        delay=steps / config.controller_hz,
        reset=reset,
        history_effect=difference.abs().tolist(),
        cue_aligned_turn=difference.tolist(),
    )


def probe_run(run, output, count, delays, device="cpu"):
    run, output = Path(run), Path(output)
    c = Config.load(run / "config.toml")
    if c.ecology_version < 3 or count < 1:
        raise ValueError("Probe requires V3+ and a positive genome count")
    if output.exists():
        raise ValueError(f"Refusing to overwrite probe: {output}")
    founders = torch.load(run / "founders.pt", map_location="cpu", weights_only=True)["genomes"]
    population = torch.load(run / "population.pt", map_location="cpu", weights_only=True)
    descendants = population["genomes"][population["generations"] > 0]
    report = dict(
        source=str(run),
        config=asdict(c),
        source_metadata=json.loads((run / "metadata.json").read_text()),
        groups={},
        interpretation=(
            "Identical current observations after distinct cue histories. A nonzero effect shows "
            "controller history dependence; a positive sign points toward the former cue. "
            "Neither establishes ecological usefulness, learned memory, or intelligence. "
            "For modular bodies this probes the shared neural template in isolation. "
            "V7 activation resets retain acquired synaptic offsets and eligibility traces."
        ),
    )
    for label, pool in (("founders", founders), ("descendants", descendants)):
        if not len(pool):
            raise ValueError(f"No genomes for {label}")
        rng = torch.Generator().manual_seed(982451653)
        chosen = pool[torch.randperm(len(pool), generator=rng)[:count]].to(device)
        rows = [cue_probe(c, chosen, delay, reset) for delay in delays for reset in (False, True)]
        report["groups"][label] = dict(genomes=[genome_hash(g) for g in chosen], trials=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    return report


@torch.no_grad()
def association_probe(config, genomes, rounds=3, mode="none"):
    """Counterbalanced cue/outcome histories and subsequent motor preferences.

    This is a controller assay, not navigation or ecological fitness. Both cue
    identities and total outcome amounts are matched; only their association
    differs. Opposite exposure orders and mirrored probe positions control for
    recency and pre-existing turning bias.
    """
    if (
        config.ecology_version < 6
        or rounds < 1
        or mode not in ("none", "no_feedback", "reset_h", "no_plasticity")
    ):
        raise ValueError("Association probes require V6+, positive rounds, and a supported mode")
    n = len(genomes)
    g = genomes.repeat_interleave(4, dim=0)
    device = g.device
    order = torch.tensor([0, 1, 0, 1], device=device).repeat(n)
    favorable = torch.tensor([0, 0, 1, 1], device=device).repeat(n)
    tau = 0.2 + 4.8 * g[:, config.brain_parameter_count + 5].sigmoid()
    state = initial_state(config, g)
    neutral = torch.zeros((len(g), config.input_size), device=device)
    neutral[:, config.input_names.index("energy")] = 0.5
    cue_indices = [config.input_names.index(f"identity_{label}_-135") for label in ("a", "b")]
    feedback = config.input_names.index("food_feedback")

    def step(inputs, current):
        return controller_step(config, g, inputs, current, tau, plasticity=mode != "no_plasticity")

    def train(good):
        nonlocal state
        for _ in range(rounds):
            for presentation in (0, 1):
                identity = (order + presentation) % 2
                inputs = neutral.clone()
                for label, index in enumerate(cue_indices):
                    inputs[identity == label, index : index + 4] = 0.6
                for tick in range(2 * config.controller_hz):
                    if tick == 2 * config.controller_hz - 1 and mode != "no_feedback":
                        inputs[:, feedback] = torch.where(identity == good, 0.5, 0.2)
                    state, _ = step(inputs, state)
                for _ in range(config.controller_hz):
                    state, _ = step(neutral, state)

    def preference():
        delayed = {key: value.clone() for key, value in state.items()}
        for _ in range(3 * config.controller_hz):
            delayed, _ = step(neutral, delayed)
        if mode == "reset_h":
            delayed["hidden"].zero_()
        turns = []
        for a_on_right in (False, True):
            inputs = neutral.clone()
            right = torch.tensor([0.0, 0.0, 0.8, 0.2], device=device)
            left = right.flip(0)
            for index, side in zip(cue_indices, (a_on_right, not a_on_right), strict=True):
                inputs[:, index : index + 4] = right if side else left
            branch = {key: value.clone() for key, value in delayed.items()}
            for _ in range(config.controller_hz):
                branch, actions = step(inputs, branch)
            turns.append(actions[:, 1] - actions[:, 0])
        return ((turns[1] - turns[0]) / 2).reshape(n, 2, 2).mean(2)

    initial = preference()
    train(favorable)
    acquired = preference()
    magnitude = (
        state["plastic"].abs().reshape(n, 4, -1).mean((1, 2)).tolist()
        if "plastic" in state
        else [0.0] * n
    )
    if config.ecology_version >= 8:
        from .topology import counts

        expressed = counts(config, genomes)[2].clamp_min(1)
        magnitude = (
            state["plastic"].abs().reshape(n, 4, -1).sum((1, 2)) / (4 * expressed)
        ).tolist()
    train(1 - favorable)
    reversed_preference = preference()
    return dict(
        mode=mode,
        rounds=rounds,
        initial_a_preference=initial.mean(1).tolist(),
        plastic_magnitude_after_training=magnitude,
        a_preference_after_a_high=acquired[:, 0].tolist(),
        a_preference_after_b_high=acquired[:, 1].tolist(),
        association_alignment=((acquired[:, 0] - acquired[:, 1]) / 2).tolist(),
        reversal_alignment=((reversed_preference[:, 1] - reversed_preference[:, 0]) / 2).tolist(),
    )


def association_run(run, output, count=64, rounds=3, device="cpu"):
    run, output = Path(run), Path(output)
    config = Config.load(run / "config.toml")
    if config.ecology_version < 6 or count < 1 or rounds < 1:
        raise ValueError("Association probes require V6+ and positive counts")
    if output.exists():
        raise ValueError(f"Refusing to overwrite probe: {output}")
    founders = torch.load(run / "founders.pt", map_location="cpu", weights_only=True)["genomes"]
    population = torch.load(run / "population.pt", map_location="cpu", weights_only=True)
    descendants = population["genomes"][population["generations"] > 0]
    report = dict(
        source=str(run),
        config=asdict(config),
        source_metadata=json.loads((run / "metadata.json").read_text()),
        groups={},
        interpretation="Matched cue/outcome amounts with counterbalanced exposure order and "
        "mirrored final stimuli. Positive alignment means motor preference follows the "
        "experienced high-value identity. This isolated-circuit assay does not establish "
        "successful navigation, learning benefits, or ecological fitness.",
    )
    for label, pool in (("founders", founders), ("descendants", descendants)):
        if not len(pool):
            raise ValueError(f"No genomes for {label}")
        rng = torch.Generator().manual_seed(982451653)
        chosen = pool[torch.randperm(len(pool), generator=rng)[:count]].to(device)
        report["groups"][label] = dict(
            genomes=[genome_hash(g) for g in chosen],
            trials=[
                association_probe(config, chosen, rounds, mode)
                for mode in (
                    ("none", "no_feedback", "reset_h", "no_plasticity")
                    if config.ecology_version >= 7
                    else ("none", "no_feedback", "reset_h")
                )
            ],
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    return report
