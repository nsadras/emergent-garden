"""Controlled neural history probes, separate from evolutionary fitness tests."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from .brain import advance
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
    hidden = torch.zeros((len(g), config.hidden_size), device=g.device)
    neutral = torch.zeros((len(g), config.input_size), device=g.device)
    neutral[:, -2] = 0.5
    for _ in range(2 * config.controller_hz):
        hidden, _ = advance(config, g, neutral, hidden, tau)
    cue = neutral.clone()
    cue[0::2, 12:16] = torch.tensor([0.2, 0.8, 0.0, 0.0], device=g.device)
    cue[1::2, 12:16] = torch.tensor([0.0, 0.0, 0.8, 0.2], device=g.device)
    for _ in range(config.controller_hz):
        hidden, _ = advance(config, g, cue, hidden, tau)
    steps = math.ceil(delay * config.controller_hz)
    for _ in range(steps):
        if reset:
            hidden.zero_()
        hidden, actions = advance(config, g, neutral, hidden, tau)
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
            "For modular bodies this probes the shared neural template in isolation."
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
