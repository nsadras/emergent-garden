"""Paired pulse responses in inherited circuits, without learning or a moving body.

Both branches receive a constant energy observation. One also receives a one-
second fresh-food field pulse. We measure the resulting activity difference,
not memory usefulness or task performance. All inherited weights remain intact.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from emergent_garden.brain import advance
from emergent_garden.config import Config
from emergent_garden.neural_timing import response_times, timing_metrics
from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256, atomic_save
from emergent_garden.topology import effective_masks
from emergent_garden.world import create_world


def quantiles(values):
    return torch.quantile(
        values.double(), torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    ).tolist()


def response(config, genomes, tau):
    c = config
    nodes = effective_masks(c, genomes)[0]
    count = nodes.sum(-1)
    x = genomes.new_zeros((len(genomes), c.input_size))
    x[:, c.input_names.index("energy")] = 0.5
    pulse = x.clone()
    # Equal receptors encode a food intensity change, without prescribed direction.
    pulse[:, :4] = 0.6
    h = genomes.new_zeros((len(genomes), c.hidden_size))
    for _ in range(5 * c.controller_hz):
        h, _ = advance(c, genomes, x, h, tau)
    control = h.clone()
    shown = nodes[0].nonzero().flatten()[:6]
    curve = [
        dict(
            time=0.0,
            activity_rms=[0.0] * 3,
            motor_rms=[0.0] * 3,
            selected_activity=[0.0] * len(shown),
        )
    ]
    records = {}
    for step in range(1, 21 * c.controller_hz + 1):
        h, actions = advance(c, genomes, pulse if step <= c.controller_hz else x, h, tau)
        control, base_actions = advance(c, genomes, x, control, tau)
        rms = ((h - control).square().sum(-1) / count).sqrt()
        motors = (actions[:, :2] - base_actions[:, :2]).square().mean(-1).sqrt()
        assert torch.isfinite(h).all() and (h.abs() <= 1).all()
        curve.append(
            dict(
                time=step / c.controller_hz,
                activity_rms=quantiles(rms),
                motor_rms=quantiles(motors),
                selected_activity=(h - control)[0, shown].tolist(),
            )
        )
        if step in [int(t * c.controller_hz) for t in (1, 2, 6, 21)]:
            records[str(step / c.controller_hz - 1)] = dict(
                seconds_after_pulse=step / c.controller_hz - 1,
                mean_activity_rms=rms.mean().item(),
                mean_motor_rms=motors.mean().item(),
                activity_quantiles=quantiles(rms),
            )
    times = response_times(c, genomes, tau)
    return dict(
        timing_range=c.neural_timing_range,
        timing_statistics=timing_metrics(c, genomes, tau, nodes),
        active_time_constants=times[nodes].tolist(),
        selected_neurons=shown.tolist(),
        selected_times=times[0, shown].tolist(),
        curve=curve,
        after_pulse=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v24.toml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    c = Config.load(args.config)
    if c.ecology_version != 24 or c.sensory_contrast or not c.initial_population:
        raise ValueError("This pulse probe requires a populated V24 absolute-sensing preset")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    script = Path(__file__).read_bytes()
    (args.output / Path(__file__).name).write_bytes(script)
    trials, founders = [], {}
    for seed in args.seeds:
        w = create_world(replace(c, initial_food=0, food_rate=0), seed=seed)
        g, tau = w.founders.clone(), w.agents["memory_tau"].clone()
        founders[seed] = g.clone()
        modes = {}
        for mode, timing_range in (("homogeneous", 1.0), ("heterogeneous", c.neural_timing_range)):
            modes[mode] = response(replace(c, neural_timing_range=timing_range), g, tau)
        assert torch.equal(g, w.founders)
        trials.append(dict(seed=seed, circuits=len(g), modes=modes))
        print(f"seed {seed}: {len(g)} paired circuit responses", flush=True)
    atomic_save(founders, args.output / "founders.pt")
    report = dict(
        completed=True,
        config=asdict(c),
        source_sha256=SOURCE_SHA256,
        script_sha256=hashlib.sha256(script).hexdigest(),
        interpretation="Random inherited founder circuits, with identical genomes and body "
        "time constants across timing treatments. Five seconds of warm-up at a constant "
        "energy observation of .5, then a one-second .6 intensity pulse at all four fresh "
        "receptors, compared with a branch that gets no pulse. No movement, evolution, "
        "acquired plasticity, exploration, or task reward is supplied. Curves report the "
        "10th/50th/90th percentiles across circuits. The six displayed neurons are the "
        "first active slots of the first founder, chosen without inspecting outcomes. "
        "Persistent or varied responses alone do not establish useful memory.",
        trials=trials,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
