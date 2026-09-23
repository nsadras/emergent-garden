"""Exercise checkpoint replay and energy conservation on CPU or CUDA.

This deliberately small world is a correctness smoke check, not a throughput
benchmark or an ecology trial. It accelerates quality reversals when available.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from emergent_garden.config import Config
from emergent_garden.storage import load_checkpoint, runtime_metadata, save_checkpoint
from emergent_garden.world import create_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    c = replace(
        Config.load(args.config),
        diameter=128.0,
        initial_population=8,
        capacity=32,
        initial_food=40,
        patch_radius=12.0,
        grid_size=32,
        smell_sigma=8.0,
        smell_cutoff=24.0,
        quality_period=0.2,
    ).validate()
    if c.ecology_version < 1:
        raise ValueError("This check requires an ecology preset (V1 or later)")
    args.output.mkdir(parents=True, exist_ok=False)
    w = create_world(c, seed=7919, device=args.device)
    w.step(60)
    save_checkpoint(w, args.output / "start.pt")
    replay = load_checkpoint(args.output / "start.pt", args.device)
    w.step(180)
    replay.step(180)
    tolerance = 1e-5 if args.device == "cuda" else 0.0
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], replay.agents[key], rtol=0, atol=tolerance)
    for key in ("food_pos", "food_energy", "food_expiry", "food_ready", "food_kind", "food_patch"):
        torch.testing.assert_close(getattr(w, key), getattr(replay, key), rtol=0, atol=tolerance)
    for original, resumed in zip(w.fields, replay.fields, strict=True):
        torch.testing.assert_close(original.grid, resumed.grid, rtol=0, atol=tolerance)
    for key in w.rng:
        assert torch.equal(w.rng[key].get_state(), replay.rng[key].get_state()), key
    for key in w.totals:
        assert abs(w.totals[key] - replay.totals[key]) < 1e-3, key
    metric = w.metrics()
    assert abs(metric["energy_balance_error"]) < 0.01
    if c.ecology_version >= 7:
        assert metric["mean_plastic_magnitude"] > 0
    save_checkpoint(w, args.output / "end.pt")
    report = dict(
        metadata=runtime_metadata(w),
        replay_passed=True,
        tensor_absolute_tolerance=tolerance,
        quality_reversals=sum(e["event"] == "quality_reversal" for e in w.events),
        metrics=metric,
    )
    if args.device == "cuda":
        report["allocated_peak_bytes"] = torch.cuda.max_memory_allocated()
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
