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
    parser.add_argument(
        "--exercise-births",
        action="store_true",
        help="Give founders reproduction energy and force structural mutation attempts",
    )
    parser.add_argument(
        "--exercise-growth",
        action="store_true",
        help="Give half the V13 founders a three-module plan and accelerate juvenile growth",
    )
    parser.add_argument(
        "--exercise-rules",
        action="store_true",
        help="Give V15 founders varied signed plasticity rules for mechanical verification",
    )
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
    if args.exercise_births and c.ecology_version >= 8:
        c = replace(c, node_mutation_probability=1.0, edge_mutation_probability=1.0)
    if args.exercise_growth:
        if c.ecology_version < 13:
            raise ValueError("Growth exercise requires V13 or later")
        c = replace(c, growth_delay=2.1, growth_reserve=10.0, module_mutation_probability=1.0)
    if args.exercise_rules and c.ecology_version < 15:
        raise ValueError("Rule exercise requires V15 or later")
    args.output.mkdir(parents=True, exist_ok=False)
    w = create_world(c, seed=7919, device=args.device)
    if args.exercise_growth:
        # Some founders mature, others juvenile: exercise births and growth.
        # These are mechanical fixtures, not ecological evaluation populations.
        w.agents["genome"][:, c.brain_parameter_count + 6] = -2
        w.agents["genome"][::2, c.brain_parameter_count + 6] = 2
        w.develop(w.agents)
    if args.exercise_rules:
        cases = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.25] * 4, [-0.25, 0.25, -0.25, 0.25], [0.0] * 4],
            device=w.device,
        )
        start = c.brain_parameter_count + 13
        w.agents["genome"][:, start : start + 4] = cases[torch.arange(w.population) % 4]
    if args.exercise_growth or args.exercise_rules:
        w.founders = w.agents["genome"].clone()
    w.step(60)
    if args.exercise_births or args.exercise_growth:
        previous = w.agents["energy"].double().sum().item()
        w.agents["energy"] = c.max_energy * w.agents["area"]
        w.initial_energy += w.agents["energy"].double().sum().item() - previous
    save_checkpoint(w, args.output / "start.pt")
    replay = load_checkpoint(args.output / "start.pt", args.device)
    w.step(180)
    replay.step(180)
    tolerance = (
        1e-5 if args.device == "cuda" and not torch.are_deterministic_algorithms_enabled() else 0.0
    )
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], replay.agents[key], rtol=0, atol=tolerance)
    for key in ("food_pos", "food_energy", "food_expiry", "food_ready", "food_kind", "food_patch"):
        torch.testing.assert_close(getattr(w, key), getattr(replay, key), rtol=0, atol=tolerance)
    for original, resumed in zip(w.fields, replay.fields, strict=True):
        torch.testing.assert_close(original.grid, resumed.grid, rtol=0, atol=tolerance)
    for key in w.rng:
        assert torch.equal(w.rng[key].get_state(), replay.rng[key].get_state()), key
    for key in w.totals:
        if tolerance:
            assert abs(w.totals[key] - replay.totals[key]) < 1e-3, key
        else:
            assert w.totals[key] == replay.totals[key], key
    if not tolerance:
        assert w.events == replay.events
    if c.ecology_version >= 9:
        torch.testing.assert_close(w.food_credit, replay.food_credit, rtol=0, atol=tolerance)
        for key, original in w.trophic.state_dict().items():
            torch.testing.assert_close(
                original, replay.trophic.state_dict()[key], rtol=0, atol=tolerance
            )
    if c.ecology_version >= 10:
        torch.testing.assert_close(w.shelter_indices, replay.shelter_indices, rtol=0, atol=0)
    metric = w.metrics()
    assert abs(metric["energy_balance_error"]) < 0.01
    if c.ecology_version >= 7:
        assert metric["mean_plastic_magnitude"] > 0
    if c.ecology_version >= 9:
        assert max(map(abs, metric["trophic_detritus_balance_error"])) < 1e-7
        torch.testing.assert_close(w.food_credit.sum(1), w.food_energy.double())
    if c.ecology_version >= 12:
        if c.exploration_max:
            assert metric["mean_motor_plastic_magnitude"] > 0
            assert metric["motor_learning_changes"] > 0
        else:
            assert metric["mean_motor_plastic_magnitude"] == 0
            assert metric["motor_learning_changes"] == 0
        if c.motor_normalized:
            assert (
                w.agents["module_motor_plastic"].norm(dim=-1).max() <= c.motor_learning_limit + 1e-6
            )
    if args.exercise_births:
        assert metric["births"] > 0
        if c.ecology_version >= 8:
            assert metric["neural_structural_births"] > 0
    if args.exercise_growth:
        assert metric["growths"] > 0
        assert metric["development_cost"] > 0
    if c.ecology_version >= 14:
        assert metric["internal_signaling_cost"] > 0
        assert metric["mean_internal_magnitude"] > 0
        assert (w.agents["module_internal"].abs() <= 1).all()
    if c.ecology_version >= 15:
        from emergent_garden.plasticity import rule_coefficients

        assert rule_coefficients(c, w.agents["genome"]).abs().sum(1).max() <= 1 + 1e-6
        assert w.agents["module_trace"].abs().max() <= 1 + 1e-6
        assert w.agents["module_plastic"].abs().max() <= c.plasticity_limit
        if args.exercise_rules:
            assert metric["mean_noncorrelation_rule_weight"] > 0
    save_checkpoint(w, args.output / "end.pt")
    report = dict(
        metadata=runtime_metadata(w),
        replay_passed=True,
        birth_exercise=args.exercise_births,
        growth_exercise=args.exercise_growth,
        rule_exercise=args.exercise_rules,
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
