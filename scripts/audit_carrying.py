"""Audit the paired V18 collection × processing experiments and continuations."""

import argparse
import json
from pathlib import Path

import torch
from audit_foraging import audit_run

from emergent_garden.digestion import capacity, loads
from emergent_garden.storage import load_checkpoint


def checked(path, duration, resumed=False):
    result = audit_run(path, duration, resumed)
    w = load_checkpoint(path / "latest.pt")
    assert len(w.food_owner) == len(w.food_energy)
    assert (loads(w) <= capacity(w).double() + 1e-8).all()
    torch.testing.assert_close(w.food_credit.sum(1), w.food_energy, rtol=0, atol=1e-8)
    metric = w.metrics()
    result["final"].update(
        {
            k: metric[k]
            for k in (
                "food_collected",
                "carried_food_energy",
                "carried_food_count",
                "free_food_energy",
                "mean_gut_fullness",
            )
        }
    )
    result.update(handling_rate=w.config.handling_rate, gut_capacity=w.config.gut_capacity)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-fast-long", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    trials = []
    for seed in (1, 2, 3):
        reference = torch.load(
            f"runs/v18-contact-pilot/seed-{seed}/founders.pt", weights_only=True
        )["genomes"]
        for treatment in ("contact", "carrying", "fast-contact", "fast-carrying"):
            path = Path(f"runs/v18-{treatment}-pilot/seed-{seed}")
            founders = torch.load(path / "founders.pt", weights_only=True)["genomes"]
            assert torch.equal(founders, reference), path
            result = checked(path, 600)
            result.update(treatment=treatment, seed=seed, matched_founders=True)
            trials.append(result)
    slow = [checked(Path(f"runs/v18-carrying-long-{seed}"), 1800, True) for seed in (1, 2, 3)]
    fast = (
        [checked(Path(f"runs/v18-fast-carrying-long-{seed}"), 1800, True) for seed in (1, 2, 3)]
        if args.include_fast_long
        else []
    )
    report = dict(
        interpretation="A 2×2 comparison of carried-food capacity (0 or 200) and processing "
        "rate (60 or 300), three matched initial genotypes per treatment. Every V18 world "
        "has 42 inputs and float64 raw-food transactions. These starts therefore differ "
        "from V16/V17 founders. Continued runs are extensions, not independent replicates. "
        "Population and births do not establish directional foraging or learning.",
        trials=trials,
        carrying_continuations=slow,
        fast_carrying_continuations=fast,
        fast_continuations_included=args.include_fast_long,
    )
    output = Path("docs/results/v18-carrying.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {len(trials)} pilots and {len(slow) + len(fast)} continuations: {output}")


if __name__ == "__main__":
    main()
