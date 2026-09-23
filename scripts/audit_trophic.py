"""Verify that a V9 traced assembly exactly reproduces its V8 reference run."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from emergent_garden.storage import load_checkpoint


def audit(reference, traced):
    old = load_checkpoint(reference / "latest.pt")
    new = load_checkpoint(traced / "latest.pt")
    assert old.config.ecology_version == 8 and new.config.ecology_version == 9
    assert old.tick == new.tick
    assert old.seed == new.seed
    assert old.totals == new.totals
    assert old.seeded_from == new.seeded_from
    for key in old.agents:
        torch.testing.assert_close(old.agents[key], new.agents[key], rtol=0, atol=0)
    for key in (
        "food_pos",
        "food_energy",
        "food_expiry",
        "food_ready",
        "food_kind",
        "food_patch",
        "patch_positions",
        "patch_phases",
        "founders",
    ):
        torch.testing.assert_close(getattr(old, key), getattr(new, key), rtol=0, atol=0)
    for first, second in zip(old.fields, new.fields, strict=True):
        torch.testing.assert_close(first.grid, second.grid, rtol=0, atol=0)
    for key in old.rng:
        assert torch.equal(old.rng[key].get_state(), new.rng[key].get_state()), key
    assert old.landscape.favorable == new.landscape.favorable
    assert old.landscape.next_tick == new.landscape.next_tick
    before = [json.loads(line) for line in (reference / "events.jsonl").read_text().splitlines()]
    after = [json.loads(line) for line in (traced / "events.jsonl").read_text().splitlines()]
    assert before == [{k: v for k, v in e.items() if k != "cause"} for e in after]
    sources = new.seeded_from["founder_sources"]
    causes = [Counter(), Counter()]
    for event in after:
        if event["event"] == "death":
            causes[sources[event["lineage"]]][event["cause"]] += 1
    metric = new.metrics()
    credit_error = (
        (new.food_credit.sum(1) - new.food_energy.double()).abs().max().item()
        if len(new.food_energy)
        else 0
    )
    balance = max(map(abs, metric["trophic_detritus_balance_error"]))
    assert credit_error < 1e-7
    assert balance < 1e-6
    differences = {}
    for key, buffer in (
        ("fresh_absorbed", "fresh_uptake"),
        ("detritus_absorbed", "detritus_uptake"),
        ("predation_absorbed", "predation_uptake"),
    ):
        differences[key] = float(getattr(new.trophic, buffer).sum()) - new.totals[key]
        # Predation's physical totals first aggregate each creature in float32;
        # the observational ledger sums pairwise meals in float64.
        assert abs(differences[key]) < 0.01
    return dict(
        reference=str(reference),
        traced=str(traced),
        physical_replay_exact=True,
        event_history_exact_except_cause=True,
        time=new.time,
        death_causes_by_source=[dict(cause) for cause in causes],
        maximum_packet_credit_error=credit_error,
        maximum_detritus_balance_error=balance,
        uptake_roundoff_differences=differences,
        metrics=metric,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--traced", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for root in (args.reference, args.traced):
        if not json.loads((root / "summary.json").read_text())["completed"]:
            raise ValueError(f"Batch is not complete: {root}")
    rows = [audit(args.reference / path.name, path) for path in sorted(args.traced.glob("seed-*"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Audited {len(rows)} exact physical replays: {args.output}")


if __name__ == "__main__":
    main()
