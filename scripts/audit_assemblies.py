"""Audit completed assembly batches or native runs, matched founders, and food credits.

Treatments with the same seed and assembly condition must share their sampled
founder genomes, source provenance, and patch layout. Different conditions may
have different founders. Energy residuals are reported against injected energy;
they are never folded into a balancing term.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from emergent_garden.storage import load_checkpoint
from emergent_garden.topology import counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    references, records = {}, []
    for root in args.runs:
        report = json.loads((root / "summary.json").read_text())
        native = "stop_reason" in report
        if native:
            if report["stop_reason"] not in ("duration", "extinction"):
                raise ValueError(f"Native run is incomplete: {root}")
            meta = json.loads((root / "metadata.json").read_text())
            trials = [dict(seed=meta["seed"], final=report)]
        else:
            if not report["completed"] or not all(row["completed"] for row in report["trials"]):
                raise ValueError(f"Assembly batch is incomplete: {root}")
            trials = report["trials"]
        for trial in trials:
            path = root if native else root / f"seed-{trial['seed']}"
            world = load_checkpoint(path / "latest.pt")
            metric = world.metrics()
            if not native:
                assert world.time >= report["duration"] or world.population == 0
            else:
                first = json.loads((path / "metrics.jsonl").read_text().splitlines()[0])
                assert first["tick"] == 0, "Native comparison requires initialization history"
            assert world.time == trial["final"]["time"]
            assert world.population == trial["final"]["population"]
            founders = torch.load(path / "founders.pt", weights_only=True)["genomes"]
            torch.testing.assert_close(founders, world.founders, rtol=0, atol=0)
            provenance = getattr(world, "seeded_from", None)
            if native:
                assert provenance is None, "This path requires native random founders"
            key = (world.seed, "native" if native else report["condition"])
            reference = references.setdefault(
                key,
                (founders, provenance, world.patch_positions, world.patch_phases),
            )
            torch.testing.assert_close(reference[0], founders, rtol=0, atol=0)
            assert reference[1] == provenance
            torch.testing.assert_close(reference[2], world.patch_positions, rtol=0, atol=0)
            torch.testing.assert_close(reference[3], world.patch_phases, rtol=0, atol=0)
            populations = [world.population]
            if not native:
                labels = torch.tensor(provenance["founder_sources"])[world.agents["lineage"]]
                populations = [(labels == group).sum().item() for group in range(2)]
                assert populations == [g["population"] for g in trial["origins"]["groups"]]
            cached_topology_exact = None
            if world.config.ecology_version >= 8:
                for name, actual in zip(
                    ("neurons", "connections", "recurrent_connections"),
                    counts(world.config, world.agents["genome"]),
                    strict=True,
                ):
                    torch.testing.assert_close(world.agents[name], actual, rtol=0, atol=0)
                cached_topology_exact = True
            development_exact = None
            if world.config.ecology_version >= 13:
                a = world.agents
                assert ((a["modules"] >= 1) & (a["modules"] <= a["target_modules"])).all()
                proposed = world.empty_agents(world.population)
                proposed["genome"] = a["genome"].clone()
                proposed["development_stage"] = a["modules"].clone()
                world.develop(proposed)
                for key in (
                    "target_modules",
                    "module_mask",
                    "module_offset",
                    "area",
                    "radius",
                    "brain_construction",
                    "brain_maintenance",
                ):
                    torch.testing.assert_close(a[key], proposed[key], rtol=0, atol=0)
                for key in (
                    "module_h",
                    "module_plastic",
                    "module_trace",
                    "module_motor_plastic",
                    "module_motor_trace",
                    "module_motor_baseline",
                    "module_actions",
                ):
                    assert a[key][~a["module_mask"]].count_nonzero() == 0
                development_exact = True
                if world.config.ecology_version >= 14:
                    assert a["module_internal"][~a["module_mask"]].count_nonzero() == 0
                    assert (a["module_internal"].abs() <= 1).all()
            packet_error = credit_error = None
            if world.config.ecology_version >= 9:
                differences = world.food_credit.sum(1) - world.food_energy.double()
                packet_error = differences.abs().max().item() if len(differences) else 0.0
                credit_error = max(map(abs, metric["trophic_detritus_balance_error"]))
                assert packet_error < 1e-7 and credit_error < 1e-6
            injected = world.initial_energy + world.totals["food_spawned"]
            error = metric["energy_balance_error"]
            assert abs(error) < max(0.01, injected * 1e-6)
            records.append(
                dict(
                    path=str(path),
                    time=world.time,
                    populations=populations,
                    matched_founders_sha256=hashlib.sha256(founders.numpy().tobytes()).hexdigest(),
                    cached_topology_exact=cached_topology_exact,
                    development_exact=development_exact,
                    energy_balance_error=error,
                    energy_injected=injected,
                    maximum_packet_credit_error=packet_error,
                    maximum_detritus_balance_error=credit_error,
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Audited {len(records)} completed runs: {args.output}")


if __name__ == "__main__":
    main()
