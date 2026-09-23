"""Audit paired V14 lifetime outcomes and summarize within each source community.

Related genotypes and repeated test environments are not independent evolution
replicates. These descriptive comparisons measure performance in the specified
assay habitat; removing a channel can perturb dynamics as well as information.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch

from emergent_garden.world import genome_hash


def summarize(path):
    report = json.loads((path / "summary.json").read_text())
    if not report["completed"]:
        raise ValueError(f"Incomplete lifetime assay: {path}")
    for key in (
        "mutation_probability",
        "trait_mutation_probability",
        "node_mutation_probability",
        "edge_mutation_probability",
        "module_mutation_probability",
    ):
        assert report["config"][key] == 0, key
    saved = torch.load(path / "genomes.pt", weights_only=True)
    assert [genome_hash(g) for g in saved["genomes"]] == report["genotype_hashes"]
    assert len(set(report["genotype_hashes"])) == len(saved["genomes"])
    assert saved["source_ids"].tolist() == report["source_ids"]
    assert (
        hashlib.sha256((path / "experiment.py").read_bytes()).hexdigest()
        == report["experiment_sha256"]
    )
    assert (
        hashlib.sha256(Path(report["source"]).read_bytes()).hexdigest()
        == report["checkpoint_sha256"]
    )
    rows = report["trials"]
    assert rows == [json.loads(line) for line in (path / "trials.jsonl").read_text().splitlines()]
    paired = {(r["genotype"], r["seed"], r["mode"]): r for r in rows}
    genotypes = range(len(saved["genomes"]))
    expected = {
        (g, seed, mode)
        for g in genotypes
        for seed in report["environment_seeds"]
        for mode in report["modes"]
    }
    assert set(paired) == expected and len(paired) == len(rows)
    for r in rows:
        assert r["target_modules"] == 2 and r["modules"] in (1, 2)
        if report.get("start_mature", False):
            assert r["initial_modules"] == r["modules"] == 2
            assert r["first_growth_time"] is None
        else:
            assert r.get("initial_modules", 1) == 1
            assert (r["first_growth_time"] is not None) == (r["modules"] == 2)
        assert (
            abs(
                r["acquired"]
                - sum(r[k] for k in ("fresh_acquired", "detritus_acquired", "meat_acquired"))
            )
            < 0.02
        )
        if r["survived"]:
            assert abs(r["age"] - report["requested_duration"]) < 0.1
        assert (
            r["first_growth_time"] == paired[r["genotype"], r["seed"], "none"]["first_growth_time"]
        ), "Treatments must agree before the first module addition enables the interface"
    keys = (
        "acquired",
        "offspring",
        "age",
        "fresh_acquired",
        "detritus_acquired",
        "meat_acquired",
        "internal_spent",
    )
    outcomes = {}
    for mode in report["modes"]:
        group = [r for r in rows if r["mode"] == mode]
        outcomes[mode] = dict(
            trials=len(group),
            survived=sum(r["survived"] for r in group),
            grown=sum(r["modules"] == 2 for r in group),
            grew_during_trial=sum(r["first_growth_time"] is not None for r in group),
            means={key: mean(r[key] for r in group) for key in keys},
        )
    differences = {}
    for mode in report["modes"]:
        if mode == "none":
            continue
        per_genotype = [
            dict(
                genotype=g,
                **{
                    key: mean(
                        paired[g, seed, "none"][key] - paired[g, seed, mode][key]
                        for seed in report["environment_seeds"]
                    )
                    for key in ("acquired", "offspring", "survived")
                },
            )
            for g in genotypes
        ]
        differences[mode] = dict(
            means={
                key: mean(r[key] for r in per_genotype)
                for key in ("acquired", "offspring", "survived")
            },
            genotype_mean_differences=per_genotype,
            genotypes_with_more_intake=sum(r["acquired"] > 0 for r in per_genotype),
            genotypes_with_more_offspring=sum(r["offspring"] > 0 for r in per_genotype),
        )
    return dict(
        path=str(path),
        source=report["source"],
        completed_trials=len(rows),
        genotypes=len(saved["genomes"]),
        environment_seeds=report["environment_seeds"],
        start_mature=report.get("start_mature", False),
        paired_first_growth_times_exact=True,
        immutable_source_and_selection_verified=True,
        outcomes=outcomes,
        full_minus_control=differences,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assays", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = [summarize(path) for path in args.assays]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(interpretation=__doc__, communities=records), indent=2) + "\n"
    )
    print(f"Audited {sum(r['completed_trials'] for r in records)} paired physical lifetimes")


if __name__ == "__main__":
    main()
