"""Extract dietary pools for fixed-genotype food-web assays of one community.

This selects existing dietary phenotypes, not species. Founder ancestry is
retained in provenance. Fresh assays reset age, anatomy, and acquired state;
their shared configuration disables every genetic mutation pathway.
"""

import argparse
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import torch

from emergent_garden.storage import load_checkpoint
from emergent_garden.world import genome_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = args.source / "latest.pt" if args.source.is_dir() else args.source
    snapshot = path.read_bytes()
    world = load_checkpoint(io.BytesIO(snapshot))
    a, c = world.agents, world.config
    if c.ecology_version < 13:
        parser.error("Use a V13+ community with juvenile development")
    masks = {"grazers": a["diet"] > 0.65, "scavengers": a["diet"] < 0.35}
    if not all(mask.any() for mask in masks.values()):
        parser.error("Both dietary pools must be present")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = replace(
        c,
        mutation_probability=0.0,
        trait_mutation_probability=0.0,
        node_mutation_probability=0.0,
        edge_mutation_probability=0.0,
        module_mutation_probability=0.0,
        timing_mutation_probability=0.0,
    )
    frozen.save(args.output / "frozen.toml")
    report = dict(
        source=str(path),
        source_time=world.time,
        source_seed=world.seed,
        source_ablation=world.ablation,
        checkpoint_sha256=hashlib.sha256(snapshot).hexdigest(),
        interpretation=__doc__,
        selection="All living bodies with diet > 0.65 or < 0.35, respectively. "
        "Selection retains genotype frequencies, including repeated clones.",
        excluded_generalists=int(((a["diet"] >= 0.35) & (a["diet"] <= 0.65)).sum()),
        pools={},
    )
    for name, mask in masks.items():
        output = args.output / name
        output.mkdir()
        c.save(output / "config.toml")
        selected = {
            key: a[value][mask].cpu().clone()
            for key, value in (
                ("genomes", "genome"),
                ("generations", "generation"),
                ("ids", "id"),
                ("acquired", "acquired"),
            )
        }
        torch.save(dict(version=1, **selected), output / "population.pt")
        lineages = a["lineage"][mask]
        report["pools"][name] = dict(
            population=int(mask.sum()),
            ids=selected["ids"].tolist(),
            generations=selected["generations"].tolist(),
            genotype_hashes=[genome_hash(g) for g in selected["genomes"]],
            lineages=lineages.tolist(),
            diets=a["diet"][mask].tolist(),
            expressed_modules=a["modules"][mask].tolist(),
            target_modules=a["target_modules"][mask].tolist(),
            founder_diets={
                str(i): float(world.founders[i, c.brain_parameter_count + 2].sigmoid())
                for i in lineages.unique().tolist()
            },
            population_sha256=hashlib.sha256((output / "population.pt").read_bytes()).hexdigest(),
        )
    experiment = Path(__file__).read_bytes()
    (args.output / "experiment.py").write_bytes(experiment)
    report["experiment_sha256"] = hashlib.sha256(experiment).hexdigest()
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Preserved pools: {[(k, v['population']) for k, v in report['pools'].items()]}")


if __name__ == "__main__":
    main()
