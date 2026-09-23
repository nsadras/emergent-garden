"""Compare fixed grazer genotypes in independent physical foraging dishes.

This is a deliberately small, resource-rich assay habitat, not the full community
ecology. A founding creature can reproduce, and its offspring share the dish,
but all genetic mutation is disabled. Each reported outcome belongs to the
original creature, including its death record if it dies before the horizon.
Matched treatments retain the same genotype, initial environment, and costs.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from emergent_garden.config import Config
from emergent_garden.experiments import trial
from emergent_garden.inheritance import upgrade_genomes
from emergent_garden.runtime import StopFlag
from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256
from emergent_garden.world import genome_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v12.toml"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--genomes", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10021, 10022, 10023])
    parser.add_argument("--seconds", type=float, default=240)
    args = parser.parse_args()
    if args.genomes < 1 or not 0 < args.seconds < float("inf"):
        parser.error("Positive genome count and finite duration required")
    torch.set_num_threads(1)
    c = replace(
        Config.load(args.config),
        diameter=128.0,
        initial_population=1,
        capacity=16,
        initial_food=160,
        food_rate=12.0,
        patches=4,
        patch_radius=12.0,
        patch_capacity=1200.0,
        grid_size=32,
        smell_sigma=8.0,
        smell_cutoff=24.0,
        quality_period=60.0,
        mutation_probability=0.0,
        trait_mutation_probability=0.0,
        node_mutation_probability=0.0,
        edge_mutation_probability=0.0,
        module_mutation_probability=0.0,
    ).validate()
    if c.ecology_version < 12:
        parser.error("Motor-learning assays require V12+")
    source_config = Config.load(args.source / "config.toml")
    population_file = args.source / "population.pt"
    pool = torch.load(population_file, map_location="cpu", weights_only=True)
    genomes = upgrade_genomes(source_config, c, pool["genomes"])
    eligible = (genomes[:, c.brain_parameter_count + 2].sigmoid() > 0.65).nonzero().flatten()
    if len(eligible) < args.genomes:
        parser.error("Source does not contain the requested number of living grazers")
    rng = torch.Generator().manual_seed(982451653)
    chosen = eligible[torch.randperm(len(eligible), generator=rng)[: args.genomes]]
    selected = genomes[chosen]
    args.output.mkdir(parents=True, exist_ok=False)
    c.save(args.output / "config.toml")
    torch.save(dict(genomes=selected, source_ids=pool["ids"][chosen]), args.output / "genomes.pt")
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    experiment = Path(__file__).read_bytes()
    (args.output / "experiment.py").write_bytes(experiment)
    report = dict(
        config=asdict(c),
        source=str(args.source),
        population_sha256=hashlib.sha256(population_file.read_bytes()).hexdigest(),
        source_sha256=SOURCE_SHA256,
        experiment_sha256=hashlib.sha256(experiment).hexdigest(),
        source_ids=pool["ids"][chosen].tolist(),
        genotype_hashes=[genome_hash(g) for g in selected],
        requested_duration=args.seconds,
        environment_seeds=args.seeds,
        completed=False,
        interpretation=__doc__ + " All genotypes come from one evolved source community; "
        "environmental repeats do not make them independent evolutionary replicates.",
        trials=[],
    )
    stop = StopFlag()
    try:
        with (args.output / "trials.jsonl").open("w") as stream:
            for index, genome in enumerate(selected):
                for seed in args.seeds:
                    for mode in ("none", "no_motor_learning", "no_exploration"):
                        if stop.requested:
                            return
                        result = trial(c, genome, seed, args.seconds, "cpu", mode, stop)
                        row = dict(genotype=index, seed=seed, mode=mode, **result)
                        report["trials"].append(row)
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        (args.output / "summary.json").write_text(
                            json.dumps(report, indent=2) + "\n"
                        )
                    print(f"Completed genotype {index}, environment {seed}", flush=True)
        report["completed"] = True
    except InterruptedError:
        pass
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        stop.close()


if __name__ == "__main__":
    main()
