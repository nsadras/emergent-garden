"""Compare fixed two-module grazer genotypes in fresh physical habitats.

The founding body normally starts as a juvenile with fresh neural and signal state.
Clonal offspring may share its dish. All mutation is disabled. Outcomes track
that original body, including death before the horizon; no outcome is restricted
to survivors. Comparisons measure use of the interface in this assay habitat,
not independent evolution of communication or general intelligence.
"""

import argparse
import hashlib
import io
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from emergent_garden.experiments import trial
from emergent_garden.runtime import StopFlag
from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256, load_checkpoint
from emergent_garden.world import genome_hash

MODES = ("none", "no_internal", "self_internal", "no_body_sense", "no_coordination")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--genomes", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10121, 10122, 10123])
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument(
        "--start-mature", action="store_true", help="Test an initially fully grown body"
    )
    args = parser.parse_args()
    if args.genomes < 1 or not 0 < args.seconds < float("inf"):
        parser.error("Positive genome count and finite duration required")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes):
        parser.error("Seeds and interventions must be unique")
    torch.set_num_threads(1)
    path = args.source / "latest.pt" if args.source.is_dir() else args.source
    snapshot = path.read_bytes()
    source = load_checkpoint(io.BytesIO(snapshot))
    if source.config.ecology_version < 14 or source.controller != "neural":
        parser.error("Need a V14+ neural population")
    if source.ablation != "none":
        parser.error("Use a population evolved with the full interface")
    c = replace(
        source.config,
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
    a = source.agents
    eligible = ((a["target_modules"] == 2) & (a["diet"] > 0.65)).nonzero().flatten().tolist()
    unique = {}
    for index in eligible:
        unique.setdefault(genome_hash(a["genome"][index]), index)
    candidates = torch.tensor(list(unique.values()), dtype=torch.long)
    if len(candidates) < args.genomes:
        parser.error("Not enough distinct living two-module grazer genotypes")
    rng = torch.Generator().manual_seed(982451653)
    chosen = candidates[torch.randperm(len(candidates), generator=rng)[: args.genomes]]
    selected = a["genome"][chosen].clone()
    args.output.mkdir(parents=True, exist_ok=False)
    c.save(args.output / "config.toml")
    torch.save(dict(genomes=selected, source_ids=a["id"][chosen]), args.output / "genomes.pt")
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    experiment = Path(__file__).read_bytes()
    (args.output / "experiment.py").write_bytes(experiment)
    report = dict(
        config=asdict(c),
        source=str(path),
        source_time=source.time,
        checkpoint_sha256=hashlib.sha256(snapshot).hexdigest(),
        source_sha256=SOURCE_SHA256,
        experiment_sha256=hashlib.sha256(experiment).hexdigest(),
        source_ids=a["id"][chosen].tolist(),
        source_generations=a["generation"][chosen].tolist(),
        genotype_hashes=[genome_hash(g) for g in selected],
        requested_duration=args.seconds,
        environment_seeds=args.seeds,
        modes=args.modes,
        start_mature=args.start_mature,
        selection="Uniform without replacement among distinct living genotypes encoding "
        "exactly two modules and fresh-food allocation above 0.65; fixed selection seed 982451653.",
        interpretation=__doc__ + " All genotypes share a source community; environmental repeats "
        "and related genotypes are not independent evolutionary replicates.",
        completed=False,
        trials=[],
    )
    if args.start_mature:
        report["interpretation"] += (
            " This supplementary assay starts the original body fully grown, with birth energy "
            "per adult area and fresh neural state. Its initial position uses the same normalized "
            "disk draw scaled to fit its adult radius. Offspring still begin as juveniles. "
            "This measures use of an available coordination interface, not developmental success."
        )
    record_keys = (
        "survived",
        "initial_modules",
        "first_growth_time",
        "modules",
        "target_modules",
        "development_spent",
        "internal_spent",
        "fresh_acquired",
        "detritus_acquired",
        "meat_acquired",
    )
    stop = StopFlag()
    try:
        with (args.output / "trials.jsonl").open("w") as stream:
            for index, genome in enumerate(selected):
                for seed in args.seeds:
                    for mode in args.modes:
                        if stop.requested:
                            return
                        result = trial(
                            c,
                            genome,
                            seed,
                            args.seconds,
                            "cpu",
                            mode,
                            stop,
                            record_keys=record_keys,
                            start_mature=args.start_mature,
                        )
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
