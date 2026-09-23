"""Compare two evolved populations alone or in an equally seeded community.

This assembles existing evolved genotypes; it does not demonstrate spontaneous
speciation. Ordinary evolution remains active. Source ancestry and diet are
reported separately, since their relationship can change during the experiment.
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from emergent_garden.config import Config
from emergent_garden.inheritance import upgrade_genomes
from emergent_garden.runtime import StopFlag
from emergent_garden.storage import RunStore
from emergent_garden.world import create_world

EXPERIMENT_SOURCE = Path(__file__).read_bytes()


def seed_assembly(world, sources, condition):
    n = world.population
    groups = (
        (n, 0)
        if condition == "first"
        else (0, n)
        if condition == "second"
        else (n // 2, n - n // 2)
    )
    pieces, identifiers, generations, origins, provenance = [], [], [], [], []
    for group, (path, count) in enumerate(zip(sources, groups, strict=True)):
        source_config = Config.load(path / "config.toml")
        population_file = path / "population.pt"
        pool = torch.load(population_file, map_location="cpu", weights_only=True)
        if not len(pool["genomes"]):
            raise ValueError(f"No living source genomes in {path}")
        rng = torch.Generator().manual_seed(world.seed + 15485863 + group * 173)
        # Always sample the same full pool per source, irrespective of treatment.
        selection = torch.randint(len(pool["genomes"]), (n,), generator=rng)[:count]
        pieces.append(upgrade_genomes(source_config, world.config, pool["genomes"][selection]))
        identifiers.extend(pool["ids"][selection].tolist())
        generations.extend(pool["generations"][selection].tolist())
        origins.extend([group] * count)
        provenance.append(
            dict(
                path=str(path),
                population_sha256=hashlib.sha256(population_file.read_bytes()).hexdigest(),
            )
        )
    rng = torch.Generator().manual_seed(world.seed + 32452843)
    order = torch.randperm(n, generator=rng)
    world.agents["genome"] = torch.cat(pieces)[order].to(world.device)
    world.develop(world.agents)
    world.agents["energy"] = world.config.birth_energy * world.agents["area"]
    world.initial_energy = world.agents["energy"].double().sum().item()
    world.move()
    world.agents["distance"].zero_()
    world.totals["distance"] = 0.0
    world.founders = world.agents["genome"].clone()
    origins = torch.tensor(origins, dtype=torch.long)[order]
    world.seeded_from = dict(
        mode="community_assembly",
        condition=condition,
        sources=provenance,
        founder_sources=origins.tolist(),
        source_ids=torch.tensor(identifiers)[order].tolist(),
        source_generations=torch.tensor(generations)[order].tolist(),
        interpretation="Equal founder numbers in mixtures; original bodies and energy endowments "
        "are retained, with zero-age brains and no acquired state. Different morphologies have "
        "different founder energy. This is deliberate community assembly, "
        "not spontaneous speciation.",
    )
    world.rebuild_fields()
    return origins.to(world.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v8.toml"))
    parser.add_argument("--sources", type=Path, nargs=2, required=True)
    parser.add_argument("--condition", choices=("first", "second", "mixed"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[151, 152, 153])
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ablation", default="none")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    torch.set_num_threads(1)
    config = Config.load(args.config)
    if config.ecology_version < 1 or config.initial_population < 2:
        parser.error("Assembly requires V1+ and at least two founders")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        config=asdict(config),
        sources=[str(p) for p in args.sources],
        condition=args.condition,
        duration=args.seconds,
        completed=False,
        trials=[],
        experiment_sha256=hashlib.sha256(EXPERIMENT_SOURCE).hexdigest(),
    )
    stop = StopFlag()
    keys = ("fresh_acquired", "detritus_acquired", "meat_acquired")
    try:
        for seed in args.seeds:
            if stop.requested:
                break
            world = create_world(config, seed, args.device, ablation=args.ablation)
            origins = seed_assembly(world, args.sources, args.condition)
            store = RunStore(args.output / f"seed-{seed}", world)
            (store.path / "experiment.py").write_bytes(EXPERIMENT_SOURCE)
            dead_uptake = [{key: 0.0 for key in keys} for _ in range(2)]
            births, deaths = [0, 0], [0, 0]
            starting = [(origins == group).sum().item() for group in range(2)]

            def snapshot(
                world=world,
                origins=origins,
                births=births,
                deaths=deaths,
                dead_uptake=dead_uptake,
                starting=starting,
            ):
                for event in world.events:
                    if event["event"] in ("birth", "death"):
                        group = int(origins[event["lineage"]])
                        if event["event"] == "birth":
                            births[group] += 1
                        else:
                            deaths[group] += 1
                            for key in keys:
                                dead_uptake[group][key] += event.get(key, 0.0)
                labels = origins[world.agents["lineage"]]
                rows = []
                for group in range(2):
                    mask = labels == group
                    population = int(mask.sum())
                    assert population == starting[group] + births[group] - deaths[group]
                    rows.append(
                        dict(
                            source=group,
                            population=population,
                            births=births[group],
                            deaths=deaths[group],
                            mean_diet=world.agents["diet"][mask].mean().item()
                            if population
                            else None,
                            uptake={
                                key: dead_uptake[group][key]
                                + world.agents[key][mask].double().sum().item()
                                if key in world.agents
                                else 0.0
                                for key in keys
                            },
                        )
                    )
                return dict(time=world.time, groups=rows)

            target = math.ceil(args.seconds * config.physics_hz)
            try:
                with (store.path / "origins.jsonl").open("w") as origin_file:
                    state = snapshot()
                    origin_file.write(json.dumps(state) + "\n")
                    store.measure(world)
                    while world.population and world.tick < target and not stop.requested:
                        world.step(min(config.physics_hz, target - world.tick))
                        state = snapshot()
                        origin_file.write(json.dumps(state) + "\n")
                        origin_file.flush()
                        store.measure(world)  # Writes and clears the events counted above.
                metric = store.measure(world)
                row = dict(
                    seed=seed,
                    origins=state,
                    final=metric,
                    completed=world.tick >= target or not world.population,
                )
                report["trials"].append(row)
                (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"{args.condition}/{seed}: t={world.time:.0f}, populations="
                    f"{[g['population'] for g in state['groups']]}, "
                    f"births={world.totals['births']}",
                    flush=True,
                )
            finally:
                store.checkpoint(world)
                store.close()
        report["completed"] = len(report["trials"]) == len(args.seeds) and all(
            row["completed"] for row in report["trials"]
        )
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        stop.close()


if __name__ == "__main__":
    main()
