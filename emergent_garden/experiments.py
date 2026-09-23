"""Independent calibration and paired founder/descendant sensory evaluations."""

import json
import math
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .config import Config
from .storage import RunStore, atomic_save
from .world import create_world as World


def calibration(
    config,
    output,
    seeds,
    seconds,
    device,
    controller="neural",
    stop=None,
    seed_from=None,
    ablation="none",
):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in seeds:
        if stop is not None and stop.requested:
            break
        world = World(config, seed, device, controller, ablation)
        if seed_from:
            from .inheritance import seed_population

            seed_population(world, seed_from)
        store = RunStore(root / f"seed-{seed}", world)
        store.measure(world)
        target = math.ceil(seconds * config.physics_hz)
        last_checkpoint = time.monotonic()
        try:
            while (
                world.tick < target
                and world.population
                and not (stop is not None and stop.requested)
            ):
                world.step(min(config.physics_hz, target - world.tick))
                store.measure(world)
                if time.monotonic() - last_checkpoint >= config.checkpoint_wall_seconds:
                    store.checkpoint(world)
                    last_checkpoint = time.monotonic()
            metric = store.measure(world)
            results.append(dict(seed=seed, **metric))
            print(
                f"seed {seed}: t={world.time:.0f}s population={world.population} "
                f"births={world.totals['births']} generation={metric['generation_max']} "
                f"speed={metric['speed']:.1f}x",
                flush=True,
            )
        finally:
            store.checkpoint(world)
            store.close()
    (root / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def trial(config, genome, seed, seconds, device, ablation, stop=None):
    config = replace(
        config,
        initial_population=1,
        mutation_probability=0.0,
        trait_mutation_probability=0.0,
        node_mutation_probability=0.0,
        edge_mutation_probability=0.0,
    )
    world = World(config, seed, device, ablation=ablation)
    world.agents["genome"][0] = genome.to(device)
    if config.ecology_version:
        world.develop(world.agents)
        world.agents["energy"] = config.birth_energy * world.agents["area"]
        world.initial_energy = world.agents["energy"].double().sum().item()
    world.founders = world.agents["genome"].clone()
    target = math.ceil(seconds * config.physics_hz)
    record = None
    while world.tick < target and world.population:
        if stop is not None and stop.requested:
            raise InterruptedError("Evaluation stopped; completed trials are saved in trials.jsonl")
        world.step(min(config.physics_hz, target - world.tick))
        for event in world.events:
            if event["event"] == "death" and event["id"] == 0:
                record = event
        world.events.clear()
    if record is None:
        index = (world.agents["id"] == 0).nonzero().flatten()
        if len(index):
            record = world.agent_record(int(index[0]))
    if record is None:
        raise RuntimeError("Evaluation lost the founding organism record")
    return {key: record[key] for key in ("acquired", "spent", "age", "offspring", "distance")}


def summary(rows):
    values = {}
    for key in ("acquired", "spent", "age", "offspring", "distance"):
        sample = [row[key] for row in rows]
        values[key] = dict(
            mean=statistics.fmean(sample),
            stdev=statistics.stdev(sample) if len(sample) > 1 else 0,
            minimum=min(sample),
            maximum=max(sample),
        )
    return values


def evaluate(run, output, count, seeds, seconds, device, stop=None):
    run, output = Path(run), Path(output)
    config = Config.load(run / "config.toml")
    founders = torch.load(run / "founders.pt", weights_only=True, map_location="cpu")["genomes"]
    population = torch.load(run / "population.pt", weights_only=True, map_location="cpu")
    descendants = population["genomes"][population["generations"] > 0]
    if not len(founders) or not len(descendants):
        raise ValueError("Evaluation requires founders and living descendants of generation > 0")
    output.mkdir(parents=True, exist_ok=False)
    rng = torch.Generator().manual_seed(982451653)
    selected = {
        "founders": founders[torch.randperm(len(founders), generator=rng)[:count]],
        "descendants": descendants[torch.randperm(len(descendants), generator=rng)[:count]],
    }
    atomic_save(selected, output / "sampled-genomes.pt")
    rows = []
    started = time.monotonic()
    with (output / "trials.jsonl").open("w") as stream:
        for group, genomes in selected.items():
            modes = ("none",) if group == "founders" else ("none", "disabled", "shuffled")
            for mode in modes:
                for index, genome in enumerate(genomes):
                    for seed in seeds:
                        row = dict(
                            group=group,
                            ablation=mode,
                            genome=index,
                            seed=seed,
                            **trial(config, genome, seed, seconds, device, mode, stop=stop),
                        )
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        rows.append(row)
                    print(
                        f"{group}/{mode}: genome {index + 1}/{len(genomes)} "
                        f"({time.monotonic() - started:.0f}s elapsed)",
                        flush=True,
                    )
    groups = {}
    for group, mode in [
        ("founders", "none"),
        ("descendants", "none"),
        ("descendants", "disabled"),
        ("descendants", "shuffled"),
    ]:
        groups[f"{group}/{mode}"] = summary(
            [row for row in rows if row["group"] == group and row["ablation"] == mode]
        )
    # Pair sensory treatments within genome and environment; retain raw trials.
    normal = {
        (r["genome"], r["seed"]): r
        for r in rows
        if r["group"] == "descendants" and r["ablation"] == "none"
    }
    paired = {}
    for mode in ("disabled", "shuffled"):
        differences = [
            normal[r["genome"], r["seed"]]["acquired"] - r["acquired"]
            for r in rows
            if r["group"] == "descendants" and r["ablation"] == mode
        ]
        paired[mode] = dict(
            mean_food_advantage=statistics.fmean(differences),
            positive_pairs=sum(x > 0 for x in differences),
            pairs=len(differences),
        )
    report = dict(
        config=asdict(config),
        samples={k: len(v) for k, v in selected.items()},
        seeds=seeds,
        duration=seconds,
        groups=groups,
        paired_sensory_effect=paired,
        interpretation="One evolutionary run; repeat across independent runs. "
        "Positive mean differences alone do not establish a reproducible advantage.",
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def community_assay(run, output, seeds, seconds, device, modes, stop=None):
    """Transplant whole sampled communities into matched, independent environments.

    Mutation is disabled for both brain and trait genes, but births remain active.
    Comparing founder/descendant communities measures collective performance, not
    individual fitness or proof of adaptive cognition. Interventions can disrupt
    ordinary controller dynamics and should be interpreted with behavioral probes.
    """
    run, output = Path(run), Path(output)
    c = replace(
        Config.load(run / "config.toml"),
        mutation_probability=0.0,
        trait_mutation_probability=0.0,
        node_mutation_probability=0.0,
        edge_mutation_probability=0.0,
    )
    if not c.ecology_version:
        raise ValueError("Use evaluate for V0; community assays require V1 or later")
    from .ecology import EcologyWorld

    founders = torch.load(run / "founders.pt", weights_only=True, map_location="cpu")["genomes"]
    final = torch.load(run / "population.pt", weights_only=True, map_location="cpu")
    descendants = final["genomes"][final["generations"] > 0]
    if not len(founders) or not len(descendants):
        raise ValueError("Assay requires founders and living descendants")
    # Validate all modes before creating output or doing expensive work.
    for mode in modes:
        EcologyWorld(replace(c, initial_population=0, initial_food=0), ablation=mode)
    output.mkdir(parents=True, exist_ok=False)
    c.save(output / "config.toml")
    atomic_save(dict(founders=founders, descendants=descendants), output / "source-genomes.pt")
    rows = []
    target = math.ceil(seconds * c.physics_hz)
    for seed in seeds:
        for group, genomes in (("founders", founders), ("descendants", descendants)):
            selection = torch.Generator().manual_seed(seed + 1299709)
            chosen = genomes[
                torch.randint(len(genomes), (c.initial_population,), generator=selection)
            ]
            for mode in ["none"] if group == "founders" else modes:
                if stop is not None and stop.requested:
                    (output / "summary.json").write_text(
                        json.dumps(dict(completed=False, trials=rows), indent=2)
                    )
                    return rows
                w = EcologyWorld(c, seed, device, ablation=mode)
                w.agents["genome"] = chosen.to(device).clone()
                w.develop(w.agents)
                w.agents["energy"] = c.birth_energy * w.agents["area"]
                w.initial_energy = w.agents["energy"].double().sum().item()
                # Transplanting changes radii; project initial overlaps before time starts.
                w.move()
                w.founders = chosen.to(device).clone()
                path = output / f"{seed}-{group}-{mode}"
                store = RunStore(path, w, run)
                try:
                    store.measure(w)
                    while w.tick < target and w.population and not (stop and stop.requested):
                        w.step(min(c.physics_hz, target - w.tick))
                        if w.tick % (c.physics_hz * 5) == 0:
                            store.measure(w)
                    row = dict(seed=seed, group=group, ablation=mode, **store.measure(w))
                    row["completed"] = w.tick >= target or not w.population
                    rows.append(row)
                    with (output / "trials.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(
                        f"assay {seed}/{group}/{mode}: population={w.population}, "
                        f"births={w.totals['births']}",
                        flush=True,
                    )
                finally:
                    store.checkpoint(w)
                    store.close()
    report = dict(
        completed=all(r["completed"] for r in rows),
        trials=rows,
        interpretation="Paired community transplants retain resource feedback and competition. "
        "Founders and descendants may differ in initial body energy investment. "
        "State resets perturb controller dynamics; they alone do not establish useful memory.",
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return rows
