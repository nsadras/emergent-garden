"""Compare timing placement and motor feedback within the same evolved genotypes."""

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from emergent_garden.neural_timing import response_times, timing_genes
from emergent_garden.runtime import StopFlag
from emergent_garden.storage import RunStore, atomic_save, load_checkpoint
from emergent_garden.topology import effective_masks
from emergent_garden.world import create_world

TREATMENTS = {
    "native": ("native", "none"),
    "mean-tau": ("mean_tau", "none"),
    "permuted": ("permuted", "none"),
    "no-motor-learning": ("native", "no_motor_learning"),
    "shuffled-return": ("native", "shuffled_motor_reward"),
}
MUTATION_KEYS = (
    "mutation_probability",
    "trait_mutation_probability",
    "node_mutation_probability",
    "edge_mutation_probability",
    "module_mutation_probability",
    "timing_mutation_probability",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform_genomes(config, genomes, mode, permutation_seed=907):
    """Only active timing genes change; equal genotypes receive equal permutations.

    The mean-tau intervention preserves each body's arithmetic mean active time
    constant. It does not preserve the mean integration factor or the recurrent
    network's eigenmodes. A permutation preserves the entire timing multiset.
    """
    if config.ecology_version < 24 or config.neural_timing_range <= 1:
        raise ValueError("Timing interventions require heterogeneous V24+ circuits")
    if genomes.ndim != 2 or genomes.shape[1] != config.parameter_count:
        raise ValueError("Genome shape does not match its configuration")
    if genomes.device.type != "cpu":
        raise ValueError("Apply assay genome interventions on CPU before transplantation")
    if mode not in ("native", "mean_tau", "permuted"):
        raise ValueError(f"Unknown timing intervention: {mode}")
    out = genomes.clone()
    if mode == "native":
        return out
    nodes = effective_masks(config, genomes)[0]
    counts = nodes.sum(-1)
    if (counts == 0).any():
        raise ValueError("Each assayed circuit needs at least one active neuron")
    genes = timing_genes(config, out)
    if mode == "mean_tau":
        scale = math.log(config.neural_timing_range)
        factors = (scale * genes.double().tanh()).exp()
        mean = (factors * nodes).sum(-1) / counts
        ratio = mean.log() / scale
        # Roundoff at the inherited bound must not produce an invalid atanh.
        bound = min(math.tanh(config.weight_limit), math.nextafter(1.0, 0.0))
        equal = ratio.clamp(-bound, bound).atanh().clamp(-config.weight_limit, config.weight_limit)
        genes[nodes] = equal[:, None].expand_as(genes)[nodes].to(genes.dtype)
    else:
        for i, genome in enumerate(genomes):
            key = hashlib.sha256(genome.contiguous().numpy().tobytes()).digest()
            key = hashlib.sha256(str(permutation_seed).encode() + b":" + key).digest()
            seed = int.from_bytes(key[:8], "little") % (2**63 - 1)
            rng = torch.Generator().manual_seed(seed)
            active = nodes[i].nonzero().flatten()
            order = torch.randperm(len(active), generator=rng)
            genes[i, active] = genes[i, active[order]].clone()
    assert torch.isfinite(out).all()
    assert (genes.abs() <= config.weight_limit).all()
    return out


def prepare_world(config, genomes, environment, ablation, device="cpu"):
    c = replace(config, **dict.fromkeys(MUTATION_KEYS, 0.0)).validate()
    if len(genomes) != c.initial_population:
        raise ValueError("The transplanted pool must match initial_population")
    world = create_world(c, seed=environment, device=device, ablation=ablation)
    world.agents["genome"] = genomes.to(device).clone()
    world.develop(world.agents)
    world.agents["energy"] = c.birth_energy * world.agents["area"]
    world.initial_energy = world.agents["energy"].double().sum().item()
    world.move()  # Repair the original positions for the transplanted body sizes.
    world.agents["distance"].zero_()
    world.totals["distance"] = 0.0
    world.founders = world.agents["genome"].clone()
    return world


def mean_times(config, genomes):
    nodes = effective_masks(config, genomes)[0]
    tau = 0.2 + 4.8 * genomes[:, config.brain_parameter_count + 5].sigmoid()
    values = response_times(config, genomes, tau)
    return (values * nodes).sum(-1) / nodes.sum(-1)


def run_assay(source, output, seeds, seconds, device="cpu", permutation_seed=907, stop=None):
    source, output = Path(source), Path(output)
    if not math.isfinite(seconds) or seconds <= 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use a positive finite duration and distinct environment seeds")
    original = load_checkpoint(source / "latest.pt")
    c = original.config
    if c.ecology_version < 24 or c.neural_timing_range <= 1 or original.ablation != "none":
        raise ValueError("Use an unablated heterogeneous V24+ source")
    a = original.agents
    mask = a["generation"] > 0
    pool = dict(genomes=a["genome"][mask], ids=a["id"][mask], generations=a["generation"][mask])
    if not len(pool["genomes"]):
        raise ValueError("Source needs living descendant genotypes")
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).read_bytes()
    (output / Path(__file__).name).write_bytes(script)
    atomic_save(pool, output / "source-genomes.pt")
    frozen = replace(c, **dict.fromkeys(MUTATION_KEYS, 0.0)).validate()
    frozen.save(output / "config.toml")
    report = dict(
        completed=False,
        source=str(source),
        source_time=original.time,
        source_checkpoint_sha256=digest(source / "latest.pt"),
        source_pool_sha256=digest(output / "source-genomes.pt"),
        script_sha256=hashlib.sha256(script).hexdigest(),
        seeds=seeds,
        seconds=seconds,
        permutation_seed=permutation_seed,
        treatments=TREATMENTS,
        interpretation="Living descendants are sampled with replacement into identical "
        "new environments. All six mutation probabilities are zero; births and selection "
        "remain active, with fresh lifetime state. Native, mean-tau, and permuted treatments "
        "differ only in active timing genes. Mean-tau preserves each body's arithmetic "
        "mean active time constant; permutation preserves its timing multiset, consistently "
        "for identical genotypes. These do not preserve the network's temporal computation. "
        "Two further native-genome controls disable motor updates or shuffle return rates. "
        "Recurrent plasticity and learning-capacity costs remain. These are conditional "
        "community interventions, not independent individual fitness tests or proof of memory.",
        trials=[],
    )
    target = math.ceil(seconds * c.physics_hz)

    def stopping():
        return stop is not None and stop.requested

    def write_summary():
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    write_summary()
    try:
        for environment in seeds:
            rng = torch.Generator().manual_seed(environment + 1299709)
            selected = torch.randint(len(pool["genomes"]), (c.initial_population,), generator=rng)
            chosen = {key: values[selected].clone() for key, values in pool.items()}
            atomic_save(chosen, output / f"selected-{environment}.pt")
            for treatment, (transform, ablation) in TREATMENTS.items():
                if stopping():
                    return report
                genomes = transform_genomes(c, chosen["genomes"], transform, permutation_seed)
                world = prepare_world(c, genomes, environment, ablation, device)
                world.seeded_from = dict(
                    mode="timing_assay",
                    source=str(source),
                    source_checkpoint_sha256=report["source_checkpoint_sha256"],
                    source_ids=chosen["ids"].tolist(),
                    source_generations=chosen["generations"].tolist(),
                    intervention=treatment,
                    permutation_seed=permutation_seed,
                )
                path = output / f"{environment}-{treatment}"
                store = RunStore(path, world)
                try:
                    atomic_save(world.state_dict(), path / "initial.pt")
                    store.measure(world)
                    while world.tick < target and world.population and not stopping():
                        world.step(min(c.physics_hz, target - world.tick))
                        store.measure(world)
                    metric = store.measure(world)
                    row = dict(
                        environment=environment,
                        treatment=treatment,
                        path=str(path),
                        completed=world.tick >= target or not world.population,
                        **metric,
                    )
                    report["trials"].append(row)
                    print(
                        f"{environment}/{treatment}: t={world.time:.1f}s "
                        f"population={world.population} births={metric['births']}",
                        flush=True,
                    )
                finally:
                    store.checkpoint(world)
                    store.close()
                write_summary()
        report["completed"] = len(report["trials"]) == len(seeds) * len(TREATMENTS) and all(
            row["completed"] for row in report["trials"]
        )
        return report
    finally:
        write_summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[901, 902])
    parser.add_argument("--seconds", type=float, default=360)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--permutation-seed", type=int, default=907)
    args = parser.parse_args()
    torch.set_num_threads(1)
    stop = StopFlag()
    try:
        run_assay(
            args.source,
            args.output,
            args.seeds,
            args.seconds,
            args.device,
            args.permutation_seed,
            stop,
        )
    finally:
        stop.close()


if __name__ == "__main__":
    main()
