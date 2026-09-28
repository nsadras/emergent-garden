"""Audit all predeclared V24 timing and feedback transplants without rerunning them."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from assay_neural_timing import MUTATION_KEYS, TREATMENTS, mean_times, transform_genomes
from audit_circuits import same_state
from audit_value_forecasts import digest, source_digest
from audit_value_prediction import checked

from emergent_garden.config import Config
from emergent_garden.history import PERFORMANCE_KEYS, read_history
from emergent_garden.neural_timing import response_times, timing_genes, timing_metrics
from emergent_garden.storage import load_checkpoint
from emergent_garden.topology import effective_masks
from emergent_garden.world import genome_hash

OUTCOMES = ("births", "population", "fresh_absorbed", "detritus_absorbed", "predation_absorbed")


def common_initial(world):
    state = world.state_dict()
    state.pop("ablation")
    state.pop("founders")
    state["agents"].pop("genome")
    state["seeded_from"].pop("intervention")
    state["rng"].pop("reward_shuffle", None)
    return state


def full_hash(genome):
    return hashlib.sha256(genome.contiguous().numpy().tobytes()).hexdigest()


def check_initial(world, original, transform, permutation_seed):
    c, a = world.config, world.agents
    expected = transform_genomes(c, original, transform, permutation_seed)
    same_state(a["genome"], expected)
    same_state(world.founders, expected)
    assert world.time == 0 and not a["generation"].count_nonzero()
    assert not a["age"].count_nonzero()
    for key, value in a.items():
        if key.startswith("module_") and key not in ("module_mask", "module_offset"):
            assert not value.count_nonzero(), key
    nodes = effective_masks(c, original)[0]
    old, new = timing_genes(c, original), timing_genes(c, expected)
    same_state(original[:, : -c.timing_count], expected[:, : -c.timing_count])
    same_state(old[~nodes], new[~nodes])
    before, after = mean_times(c, original), mean_times(c, expected)
    torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-6)
    for i, mask in enumerate(nodes):
        if transform == "mean_tau":
            assert new[i, mask].unique().numel() == 1
        elif transform == "permuted":
            same_state(old[i, mask].sort().values, new[i, mask].sort().values)
    return dict(
        genomes=len(expected),
        unique_genotypes=len({full_hash(g) for g in expected}),
        active_timing_genes=int(nodes.sum()),
        changed_active_timing_genes=int(((old != new) & nodes).sum()),
        changed_genomes=int((old != new).any(-1).sum()),
        maximum_mean_time_difference_seconds=float((before - after).abs().max()),
        fresh_acquired_state=True,
        unchanged_non_timing_and_dormant_genes=True,
        timing_statistics=timing_metrics(c, expected, a["memory_tau"], nodes),
    )


def check_trial(path, row, source, chosen, frozen, seconds, permutation_seed):
    treatment = row["treatment"]
    transform, ablation = TREATMENTS[treatment]
    initial = load_checkpoint(path / "initial.pt")
    assert asdict(initial.config) == asdict(frozen)
    assert initial.seed == row["environment"] and initial.ablation == ablation
    assert initial.seeded_from == dict(
        mode="timing_assay",
        source=source["source"],
        source_checkpoint_sha256=source["source_checkpoint_sha256"],
        source_ids=chosen["ids"].tolist(),
        source_generations=chosen["generations"].tolist(),
        intervention=treatment,
        permutation_seed=permutation_seed,
    )
    intervention = check_initial(initial, chosen["genomes"], transform, permutation_seed)
    founders = torch.load(path / "founders.pt", weights_only=True)["genomes"]
    same_state(founders, initial.founders)
    result = checked(path, seconds)
    final = load_checkpoint(path / "latest.pt")
    assert final.config == frozen and final.ablation == ablation
    assert final.seeded_from == initial.seeded_from
    _, history, events = read_history(path)
    metrics = final.metrics()
    for key, value in metrics.items():
        if key not in PERFORMANCE_KEYS:
            assert row[key] == value == history[-1][key], (path, key)
    genomes = final.agents["genome"]
    inherited = {full_hash(g) for g in founders}
    assert all(full_hash(g) in inherited for g in genomes), path
    short_hashes = {genome_hash(g) for g in founders}
    births = [e for e in events if e["event"] == "birth"]
    assert all(e["genome_hash"] in short_hashes for e in births), path
    assert not any(e["neural_structure_changed"] or e["module_event"] for e in births)
    lookup = {r["tick"]: r for r in history}
    timings = timing_metrics(
        frozen, genomes, final.agents["memory_tau"], effective_masks(frozen, genomes)[0]
    )
    for key, value in timings.items():
        assert value == history[-1][key], (path, key)
        result["final"][key] = value
        for checkpoint in result["checkpoints"]:
            checkpoint[key] = lookup[checkpoint["tick"]][key]
    tau = response_times(frozen, genomes, final.agents["memory_tau"])
    assert torch.isfinite(tau).all() and (tau > 0).all()
    genes = timing_genes(frozen, genomes)
    assert torch.isfinite(genes).all() and (genes.abs() <= frozen.weight_limit).all()
    for segment in result["segments"]:
        metadata = segment["metadata"]
        assert metadata["ablation"] == ablation
        assert source_digest(Path(segment["path"]) / "source.zip") == metadata["source_sha256"]
    result.update(
        environment=row["environment"],
        treatment=treatment,
        initial_sha256=digest(path / "initial.pt"),
        final_sha256=digest(path / "latest.pt"),
        intervention=intervention,
        matching_common_initial_state=True,
        frozen_final_genotypes=True,
        frozen_birth_genome_digests=True,
    )
    return result, common_initial(initial)


def collect(path):
    report = json.loads((path / "summary.json").read_text())
    assert report["completed"], path
    assert report["seeds"] == [901, 902] and report["seconds"] == 360, path
    assert report["permutation_seed"] == 907
    assert report["treatments"] == {k: list(v) for k, v in TREATMENTS.items()}
    assert digest(path / "assay_neural_timing.py") == report["script_sha256"]
    assert digest(path / "source-genomes.pt") == report["source_pool_sha256"]
    source_path = Path(report["source"]) / "latest.pt"
    assert digest(source_path) == report["source_checkpoint_sha256"]
    source = load_checkpoint(source_path)
    assert source.time == report["source_time"] == 1800
    assert source.ablation == "none" and source.config.ecology_version == 24
    assert source.config.timing_mutation_probability == 0
    assert source.config.neural_timing_range == 4
    frozen = replace(source.config, **dict.fromkeys(MUTATION_KEYS, 0.0)).validate()
    assert Config.load(path / "config.toml") == frozen
    pool = torch.load(path / "source-genomes.pt", weights_only=True)
    mask = source.agents["generation"] > 0
    for recorded, key in (("genomes", "genome"), ("ids", "id"), ("generations", "generation")):
        same_state(pool[recorded], source.agents[key][mask])
    expected_pairs = {(env, treatment) for env in report["seeds"] for treatment in TREATMENTS}
    actual_pairs = {(r["environment"], r["treatment"]) for r in report["trials"]}
    assert len(report["trials"]) == len(actual_pairs) == len(expected_pairs)
    assert actual_pairs == expected_pairs and all(r["completed"] for r in report["trials"])
    records, references, samples = [], {}, {}
    for environment in report["seeds"]:
        rng = torch.Generator().manual_seed(environment + 1299709)
        indices = torch.randint(len(pool["genomes"]), (frozen.initial_population,), generator=rng)
        samples[environment] = torch.load(path / f"selected-{environment}.pt", weights_only=True)
        for key, values in pool.items():
            same_state(samples[environment][key], values[indices])
    for row in report["trials"]:
        environment = row["environment"]
        trial_path = path / f"{environment}-{row['treatment']}"
        assert Path(row["path"]) == trial_path
        record, physical = check_trial(
            trial_path,
            row,
            report,
            samples[environment],
            frozen,
            report["seconds"],
            report["permutation_seed"],
        )
        if environment in references:
            same_state(physical, references[environment])
        else:
            references[environment] = physical
        record.update(source_seed=source.seed, source=str(source_path.parent))
        records.append(record)
    metadata = {key: value for key, value in report.items() if key != "trials"}
    metadata.update(
        path=str(path),
        summary_sha256=digest(path / "summary.json"),
        source_seed=source.seed,
        pool_genomes=len(pool["genomes"]),
        unique_pool_genotypes=len({full_hash(g) for g in pool["genomes"]}),
        selected_sha256={env: digest(path / f"selected-{env}.pt") for env in report["seeds"]},
        frozen_configuration=asdict(frozen),
    )
    return metadata, records


def contrasts(trials):
    lookup = {(r["source_seed"], r["environment"], r["treatment"]): r for r in trials}
    results = []
    for (seed, env, treatment), trial in lookup.items():
        if treatment == "native":
            continue
        native = lookup[seed, env, "native"]
        differences = {}
        for key in OUTCOMES:
            a, b = native["final"][key], trial["final"][key]
            differences[key] = dict(
                native=a,
                control=b,
                native_minus_control=a - b,
                relative_to_control=(a - b) / b if b else None,
            )
        results.append(
            dict(source_seed=seed, environment=env, control=treatment, outcomes=differences)
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assays",
        type=Path,
        nargs="+",
        default=[Path(f"runs/v24-timing-assay-{seed}") for seed in (1, 2, 3)],
    )
    parser.add_argument("--output", type=Path, default=Path("docs/results/v24-timing-assays.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources, trials = [], []
    for path in args.assays:
        source, records = collect(path)
        sources.append(source)
        trials.extend(records)
    assert len({s["source_seed"] for s in sources}) == len(sources)
    assert len({s["script_sha256"] for s in sources}) == 1
    assert len({r["segments"][0]["metadata"]["source_sha256"] for r in trials}) == 1
    report = dict(
        completed=True,
        interpretation="Same sampled evolved genotypes, fresh acquired state, matched initial "
        "physical worlds, and frozen mutation. Three source communities and two environments "
        "per source are conditional comparisons; individuals and cloned offspring are not "
        "independent replicates. Assay lineage labels name newly sampled founders, not original "
        "ancestry. Mean-tau preserves each body's arithmetic mean intrinsic time constant; "
        "permutation preserves each brain's full timing multiset. Neither guarantees matching "
        "recurrent computation. Learning controls retain recurrent plasticity and energetic "
        "capacity costs. Shuffling cannot mix singleton updates. Contrasts are native minus "
        "each control. No outcome alone demonstrates useful temporal memory or intelligence.",
        sources=sources,
        trials=trials,
        contrasts=contrasts(trials),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {len(trials)} complete timing/feedback transplants: {args.output}")


if __name__ == "__main__":
    main()
