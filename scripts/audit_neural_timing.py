"""Audit V24's matched timing populations and exact neutral compatibility."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from audit_circuits import physical, same_state
from audit_value_forecasts import digest, source_digest
from audit_value_prediction import checked

from emergent_garden.config import Config
from emergent_garden.history import read_history
from emergent_garden.neural_timing import response_times, timing_genes, timing_metrics
from emergent_garden.storage import load_checkpoint
from emergent_garden.topology import effective_masks

GROUPS = {
    "neutral": "v24-neutral",
    "homogeneous": "v24-baseline",
    "inherited": "v24-inherited",
    "evolving": "v24",
}


def common_state(world):
    state = world.state_dict()
    state.pop("config")
    state["rng"].pop("neural_timing", None)
    state["events"] = [{k: v for k, v in e.items() if k != "genome_hash"} for e in state["events"]]
    if world.config.timing_count:
        stop = world.config.parameter_count - world.config.timing_count
        state["founders"] = state["founders"][:, :stop]
        state["agents"]["genome"] = state["agents"]["genome"][:, :stop]
    return state


def common_metrics(rows):
    return physical(
        [
            {
                k: v
                for k, v in row.items()
                if not k.startswith("neural_timing_") and k != "genome_variance"
            }
            for row in rows
        ]
    )


def parity(old, new):
    _, a, events_a = read_history(old)
    _, b, events_b = read_history(new)
    assert common_metrics(a) == common_metrics(b), (old, new)
    # Digests and total-gene variance change when neutral genes are appended.
    events_a, events_b = (
        [{k: v for k, v in e.items() if k != "genome_hash"} for e in events]
        for events in (events_a, events_b)
    )
    assert events_a == events_b, (old, new)
    left, right = load_checkpoint(old / "latest.pt"), load_checkpoint(new / "latest.pt")
    same_state(common_state(left), common_state(right))
    return dict(
        old=str(old),
        new=str(new),
        measurements=len(a),
        events=len(events_a),
        exact_common_state=True,
        exact_common_measurements=True,
    )


def timing_run(path, duration, resumed=False):
    result = checked(path, duration, resumed)
    world = load_checkpoint(path / "latest.pt")
    c, a = world.config, world.agents
    assert c.ecology_version == 24 and world.ablation == "none", path
    _, rows, _ = read_history(path, resumed)
    metrics = timing_metrics(c, a["genome"], a["memory_tau"], effective_masks(c, a["genome"])[0])
    lookup = {r["tick"]: r for r in rows}
    for key, value in metrics.items():
        assert value == rows[-1][key], (path, key)
        result["final"][key] = value
        for row in result["checkpoints"]:
            row[key] = lookup[row["tick"]][key]
    genes = timing_genes(c, a["genome"])
    assert torch.isfinite(genes).all() and (genes.abs() <= c.weight_limit).all(), path
    times = response_times(c, a["genome"], a["memory_tau"])
    assert torch.isfinite(times).all() and (times > 0).all(), path
    if c.neural_timing_range == 1:
        assert metrics["neural_timing_mean_within_log_std"] == 0, path
    elif world.population:
        assert metrics["neural_timing_mean_within_log_std"] > 0, path
    for segment in result["segments"]:
        source = segment["metadata"]["source_sha256"]
        assert source_digest(Path(segment["path"]) / "source.zip") == source, segment["path"]
    return result


def pulse_probe(path):
    report = json.loads((path / "summary.json").read_text())
    assert report["completed"]
    assert digest(path / "probe_neural_timing.py") == report["script_sha256"]
    assert source_digest(path / "source.zip") == report["source_sha256"]
    c = Config.from_dict(report["config"])
    founders = torch.load(path / "founders.pt", weights_only=True)
    edges = torch.logspace(torch.log10(torch.tensor(0.05)), torch.log10(torch.tensor(20.0)), 41)
    records = []
    for trial in report["trials"]:
        seed = trial["seed"]
        genomes = founders[seed]
        native = torch.load(f"runs/v24-evolving-pilot/seed-{seed}/founders.pt", weights_only=True)
        assert torch.equal(genomes, native["genomes"]), seed
        nodes = effective_masks(c, genomes)[0]
        tau = 0.2 + 4.8 * genomes[:, c.brain_parameter_count + 5].sigmoid()
        modes = {}
        for mode, result in trial["modes"].items():
            configured = replace(c, neural_timing_range=result["timing_range"])
            times = response_times(configured, genomes, tau)[nodes]
            assert times.tolist() == result["active_time_constants"]
            assert timing_metrics(configured, genomes, tau, nodes) == result["timing_statistics"]
            counts = torch.histogram(times, edges).hist
            assert int(counts.sum()) == int(nodes.sum())
            modes[mode] = {
                key: value
                for key, value in result.items()
                if key not in ("active_time_constants", "curve")
            }
            modes[mode]["time_histogram"] = dict(edges=edges.tolist(), counts=counts.int().tolist())
            modes[mode]["curve"] = result["curve"][::5]
            assert modes[mode]["curve"][-1]["time"] == 21
        records.append(dict(seed=seed, circuits=trial["circuits"], modes=modes))
    return dict(
        interpretation=report["interpretation"],
        source=str(path),
        report_sha256=digest(path / "summary.json"),
        source_sha256=report["source_sha256"],
        script_sha256=report["script_sha256"],
        founders_sha256=digest(path / "founders.pt"),
        matched_native_founders=True,
        curve_sampling="Every fifth controller update, 2 Hz",
        trials=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-long", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    configs = {group: asdict(Config.load(f"configs/{name}.toml")) for group, name in GROUPS.items()}
    shared = []
    for group in GROUPS:
        values = configs[group].copy()
        assert values.pop("initial_timing_sigma") == (0 if group == "neutral" else 0.5)
        assert values.pop("neural_timing_range") == (
            1 if group in ("neutral", "homogeneous") else 4
        )
        assert values.pop("timing_mutation_probability") == (0.1 if group == "evolving" else 0)
        shared.append(values)
    assert all(values == shared[0] for values in shared)
    old_config = asdict(Config.load("configs/v22-baseline.toml"))
    neutral = configs["neutral"].copy()
    old_config.pop("ecology_version")
    neutral.pop("ecology_version")
    assert old_config == neutral
    trials, compatibility = [], []
    for seed in (1, 2, 3):
        reference = None
        for group in GROUPS:
            path = Path(f"runs/v24-{group}-pilot/seed-{seed}")
            assert asdict(Config.load(path / "config.toml")) == configs[group], path
            founders = torch.load(path / "founders.pt", weights_only=True)["genomes"]
            if group == "neutral":
                assert not founders[:, -32:].count_nonzero(), path
                reference = founders
            else:
                assert torch.equal(founders[:, :-32], reference[:, :-32]), path
                if group == "homogeneous":
                    reference = founders
                else:
                    assert torch.equal(founders, reference), path
            row = timing_run(path, 600)
            row.update(
                treatment=group,
                seed=seed,
                matching_founder_prefix=True,
                matching_full_timing_founders=group != "neutral",
            )
            trials.append(row)
        old = Path(f"runs/v22-baseline-pilot/seed-{seed}")
        new = Path(f"runs/v24-neutral-pilot/seed-{seed}")
        compatibility.append(parity(old, new))
        compatibility.append(parity(new, Path(f"runs/v24-homogeneous-pilot/seed-{seed}")))
    continuations = []
    if args.include_long:
        for group in ("homogeneous", "inherited", "evolving"):
            for seed in (1, 2, 3):
                path = Path(f"runs/v24-{group}-long-{seed}")
                assert asdict(Config.load(path / "config.toml")) == configs[group], path
                row = timing_run(path, 1800, True)
                row.update(treatment=group, seed=seed)
                continuations.append(row)
    report = dict(
        interpretation="Three matched starts compare homogeneous per-body response times, "
        "inherited per-neuron timing without direct timing mutation, and inherited timing "
        "with per-gene birth mutation. The latter two retain ordinary weight, body, and "
        "structural mutations. Their founders, including timing genes, match exactly. "
        "All retain the V22 disabled-critic reference's gentle motor learning. No extra "
        "timing energy charge is applied. This is an evolutionary community screen, not "
        "a controlled demonstration of useful temporal memory or individual fitness.",
        statistics="Time summaries count each active neuron once per organism, not once "
        "per module; within-brain log standard deviations weight organisms equally. "
        "Dormant slots are excluded. Neutral comparisons exclude version metadata, "
        "the appended genes, their separate RNG, derived timing summaries, full-genome "
        "variance, and genome digests. All remaining state, event fields, and physical "
        "measurements must match exactly.",
        configurations=configs,
        trials=trials,
        neutral_compatibility=compatibility,
        continuations=continuations,
        continuations_included=args.include_long,
    )
    output = Path("docs/results/v24-neural-timing.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    pulse = pulse_probe(Path("runs/v24-pulse-responses"))
    Path("docs/results/v24-pulse-responses.json").write_text(json.dumps(pulse, indent=2) + "\n")
    print(
        f"Audited {len(trials)} pilots, {len(continuations)} continuations, "
        f"and {len(compatibility)} exact comparisons: {output}"
    )


if __name__ == "__main__":
    main()
