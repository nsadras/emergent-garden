"""Audit V25's native delayed task and matched ecological learning controls."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from audit_circuits import physical, same_state
from audit_value_forecasts import digest, source_digest
from audit_value_prediction import checked
from probe_native_recurrent import MODES, evaluate

from emergent_garden.brain import initial_state
from emergent_garden.config import Config
from emergent_garden.history import read_history
from emergent_garden.neural_timing import response_times, timing_genes, timing_metrics
from emergent_garden.recurrent_learning import recurrent_metrics, recurrent_shapes
from emergent_garden.storage import load_checkpoint
from emergent_garden.topology import effective_masks

GROUPS = {
    "baseline": ("v25-baseline", "none"),
    "noise-only": ("v25", "no_recurrent_learning"),
    "learning": ("v25", "none"),
    "shuffled": ("v25", "shuffled_recurrent_reward"),
}
RECURRENT_KEYS = {
    "mean_recurrent_plastic_magnitude",
    "recurrent_rows_saturated_fraction",
    "recurrent_noise_rms",
    "mean_recurrent_learning_rate",
    "recurrent_learning_changes",
}


def native_probe(path):
    report = json.loads((path / "summary.json").read_text())
    assert report["completed"], path
    assert digest(path / "probe_native_recurrent.py") == report["script_sha256"]
    assert source_digest(path / "source.zip") == report["source_sha256"]
    assert digest(path / "genomes.pt") == report["genomes_sha256"]
    c = Config.from_dict(report["native_config"])
    assert c == Config.load(path / "config.toml")
    genomes = torch.load(path / "genomes.pt", weights_only=True)
    assert genomes.shape == (report["circuits"], c.parameter_count)
    assert torch.isfinite(genomes).all() and (genomes.abs() <= c.weight_limit).all()
    expected = {(seed, mode) for seed in report["seeds"] for mode in MODES}
    assert {(row["seed"], row["mode"]) for row in report["trials"]} == expected
    assert len(report["trials"]) == len(expected)
    for row in report["trials"]:
        seed, mode = row["seed"], row["mode"]
        saved = path / f"final-{seed}-{mode}.pt"
        assert digest(saved) == row["final_state_sha256"], saved
        state = torch.load(saved, weights_only=True)
        empty = initial_state(c, genomes)
        assert state.keys() == empty.keys()
        for key, value in state.items():
            assert value.shape == empty[key].shape and torch.isfinite(value).all(), key
            if key not in ("recurrent_plastic", "recurrent_baseline"):
                assert torch.equal(value, empty[key]), key
        lengths = state["recurrent_plastic"].norm(dim=-1)
        assert lengths.max() <= c.recurrent_learning_limit + 1e-12, saved
        assert row["completed"] and row["curve"][-1]["episode"] == report["episodes"]
        assert row["curve"][0]["episode"] == 0
        assert len({p["episode"] for p in row["curve"]}) == len(row["curve"])
        for point in row["curve"]:
            errors = torch.tensor(point["per_circuit_mse"], dtype=torch.float64)
            assert errors.shape == (report["circuits"],)
            assert torch.isfinite(errors).all() and (errors >= 0).all()
            assert errors.mean().item() == point["mse"]
            assert 0 <= point["saturated_row_fraction"] <= 1
            assert 0 <= point["mean_offset_norm"] <= c.recurrent_learning_limit + 1e-12
        cues = torch.tensor([-1.0, 1.0], dtype=torch.float64).repeat(16)
        cues = cues.expand(report["circuits"], -1)
        noise = torch.randn(
            24,
            report["circuits"],
            32,
            2,
            generator=torch.Generator().manual_seed(seed + 1009),
            dtype=torch.float64,
        )
        error = evaluate(c, genomes, state["recurrent_plastic"], cues, noise)
        assert error.tolist() == row["curve"][-1]["per_circuit_mse"]
        assert lengths.mean().item() == row["curve"][-1]["mean_offset_norm"]
        assert (lengths >= 0.999 * c.recurrent_learning_limit).double().mean().item() == row[
            "curve"
        ][-1]["saturated_row_fraction"]
        initial_error = evaluate(c, genomes, empty["recurrent_plastic"], cues, noise)
        assert initial_error.tolist() == row["curve"][0]["per_circuit_mse"]
        if mode == "no-updates":
            assert torch.equal(error, initial_error)
            assert not state["recurrent_plastic"].count_nonzero()
    comparisons = []
    for seed in report["seeds"]:
        trials = {r["mode"]: r for r in report["trials"] if r["seed"] == seed}
        initial = [r["curve"][0]["per_circuit_mse"] for r in trials.values()]
        assert initial[0] == initial[1] == initial[2]
        final = {mode: r["curve"][-1]["mse"] for mode, r in trials.items()}
        comparisons.append(
            dict(
                seed=seed,
                final_mse=final,
                learning_reduction=1 - final["learning"] / final["no-updates"],
            )
        )
    return dict(
        **report,
        path=str(path),
        report_sha256=digest(path / "summary.json"),
        exact_endpoint_validation_replay=True,
        comparisons=comparisons,
    )


def recurrent_run(path, duration):
    result = checked(path, duration)
    w = load_checkpoint(path / "latest.pt")
    c, a = w.config, w.agents
    assert c.ecology_version == 25, path
    _, rows, events = read_history(path)
    nodes, _, edges, _ = effective_masks(c, a["genome"])
    modules = a["module_mask"]
    for key, shape in recurrent_shapes(c.hidden_size).items():
        value = a[f"module_{key}"]
        assert value.shape == (*modules.shape, *shape), (path, key)
        assert torch.isfinite(value).all(), (path, key)
        assert not value[~modules].count_nonzero(), (path, key)
        if not c.recurrent_noise_sigma:
            assert not value.count_nonzero(), (path, key)
    for key in ("recurrent_plastic", "recurrent_trace"):
        assert not a[f"module_{key}"][~(modules[:, :, None, None] & edges[:, None])].count_nonzero()
        if w.ablation == "no_recurrent_learning":
            assert not a[f"module_{key}"].count_nonzero(), path
    assert not a["module_recurrent_applied_noise"][
        ~(modules[:, :, None] & nodes[:, None])
    ].count_nonzero()
    if w.population:
        assert a["module_recurrent_plastic"].norm(dim=-1).max() <= c.recurrent_learning_limit + 1e-6
    if not c.recurrent_noise_sigma or w.ablation == "no_recurrent_learning":
        assert w.totals["recurrent_learning_changes"] == 0, path
    else:
        assert w.totals["recurrent_learning_changes"] > 0, path
    times = response_times(c, a["genome"], a["memory_tau"])
    genes = timing_genes(c, a["genome"])
    assert torch.isfinite(times).all() and (times > 0).all(), path
    assert torch.isfinite(genes).all() and (genes.abs() <= c.weight_limit).all(), path
    metrics = recurrent_metrics(c, a) | timing_metrics(c, a["genome"], a["memory_tau"], nodes)
    metrics["recurrent_learning_changes"] = w.totals["recurrent_learning_changes"]
    lookup = {r["tick"]: r for r in rows}
    for key, value in metrics.items():
        assert value == rows[-1][key], (path, key)
        result["final"][key] = value
        for point in result["checkpoints"]:
            point[key] = lookup[point["tick"]][key]
    for segment in result["segments"]:
        assert (
            source_digest(Path(segment["path"]) / "source.zip")
            == segment["metadata"]["source_sha256"]
        )
    assert rows[0]["time"] == 0
    founders = len(w.founders)
    deaths = {
        int(e["id"]): e["time"] for e in events if e["event"] == "death" and e["id"] < founders
    }
    survivors = {int(i): w.time for i in a["id"] if i < founders}
    assert not deaths.keys() & survivors.keys()
    assert deaths.keys() | survivors.keys() == set(range(founders))
    lifetimes = sorted((deaths | survivors).values())
    result["founder_survival"] = dict(
        interpretation="The original matched founder cohort only, not independent replicates. "
        "Living founders are right-censored at the run endpoint. Median is the time when "
        "at least half this cohort has died, if observed. Event times avoid accumulated "
        "float32 age rounding. This is not a causal test of learning speed.",
        cohort_size=founders,
        alive_at_end=len(survivors),
        median_seconds=lifetimes[(founders - 1) // 2]
        if len(deaths) >= (founders + 1) // 2
        else None,
        survived_at_least={
            str(t): sum(age >= t for age in lifetimes) / founders
            for t in (30, 60, 120, 180)
            if t <= w.time
        },
    )
    result["maximum_recorded_recurrent_saturation"] = max(
        row["recurrent_rows_saturated_fraction"] for row in rows
    )
    result["checkpoint_sha256"] = digest(path / "latest.pt")
    return result


def neutral_parity(old, new):
    a, b = (asdict(Config.load(path / "config.toml")) for path in (old, new))
    assert a.pop("ecology_version") == 24
    assert b.pop("ecology_version") == 25
    assert a == b
    _, old_rows, old_events = read_history(old)
    _, new_rows, new_events = read_history(new)
    assert physical(old_rows) == physical(
        [
            {key: value for key, value in row.items() if key not in RECURRENT_KEYS}
            for row in new_rows
        ]
    ), (old, new)
    # No new genes in V25: whole-genome digests and genetic variance must match too.
    assert old_events == new_events, (old, new)
    a, b = (load_checkpoint(path / "latest.pt").state_dict() for path in (old, new))
    a.pop("config")
    c = b.pop("config")
    b["rng"].pop("recurrent_exploration")
    assert b["totals"].pop("recurrent_learning_changes") == 0
    for key in recurrent_shapes(c["hidden_size"]):
        assert not b["agents"].pop(f"module_{key}").count_nonzero()
    same_state(a, b)
    return dict(
        old=str(old),
        new=str(new),
        measurements=len(new_rows),
        events=len(new_events),
        exact_common_state=True,
        exact_common_measurements=True,
        exact_events=True,
    )


def inert_rate_parity(old, new):
    """A disabled learner's configured rate must not alter the entire 600-second run."""
    before, after = (asdict(Config.load(path / "config.toml")) for path in (old, new))
    assert before.pop("recurrent_learning_rate") == 0.001
    assert after.pop("recurrent_learning_rate") == 0.001 / 3
    assert before == after
    _, left, events_a = read_history(old)
    _, right, events_b = read_history(new)

    def measurements(rows):
        return physical(
            [{k: v for k, v in row.items() if k != "mean_recurrent_learning_rate"} for row in rows]
        )

    assert measurements(left) == measurements(right), (old, new)
    assert events_a == events_b, (old, new)
    a, b = (load_checkpoint(path / "latest.pt").state_dict() for path in (old, new))
    assert a["ablation"] == b["ablation"] == "no_recurrent_learning"
    a.pop("config")
    b.pop("config")
    same_state(a, b)
    return dict(
        reference=str(old),
        replay=str(new),
        measurements=len(right),
        events=len(events_b),
        exact_common_state=True,
        exact_common_measurements=True,
        exact_events=True,
        reference_checkpoint_sha256=digest(old / "latest.pt"),
        replay_checkpoint_sha256=digest(new / "latest.pt"),
    )


def control_replays():
    trials = []
    for seed in (1, 2, 3):
        old = Path(f"runs/v25-quiet-noise-only-pilot/seed-{seed}")
        new = Path(f"runs/v25-quiet-gentle-control-replay-pilot/seed-{seed}")
        row = recurrent_run(new, 600)
        row.update(
            seed=seed, parity=inert_rate_parity(old, new), independent_ecological_replicate=False
        )
        trials.append(row)
    report = dict(
        interpretation="Three full 600-second replays lower only the disabled recurrent "
        "learner's configured maximum rate from .001 to .001/3. Every physical measurement, "
        "event, genotype, and complete final state must match the original quiet noise-only "
        "control. Only configuration and potential-rate telemetry are excluded. These are "
        "verification replays, not additional ecological replicates.",
        trials=trials,
    )
    output = Path("docs/results/v25-gentle-control-replays.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified three full control replays: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe-only", action="store_true")
    mode.add_argument(
        "--quiet",
        action="store_true",
        help="Audit nine quieter pilots and the three reused baselines",
    )
    mode.add_argument(
        "--gentle",
        action="store_true",
        help="Audit six gentler pilots, six reused references, and three replay checks",
    )
    mode.add_argument("--control-replays-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.control_replays_only:
        control_replays()
        return
    probe = native_probe(Path("runs/v25-native-recurrent-probe"))
    Path("docs/results/v25-native-recurrent-probe.json").write_text(
        json.dumps(probe, indent=2) + "\n"
    )
    print("Audited nine native delayed-task runs and exact endpoint validation replay")
    if args.probe_only:
        return
    quieter = args.quiet or args.gentle
    groups = {
        group: ("v25-quiet" if quieter and group != "baseline" else preset, mode)
        for group, (preset, mode) in GROUPS.items()
    }
    prefix = "v25-quiet" if quieter else "v25"
    replay_checks = None
    if args.gentle:
        prefix = "v25-quiet-gentle"
        for group in ("learning", "shuffled"):
            groups[group] = (prefix, groups[group][1])
        replay_checks = control_replays()
    sigma = 0.05 if quieter else 0.15
    configurations = {
        group: asdict(Config.load(f"configs/{preset}.toml"))
        for group, (preset, _) in groups.items()
    }
    shared = []
    for group, configured in configurations.items():
        values = configured.copy()
        assert values.pop("recurrent_noise_sigma") == (0 if group == "baseline" else sigma)
        rate = 0 if group == "baseline" else 0.001
        if args.gentle and group in ("learning", "shuffled"):
            rate /= 3
        assert values.pop("recurrent_learning_rate") == rate
        shared.append(values)
    assert all(values == shared[0] for values in shared)
    trials, compatibility = [], []
    for seed in (1, 2, 3):
        old = Path(f"runs/v24-inherited-pilot/seed-{seed}")
        reference = torch.load(old / "founders.pt", weights_only=True)["genomes"]
        for group, (_, mode) in groups.items():
            root = "v25" if group == "baseline" else prefix
            if args.gentle and group == "noise-only":
                root = "v25-quiet"
            path = Path(f"runs/{root}-{group}-pilot/seed-{seed}")
            assert asdict(Config.load(path / "config.toml")) == configurations[group], path
            founders = torch.load(path / "founders.pt", weights_only=True)["genomes"]
            assert torch.equal(founders, reference), path
            row = recurrent_run(path, 600)
            assert row["segments"][0]["metadata"]["ablation"] == mode, path
            row.update(treatment=group, seed=seed, matched_complete_founders=True)
            if quieter:
                row["reused_reference"] = group == "baseline" or (
                    args.gentle and group == "noise-only"
                )
            trials.append(row)
        compatibility.append(neutral_parity(old, Path(f"runs/v25-baseline-pilot/seed-{seed}")))
    comparisons = []
    controls = ("noise-only", "shuffled", "baseline") if quieter else ("noise-only", "shuffled")
    for control in controls:
        pairs = []
        for seed in (1, 2, 3):
            by_mode = {r["treatment"]: r["final"] for r in trials if r["seed"] == seed}
            delta = {
                key: by_mode["learning"][key] - by_mode[control][key]
                for key in ("births", "fresh_absorbed", "population")
            }
            pairs.append(
                dict(
                    seed=seed,
                    differences=delta,
                    both_improved=delta["births"] > 0 and delta["fresh_absorbed"] > 0,
                )
            )
        comparisons.append(
            dict(control=control, pairs=pairs, passed=sum(p["both_improved"] for p in pairs) >= 2)
        )
    report = dict(
        interpretation="Three matched founder starts compare disabled recurrent learning/noise, "
        "hidden noise only, own-return learning, and shuffled recurrent returns. The older "
        "motor/local learning, heterogeneous inherited timing, ecology, and capacity charges "
        "are retained. No new energy charge is added for the acquired recurrent matrices. "
        "These evolving communities are a screen, not fixed-genotype evidence of useful "
        "learning. The prospective continuation rule requires births AND fresh absorption "
        "to improve in at least two matched starts against each listed control. Neutral "
        "compatibility excludes only version/configuration, the new RNG, four zero acquired "
        "states, four new metrics, and one zero cumulative counter. Genomes and event digests "
        "must match exactly. Extinctions remain in all comparisons.",
        configurations=configurations,
        trials=trials,
        neutral_compatibility=compatibility,
        comparisons=comparisons,
        continuation_screen_passed=all(r["passed"] for r in comparisons),
    )
    if args.quiet:
        report["interpretation"] += (
            " This quieter screen changes only hidden sigma to .05, retaining the .001 maximum "
            "rate. Its likelihood scores consequently scale differently. The three mechanism-off "
            "references are reused from the original V25 screen, not new runs. The prospective "
            "continuation criterion additionally requires beating that reference in two starts."
        )
        report["new_runs"] = 9
        report["reused_references"] = 3
    if args.gentle:
        report["interpretation"] += (
            " This gentler screen keeps sigma .05 and lowers maximum rate to .001/3. "
            "Six own-return/shuffled trials are new; the three quiet noise-only controls and "
            "three original mechanism-off baselines are reused. Three full replays verify "
            "that the noise-only reference's unused higher rate has no physical effect. "
            "The continuation criterion must also beat the mechanism-off baseline in two starts."
        )
        report.update(new_runs=6, reused_references=6, control_replay_checks=replay_checks)
    output = Path(f"docs/results/{prefix}-recurrent-learning.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    label = "9 new pilots and 3 reused baselines" if args.quiet else f"{len(trials)} pilots"
    if args.gentle:
        label = "6 new pilots, 6 reused references, and 3 verification replays"
    print(f"Audited {label}, {len(compatibility)} exact neutral comparisons: {output}")
    print(f"Continuation screen passed: {report['continuation_screen_passed']}")


if __name__ == "__main__":
    main()
