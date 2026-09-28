"""Paired interventions on the acquired state of the same living organisms."""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from .ecology import EcologyWorld
from .storage import RunStore, load_checkpoint, save_checkpoint


def fork_challenge(source, mode, reverse, target_tick):
    if source.config.ecology_version < 7 or source.controller != "neural":
        raise ValueError("Acquired-state challenges require a V7+ neural population")
    if source.ablation != "none":
        raise ValueError("Use an unablated population as the challenge source")
    if mode not in ("intact", "erase_plastic", "erase_activity", "no_plasticity"):
        raise ValueError(f"Unsupported challenge intervention: {mode}")
    w = EcologyWorld.from_state(source.state_dict(), source.device)
    w.config = replace(
        w.config,
        mutation_probability=0.0,
        trait_mutation_probability=0.0,
        node_mutation_probability=0.0,
        edge_mutation_probability=0.0,
        module_mutation_probability=0.0,
        timing_mutation_probability=0.0,
    )
    if mode in ("erase_plastic", "no_plasticity"):
        w.agents["module_plastic"].zero_()
        w.agents["module_trace"].zero_()
        if w.config.ecology_version >= 12:
            for key in ("module_motor_plastic", "module_motor_trace", "module_motor_baseline"):
                w.agents[key].zero_()
        if w.config.ecology_version >= 22:
            from .value import value_shapes

            for key in value_shapes(w.config.hidden_size):
                w.agents[f"module_{key}"].zero_()
        if w.config.ecology_version >= 25:
            from .recurrent_learning import recurrent_shapes

            for key in recurrent_shapes(w.config.hidden_size):
                w.agents[f"module_{key}"].zero_()
    if mode == "erase_activity":
        w.agents["module_h"].zero_()
        w.agents["h"].zero_()
        if w.config.ecology_version >= 14:
            w.agents["module_internal"].zero_()
    if mode == "no_plasticity":
        w.ablation = "no_plasticity"
    # The challenge has a single controlled quality transition. Hide subsequent
    # spontaneous reversals beyond the readout window for both paired conditions.
    w.landscape.next_tick = target_tick + 1
    if reverse:
        w.landscape.favorable = 1 - w.landscape.favorable
        w.totals["quality_reversals"] += 1
    w.events.append(dict(event="challenge_start", time=w.time, intervention=mode, reversal=reverse))
    return w


def challenge_run(run, output, seconds=180.0, device="cpu", stop=None):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Challenge duration must be finite and positive")
    run, output = Path(run), Path(output)
    checkpoint = run / "latest.pt" if run.is_dir() else run
    source = load_checkpoint(checkpoint, device)
    if not source.population:
        raise ValueError("Challenge requires a living population")
    target = source.tick + math.ceil(seconds * source.config.physics_hz)
    fork_challenge(source, "intact", False, target)  # Validate before creating artifacts.
    output.mkdir(parents=True, exist_ok=False)
    initial = {
        int(identifier): source.agent_record(i) for i, identifier in enumerate(source.agents["id"])
    }
    report = dict(
        source=str(checkpoint),
        source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        initial_time=source.time,
        initial_population=source.population,
        requested_duration=seconds,
        completed=False,
        trials=[],
        interpretation="Clones of one living community share genotypes, bodies, ages, energy, "
        "positions, food, fields, and random states. Mutation is disabled. Only the specified "
        "acquired state and hidden food quality are changed; spontaneous reversals are held "
        "until after the window. Erasure can disrupt ordinary dynamics. A performance change "
        "shows dependence on state, not by itself adaptive learning or generalization. "
        "These branches are paired interventions, not independent evolutionary replicates.",
    )
    keys = (
        "acquired",
        "fresh_acquired",
        "high_acquired",
        "low_acquired",
        "spent",
        "offspring",
        "distance",
    )
    for reverse in (False, True):
        for mode in ("intact", "erase_plastic", "erase_activity", "no_plasticity"):
            if stop is not None and stop.requested:
                (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                return report
            w = fork_challenge(source, mode, reverse, target)
            label = f"{'reversed' if reverse else 'unchanged'}-{mode}"
            store = RunStore(output / label, w, checkpoint)
            save_checkpoint(w, store.path / "start.pt")
            start_metric = store.measure(w)
            ended = {}
            try:
                while w.population and w.tick < target and not (stop and stop.requested):
                    w.step(min(w.config.physics_hz, target - w.tick))
                    for event in w.events:
                        if event["event"] == "death" and event["id"] in initial:
                            ended[event["id"]] = event
                    if (w.tick - source.tick) % (5 * w.config.physics_hz) == 0:
                        store.measure(w)
                survivors = set()
                for i, identifier in enumerate(w.agents["id"].tolist()):
                    if identifier in initial:
                        survivors.add(identifier)
                        ended[identifier] = w.agent_record(i)
                if ended.keys() != initial.keys():
                    raise RuntimeError("Challenge lost an original organism's record")
                cohort = [
                    dict(
                        id=identifier,
                        survived=identifier in survivors,
                        **{key: ended[identifier][key] - before[key] for key in keys},
                    )
                    for identifier, before in initial.items()
                ]
                final = store.measure(w)
                row = dict(
                    intervention=mode,
                    reversal=reverse,
                    duration=w.time - source.time,
                    completed=w.tick >= target or not w.population,
                    population=w.population,
                    cohort_survivors=len(survivors),
                    cohort_totals={key: sum(agent[key] for agent in cohort) for key in keys},
                    cohort=cohort,
                    community_gain={key: final[key] - start_metric[key] for key in source.totals},
                )
                report["trials"].append(row)
                (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"challenge {label}: cohort survivors={len(survivors)}/{source.population}, "
                    f"cohort food={row['cohort_totals']['acquired']:.1f}",
                    flush=True,
                )
            finally:
                store.checkpoint(w)
                store.close()
    report["completed"] = all(row["completed"] for row in report["trials"])
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
