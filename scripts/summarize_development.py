"""Reconstruct V13 body stages and parenthood from complete recorded histories.

These descriptive counts measure access to body plans and reproduction. They do
not assign fitness to module counts or treat relatives as independent trials.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from emergent_garden.history import read_history
from emergent_garden.morphology import module_count
from emergent_garden.storage import load_checkpoint


def summarize(path, follow_resumes=False):
    segments, rows, events = read_history(path, follow_resumes)
    if rows[0]["tick"] != 0:
        raise ValueError(f"Need the full history from initialization: {path}")
    w = load_checkpoint(path / "latest.pt")
    if w.config.ecology_version < 13 or w.time != rows[-1]["time"]:
        raise ValueError(f"Need a consistent V13+ final checkpoint: {path}")
    plans = module_count(w.founders[:, w.config.brain_parameter_count + 6].sigmoid()).tolist()
    stage = {i: 1 for i in range(len(plans))}
    plan = dict(enumerate(plans))
    birth_plans, parent_modules, growth_to = Counter(), Counter(), Counter()
    reproductive_parents = {k: set() for k in (1, 2, 3)}
    grown, energy, explicit_events = set(), 0.0, 0
    first_larger_parent = None
    for e in events:
        if e["event"] == "growth":
            identifier = e["id"]
            assert stage[identifier] == e["from_modules"]
            assert e["modules"] == stage[identifier] + 1 <= plan[identifier]
            stage[identifier] = e["modules"]
            growth_to[e["modules"]] += 1
            grown.add(identifier)
            energy += e["energy_cost"]
        elif e["event"] == "birth":
            parent = e["parent"]
            assert stage[parent] == plan[parent], "An immature body reproduced"
            parent_modules[stage[parent]] += 1
            reproductive_parents[stage[parent]].add(parent)
            if stage[parent] > 1 and first_larger_parent is None:
                first_larger_parent = dict(time=e["time"], parent=parent, child=e["id"])
            assert e["id"] not in stage
            stage[e["id"]], plan[e["id"]] = e["modules"], e["target_modules"]
            birth_plans[e["target_modules"]] += 1
            explicit_events += int(e["module_event"])
        elif e["event"] == "death":
            assert stage.pop(e["id"]) == e["modules"]
            assert plan[e["id"]] == e["target_modules"]
    assert len(stage) == w.population
    assert sum(parent_modules.values()) == w.totals["births"]
    assert sum(growth_to.values()) == w.totals["growths"]
    assert explicit_events == w.totals["module_event_births"]
    assert energy == w.totals["development_cost"]
    for i, identifier in enumerate(w.agents["id"].tolist()):
        assert stage[identifier] == int(w.agents["modules"][i])
        assert plan[identifier] == int(w.agents["target_modules"][i])
    return dict(
        path=str(path),
        history_segments=[str(s["path"]) for s in segments],
        time=w.time,
        ablation=w.ablation,
        population=w.population,
        births=w.totals["births"],
        founder_plans=[plans.count(k) for k in (1, 2, 3)],
        birth_plans=[birth_plans[k] for k in (1, 2, 3)],
        births_by_parent_modules=[parent_modules[k] for k in (1, 2, 3)],
        distinct_reproductive_parents=[len(reproductive_parents[k]) for k in (1, 2, 3)],
        growth_to_modules=[growth_to[k] for k in (1, 2, 3)],
        distinct_grown_creatures=len(grown),
        development_energy=energy,
        living_module_histogram=rows[-1]["module_histogram"],
        living_target_histogram=rows[-1]["target_module_histogram"],
        first_larger_parent=first_larger_parent,
        event_reconstruction_passed=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--follow-resumes", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = []
    for root in args.runs:
        paths = [root] if (root / "metrics.jsonl").exists() else sorted(root.glob("seed-*"))
        for path in paths:
            records.append(summarize(path, args.follow_resumes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(interpretation=__doc__, runs=records), indent=2) + "\n")
    print(f"Reconstructed {len(records)} complete developmental histories: {args.output}")


if __name__ == "__main__":
    main()
