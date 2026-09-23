"""Check affordability of a changed body-module count without changing other genes.

This counterfactual audit does not mutate or evolve the source population. It
tests the developmental and birth-cost laws, including neural construction,
for a parent at its reproduction threshold or maximum energy store.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from emergent_garden.storage import SOURCE_SHA256, load_checkpoint


def audit(path):
    checkpoint = path / "latest.pt" if path.is_dir() else path
    world = load_checkpoint(checkpoint)
    c, a = world.config, world.agents
    if c.ecology_version < 4 or not world.population:
        raise ValueError("Need a living population with developmental modules")
    rows = []
    for source_count in (1, 2, 3):
        parents = a["modules"] == source_count
        if not parents.any():
            continue
        for target_count in (1, 2, 3):
            if target_count == source_count:
                continue
            child = world.empty_agents(int(parents.sum()))
            child["genome"] = a["genome"][parents].clone()
            p = (target_count - 0.5) / 3
            child["genome"][:, c.brain_parameter_count + 6] = math.log(p / (1 - p))
            if c.ecology_version >= 13:
                # This audit asks about a fully formed child, not its juvenile.
                child["development_stage"].fill_(target_count)
            world.develop(child)
            assert (child["modules"] == target_count).all()
            torch.testing.assert_close(
                child["core_radius"], a["core_radius"][parents], rtol=0, atol=0
            )
            neural = child.get("brain_construction", torch.zeros_like(child["area"]))
            debit = c.reproduction_debit * child["area"] + neural
            threshold = c.reproduction_threshold * a["area"][parents]
            maximum = c.max_energy * a["area"][parents]
            ratio = debit / maximum
            rows.append(
                dict(
                    from_modules=source_count,
                    to_modules=target_count,
                    parent_count=len(debit),
                    affordable_at_threshold=int((debit < threshold).sum()),
                    affordable_at_full_energy=int((debit < maximum).sum()),
                    debit_over_parent_maximum=dict(
                        minimum=ratio.min().item(),
                        median=ratio.median().item(),
                        maximum=ratio.max().item(),
                    ),
                    source_ids=a["id"][parents].tolist(),
                    parent_maximum_energy=maximum.tolist(),
                    child_debit=debit.tolist(),
                )
            )
    return dict(
        source=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        time=world.time,
        ecology_version=c.ecology_version,
        changes=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = dict(
        source_sha256=SOURCE_SHA256,
        experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation=__doc__ + " Combined mutations to core size, spacing or neural structure "
        "may alter affordability; this audit holds those traits fixed.",
        populations=[audit(path) for path in args.runs],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for population in report["populations"]:
        for row in population["changes"]:
            print(
                population["source"],
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("source_ids", "parent_maximum_energy", "child_debit")
                },
            )


if __name__ == "__main__":
    main()
