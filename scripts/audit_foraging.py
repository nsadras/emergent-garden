"""Audit the September 23 foraging pilots and export compact, reproducible evidence.

Run from the repository root with uv run python scripts/audit_foraging.py.
Large checkpoints and exact source archives stay under runs/.
"""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from emergent_garden.config import Config
from emergent_garden.history import PERFORMANCE_KEYS, read_history
from emergent_garden.storage import load_checkpoint

GROUPS = (
    "v16-user-baseline",
    "v16-dispersed",
    "v16-compact",
    "v16-fine-food",
    "v16-compact-fine-food",
    "v16-fast-feeding",
    "v17-absolute",
    "v17-contrast",
)
MEASUREMENTS = (
    "time",
    "tick",
    "population",
    "births",
    "deaths",
    "generation_max",
    "lineages",
    "fresh_absorbed",
    "detritus_absorbed",
    "predation_absorbed",
    "food_spawned",
    "food_expired",
    "food_energy",
    "fresh_processed",
    "detritus_processed",
    "energy_balance_error",
    "fertility_balance_error",
    "trophic_detritus_balance_error",
    "mean_diet",
    "grazers",
    "scavengers",
    "generalists",
    "module_histogram",
    "distance",
)


def select(row):
    return {key: row[key] for key in MEASUREMENTS}


def audit_run(path, duration, follow_resumes=False):
    segments, rows, events = read_history(path, follow_resumes)
    final = rows[-1]
    assert final["time"] == duration or final["population"] == 0, path
    world = load_checkpoint(path / "latest.pt")
    measured = world.metrics()
    for key in MEASUREMENTS:
        assert measured[key] == final[key], (path, key)
    assert world.totals["births"] == sum(e["event"] == "birth" for e in events), path
    assert world.totals["deaths"] == sum(e["event"] == "death" for e in events), path
    assert (
        abs(final["energy_balance_error"]) / (world.initial_energy + final["food_spawned"]) < 1e-6
    ), path
    assert max(abs(r["fertility_balance_error"]) for r in rows) < 1e-7, path
    assert max(abs(x) for r in rows for x in r["trophic_detritus_balance_error"]) < 1e-6, path
    births = [e for e in events if e["event"] == "birth"]
    return dict(
        path=str(path),
        complete=True,
        segments=[dict(path=str(s["path"]), metadata=s["metadata"]) for s in segments],
        final=select(final),
        first_birth=births[0]["time"] if births else None,
        checkpoints=[select(r) for r in rows if r["time"] % 60 == 0],
        max_absolute_energy_residual=max(abs(r["energy_balance_error"]) for r in rows),
    )


def main():
    torch.set_num_threads(1)
    baseline = asdict(Config.load("configs/v16-user-baseline.toml"))
    treatments, trials = {}, []
    for group in GROUPS:
        config = asdict(Config.load(f"configs/{group}.toml"))
        treatments[group] = {k: v for k, v in config.items() if baseline[k] != v}
        for seed in (1, 2, 3):
            path = Path(f"runs/{group}-pilot/seed-{seed}")
            actual = asdict(Config.load(path / "config.toml"))
            assert actual == config, path
            founders = torch.load(path / "founders.pt", weights_only=True)["genomes"]
            reference = torch.load(
                f"runs/v16-user-baseline-pilot/seed-{seed}/founders.pt", weights_only=True
            )["genomes"]
            assert torch.equal(founders, reference), path
            trial = audit_run(path, 600)
            trial.update(group=group, seed=seed, matched_founders=True)
            trials.append(trial)
    parity = []
    for seed in (1, 2, 3):
        _, a, _ = read_history(f"runs/v16-compact-pilot/seed-{seed}")
        _, b, _ = read_history(f"runs/v17-absolute-pilot/seed-{seed}")

        def physical(rows):
            return [
                {k: v for k, v in r.items() if k not in PERFORMANCE_KEYS | {"ecology_version"}}
                for r in rows
            ]

        assert physical(a) == physical(b), seed
        parity.append(dict(seed=seed, measurements=len(a), exactly_equal=True))
    long = [audit_run(Path(f"runs/v16-fast-feeding-long-{seed}"), 1800, True) for seed in (1, 2, 3)]
    report = dict(
        interpretation="Matched inherited founders across seeds; world and input laws vary. "
        "Three seeds per pilot are screening evidence, not a general advantage. "
        "Smaller food also increases odor per unit energy. Dispersal changes realized "
        "production because background proposals bypass source stock caps. "
        "Longer trials continue all three fast-feeding starts, not independent replicates.",
        baseline=baseline,
        changes=treatments,
        trials=trials,
        legacy_encoding_parity=parity,
        fast_feeding_continuations=long,
    )
    output = Path("docs/results/foraging-pilots.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {len(trials)} pilots and {len(long)} continuations: {output}")
    for row in long:
        print(row["path"], row["final"]["time"], row["final"]["population"], row["final"]["births"])


if __name__ == "__main__":
    main()
