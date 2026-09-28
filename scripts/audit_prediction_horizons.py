"""Audit passive short/long predictions on identical physical experience."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from audit_circuits import same_state
from audit_sensory_values import paired_forecasts
from audit_value_prediction import checked

from emergent_garden.config import Config


def native_pilots():
    groups = {
        "long": ("v23-sensory-pilot", "v23", "none"),
        "long-shuffled": ("v23-shuffled-pilot", "v23", "shuffled_motor_reward"),
        "short": ("v23-short-pilot", "v23-short", "none"),
        "short-shuffled": ("v23-short-shuffled-pilot", "v23-short", "shuffled_motor_reward"),
    }
    configs = {name: asdict(Config.load(f"configs/{preset}.toml"))
               for name, (_, preset, _) in groups.items()}
    a, b = configs["long"].copy(), configs["short"].copy()
    assert a.pop("motor_value_horizon") == 20 and b.pop("motor_value_horizon") == 2
    assert a == b
    trials = []
    for seed in (1, 2, 3):
        reference = torch.load(f"runs/v23-sensory-pilot/seed-{seed}/founders.pt",
                               weights_only=True)["genomes"]
        for group, (prefix, _, mode) in groups.items():
            path = Path(f"runs/{prefix}/seed-{seed}")
            assert asdict(Config.load(path / "config.toml")) == configs[group]
            assert torch.equal(reference, torch.load(path / "founders.pt",
                                                     weights_only=True)["genomes"])
            row = checked(path, 600)
            assert row["segments"][0]["metadata"]["ablation"] == mode
            row.update(treatment=group, seed=seed, matched_founders=True)
            trials.append(row)
    return dict(
        interpretation="Matched native V23 sensory predictors compare 20-second and two-second "
        "discount horizons, each with its own shuffled-return control. All other settings "
        "and founder genomes match. The six long-horizon runs are the original V23 pilots, "
        "not additional replicates. Evolution, ecological feedback, and selection continue; "
        "these are not fixed-genotype fitness comparisons.",
        configurations=configs, trials=trials,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long", type=Path,
                        default=Path("runs/v23-paired-forecasts/summary.json"))
    parser.add_argument("--short", type=Path,
                        default=Path("runs/v23-paired-short-forecasts/summary.json"))
    parser.add_argument("--native", action="store_true", help="Also audit native ecological pilots")
    args = parser.parse_args()
    torch.set_num_threads(1)
    long, short = paired_forecasts(args.long), paired_forecasts(args.short)
    assert long["prediction_rate"] == short["prediction_rate"] == 0.02
    assert len(long["trials"]) == len(short["trials"]) == 3
    for old, new in zip(long["trials"], short["trials"], strict=True):
        assert old["source_checkpoint_sha256"] == new["source_checkpoint_sha256"]
        assert old["prediction_horizon"] == 20 and new["prediction_horizon"] == 2
        assert (old["warmup_seconds"], old["forecast_seconds"], old["forecast_start"]) == (
            new["warmup_seconds"], new["forecast_seconds"], new["forecast_start"]
        )
        for mode in ("hidden", "sensory"):
            for group in ("all_adults", "adults_at_least_30_seconds_old"):
                a, b = (r["predictors"][mode][group] for r in (old, new))
                assert a["count"] == b["count"] and a["deaths"] == b["deaths"]
        for name in ("forecast-start.pt", "end.pt"):
            a, b = (Path(r["forecast_checkpoint"]).parent / name for r in (old, new))
            same_state(torch.load(a, weights_only=True), torch.load(b, weights_only=True))
    report = dict(
        interpretation="The predictors use the same physical trajectories and cohorts under "
        "two discount horizons. Native motor control, learning, genes, and ecology do not "
        "change. Each head is scored against returns under its own trained horizon. "
        "Absolute MSE across horizons measures different targets and cannot be ranked as "
        "if it measured the same prediction task. Relative errors use a matched zero or "
        "recent-return-rate predictor. These are three shared communities, not independent "
        "creature-level replicates or evidence of useful motor adaptation.",
        exact_physical_state_parity=True, long=long, short=short,
    )
    output = Path("docs/results/prediction-horizons.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited three paired prediction-horizon comparisons: {output}")
    if args.native:
        report = native_pilots()
        output = Path("docs/results/v23-short-pilots.json")
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Audited {len(report['trials'])} native comparisons: {output}")


if __name__ == "__main__":
    main()
