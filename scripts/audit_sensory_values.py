"""Audit V23's matched pilots and passive prediction comparisons."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from audit_circuits import physical, same_state
from audit_value_forecasts import digest, source_digest
from audit_value_prediction import checked
from probe_value_forecasts import describe

from emergent_garden.config import Config
from emergent_garden.history import read_history
from emergent_garden.storage import load_checkpoint


def paired_forecasts(path):
    report = json.loads(path.read_text())
    assert report["completed"] and report["trials"][0]["exactly_passive"], path
    assert source_digest(path.parent / "source.zip") == report["source_sha256"], path
    for name, expected in report["scripts_sha256"].items():
        assert digest(path.parent / name) == expected, name
    records = []
    for trial in report["trials"]:
        source = Path(trial["source"]) / "latest.pt"
        assert digest(source) == trial["source_checkpoint_sha256"], source
        checkpoint = Path(trial["forecast_checkpoint"])
        w = load_checkpoint(checkpoint)
        assert w.time == trial["forecast_start"], checkpoint
        assert trial["warmup_completed"], checkpoint
        a, c = w.agents, w.config
        assert c.motor_value_centered == 0 and w.ablation == "none", checkpoint
        horizon = trial.get("prediction_horizon", c.motor_value_horizon)
        prediction_config = replace(c, motor_value_horizon=horizon)
        adults = (a["modules"] == a["target_modules"]).nonzero().flatten().tolist()
        values = torch.load(checkpoint.parent / "end-values.pt", weights_only=True)
        record = {k: v for k, v in trial.items() if k != "predictors"}
        record["prediction_horizon"] = horizon
        record["predictors"] = {}
        observed = []
        for mode, result in trial["predictors"].items():
            rows = result["rows"]
            assert len(rows) == result["mature_cohort"] == len(adults), checkpoint
            lookup = {r["id"]: r for r in rows}
            assert lookup.keys() == {int(a["id"][i]) for i in adults}, checkpoint
            for i in adults:
                row = lookup[int(a["id"][i])]
                assert row["age"] == float(a["age"][i]), checkpoint
                assert row["area"] == float(a["area"][i]), checkpoint
            selected = [r for r in rows if r["prediction"] is not None]
            assert len(rows) - len(selected) == result["died_before_prediction"], checkpoint
            for group, cohort in (
                ("all_adults", selected),
                ("adults_at_least_30_seconds_old", [r for r in selected if r["age"] >= 30]),
            ):
                assert result[group] == describe(cohort, prediction_config,
                                                 trial["forecast_seconds"]), mode
            norm = values[mode]["motor_value_weights"].norm(dim=-1).max().item()
            assert norm == result["maximum_norm"] and norm <= c.motor_value_limit + 1e-6, mode
            for value in values[mode].values():
                assert torch.isfinite(value).all(), mode
            observed.append([
                tuple(r[k] for k in ("id", "age", "area", "start", "discounted_return", "died_at"))
                for r in rows
            ])
            record["predictors"][mode] = {k: v for k, v in result.items() if k != "rows"}
        assert observed[0] == observed[1], checkpoint
        records.append(record)
    return dict(
        interpretation=report["interpretation"], report=str(path), report_sha256=digest(path),
        source_sha256=report["source_sha256"], scripts_sha256=report["scripts_sha256"],
        prediction_rate=report["prediction_rate"], trials=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path)
    parser.add_argument("--paired-fast", type=Path)
    args = parser.parse_args()
    if args.paired_fast and not args.paired:
        parser.error("The faster-rate comparison also requires the original paired report")
    torch.set_num_threads(1)
    groups = {
        "hidden": ("v23-baseline", "none"),
        "sensory": ("v23", "none"),
        "shuffled": ("v23", "shuffled_motor_reward"),
    }
    configs = {key: asdict(Config.load(f"configs/{preset}.toml"))
               for key, (preset, _) in groups.items()}
    a, b = configs["hidden"].copy(), configs["sensory"].copy()
    assert a.pop("motor_value_inputs") == 0
    assert b.pop("motor_value_inputs") == 1
    assert a == b
    trials, parity = [], []
    for seed in (1, 2, 3):
        old = Path(f"runs/v22-raw-pilot/seed-{seed}")
        reference = torch.load(old / "founders.pt", weights_only=True)["genomes"]
        for group, (_, ablation) in groups.items():
            path = Path(f"runs/v23-{group}-pilot/seed-{seed}")
            assert asdict(Config.load(path / "config.toml")) == configs[group], path
            founders = torch.load(path / "founders.pt", weights_only=True)["genomes"]
            assert torch.equal(founders, reference), path
            row = checked(path, 600)
            assert row["segments"][0]["metadata"]["ablation"] == ablation, path
            row.update(treatment=group, seed=seed, matched_founders=True)
            trials.append(row)
        new = Path(f"runs/v23-hidden-pilot/seed-{seed}")
        old_config = asdict(Config.load(old / "config.toml"))
        new_config = asdict(Config.load(new / "config.toml"))
        old_config.pop("ecology_version")
        new_config.pop("ecology_version")
        assert old_config == new_config
        _, old_rows, _ = read_history(old)
        _, new_rows, _ = read_history(new)
        assert physical(old_rows) == physical(new_rows), seed
        a = torch.load(old / "latest.pt", weights_only=True)
        b = torch.load(new / "latest.pt", weights_only=True)
        a.pop("config")
        b.pop("config")
        same_state(a, b)
        parity.append(dict(seed=seed, measurements=len(new_rows), exact_common_final_state=True))
    report = dict(
        interpretation="Three matched starts compare direct sensory access for the acquired "
        "critic with the V22 hidden-only predictor and a shuffled-return sensory control. "
        "The inherited circuits, actor features, mutation, exploration, and energy laws "
        "are unchanged. Whole-vector normalization changes feature scale as well as access "
        "to sensory information. There is no additional critic-specific energy cost. "
        "These are evolving communities, not fixed-genotype fitness comparisons.",
        configurations=configs, trials=trials, neutral_v22_parity=parity,
        paired_forecasts=paired_forecasts(args.paired) if args.paired else None,
    )
    if args.paired_fast:
        fast = report["paired_fast_forecasts"] = paired_forecasts(args.paired_fast)
        previous = report["paired_forecasts"]
        assert previous["prediction_rate"] == 0.02 and fast["prediction_rate"] == 0.1
        assert len(previous["trials"]) == len(fast["trials"])
        for old, new in zip(previous["trials"], fast["trials"], strict=True):
            assert old["source_checkpoint_sha256"] == new["source_checkpoint_sha256"]
            assert old["prediction_horizon"] == new["prediction_horizon"]
            assert (old["forecast_start"], old["forecast_seconds"]) == (
                new["forecast_start"], new["forecast_seconds"]
            )
            old_path, new_path = (Path(r["forecast_checkpoint"]).parent for r in (old, new))
            for name in ("forecast-start.pt", "end.pt"):
                same_state(torch.load(old_path / name, weights_only=True),
                           torch.load(new_path / name, weights_only=True))
        fast["exact_physical_parity_with_original_rate"] = True
    path = Path("docs/results/v23-sensory-values.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {len(trials)} pilots; paired forecasts: {bool(args.paired)}; {path}")


if __name__ == "__main__":
    main()
