"""Collect and check recorded forecasts without rerunning their communities."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import torch
from probe_value_forecasts import describe

from emergent_garden.storage import load_checkpoint


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(path):
    result = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.startswith("emergent_garden/") and name.endswith(".py"):
                result.update(Path(name).name.encode())
                result.update(archive.read(name))
    return result.hexdigest()


def collect(paths):
    records = []
    for path in paths:
        report = json.loads(path.read_text())
        assert report["completed"], path
        assert report["trials"] and report["trials"][0]["exactly_passive"], path
        assert digest(path.parent / "experiment.py") == report["experiment_sha256"], path
        assert source_digest(path.parent / "source.zip") == report["source_sha256"], path
        for trial in report["trials"]:
            checkpoint = Path(trial["source"]) / "latest.pt"
            assert digest(checkpoint) == trial["source_checkpoint_sha256"], checkpoint
            w = load_checkpoint(checkpoint)
            a, c, rows = w.agents, w.config, trial["rows"]
            adults = (a["modules"] == a["target_modules"]).nonzero().flatten().tolist()
            assert {r["id"] for r in rows} == {int(a["id"][i]) for i in adults}, path
            assert len(rows) == trial["mature_cohort"] == len(adults), path
            lookup = {r["id"]: r for r in rows}
            for i in adults:
                row = lookup[int(a["id"][i])]
                assert row["age"] == float(a["age"][i]), path
                assert row["area"] == float(a["area"][i]), path
            assert trial["centered"] == bool(c.motor_value_centered), path
            assert trial["horizon"] == c.motor_value_horizon, path
            selected = [r for r in rows if r["prediction"] is not None]
            assert trial["died_before_prediction"] == len(rows) - len(selected), path
            for row in selected:
                assert w.time <= row["start"] <= w.time + 1 / c.controller_hz + 1e-8, path
                assert abs(row["prediction"]) <= c.motor_value_limit + 1e-6, path
                if row["died_at"] is not None:
                    assert row["tail"] == 0, path
            for group, cohort in (
                ("all_adults", selected),
                ("adults_at_least_30_seconds_old", [r for r in selected if r["age"] >= 30]),
            ):
                assert trial[group] == describe(cohort, c, trial["forecast_seconds"]), (path, group)
            record = {key: value for key, value in trial.items() if key != "rows"}
            record.update(
                report=str(path),
                report_sha256=digest(path),
                observer_source_sha256=report["source_sha256"],
                experiment_sha256=report["experiment_sha256"],
            )
            records.append(record)
    return dict(
        interpretation="Predictions were recorded before observing each mature creature's "
        "subsequent forecast window. Deaths remain in the cohorts. The primary target is "
        "the observed discounted finite-window return; learned tails are reported separately. "
        "Creatures sharing a community are not independent replicates. False in exactly_passive "
        "means that fork was not independently replayed without the observer. The first fork "
        "of each report was replayed and checked for exact full-state equality. No ecological "
        "continuations or fixed-genotype fitness assays are implied by these diagnostic forks.",
        trials=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summaries", type=Path, nargs="+",
        default=[Path(f"runs/v22-{mode}-forecasts/summary.json") for mode in ("centered", "raw")],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/results/v22-value-forecasts.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = collect(args.summaries)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {len(report['trials'])} forecast cohorts: {args.output}")


if __name__ == "__main__":
    main()
