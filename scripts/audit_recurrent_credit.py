"""Validate passive V25 credit measurements, archives, and age-bucket accounting."""

import argparse
import json
import math
from pathlib import Path

import torch
from audit_value_forecasts import digest, source_digest

from emergent_garden.storage import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/v25-recurrent-credit-probe"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/results/v25-recurrent-credit.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = args.input
    report = json.loads((path / "summary.json").read_text())
    assert report["completed"] and len(report["trials"]) == 3
    assert report["seconds"] == 30
    assert {r["source"] for r in report["trials"]} == {
        f"runs/v25-learning-pilot/seed-{seed}" for seed in (1, 2, 3)
    }
    assert digest(path / "probe_recurrent_credit.py") == report["script_sha256"]
    assert source_digest(path / "source.zip") == report["source_sha256"]
    for row in report["trials"]:
        source = Path(row["source"]) / "latest.pt"
        final = path / row["output"] / "final.pt"
        assert digest(source) == row["source_checkpoint_sha256"]
        assert digest(final) == row["final_checkpoint_sha256"]
        assert row["completed"] and row["exactly_passive"]
        assert row["exact_hidden_and_inherited_motor_reconstruction"]
        assert row["maximum_motor_reconstruction_error"] <= 1e-7
        assert row["controller_calls"] > 0
        original, result = load_checkpoint(source), load_checkpoint(final)
        assert original.metrics() == row["initial"]
        assert result.metrics() == row["final"]
        assert result.time == original.time + 30 or not result.population
        metric = row["final"]
        assert (
            abs(metric["energy_balance_error"]) / (result.initial_energy + metric["food_spawned"])
            < 1e-6
        )
        assert abs(metric["fertility_balance_error"]) < 1e-7
        assert max(map(abs, metric["trophic_detritus_balance_error"])) < 1e-6
        all_values = row["groups"]["all"]
        others = [row["groups"][key] for key in ("under_30", "30_to_120", "at_least_120")]
        for group in row["groups"].values():
            assert group.keys() == all_values.keys()
            for values in group.values():
                assert values["count"] >= 0
                if not values["count"]:
                    assert all(v is None for k, v in values.items() if k != "count")
                    continue
                assert all(math.isfinite(v) for v in values.values())
                assert 0 <= values["mean_absolute"] <= values["rms"] + 1e-12
                assert values["minimum"] <= values["mean"] + 1e-12 <= values["maximum"] + 2e-12
            modules = group["reward"]["count"]
            assert group["offset_motor_effect"]["count"] == 2 * modules
            assert group["noise_motor_effect"]["count"] == 2 * modules
            assert group["hidden_noise"]["count"] == group["hidden"]["count"]
            assert group["acquired_weight"]["count"] == group["applied_weight_change"]["count"]
            if modules:
                assert 0 <= group["clipped"]["mean"] <= 1
                assert group["elapsed"]["minimum"] > 0
                assert group["advantage"]["minimum"] >= -1
                assert group["advantage"]["maximum"] <= 1
                assert (
                    group["advantage"]["mean_absolute"]
                    <= group["raw_advantage"]["mean_absolute"] + 1e-12
                )
        for key, values in all_values.items():
            assert values["count"] == sum(group[key]["count"] for group in others)
            for field, power in (("mean", 1), ("mean_absolute", 1), ("rms", 2)):
                total = sum(
                    group[key]["count"] * group[key][field] ** power
                    for group in others
                    if group[key]["count"]
                )
                assert math.isclose(
                    total, values["count"] * values[field] ** power, rel_tol=1e-10, abs_tol=1e-9
                ), (row["source"], key, field)
        advantage = all_values["raw_advantage"]["mean_absolute"]
        noise = all_values["noise_motor_effect"]["rms"]
        row["derived"] = dict(
            absolute_advantage_mass_clipped=1 - all_values["advantage"]["mean_absolute"] / advantage
            if advantage
            else 0,
            offset_to_noise_motor_effect_ratio=all_values["offset_motor_effect"]["rms"] / noise
            if noise
            else None,
        )
        print(row["source"], row["derived"])
    report.update(
        report_sha256=digest(path / "summary.json"),
        path=str(path),
        archives_checked=True,
        endpoint_metrics_checked=True,
        age_bucket_accounting_checked=True,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited three passive diagnostic forks: {args.output}")


if __name__ == "__main__":
    main()
