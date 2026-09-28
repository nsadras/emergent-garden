"""Reconstruct passive forecast targets and their paired error comparisons."""

import argparse
import json
import math
from pathlib import Path

import torch
from audit_value_forecasts import digest, source_digest

from emergent_garden.storage import load_checkpoint
from emergent_garden.topology import effective_masks


def audit_trial(trial, report):
    root = Path(trial["output"])
    assert trial["exactly_passive"] and trial["complete_founders_match"]
    assert digest(Path(trial["source"]) / "latest.pt") == trial["source_checkpoint_sha256"]
    for name, expected in trial["artifacts_sha256"].items():
        assert digest(root / name) == expected
    initial, final = (load_checkpoint(root / name) for name in ("initial.pt", "end.pt"))
    assert initial.tick == 0 and initial.seed == trial["seed"]
    assert final.time == trial["end_time"]
    intervals = torch.load(root / "intervals.pt", weights_only=True)
    states = torch.load(root / "learners.pt", weights_only=True)
    rows = json.loads((root / "predictions.json").read_text())
    c = initial.config
    assert torch.equal(
        intervals["reward"], intervals["raw"] / (c.feedback_scale * intervals["area"])
    )
    assert (intervals["area"] > 0).all()
    assert torch.isfinite(intervals["reward"]).all()
    assert (intervals["start"] <= intervals["end"]).all()
    assert (intervals["end"] - intervals["start"]).max() <= c.physics_hz // c.controller_hz
    assert int(intervals["terminal"].sum()) == trial["deaths"] == final.totals["deaths"]
    assert int((~intervals["terminal"]).sum()) == trial["updates"]
    assert trial["observed_births"] == c.initial_population + final.totals["births"]
    lifetime = {}
    for event in final.events:
        if event["event"] == "birth":
            lifetime[event["id"]] = round(event["time"] * c.physics_hz)
    for identifier in initial.agents["id"].tolist():
        lifetime[identifier] = 0
    assert len(rows) == len({(row["id"], row["cohort_age"]) for row in rows})
    maximum_error = 0.0
    for row in rows:
        assert row["complete"]
        start, end = row["start_tick"], row["end_tick"]
        assert start < report["prediction_seconds"] * c.physics_hz
        assert end - start == report["window"] * c.physics_hz
        age = (start - lifetime[row["id"]]) * c.dt
        assert age == row["actual_age"]
        assert row["cohort_age"] <= age + 1e-9 < row["cohort_age"] + 1 / c.controller_hz + 1e-9
        selected = (intervals["id"] == row["id"]) & (intervals["start"] >= start)
        selected &= (intervals["start"] < end) & (intervals["end"] <= end)
        left, right = intervals["start"][selected], intervals["end"][selected]
        reward = intervals["reward"][selected].tolist()
        terminal = intervals["terminal"][selected]
        assert len(left) == row["intervals"] and len(left)
        assert int(left[0]) == start
        assert torch.equal(left[1:], right[:-1]), (row["id"], start)
        assert int(right[-1]) == end or bool(terminal[-1])
        expected_terminal = int(right[-1]) if bool(terminal[-1]) else None
        assert row["terminal_tick"] == expected_terminal
        target = sum(
            value * math.exp(-(tick - start) / (c.physics_hz * report["horizon"]))
            for value, tick in zip(reward, left.tolist(), strict=True)
        )
        difference = abs(target - row["discounted_return"])
        maximum_error = max(maximum_error, difference)
        assert difference < 1e-12
        assert all(math.isfinite(value) for value in row["predictions"].values())
        assert all(
            abs(row["predictions"][key]) <= 4 + 1e-6
            for key in ("fixed", "adaptive", "shuffled", "sensory")
        )
    for mode in ("fixed", "adaptive", "shuffled", "sensory"):
        state = states[mode]
        assert all(torch.isfinite(value).all() for value in state.values())
        assert state["readout"].norm(dim=-1).max() <= 4 + 1e-6
        if mode == "sensory":
            continue
        assert state["offsets"].norm(dim=-1).max() <= 0.3 + 1e-6
        if mode == "fixed":
            assert not state["offsets"].count_nonzero()
        nodes, mi, mr, _ = effective_masks(c, final.agents["genome"])
        mask = torch.cat((mi, mr, nodes[..., None]), -1)
        ids = final.agents["id"]
        assert not state["offsets"][ids][~mask].count_nonzero()
        assert not state["eligibility"][ids][~mask].count_nonzero()
        assert not state["hidden"][ids][~nodes].count_nonzero()
    # Independently calculate the aggregate errors from the saved pre-outcome predictions.
    for cohort, reported in trial["cohorts"].items():
        chosen = [row for row in rows if row["cohort_age"] == int(cohort)]
        assert len(chosen) == reported["count"]
        if not chosen:
            continue
        assert sum(row["terminal_tick"] is not None for row in chosen) == reported["deaths"]
        for mode, mse in reported["mse"].items():
            expected = sum(
                (row["predictions"][mode] - row["discounted_return"]) ** 2 for row in chosen
            ) / len(chosen)
            assert math.isclose(expected, mse, rel_tol=1e-12, abs_tol=1e-12)
    return trial | dict(
        forecasts_reconstructed=len(rows),
        maximum_target_reconstruction_error=maximum_error,
        interval_normalization_exact=True,
        finite_states_and_bounds=True,
        final_living_masks_exact=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=Path("runs/learned-representation-pilot/summary.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/results/learned-representations.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = json.loads(args.summary.read_text())
    assert report["completed"]
    assert source_digest(args.summary.parent / "source.zip") == report["source_sha256"]
    for name, expected in report["scripts_sha256"].items():
        assert digest(args.summary.parent / name) == expected
    trials = [audit_trial(trial, report) for trial in report["trials"]]
    comparisons = []
    for trial in trials:
        primary = trial["cohorts"]["30"]
        if not primary["count"]:
            continue
        mse = primary["mse"]
        comparisons.append(
            dict(
                seed=trial["seed"],
                count=primary["count"],
                relative_improvement={
                    mode: (mse[mode] - mse["adaptive"]) / mse[mode]
                    for mode in mse
                    if mode != "adaptive" and mse[mode] > 0
                },
                passed=all(
                    mse["adaptive"] < mse[control] for control in ("fixed", "shuffled", "zero")
                ),
            )
        )
    complete_screen = report["prediction_seconds"] == 600 and {t["seed"] for t in trials} == {
        1,
        2,
        3,
    }
    result = report | dict(
        trials=trials,
        comparisons=comparisons,
        complete_screen=complete_screen,
        prospective_screen_passed=complete_screen
        and sum(row["passed"] for row in comparisons) >= 2,
        raw_report=str(args.summary),
        raw_report_sha256=digest(args.summary),
        limitations="Passive predictors observe the same native histories and do not control "
        "movement. Forecasts within each community are correlated; reaching an age selects "
        "survivors. Targets include recorded terminal returns but truncate at 20 seconds. "
        "A prediction improvement alone does not establish a behavioral or ecological benefit.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    passed = result["prospective_screen_passed"]
    print(f"Audited {len(trials)} passive replays; screen passed: {passed}")
    print(comparisons)


if __name__ == "__main__":
    main()
