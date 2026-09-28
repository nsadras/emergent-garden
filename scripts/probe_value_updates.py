"""Replay a paired predictor experiment while measuring TD clipping and bounds.

The instrumented world's complete final state and both shadow predictors must
match the original experiment exactly. Statistics count ready core-module
transitions, not independent organisms. Positive/negative error mass describes
the scalar feedback before multiplication by eligibility and the learning rate;
it is not a physical energy account or a measured synaptic update direction.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from audit_circuits import same_state
from probe_paired_values import PairedValueObserver

from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256, load_checkpoint


class UpdateObserver(PairedValueObserver):
    def __init__(self, world, rate):
        super().__init__(world, rate)
        self.statistics = {
            mode: dict(
                transitions=0, positive_clipped=0, negative_clipped=0,
                positive_error_mass=0.0, negative_error_mass=0.0,
                removed_positive_mass=0.0, removed_negative_mass=0.0,
                positive_return_mass=0.0, negative_return_mass=0.0,
                returns_above_one=0, returns_below_minus_one=0,
                maximum_return=None, minimum_return=None,
                maximum_error=None, minimum_error=None,
                weight_rows_at_bound=0,
            ) for mode in self.states
        }

    def before_controller(self, world, index, inputs, module_inputs=None):
        super().before_controller(world, index, inputs, module_inputs)
        ids = self.pending[1]
        self.ready = {mode: state["motor_value_ready"][ids].clone()
                      for mode, state in self.states.items()}

    def after_controller(self, world, noise=None):
        super().after_controller(world, noise)
        _, ids, _, reward, elapsed = self.pending
        for mode, state in self.states.items():
            ready = self.ready[mode] & (elapsed > 0)
            selected = ids[ready]
            if not len(selected):
                continue
            s = self.statistics[mode]
            error = state["motor_value_error"][selected].double()
            returns = reward[ready].double()
            s["transitions"] += len(selected)
            s["positive_clipped"] += int((error > 1).sum())
            s["negative_clipped"] += int((error < -1).sum())
            s["positive_error_mass"] += error.clamp_min(0).sum().item()
            s["negative_error_mass"] += (-error).clamp_min(0).sum().item()
            s["removed_positive_mass"] += (error - 1).clamp_min(0).sum().item()
            s["removed_negative_mass"] += (-error - 1).clamp_min(0).sum().item()
            s["positive_return_mass"] += returns.clamp_min(0).sum().item()
            s["negative_return_mass"] += (-returns).clamp_min(0).sum().item()
            s["returns_above_one"] += int((returns > 1).sum())
            s["returns_below_minus_one"] += int((returns < -1).sum())
            for name, values in (("return", returns), ("error", error)):
                high, low = values.max().item(), values.min().item()
                previous_high, previous_low = s[f"maximum_{name}"], s[f"minimum_{name}"]
                s[f"maximum_{name}"] = high if previous_high is None else max(previous_high, high)
                s[f"minimum_{name}"] = low if previous_low is None else min(previous_low, low)
            norm = state["motor_value_weights"][selected].norm(dim=-1)
            s["weight_rows_at_bound"] += int((norm >= 0.999 * world.config.motor_value_limit).sum())

    def results(self):
        rows = {}
        for mode, data in self.statistics.items():
            row = data.copy()
            count = max(row["transitions"], 1)
            for key in ("positive_clipped", "negative_clipped", "weight_rows_at_bound"):
                row[f"{key}_fraction"] = row[key] / count
            for sign in ("positive", "negative"):
                total = row[f"{sign}_error_mass"]
                removed = row[f"removed_{sign}_mass"]
                row[f"{sign}_error_mass_removed_fraction"] = removed / total if total else 0.0
                assert 0 <= removed <= total, (mode, sign)
            row["mean_raw_error"] = (
                row["positive_error_mass"] - row["negative_error_mass"]
            ) / count
            row["mean_clipped_error"] = (
                row["positive_error_mass"] - row["removed_positive_mass"]
                - row["negative_error_mass"] + row["removed_negative_mass"]
            ) / count
            rows[mode] = row
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    assert reference["completed"], args.reference
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    scripts = {}
    for name in ("probe_value_updates.py", "probe_paired_values.py", "probe_value_forecasts.py"):
        data = Path(__file__).with_name(name).read_bytes()
        (args.output / name).write_bytes(data)
        scripts[name] = hashlib.sha256(data).hexdigest()
    report = dict(
        interpretation=__doc__, reference=str(args.reference),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        source_sha256=SOURCE_SHA256, scripts_sha256=scripts,
        prediction_rate=reference["prediction_rate"], completed=False, trials=[],
    )
    for original in reference["trials"]:
        source = Path(original["source"]) / "latest.pt"
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest() == original["source_checkpoint_sha256"]
        )
        w = load_checkpoint(source)
        if w.config.ecology_version not in (22, 23):
            raise ValueError("This diagnostic measures V22/V23's fixed [-1, 1] TD error clipping")
        folder = Path(original["forecast_checkpoint"]).parent
        expected = torch.load(folder / "end.pt", weights_only=True)
        expected_values = torch.load(folder / "end-values.pt", weights_only=True)
        observer = w.controller_observer = UpdateObserver(w, reference["prediction_rate"])
        while w.population and w.tick < expected["tick"]:
            w.step()
        same_state(w.state_dict(), expected)
        observer.reserve(torch.tensor([len(expected_values["hidden"]["motor_value_weights"]) - 1]))
        same_state(observer.states, expected_values)
        row = dict(
            source=str(source), end_time=w.time, exact_world_replay=True,
            exact_predictor_replay=True, statistics=observer.results(),
        )
        report["trials"].append(row)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(source, row["statistics"], flush=True)
    report["completed"] = True
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
