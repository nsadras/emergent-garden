"""Measure forecasts made by mature creatures before observing their next returns.

Each initial adult contributes one forecast from its core module's next actual
controller update. Follow all of them, including deaths, without changing the
world or its learning. Integrate acquired minus spent minus bite losses, using
the same body-size normalization and controller discount intervals as learning.
Centered targets subtract each future interval's actual running baseline.
The finite-window target is reported both alone and with a discounted learned
tail for survivors. The tail remains an estimate, not an observed future return.
These are descriptive forecasts in six evolved communities, not independent
per-creature fitness tests or evidence of useful motor adaptation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from audit_circuits import same_state

from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256, load_checkpoint, save_checkpoint
from emergent_garden.world import World


class ForecastObserver:
    def __init__(self, world, seconds, readout=None, *, horizon=None):
        a = world.agents
        self.readout = readout or self.native_readout
        self.horizon = world.config.motor_value_horizon if horizon is None else horizon
        if not math.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("Prediction horizon must be positive and finite")
        self.seconds, self.pending = seconds, None
        _, norms = self.readout(world)
        adults = (a["modules"] == a["target_modules"]).nonzero().flatten().tolist()
        self.rows = {
            int(a["id"][i]): dict(
                id=int(a["id"][i]),
                age=float(a["age"][i]),
                area=float(a["area"][i]),
                norm=float(norms[i]),
                start=None,
                prediction=None,
                baseline=None,
                current_baseline=0.0,
                last_control=None,
                tail=None,
                discounted_return=0.0,
                last_net=float(a["acquired"][i]) - float(a["spent"][i]) - float(a["bitten"][i]),
                died_at=None,
            )
            for i in adults
        }

    @staticmethod
    def native_readout(world):
        a = world.agents
        return (
            a["module_motor_value_prediction"][:, 0],
            a["module_motor_value_weights"][:, 0].norm(dim=-1),
        )

    def before_controller(self, world, index, inputs, module_inputs=None):
        self.pending = index

    def after_controller(self, world, noise=None):
        a, t = world.agents, world.time
        predictions, _ = self.readout(world)
        for i in self.pending.tolist():
            row = self.rows.get(int(a["id"][i]))
            if row is None:
                continue
            prediction = float(predictions[i])
            baseline = float(a["module_motor_baseline"][i, 0])
            row["current_baseline"], row["last_control"] = baseline, t
            if row["start"] is None:
                row.update(start=t, prediction=prediction, baseline=baseline)
            elif row["tail"] is None and t >= row["start"] + self.seconds - 1e-8:
                row["tail"] = prediction

    def record_step(self, world, before_time, event_offset):
        a, c = world.agents, world.config
        net = a["acquired"].double() - a["spent"].double() - a["bitten"].double()
        observed = dict(zip(a["id"].tolist(), net.tolist(), strict=True))
        for event in world.events[event_offset:]:
            if event["event"] == "death" and event["id"] in self.rows:
                row = self.rows[event["id"]]
                if row["start"] is None or before_time < row["start"] + self.seconds - 1e-8:
                    row["died_at"], row["tail"] = event["time"], 0.0
                observed[event["id"]] = event["acquired"] - event["spent"] - event["bitten"]
        for identifier, total in observed.items():
            row = self.rows.get(identifier)
            if row is None:
                continue
            increment = (total - row["last_net"]) / (c.feedback_scale * row["area"])
            row["last_net"] = total
            if row["start"] is None or before_time >= row["start"] + self.seconds - 1e-8:
                continue
            if c.motor_value_centered:
                increment -= row["current_baseline"] * c.dt
            # Rewards during one held motor interval share the learner's discount.
            discount = math.exp(-(row["last_control"] - row["start"]) / self.horizon)
            row["discounted_return"] += discount * increment


def describe(rows, c, seconds):
    if not rows:
        return dict(count=0)
    coefficient = (1 / c.controller_hz) / -math.expm1(
        -1 / (c.controller_hz * c.motor_value_horizon)
    )
    coefficient *= -math.expm1(-seconds / c.motor_value_horizon)
    prediction = torch.tensor([r["prediction"] for r in rows], dtype=torch.float64)
    observed = torch.tensor([r["discounted_return"] for r in rows], dtype=torch.float64)
    tail = torch.tensor([r["tail"] for r in rows], dtype=torch.float64)
    target = observed + math.exp(-seconds / c.motor_value_horizon) * tail
    constant = torch.tensor([r["baseline"] for r in rows], dtype=torch.float64) * coefficient
    if c.motor_value_centered:
        constant.zero_()
    return dict(
        count=len(rows),
        deaths=sum(r["died_at"] is not None for r in rows),
        mean_prediction=prediction.mean().item(),
        mean_observed_return=observed.mean().item(),
        mse_observed=(prediction - observed).square().mean().item(),
        zero_mse_observed=observed.square().mean().item(),
        running_rate_mse_observed=(constant - observed).square().mean().item(),
        mse_with_estimated_tail=(prediction - target).square().mean().item(),
        zero_mse_with_estimated_tail=target.square().mean().item(),
        maximum_tail_contribution=(math.exp(-seconds / c.motor_value_horizon) * tail)
        .abs()
        .max()
        .item(),
        prediction_return_correlation=torch.corrcoef(torch.stack((prediction, observed)))[
            0, 1
        ].item()
        if len(rows) > 1 and prediction.std() > 0 and observed.std() > 0
        else None,
    )


def trial(source, output, seconds, verify):
    w = load_checkpoint(source / "latest.pt")
    c = w.config
    if (
        c.ecology_version < 22
        or not c.motor_value_rate
        or w.ablation != "none"
        or w.controller != "neural"
    ):
        raise ValueError("Use an unablated V22 population with value learning enabled")
    observer = ForecastObserver(w, seconds)
    if not observer.rows:
        raise ValueError("This diagnostic requires an initial mature cohort")
    plain = World.from_state(w.state_dict()) if verify else None
    w.controller_observer = observer
    start_time = w.time
    target = w.tick + math.ceil(seconds * c.physics_hz) + 2 * (c.physics_hz // c.controller_hz)
    while w.population and w.tick < target:
        before, events = w.time, len(w.events)
        w.step()
        observer.record_step(w, before, events)
        if plain is not None:
            plain.step()
    if plain is not None:
        same_state(w.state_dict(), plain.state_dict())
    rows = list(observer.rows.values())
    selected = [r for r in rows if r["prediction"] is not None]
    assert all(r["tail"] is not None for r in selected)
    for row in rows:
        for key in ("current_baseline", "last_control", "last_net"):
            row.pop(key)
    output.mkdir(parents=True, exist_ok=False)
    save_checkpoint(w, output / "end.pt")
    return dict(
        source=str(source),
        source_metadata=json.loads((source / "metadata.json").read_text()),
        source_checkpoint_sha256=hashlib.sha256((source / "latest.pt").read_bytes()).hexdigest(),
        start_time=start_time,
        end_time=w.time,
        horizon=c.motor_value_horizon,
        centered=bool(c.motor_value_centered),
        forecast_seconds=seconds,
        exactly_passive=verify,
        mature_cohort=len(rows),
        died_before_prediction=len(rows) - len(selected),
        all_adults=describe(selected, c, seconds),
        adults_at_least_30_seconds_old=describe(
            [r for r in selected if r["age"] >= 30], c, seconds
        ),
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=100)
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("The forecast window must be positive")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    experiment = Path(__file__).read_bytes()
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    (args.output / "experiment.py").write_bytes(experiment)
    report = dict(
        interpretation=__doc__,
        numerical_note="Returns use differences of the simulation's cumulative float32 "
        "counters, accumulated in float64. Cohorts start mature to keep body normalization "
        "constant. The 30-second age subgroup is chosen before observing future outcomes.",
        source_sha256=SOURCE_SHA256,
        experiment_sha256=hashlib.sha256(experiment).hexdigest(),
        completed=False,
        trials=[],
    )
    for index, source in enumerate(args.sources):
        row = trial(source, args.output / f"trial-{index + 1}", args.seconds, verify=index == 0)
        report["trials"].append(row)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(source, row["all_adults"], flush=True)
    report["completed"] = True
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
