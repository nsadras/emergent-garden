"""Compare fresh hidden-only and sensory critics on identical physical experience.

Both shadow predictors see the same core module, returns, and inherited learning
rate. They never control actions. After a warmup, record one prediction per
initially mature creature and follow its discounted returns, including deaths.
The original community's native learning, reproduction, and mutation continue.
This is a paired prediction diagnostic, not a motor-learning or fitness assay.
"""

import argparse
import hashlib
import json
import math
from functools import partial
from pathlib import Path

import torch
from audit_circuits import same_state
from probe_value_forecasts import ForecastObserver, describe

from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256, load_checkpoint, save_checkpoint
from emergent_garden.value import advance_value, sensory_features, value_state
from emergent_garden.world import World


class PairedValueObserver:
    def __init__(self, world, rate=0.02):
        c = world.config
        if c.ecology_version < 22 or c.motor_value_centered or world.ablation != "none":
            raise ValueError("Use an unablated V22+ community with the direct return target")
        self.rate, self.pending, self.forecasts = rate, None, {}
        reference = world.agents["energy"][:0]
        self.states = {
            "hidden": value_state(c.hidden_size, reference),
            "sensory": value_state(c.hidden_size, reference, c.input_size),
        }
        self.slots = 0

    def reserve(self, ids):
        required = int(ids.max()) + 1 if len(ids) else 0
        if required <= self.slots:
            return
        size = max(required, 256, self.slots * 2)
        for state in self.states.values():
            for key, previous in state.items():
                new = previous.new_zeros((size, *previous.shape[1:]))
                new[:self.slots] = previous
                state[key] = new
        self.slots = size

    def readout(self, mode, world):
        ids = world.agents["id"]
        self.reserve(ids)
        state = self.states[mode]
        return state["motor_value_prediction"][ids], state["motor_value_weights"][ids].norm(dim=-1)

    def before_controller(self, world, index, inputs, module_inputs=None):
        a, c = world.agents, world.config
        ids = a["id"][index]
        self.reserve(ids)
        reward = a["motor_reward"][index] / (c.feedback_scale * a["area"][index])
        elapsed = (world.tick - a["last_motor_tick"][index]) * c.dt
        self.pending = index, ids, module_inputs[:, 0].clone(), reward, elapsed
        for forecast in self.forecasts.values():
            forecast.before_controller(world, index, inputs, module_inputs)

    def after_controller(self, world, noise=None):
        a, c = world.agents, world.config
        index, ids, inputs, reward, elapsed = self.pending
        hidden = a["module_h"][index, 0]
        features = torch.cat((hidden, torch.ones_like(hidden[:, :1])), -1)
        features /= features.norm(dim=-1, keepdim=True).clamp_min(1)
        rate = self.rate * a["genome"][index, c.brain_parameter_count + 11].sigmoid()
        for mode, current in (("hidden", features), ("sensory", sensory_features(hidden, inputs))):
            state = self.states[mode]
            updated, _ = advance_value(
                {key: value[ids] for key, value in state.items()},
                current, reward, elapsed, rate, c.motor_value_horizon,
                c.motor_value_trace_tau, c.motor_value_limit,
            )
            for key, value in updated.items():
                state[key][ids] = value
        for forecast in self.forecasts.values():
            forecast.after_controller(world)

    def start_forecasts(self, world, seconds):
        self.forecasts = {
            mode: ForecastObserver(world, seconds, partial(self.readout, mode))
            for mode in self.states
        }

    def record_step(self, world, before_time, event_offset):
        for forecast in self.forecasts.values():
            forecast.record_step(world, before_time, event_offset)


def trial(source, output, warmup, seconds, rate, verify):
    w = load_checkpoint(source / "latest.pt")
    observer = PairedValueObserver(w, rate)
    plain = World.from_state(w.state_dict()) if verify else None
    w.controller_observer = observer
    start_time = w.time
    warmup_tick = w.tick + math.ceil(warmup * w.config.physics_hz)
    while w.population and w.tick < warmup_tick:
        w.step()
        if plain is not None:
            plain.step()
    output.mkdir(parents=True, exist_ok=False)
    save_checkpoint(w, output / "forecast-start.pt")
    observer.start_forecasts(w, seconds)
    torch.save(observer.states, output / "forecast-values.pt")
    forecast_time = w.time
    target = w.tick + math.ceil(seconds * w.config.physics_hz)
    target += 2 * (w.config.physics_hz // w.config.controller_hz)
    while w.population and w.tick < target:
        before, events = w.time, len(w.events)
        w.step()
        observer.record_step(w, before, events)
        if plain is not None:
            plain.step()
    if plain is not None:
        same_state(w.state_dict(), plain.state_dict())
    predictors = {}
    for mode, forecast in observer.forecasts.items():
        rows = list(forecast.rows.values())
        selected = [r for r in rows if r["prediction"] is not None]
        assert all(r["tail"] is not None for r in selected)
        for row in rows:
            for key in ("current_baseline", "last_control", "last_net"):
                row.pop(key)
        state = observer.states[mode]
        assert all(torch.isfinite(value).all() for value in state.values())
        maximum = state["motor_value_weights"].norm(dim=-1).max().item() if observer.slots else 0
        assert maximum <= w.config.motor_value_limit + 1e-6
        predictors[mode] = dict(
            mature_cohort=len(rows), died_before_prediction=len(rows) - len(selected),
            maximum_norm=maximum, all_adults=describe(selected, w.config, seconds),
            adults_at_least_30_seconds_old=describe(
                [r for r in selected if r["age"] >= 30], w.config, seconds
            ), rows=rows,
        )
    # These must be exactly the same observed outcomes, not merely similar cohorts.
    columns = ("id", "age", "area", "start", "discounted_return", "died_at")
    assert [tuple(r[k] for k in columns) for r in predictors["hidden"]["rows"]] == [
        tuple(r[k] for k in columns) for r in predictors["sensory"]["rows"]
    ]
    save_checkpoint(w, output / "end.pt")
    torch.save(observer.states, output / "end-values.pt")
    return dict(
        source=str(source),
        source_metadata=json.loads((source / "metadata.json").read_text()),
        source_checkpoint_sha256=hashlib.sha256((source / "latest.pt").read_bytes()).hexdigest(),
        start_time=start_time, forecast_start=forecast_time, end_time=w.time,
        forecast_checkpoint=str(output / "forecast-start.pt"),
        warmup_seconds=warmup, forecast_seconds=seconds,
        warmup_completed=forecast_time >= warmup_tick * w.config.dt,
        exactly_passive=verify, predictors=predictors,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seconds", type=int, default=100)
    parser.add_argument("--rate", type=float, default=0.02)
    args = parser.parse_args()
    if args.warmup < 0 or args.seconds < 1 or not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("Warmup must be nonnegative; forecast duration and rate must be positive")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    scripts = {}
    for name in ("probe_paired_values.py", "probe_value_forecasts.py"):
        data = Path(__file__).with_name(name).read_bytes()
        (args.output / name).write_bytes(data)
        scripts[name] = hashlib.sha256(data).hexdigest()
    report = dict(
        interpretation=__doc__, source_sha256=SOURCE_SHA256, scripts_sha256=scripts,
        prediction_rate=args.rate, completed=False, trials=[],
    )
    for index, source in enumerate(args.sources):
        result = trial(source, args.output / f"trial-{index + 1}", args.warmup, args.seconds,
                       args.rate, verify=index == 0)
        report["trials"].append(result)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(source, {k: v["all_adults"] for k, v in result["predictors"].items()}, flush=True)
    report["completed"] = True
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
