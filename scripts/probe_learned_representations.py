"""Compare passive neural representations on identical native sensory histories.

Every predictor is private to one creature and starts fresh at birth. Predictions
are recorded before their returns, including partial fatal intervals. None of
these learners changes physical actions, genomes, energy, or world random streams.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from audit_circuits import same_state
from representation_learning import (
    credit,
    inherited_circuit,
    prepare,
    project_rows,
    representation_state,
)

from emergent_garden.config import Config
from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256, load_checkpoint, save_checkpoint
from emergent_garden.world import World, create_world

MODES = ("fixed", "adaptive", "shuffled", "sensory")


class ForecastLedger:
    def __init__(self, hz, horizon=2.0, window=20.0):
        self.hz, self.horizon = hz, horizon
        self.window_ticks = round(window * hz)
        self.rows, self.pending = [], {}

    def start(self, identifier, tick, cohort_age, actual_age, predictions):
        row = dict(
            id=identifier,
            start_tick=tick,
            end_tick=tick + self.window_ticks,
            cohort_age=cohort_age,
            actual_age=actual_age,
            predictions=dict(predictions),
            discounted_return=0.0,
            intervals=0,
            terminal_tick=None,
            complete=False,
        )
        self.rows.append(row)
        self.pending.setdefault(identifier, []).append(row)

    def receive(self, identifier, start, end, reward, terminal=False):
        """A received return belongs to the action held from start until end."""
        keep = []
        for row in self.pending.get(identifier, []):
            if start >= row["start_tick"] and start < row["end_tick"]:
                # Forecasts start on regular boundaries and use whole intervals.
                # Silently prorating a boundary-crossing return would leak timing.
                assert end <= row["end_tick"], (identifier, start, end, row)
                discount = math.exp(-(start - row["start_tick"]) / (self.hz * self.horizon))
                row["discounted_return"] += discount * reward
                row["intervals"] += 1
            if terminal or end >= row["end_tick"]:
                row["complete"] = True
                if terminal and end <= row["end_tick"]:
                    row["terminal_tick"] = end
            else:
                keep.append(row)
        if keep:
            self.pending[identifier] = keep
        else:
            self.pending.pop(identifier, None)

    def describe(self):
        assert not self.pending
        groups = {}
        for age in (5, 30, 60):
            rows = [row for row in self.rows if row["cohort_age"] == age]
            assert all(row["complete"] for row in rows)
            if not rows:
                groups[str(age)] = dict(count=0)
                continue
            targets = torch.tensor([r["discounted_return"] for r in rows], dtype=torch.float64)
            groups[str(age)] = dict(
                count=len(rows),
                deaths=sum(r["terminal_tick"] is not None for r in rows),
                mean_observed_return=targets.mean().item(),
                mse={
                    mode: (
                        torch.tensor([r["predictions"][mode] for r in rows], dtype=torch.float64)
                        - targets
                    )
                    .square()
                    .mean()
                    .item()
                    for mode in (*MODES, "zero", "running_rate")
                },
            )
        return groups


class RepresentationObserver:
    def __init__(self, world, forecast_until=600):
        self.c, self.slots, self.states = world.config, 0, {}
        self.until_tick = round(forecast_until * self.c.physics_hz)
        self.ledger = ForecastLedger(self.c.physics_hz)
        self.births = {int(i): 0 for i in world.agents["id"].tolist()}
        self.deaths, self.forecasted = set(), set()
        self.rng = torch.Generator(device=world.device).manual_seed(world.seed + 4194301)
        self.intervals = {
            key: [] for key in ("id", "start", "end", "reward", "raw", "area", "terminal")
        }
        self.update_count = self.shuffled_count = self.singleton_count = 0
        self.changed = dict.fromkeys(MODES[:-1], 0.0)
        self.peak_saturation = dict.fromkeys(MODES[:-1], 0.0)
        self.peak_norm = dict.fromkeys(MODES, 0.0)
        self.clip_count = dict.fromkeys(MODES, 0)

    def reserve(self, ids, reference):
        needed = int(ids.max()) + 1 if len(ids) else 0
        if needed <= self.slots:
            return
        size = max(needed, 256, self.slots * 2)
        if not self.states:
            c = self.c
            dummy = dict(
                weights=reference.new_zeros((0, c.hidden_size, c.input_size + c.hidden_size + 1)),
                nodes=torch.zeros((0, c.hidden_size), dtype=torch.bool, device=reference.device),
            )
            self.states = {mode: representation_state(dummy) for mode in MODES[:-1]}
            self.states["sensory"] = dict(
                readout=reference.new_zeros((0, c.input_size + 1)),
                previous=reference.new_zeros((0, c.input_size + 1)),
                ready=torch.zeros(0, dtype=torch.bool, device=reference.device),
            )
            self.states["baseline"] = dict(rate=reference.new_zeros(0))
        for state in self.states.values():
            for key, before in state.items():
                after = before.new_zeros((size, *before.shape[1:]))
                after[: self.slots] = before
                state[key] = after
        self.slots = size

    def receive(self, ids, start, end, raw, area, terminal=False):
        reward = raw / (self.c.feedback_scale * area)
        end_values = torch.full_like(ids, end)
        terminal_values = torch.full_like(ids, terminal, dtype=torch.bool)
        for key, value in dict(
            id=ids,
            start=start,
            end=end_values,
            reward=reward,
            raw=raw,
            area=area,
            terminal=terminal_values,
        ).items():
            self.intervals[key].append(value.detach().cpu().clone())
        for identifier, left, value in zip(
            ids.tolist(), start.tolist(), reward.tolist(), strict=True
        ):
            self.ledger.receive(identifier, left, end, value, terminal)
        return reward

    def before_controller(self, world, index, inputs, module_inputs=None):
        a, c = world.agents, self.c
        ids = a["id"][index]
        self.reserve(ids, a["genome"])
        current = module_inputs[:, 0]
        reward = self.receive(
            ids, a["last_motor_tick"][index], world.tick, a["motor_reward"][index], a["area"][index]
        )
        elapsed = (world.tick - a["last_motor_tick"][index]) * c.dt
        circuit = inherited_circuit(c, a["genome"][index], a["memory_tau"][index])
        readout_rate = 0.05 * a["genome"][index, c.brain_parameter_count + 11].sigmoid()
        representation_rate = 0.01 * a["genome"][index, c.brain_parameter_count + 9].sigmoid()
        predictions = {}
        for mode in MODES[:-1]:
            previous = {key: value[ids] for key, value in self.states[mode].items()}
            observed = prepare(circuit, previous, current, reward, elapsed, 2.0)
            feedback = None
            if mode == "shuffled":
                feedback = observed["error"].clone()
                ready = previous["ready"].nonzero().flatten()
                if len(ready) > 1:
                    shift = int(torch.randint(1, len(ready), (), generator=self.rng))
                    feedback[ready] = feedback[ready.roll(shift)]
                    self.shuffled_count += len(ready)
                else:
                    self.singleton_count += len(ready)
            new = credit(
                circuit,
                previous,
                observed,
                readout_rate,
                torch.zeros_like(representation_rate) if mode == "fixed" else representation_rate,
                representation_error=feedback,
            )
            for key, value in new.items():
                self.states[mode][key][ids] = value
            predictions[mode] = observed["prediction"]
            self.changed[mode] += (new["offsets"] - previous["offsets"]).abs().double().sum().item()
            norms = new["offsets"].norm(dim=-1)[circuit["nodes"]]
            self.peak_saturation[mode] = max(
                self.peak_saturation[mode], (norms >= 0.3 - 1e-6).double().mean().item()
            )
            self.peak_norm[mode] = max(
                self.peak_norm[mode], new["readout"].norm(dim=-1).max().item()
            )
            self.clip_count[mode] += int((observed["error"].abs() > 1).sum())
        sensory = self.states["sensory"]
        joined = torch.cat((current, torch.ones_like(current[:, :1])), -1)
        features = joined / joined.norm(dim=-1, keepdim=True)
        weights = sensory["readout"][ids]
        prediction = (weights * features).sum(-1)
        error = (
            reward
            + torch.exp(-elapsed / 2.0) * prediction
            - (weights * sensory["previous"][ids]).sum(-1)
        ) * sensory["ready"][ids]
        weights = project_rows(
            weights + (readout_rate * error.clamp(-1, 1))[:, None] * sensory["previous"][ids], 4.0
        )
        sensory["readout"][ids], sensory["previous"][ids] = weights, features
        sensory["ready"][ids] = True
        predictions["sensory"] = prediction
        self.peak_norm["sensory"] = max(
            self.peak_norm["sensory"], weights.norm(dim=-1).max().item()
        )
        self.clip_count["sensory"] += int((error.abs() > 1).sum())
        baseline = self.states["baseline"]["rate"][ids]
        rate = reward / elapsed.clamp_min(1e-9)
        baseline += (1 - torch.exp(-elapsed / 10.0)) * (rate - baseline)
        self.states["baseline"]["rate"][ids] = baseline
        coefficient = (1 / c.controller_hz) / -math.expm1(-1 / (c.controller_hz * 2.0))
        predictions["running_rate"] = baseline * coefficient * -math.expm1(-20 / 2.0)
        predictions["zero"] = torch.zeros_like(reward)
        values = {mode: value.tolist() for mode, value in predictions.items()}
        if world.tick < self.until_tick:
            for j, identifier in enumerate(ids.tolist()):
                age = (world.tick - self.births[identifier]) * c.dt
                for cohort in (5, 30, 60):
                    if age + 1e-9 < cohort or (identifier, cohort) in self.forecasted:
                        continue
                    self.forecasted.add((identifier, cohort))
                    self.ledger.start(
                        identifier,
                        world.tick,
                        cohort,
                        age,
                        {mode: value[j] for mode, value in values.items()},
                    )
        self.update_count += len(ids)

    def after_controller(self, world, noise=None):
        pass

    def before_death(self, world):
        a = world.agents
        index = (a["energy"] <= 0).nonzero().flatten()
        if not len(index):
            return
        ids = a["id"][index]
        self.receive(
            ids,
            a["last_motor_tick"][index],
            world.tick + 1,
            a["motor_reward"][index],
            a["area"][index],
            terminal=True,
        )
        self.deaths.update(ids.tolist())
        # No terminal training is applied. Dead acquired state is retained only
        # as diagnostic output; offspring use new identifiers and zero state.

    def record_births(self, world, event_start):
        for event in world.events[event_start:]:
            if event["event"] == "birth":
                self.births[event["id"]] = round(event["time"] * self.c.physics_hz)

    def finish(self, output):
        for mode, state in self.states.items():
            assert all(torch.isfinite(value).all() for value in state.values()), mode
        assert self.changed["fixed"] == 0
        assert all(value <= 4.0 + 1e-6 for value in self.peak_norm.values())
        torch.save(self.states | {"shuffle_rng": self.rng.get_state()}, output / "learners.pt")
        intervals = {key: torch.cat(values) for key, values in self.intervals.items()}
        torch.save(intervals, output / "intervals.pt")
        rows = self.ledger.rows
        (output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
        return dict(
            cohorts=self.ledger.describe(),
            observed_births=len(self.births),
            deaths=len(self.deaths),
            updates=self.update_count,
            shuffled_updates=self.shuffled_count,
            unshuffled_singleton_updates=self.singleton_count,
            absolute_offset_changes=self.changed,
            peak_saturated_row_fraction=self.peak_saturation,
            peak_readout_norm=self.peak_norm,
            clipped_errors=self.clip_count,
            artifacts_sha256={
                name: digest(output / name)
                for name in ("learners.pt", "intervals.pt", "predictions.json")
            },
        )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trial(seed, output, seconds):
    source = Path(f"runs/v25-baseline-pilot/seed-{seed}")
    c = Config.load(source / "config.toml")
    world = create_world(c, seed=seed, device="cpu")
    founders = torch.load(source / "founders.pt", weights_only=True)["genomes"]
    assert torch.equal(world.founders, founders)
    plain = World.from_state(world.state_dict())
    observer = RepresentationObserver(world, seconds)
    world.controller_observer = observer
    original_remove = world.remove_dead

    def observed_remove(cause="unspecified"):
        observer.before_death(world)
        original_remove(cause)

    world.remove_dead = observed_remove
    output.mkdir(parents=True, exist_ok=False)
    save_checkpoint(world, output / "initial.pt")
    endpoint = math.ceil((seconds + 20) * c.physics_hz)
    while world.population and world.tick <= endpoint:
        event_start = len(world.events)
        world.step()
        observer.record_births(world, event_start)
        plain.step()
        if world.tick % (60 * c.physics_hz) == 0:
            print(
                f"seed {seed}: {world.time:.0f}s, population={world.population}, "
                f"forecasts={len(observer.ledger.rows)}",
                flush=True,
            )
        if world.tick == round(seconds * c.physics_hz):
            same_state(world.state_dict(), plain.state_dict())
            save_checkpoint(world, output / "prediction-end.pt")
            if seconds == 600:
                reference = load_checkpoint(source / "latest.pt").state_dict()
                candidate = world.state_dict()
                reference.pop("events")
                candidate.pop("events")
                same_state(candidate, reference)
                # The calibration logger drains events, so compare its event log.
                recorded = [
                    json.loads(line) for line in (source / "events.jsonl").read_text().splitlines()
                ]
                assert world.events == recorded
    same_state(world.state_dict(), plain.state_dict())
    result = observer.finish(output)
    save_checkpoint(world, output / "end.pt")
    result.update(
        seed=seed,
        source=str(source),
        source_checkpoint_sha256=digest(source / "latest.pt"),
        output=str(output),
        end_time=world.time,
        exactly_passive=True,
        reference_600_second_parity=seconds == 600,
        complete_founders_match=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args()
    if args.seconds <= 0 or set(args.seeds) - {1, 2, 3}:
        parser.error("Use positive seconds and recorded baseline seeds 1, 2, or 3")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    scripts = {}
    for name in (Path(__file__).name, "representation_learning.py", "audit_circuits.py"):
        data = Path(__file__).with_name(name).read_bytes()
        (args.output / name).write_bytes(data)
        scripts[name] = hashlib.sha256(data).hexdigest()
    report = dict(
        interpretation=__doc__,
        source_sha256=SOURCE_SHA256,
        scripts_sha256=scripts,
        completed=False,
        prediction_seconds=args.seconds,
        horizon=2,
        window=20,
        readout_rate=0.05,
        representation_rate=0.01,
        readout_limit=4,
        representation_limit=0.3,
        baseline_tau=10,
        trials=[],
    )
    for seed in args.seeds:
        result = trial(seed, args.output / f"seed-{seed}", args.seconds)
        report["trials"].append(result)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(seed, result["cohorts"], flush=True)
    report["completed"] = True
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
