"""Passively measure native recurrent credit and instantaneous output effects.

Counterfactuals change one current transition only. They retain the observed
history, motor offsets, and motor noise; they do not estimate ecological benefit.
Each instrumented fork must match a complete uninstrumented replay exactly.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from audit_circuits import same_state
from audit_value_forecasts import digest

from emergent_garden import ecology
from emergent_garden.brain import advance
from emergent_garden.inheritance import brain_parts
from emergent_garden.morphology import MAX_MODULES
from emergent_garden.runtime import StopFlag
from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256, load_checkpoint, save_checkpoint
from emergent_garden.topology import effective_masks


class Moments:
    def __init__(self):
        self.count = 0
        self.total = self.squares = self.absolute = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value):
        value = value.detach().double()
        if not value.numel():
            return
        assert torch.isfinite(value).all()
        self.count += value.numel()
        self.total += value.sum().item()
        self.squares += value.square().sum().item()
        self.absolute += value.abs().sum().item()
        self.minimum = min(self.minimum, value.min().item())
        self.maximum = max(self.maximum, value.max().item())

    def result(self):
        if not self.count:
            return dict(
                count=0, mean=None, rms=None, mean_absolute=None, minimum=None, maximum=None
            )
        return dict(
            count=self.count,
            mean=self.total / self.count,
            rms=math.sqrt(self.squares / self.count),
            mean_absolute=self.absolute / self.count,
            minimum=self.minimum,
            maximum=self.maximum,
        )


class CreditProbe:
    def __init__(self, world, original):
        self.world, self.original = world, original
        self.modules = self.ages = None
        self.records = {name: {} for name in ("all", "under_30", "30_to_120", "at_least_120")}
        self.calls = 0
        self.maximum_motor_reconstruction_error = 0.0

    def before_controller(self, world, index, inputs, module_inputs=None):
        assert world is self.world and module_inputs is not None
        self.modules = world.agents["module_mask"][index].flatten()
        self.ages = world.agents["age"][index, None].expand(-1, MAX_MODULES).reshape(-1)

    def after_controller(self, world, motor_noise=None):
        self.modules = self.ages = None

    def step(self, c, genomes, inputs, state, tau, plasticity=True, **kwargs):
        assert self.modules is not None and len(self.modules) == len(genomes)
        assert plasticity and kwargs["recurrent_learning"] and c.recurrent_noise_sigma
        assert c.motor_noise_tau == 0 and c.motor_value_rate == 0
        updated, actions = self.original(c, genomes, inputs, state, tau, plasticity, **kwargs)
        nodes, mi, mr, _ = effective_masks(c, genomes)
        wi, wr, _, _, _ = brain_parts(c, genomes)
        previous = state["hidden"] * nodes
        old_local = state["plastic"]
        learned = updated["recurrent_plastic"]
        applied_noise = updated["recurrent_applied_noise"]
        hidden, logits = advance(
            c,
            genomes,
            inputs,
            state["hidden"],
            tau,
            old_local + learned,
            True,
            hidden_noise=applied_noise,
        )
        assert torch.equal(hidden, updated["hidden"])
        assert torch.equal(logits[:, :2], updated["motor_previous_base"])

        def motors(activity, base):
            features = torch.cat((activity, torch.ones_like(activity[:, :1])), -1)
            if c.motor_normalized:
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1)
            offsets = (updated["motor_plastic"] @ features[..., None]).squeeze(-1)
            return (base[:, :2] + (offsets + updated["motor_applied_noise"])).sigmoid()

        reconstructed = motors(hidden, logits)
        error = (reconstructed - actions[:, :2]).abs().max().item()
        self.maximum_motor_reconstruction_error = max(
            self.maximum_motor_reconstruction_error, error
        )
        torch.testing.assert_close(reconstructed, actions[:, :2], rtol=0, atol=1e-7)
        no_weights_h, no_weights_l = advance(
            c, genomes, inputs, state["hidden"], tau, old_local, True, hidden_noise=applied_noise
        )
        no_noise_h, no_noise_l = advance(
            c, genomes, inputs, state["hidden"], tau, old_local + learned, True
        )
        motor_without_weights = motors(no_weights_h, no_weights_l)
        motor_without_noise = motors(no_noise_h, no_noise_l)

        field_count = 4 * len(c.field_names)
        common = inputs.new_zeros(inputs.shape)
        if c.sensory_contrast:
            common[:, :field_count:4] = inputs[:, :field_count:4]
        else:
            common[:, :field_count] = (
                inputs[:, :field_count]
                .reshape(-1, field_count // 4, 4)
                .mean(-1)
                .repeat_interleave(4, -1)
            )
        contrast = inputs.new_zeros(inputs.shape)
        contrast[:, :field_count] = inputs[:, :field_count] - common[:, :field_count]
        wi = wi * mi

        def drive(matrix, activity):
            return (matrix @ activity[..., None]).squeeze(-1)

        neuron_values = dict(
            input_drive=drive(wi, inputs),
            field_mean_drive=drive(wi, common),
            field_contrast_drive=drive(wi, contrast),
            inherited_recurrent_drive=drive(wr * mr, previous),
            local_recurrent_drive=drive(old_local * mr, previous),
            acquired_recurrent_drive=drive(learned * mr, previous),
            hidden_noise=applied_noise,
            hidden=hidden,
            offset_hidden_effect=hidden - no_weights_h,
            noise_hidden_effect=hidden - no_noise_h,
        )
        reward, elapsed = kwargs["recurrent_reward"], kwargs["elapsed"]
        raw_advantage = (reward / elapsed.clamp_min(1e-9) - state["recurrent_baseline"]) * elapsed
        advantage = raw_advantage.clamp(-1, 1)
        rate = c.recurrent_learning_rate * genomes[:, c.brain_parameter_count + 9].sigmoid()
        reinforcement = (rate * advantage)[:, None, None] * state["recurrent_trace"]
        decay = torch.exp(-math.log(2) * elapsed / c.recurrent_half_life)
        edge_values = dict(
            inherited_weight=wr,
            acquired_weight=learned,
            old_score_trace=state["recurrent_trace"],
            reinforcement_proposal=reinforcement,
            decay_removed=state["recurrent_plastic"] * (1 - decay)[:, None, None],
            applied_weight_change=learned - state["recurrent_plastic"],
        )
        module_values = dict(
            reward=reward,
            raw_advantage=raw_advantage,
            advantage=advantage,
            clipped=(raw_advantage.abs() > 1),
            elapsed=elapsed,
            rate=rate,
            age=self.ages,
        )
        motor_values = dict(
            offset_motor_effect=reconstructed - motor_without_weights,
            noise_motor_effect=reconstructed - motor_without_noise,
            motor_noise=updated["motor_applied_noise"],
        )
        choices = dict(
            all=self.modules,
            under_30=self.modules & (self.ages < 30),
            **{"30_to_120": self.modules & (self.ages >= 30) & (self.ages < 120)},
            at_least_120=self.modules & (self.ages >= 120),
        )
        for group, chosen in choices.items():
            masks = (nodes & chosen[:, None], mr & chosen[:, None, None], chosen, chosen)
            for values, mask in zip(
                (neuron_values, edge_values, module_values, motor_values), masks, strict=True
            ):
                for key, value in values.items():
                    self.records[group].setdefault(key, Moments()).add(value[mask])
        self.calls += 1
        return updated, actions

    def result(self):
        return {
            group: {key: value.result() for key, value in record.items()}
            for group, record in self.records.items()
        }


def trial(source, output, seconds, stop):
    checkpoint = source / "latest.pt"
    source_hash = digest(checkpoint)
    world = load_checkpoint(checkpoint)
    assert world.config.ecology_version == 25 and world.ablation == "none"
    assert world.controller == "neural"
    initial_tick = world.tick
    target = initial_tick + math.ceil(seconds * world.config.physics_hz)
    initial_metrics = world.metrics()
    original = ecology.controller_step
    probe = CreditProbe(world, original)
    world.controller_observer = probe
    ecology.controller_step = probe.step
    try:
        while world.tick < target and world.population and not stop.requested:
            world.step(min(world.config.physics_hz, target - world.tick))
    finally:
        ecology.controller_step = original
        del world.controller_observer
    replay = load_checkpoint(checkpoint)
    replay.step(world.tick - initial_tick)
    same_state(world.state_dict(), replay.state_dict())
    assert digest(checkpoint) == source_hash
    output.mkdir(parents=True, exist_ok=False)
    save_checkpoint(world, output / "final.pt")
    return dict(
        source=str(source),
        source_checkpoint_sha256=source_hash,
        final_checkpoint_sha256=digest(output / "final.pt"),
        completed=world.tick >= target or not world.population,
        exactly_passive=True,
        controller_calls=probe.calls,
        maximum_motor_reconstruction_error=probe.maximum_motor_reconstruction_error,
        exact_hidden_and_inherited_motor_reconstruction=True,
        initial=initial_metrics,
        final=world.metrics(),
        groups=probe.result(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        default=[Path(f"runs/v25-learning-pilot/seed-{seed}") for seed in (1, 2, 3)],
    )
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be positive and finite")
    if len(set(args.sources)) != len(args.sources):
        parser.error("sources must be distinct")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).read_bytes()
    (args.output / Path(__file__).name).write_bytes(script)
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    report = dict(
        completed=False,
        seconds=args.seconds,
        script_sha256=hashlib.sha256(script).hexdigest(),
        source_sha256=SOURCE_SHA256,
        trials=[],
        interpretation="Passive forks of all three own-return pilot endpoints. "
        "Measurements are pooled module/neuron/edge-update samples, not independent "
        "organisms or replicates. Age buckets use current module owner's age. "
        "Instantaneous counterfactuals omit only current recurrent offsets or noise; "
        "they hold observed history, motor weights, and motor noise fixed and do not "
        "advance the environment. Differences measure expression, not usefulness. "
        "Each entire fork, including events and random streams, matches a plain replay.",
    )
    stop = StopFlag()
    try:
        for number, source in enumerate(args.sources):
            if stop.requested:
                break
            row = trial(source, args.output / f"trial-{number}", args.seconds, stop)
            row["output"] = f"trial-{number}"
            report["trials"].append(row)
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            pooled = row["groups"]["all"]
            print(
                f"{source}: exact replay; offset/noise motor RMS "
                f"{pooled['offset_motor_effect']['rms']:.6f} / "
                f"{pooled['noise_motor_effect']['rms']:.6f}",
                flush=True,
            )
        report["completed"] = len(report["trials"]) == len(args.sources) and all(
            row["completed"] for row in report["trials"]
        )
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        stop.close()


if __name__ == "__main__":
    main()
