"""Test actual recurrent adaptation with past-experience baselines in a supplied task.

This independent two-neuron fixture has no food, bodies, births, or evolution.
Training supplies only an episode's terminal score. Each circuit learns its own
bounded offsets, without an oracle baseline or gradients through its trajectory.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

MODES = ("no-updates", "full-trace", "shuffled-full", "decayed-trace", "shuffled-decayed")


def rollout(weights, cue, noise, decay=1.0):
    """Batched fixed-weight trajectories and detached conditional score traces."""
    alpha = -torch.expm1(-0.1 / weights.new_tensor([0.7, 2.0]))
    h = weights.new_zeros((*cue.shape, 2))
    trace = weights.new_zeros((*cue.shape, 2, 2))
    for step, epsilon in enumerate(noise):
        trace = decay * trace + (epsilon / 0.15)[..., :, None] * h[..., None, :]
        inputs = cue[..., None] * weights.new_tensor([0.8, 0.2]) if step < 6 else 0
        drive = (weights @ h[..., None]).squeeze(-1) + inputs + 0.15 * epsilon
        h = (1 - alpha) * h + alpha * drive.tanh()
    output = h @ weights.new_tensor([0.7, -0.4])
    reward = -0.5 * (output - 0.12 * cue).square()
    return reward, trace


def evaluate(weights, cue, noise):
    # Validation noise never changes the training RNG or supplies an update.
    reward, _ = rollout(weights[:, :, None], cue, noise)
    return (-2 * reward).mean(-1)


def run_seed(seed, circuits, episodes, rate, row_limit):
    rng = torch.Generator().manual_seed(seed)
    shuffle_rng = torch.Generator().manual_seed(seed + 449)
    evaluation_rng = torch.Generator().manual_seed(seed + 1009)
    base = torch.tensor([[0.4, -0.2], [0.1, 0.5]], dtype=torch.float64)
    offsets = torch.zeros(len(MODES), circuits, 2, 2, dtype=torch.float64)
    baseline = torch.zeros(len(MODES), circuits, dtype=torch.float64)
    validation_cues = torch.tensor([-1.0, 1.0], dtype=torch.float64).repeat(16)
    validation_cues = validation_cues.expand(len(MODES), circuits, -1)
    validation_noise = torch.randn(
        24, 1, circuits, 32, 2, generator=evaluation_rng, dtype=torch.float64
    ).expand(-1, len(MODES), -1, -1, -1)
    decay = torch.ones(len(MODES), 1, 1, 1, dtype=torch.float64)
    decay[3:] = math.exp(-0.1 / 2)
    rows = []

    def record(episode):
        mse = evaluate(base + offsets, validation_cues, validation_noise)
        rows.append(
            dict(
                episode=episode,
                modes={
                    mode: dict(
                        mse=float(mse[i].mean()),
                        per_circuit_mse=mse[i].tolist(),
                        mean_offset_norm=float(offsets[i].norm(dim=-1).mean()),
                        saturated_row_fraction=float(
                            (offsets[i].norm(dim=-1) >= row_limit - 1e-10).double().mean()
                        ),
                    )
                    for i, mode in enumerate(MODES)
                },
            )
        )

    record(0)
    for episode in range(episodes):
        cue = torch.where(torch.rand(circuits, generator=rng) < 0.5, -1.0, 1.0).double()
        cue = cue.expand(len(MODES), -1)
        noise = torch.randn(24, 1, circuits, 2, generator=rng, dtype=torch.float64)
        noise = noise.expand(-1, len(MODES), -1, -1)
        reward, trace = rollout(base + offsets, cue, noise, decay)
        feedback = reward.clone()
        for index in (2, 4):
            shift = int(torch.randint(1, circuits, (), generator=shuffle_rng))
            feedback[index] = reward[index].roll(shift)
        # Baseline comes only from past episodes, before observing this noise.
        advantage = feedback - baseline
        baseline += (1 - math.exp(-1 / 20)) * advantage
        offsets[1:] += rate * advantage[1:, :, None, None] * trace[1:]
        length = offsets.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        offsets *= (row_limit / length).clamp_max(1)
        if (episode + 1) % 64 == 0 or episode + 1 == episodes:
            record(episode + 1)
    assert not offsets[0].count_nonzero()
    assert torch.isfinite(offsets).all() and torch.isfinite(baseline).all()
    assert offsets.norm(dim=-1).max() <= row_limit + 1e-12
    initial = [rows[0]["modes"][mode]["per_circuit_mse"] for mode in MODES]
    assert all(values == initial[0] for values in initial)
    assert rows[-1]["modes"]["no-updates"] == rows[0]["modes"]["no-updates"]
    return dict(seed=seed, curve=rows), dict(offsets=offsets, baseline=baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--circuits", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--row-limit", type=float, default=0.5)
    args = parser.parse_args()
    if (
        args.circuits < 2
        or args.episodes < 1
        or not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or not math.isfinite(args.rate)
        or args.rate <= 0
        or not math.isfinite(args.row_limit)
        or args.row_limit <= 0
    ):
        parser.error("Use distinct seeds, at least two circuits, and positive finite settings")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).read_bytes()
    (args.output / Path(__file__).name).write_bytes(script)
    report = dict(
        completed=False,
        script_sha256=hashlib.sha256(script).hexdigest(),
        torch=str(torch.__version__),
        config={key: value for key, value in vars(args).items() if key != "output"},
        modes=MODES,
        interpretation="Two-neuron supplied delayed-cue task, not a native controller or an "
        "ecological result. Each circuit updates its own recurrent offsets after a complete "
        "2.4-second noisy trajectory. Its baseline uses only past episodes. Five conditions "
        "share initial weights, cues, and Gaussian samples. Full traces use the episodic "
        "score; two-second trace decay is an approximation. Shuffled feedback uses another "
        "circuit's terminal score, with nonzero cyclic shifts. Validation uses an independent "
        "fixed bank of 32 noisy trajectories per circuit. No validation score trains weights. "
        "This does not test continuous online updates, native energy feedback, or fitness.",
        trials=[],
    )
    try:
        for seed in args.seeds:
            row, final = run_seed(seed, args.circuits, args.episodes, args.rate, args.row_limit)
            state_path = args.output / f"final-{seed}.pt"
            torch.save(final, state_path)
            row["final_state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
            report["trials"].append(row)
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        final={mode: row["curve"][-1]["modes"][mode]["mse"] for mode in MODES},
                    )
                ),
                flush=True,
            )
        report["completed"] = True
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
