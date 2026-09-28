"""Check exact and local sensitivities through native leaky, masked neurons.

Fixed-weight derivatives are implementation checks, not demonstrations of
learning or ecological fitness. Local sensitivity drops recurrent propagation.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
from representation_learning import inherited_circuit, transition, value_features

from emergent_garden.brain import advance
from emergent_garden.config import Config
from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256
from emergent_garden.topology import mask_parts
from emergent_garden.world import create_world


def fixture(seed, zero_recurrence=False):
    c = replace(
        Config.load("configs/v25-baseline.toml"),
        hidden_size=6,
        initial_neurons=6,
        initial_neuron_spread=0,
        initial_population=1,
        min_neurons=2,
    ).validate()
    world = create_world(c, seed=seed, device="cpu")
    genome = world.founders.double()
    nodes, mi, mr, _ = mask_parts(c, genome)
    nodes[:, -1] = 0
    mi[:, 0, 3] = 0
    mr[:, 2, 1] = 0
    mr[:, 0, 0] = 1
    tau = world.agents["memory_tau"].double()
    circuit = inherited_circuit(c, genome, tau)
    h, ni = c.hidden_size, c.input_size
    weights = circuit["weights"].clone()
    if zero_recurrence:
        weights[:, :, ni:-1] = 0
    weights.requires_grad_()
    circuit["weights"] = weights
    # Assign the same packed variables into the actual native genome layout.
    native = genome.clone()
    native[:, : h * ni] = weights[:, :, :ni].flatten(1)
    native[:, h * ni : h * (ni + h)] = weights[:, :, ni:-1].flatten(1)
    native[:, h * (ni + h) : h * (ni + h + 1)] = weights[:, :, -1]
    rng = torch.Generator().manual_seed(seed + 83041)
    inputs = torch.randn((40, 1, ni), generator=rng, dtype=torch.float64) * 0.3
    readout = torch.randn((1, h + 1), generator=rng, dtype=torch.float64)
    hidden = torch.zeros((1, h), dtype=torch.float64)
    actual = hidden.clone()
    local = torch.zeros_like(weights)
    exact = weights.new_zeros((h, h, ni + h + 1))
    identity = torch.eye(h, dtype=torch.float64)
    records = []
    maximum_transition_error = 0.0
    for step, sample in enumerate(inputs, 1):
        previous = actual.detach()
        actual, _ = advance(c, native, sample, actual, tau)
        with torch.no_grad():
            hidden, local, direct, gain = transition(
                circuit, hidden, sample, torch.zeros_like(weights), local
            )
            jacobian = torch.diag(((1 - circuit["alpha"]) * circuit["nodes"])[0])
            recurrent = (weights * circuit["mask"])[0, :, ni:-1]
            jacobian += gain[0, :, None] * recurrent
            exact = torch.einsum("ij,jab->iab", jacobian, exact)
            exact += identity[:, :, None] * direct[0, None]
            maximum_transition_error = max(
                maximum_transition_error, (actual - hidden).abs().max().item()
            )
            torch.testing.assert_close(hidden, actual, rtol=1e-12, atol=1e-12)
        if step not in (1, 4, 12, 40):
            continue
        _, predicted, dh = value_features(actual, readout)
        autograd = torch.autograd.grad(predicted.sum(), weights, retain_graph=True)[0][0]
        analytic = torch.einsum("i,iab->ab", dh.detach()[0], exact)
        approximate = dh.detach()[0, :, None] * local[0]
        torch.testing.assert_close(analytic, autograd, rtol=1e-11, atol=1e-12)
        if zero_recurrence or step == 1:
            torch.testing.assert_close(approximate, autograd, rtol=1e-11, atol=1e-12)
        # Independently detach history: only the most recent transition varies.
        conditional, _ = advance(c, native, sample, previous, tau)
        _, value, last_dh = value_features(conditional, readout)
        one_step = torch.autograd.grad(value.sum(), weights, retain_graph=True)[0][0]
        local_direct = last_dh.detach()[0, :, None] * direct[0]
        torch.testing.assert_close(local_direct, one_step, rtol=1e-11, atol=1e-12)
        for derivative in (autograd, analytic, approximate, one_step):
            assert not derivative[~circuit["mask"][0]].count_nonzero()
        records.append(
            dict(
                steps=step,
                exact_maximum_error=(analytic - autograd).abs().max().item(),
                conditional_maximum_error=(local_direct - one_step).abs().max().item(),
                local_relative_error=((approximate - autograd).norm() / autograd.norm()).item(),
                local_cosine=torch.nn.functional.cosine_similarity(
                    approximate.flatten(), autograd.flatten(), dim=0
                ).item(),
                autograd_norm=autograd.norm().item(),
                inactive_derivatives_zero=True,
            )
        )
    return dict(
        seed=seed,
        zero_recurrence=zero_recurrence,
        active_neurons=int(circuit["nodes"].sum()),
        alpha=circuit["alpha"][0].tolist(),
        maximum_transition_error=maximum_transition_error,
        samples=records,
    )


def normalized_readout_check():
    readout = torch.tensor([[0.2, -0.4, 0.9, 0.3]], dtype=torch.float64)
    errors = []
    for values in ([0.0, 0.0, 0.0], [0.2, -0.5, 0.8]):
        hidden = torch.tensor([values], dtype=torch.float64, requires_grad=True)
        _, prediction, derivative = value_features(hidden, readout)
        expected = torch.autograd.grad(prediction.sum(), hidden)[0]
        torch.testing.assert_close(derivative, expected, rtol=1e-12, atol=1e-12)
        errors.append((derivative - expected).abs().max().item())
    return dict(maximum_absolute_error=max(errors), includes_zero_activity=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    scripts = {}
    for name in (Path(__file__).name, "representation_learning.py"):
        data = Path(__file__).with_name(name).read_bytes()
        (args.output / name).write_bytes(data)
        scripts[name] = hashlib.sha256(data).hexdigest()
    trials = [fixture(seed, zero) for seed in (11, 12, 13) for zero in (False, True)]
    report = dict(
        interpretation=__doc__,
        completed=True,
        source_sha256=SOURCE_SHA256,
        scripts_sha256=scripts,
        dtype="float64",
        torch=str(torch.__version__),
        normalization=normalized_readout_check(),
        trials=trials,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for trial in trials:
        print(
            trial["seed"],
            "zero recurrence",
            trial["zero_recurrence"],
            "last comparison",
            trial["samples"][-1],
            flush=True,
        )


if __name__ == "__main__":
    main()
