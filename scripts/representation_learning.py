"""Passive representation learners; never installed in a native controller.

The local sensitivity retains a neuron's leak and drops all recurrent Jacobian
terms. It is an approximation even with fixed weights. TD targets and imposed
sensory histories are held fixed during the semi-gradient update.
"""

import torch

from emergent_garden.inheritance import brain_parts
from emergent_garden.neural_timing import integration_factors
from emergent_garden.topology import effective_masks


def inherited_circuit(config, genomes, tau):
    wi, wr, bias, _, _ = brain_parts(config, genomes)
    nodes, mi, mr, _ = effective_masks(config, genomes)
    return dict(
        weights=torch.cat((wi, wr, bias[..., None]), -1).clone(),
        mask=torch.cat((mi, mr, nodes[..., None]), -1),
        nodes=nodes,
        alpha=integration_factors(config, genomes, tau),
    )


def representation_state(circuit):
    weights, nodes = circuit["weights"], circuit["nodes"]
    return dict(
        hidden=weights.new_zeros(nodes.shape),
        offsets=torch.zeros_like(weights),
        eligibility=torch.zeros_like(weights),
        readout=weights.new_zeros((len(weights), nodes.shape[-1] + 1)),
        ready=torch.zeros(len(weights), dtype=torch.bool, device=weights.device),
    )


def transition(circuit, hidden, inputs, offsets, eligibility):
    """Native leaky/masked dynamics with a separate local sensitivity trace."""
    nodes, alpha, mask = (circuit[key] for key in ("nodes", "alpha", "mask"))
    previous = hidden * nodes
    weights = (circuit["weights"] + offsets) * mask
    ni = inputs.shape[-1]
    drive = (weights[:, :, :ni] @ inputs[..., None]).squeeze(-1)
    drive += (weights[:, :, ni:-1] @ previous[..., None]).squeeze(-1) + weights[:, :, -1]
    activation = drive.tanh()
    current = ((1 - alpha) * previous + alpha * activation) * nodes
    presynaptic = torch.cat((inputs, previous, torch.ones_like(previous[:, :1])), -1)
    sensitivity = nodes * alpha * (1 - activation.square())
    direct = sensitivity[..., None] * presynaptic[:, None, :] * mask
    local = ((1 - alpha)[..., None] * eligibility + direct) * mask
    return current, local, direct, sensitivity


def value_features(hidden, readout):
    """Normalized features and the readout derivative with respect to hidden state."""
    joined = torch.cat((hidden, torch.ones_like(hidden[:, :1])), -1)
    norm = joined.norm(dim=-1, keepdim=True)
    features = joined / norm
    prediction = (readout * features).sum(-1)
    derivative = readout[:, :-1] / norm - prediction[:, None] * hidden / norm.square()
    return features, prediction, derivative


def prepare(circuit, state, inputs, reward, elapsed, horizon):
    """Observe the new state and form an error before crediting the old state."""
    hidden, eligibility, _, _ = transition(
        circuit, state["hidden"], inputs, state["offsets"], state["eligibility"]
    )
    previous, previous_value, derivative = value_features(state["hidden"], state["readout"])
    _, prediction, _ = value_features(hidden, state["readout"])
    error = reward + torch.exp(-elapsed / horizon) * prediction - previous_value
    error *= state["ready"]
    return dict(
        hidden=hidden,
        eligibility=eligibility,
        previous=previous,
        derivative=derivative,
        prediction=prediction,
        error=error,
    )


def project_rows(values, limit):
    return values * (limit / values.norm(dim=-1, keepdim=True).clamp_min(1e-20)).clamp_max(1)


def credit(
    circuit,
    state,
    observed,
    readout_rate,
    representation_rate,
    readout_limit=4.0,
    representation_limit=0.3,
    *,
    representation_error=None,
):
    """Train only the previous representation; current predictions stay pre-update.

    A shuffled control supplies a different representation_error but preserves
    owner-specific value learning. The supplied rates are already trait-scaled.
    """
    error = observed["error"]
    feedback = error if representation_error is None else representation_error
    # Readiness belongs to the recipient, even when its feedback is shuffled.
    feedback = feedback.clamp(-1, 1) * state["ready"]
    offsets = state["offsets"] + (
        (representation_rate * feedback)[:, None, None]
        * observed["derivative"][..., None]
        * state["eligibility"]
    )
    offsets = project_rows(offsets * circuit["mask"], representation_limit)
    readout = state["readout"] + (
        (readout_rate * error.clamp(-1, 1))[:, None] * observed["previous"]
    )
    return dict(
        hidden=observed["hidden"],
        eligibility=observed["eligibility"],
        offsets=offsets,
        readout=project_rows(readout, readout_limit),
        ready=torch.ones_like(state["ready"]),
    )
