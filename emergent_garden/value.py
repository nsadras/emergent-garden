"""Acquired linear prediction of energetic returns with TD eligibility traces.

Transitions arrive before the next action. The old trace belongs to the previous
observation; the current observation enters eligibility only after this update.
Callers supply integrated returns (optionally centered) and normalized features.
"""

import torch


def value_shapes(hidden_size, input_size=0):
    features = hidden_size + input_size + 1
    return {
        "motor_value_weights": (features,),
        "motor_value_trace": (features,),
        "motor_value_previous": (features,),
        "motor_value_ready": (),
        "motor_value_prediction": (),
        "motor_value_error": (),
    }


def value_state(hidden_size, reference, input_size=0):
    state = {
        key: reference.new_zeros((len(reference), *shape))
        for key, shape in value_shapes(hidden_size, input_size).items()
    }
    state["motor_value_ready"] = state["motor_value_ready"].bool()
    return state


def sensory_features(hidden, inputs):
    """Use the actual controller observations, alongside activity and one bias."""
    features = torch.cat((hidden, inputs, torch.ones_like(hidden[:, :1])), -1)
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1)


def advance_value(state, features, reward, elapsed, rate, horizon, trace_tau, limit, terminal=None):
    """Return new critic state and the pre-update TD error for the policy.

    Weights and trace are per module. The error used in the weight update is
    bounded to [-1, 1], and the acquired weight row is norm-bounded. `terminal`
    supports isolated episodic fixtures; a living ecological module uses False.
    The prediction telemetry describes the next state before this update.
    """
    if terminal is None:
        terminal = torch.zeros_like(reward, dtype=torch.bool)
    weights = state["motor_value_weights"]
    previous = (weights * state["motor_value_previous"]).sum(-1)
    prediction = (weights * features).sum(-1)
    discount = torch.exp(-elapsed / horizon) * ~terminal
    error = (reward + discount * prediction - previous) * state["motor_value_ready"]
    weights = weights + (rate * error.clamp(-1, 1))[:, None] * state["motor_value_trace"]
    weights = weights * (limit / weights.norm(dim=-1, keepdim=True).clamp_min(1e-20)).clamp_max(1)
    trace = state["motor_value_trace"] * torch.exp(-elapsed / trace_tau)[:, None] + features
    keep = ~terminal
    return dict(
        motor_value_weights=weights,
        motor_value_trace=trace * keep[:, None],
        motor_value_previous=features * keep[:, None],
        motor_value_ready=keep,
        motor_value_prediction=prediction * keep,
        motor_value_error=error,
    ), error
