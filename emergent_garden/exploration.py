"""Temporally correlated motor policies with their conditional likelihood score.

This module supplies no steering direction or random samples. It receives the
same independent standard-normal innovations used by the older controller.
The acquired linear readout contributes to both current and historical means.
"""

import torch


def history_shapes(hidden_size):
    return {
        "motor_previous_features": (hidden_size + 1,),
        "motor_previous_base": (2,),
        "motor_previous_logits": (2,),
        "motor_history_ready": (),
        "motor_applied_noise": (2,),
    }


def history_state(hidden_size, reference):
    result = {
        key: reference.new_zeros((len(reference), *shape))
        for key, shape in history_shapes(hidden_size).items()
    }
    result["motor_history_ready"] = result["motor_history_ready"].bool()
    return result


def sample_motors(features, base, plastic, state, noise, sigma, elapsed, tau, enabled=True):
    """Return sampled logits, score factors, and the next disposable history.

    Conditional on the observed history, the score for each motor row is
    `score[..., None] * credit_features[:, None, :]`. With correlation, both
    current and previous means use the current acquired weights. The history
    samples themselves remain fixed when differentiating this density.
    """
    active = (sigma > 0) & enabled
    if tau and enabled:
        ready = state["motor_history_ready"] & active
        rho = torch.where(ready, torch.exp(-elapsed / tau), 0)
        # expm1 retains variance when elapsed/tau is small. A first observation
        # uses full stationary variance, rather than warming up from zero noise.
        variance = torch.where(ready, -torch.expm1(-2 * elapsed / tau), 1)
        deviation = sigma * variance.clamp_min(0).sqrt()
        previous_mean = state["motor_previous_base"] + (
            plastic @ state["motor_previous_features"][..., None]
        ).squeeze(-1)
        perturbation = rho[:, None] * (state["motor_previous_logits"] - previous_mean)
        perturbation += deviation[:, None] * noise
        credit_features = features - rho[:, None] * state["motor_previous_features"]
    else:
        deviation = sigma if enabled else torch.zeros_like(sigma)
        perturbation = deviation[:, None] * noise
        credit_features = features
    score = torch.where((deviation > 0)[:, None], noise / deviation[:, None].clamp_min(1e-20), 0)
    # Retain the older controller's floating-point addition order at tau=0.
    sampled = base + ((plastic @ features[..., None]).squeeze(-1) + perturbation)
    history = dict(
        motor_previous_features=features,
        motor_previous_base=base.clone(),
        motor_previous_logits=sampled,
        motor_history_ready=active,
        motor_applied_noise=perturbation,
    )
    return sampled, score, credit_features, history
