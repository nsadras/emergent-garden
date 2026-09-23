"""Local, acquired motor readouts with exploratory eligibility traces.

Gaussian exploration is applied to the two motor logits. For independent
samples, noise / standard_deviation times presynaptic activity gives a motor
connection's likelihood score. V21's correlated policy also differentiates its
historical mean. A body's later net energetic return modulates the accumulated
eligibility trace.

This is a bounded online policy-gradient approximation, not backpropagation
through the environment or a guarantee of useful behavior. Both readouts and
traces are transient; the genome only supplies learning and exploration traits.
"""

import math

import torch

from .topology import effective_masks


def shuffled_returns(reward, elapsed, generator):
    """Assign another body's normalized energetic return rate to each learner.

    A random nonzero cyclic shift is a derangement: every body receives a
    different body's rate, with the batch's rate distribution preserved. Scale
    by the recipient's elapsed time so off-phase newborns keep valid units.
    This intervention changes learning signals, never physical energy transfers.
    A batch containing only one body cannot be shuffled.
    """
    if len(reward) < 2:
        return reward
    shift = torch.randint(1, len(reward), (), generator=generator, device=reward.device)
    indices = (torch.arange(len(reward), device=reward.device) + shift) % len(reward)
    return (reward / elapsed.clamp_min(1e-9))[indices] * elapsed


def motor_state(config, genomes):
    n, h = len(genomes), config.hidden_size
    state = dict(
        motor_plastic=genomes.new_zeros((n, 2, h + 1)),
        motor_trace=genomes.new_zeros((n, 2, h + 1)),
        motor_baseline=genomes.new_zeros(n),
    )
    if config.ecology_version >= 21:
        from .exploration import history_state

        state.update(history_state(h, genomes))
    return state


def motor_policy(
    config,
    genomes,
    hidden,
    logits,
    state,
    noise,
    reward,
    elapsed,
    learning=True,
    *,
    exploration_enabled=True,
):
    """Credit previous actions before choosing the next exploratory action.

    `reward` is integrated over the elapsed interval and already normalized by
    body size and feedback scale. It must exclude reproduction transfers.
    `noise` is supplied by the caller's checkpointed random stream. Zero noise
    contributes no new eligibility. With correlated exploration, its historical
    contribution persists unless `exploration_enabled=False` also removes it.
    """
    c = config
    traits = genomes[:, c.brain_parameter_count + 11 : c.brain_parameter_count + 13].sigmoid()
    rate = c.motor_learning_rate * traits[:, 0]
    sigma = c.exploration_min + (c.exploration_max - c.exploration_min) * traits[:, 1]
    connections = effective_masks(c, genomes)[3][:, :2]
    mask = torch.cat((connections, torch.ones_like(connections[:, :, :1])), -1)
    # Compare rates so an off-phase newborn's shorter first interval does not
    # distort its baseline. Multiply back by time for the integrated advantage.
    reward_rate = reward / elapsed.clamp_min(1e-9)
    advantage = ((reward_rate - state["motor_baseline"]) * elapsed).clamp(-1, 1)
    beta = 1 - torch.exp(-elapsed / c.motor_baseline_tau)
    baseline = state["motor_baseline"] + beta * (reward_rate - state["motor_baseline"])
    if learning:
        decay = torch.exp(-math.log(2) * elapsed / c.motor_half_life)
        plastic = state["motor_plastic"] * decay[:, None, None]
        plastic += (rate * advantage)[:, None, None] * state["motor_trace"]
        plastic = plastic * mask
        if c.motor_normalized:
            length = plastic.norm(dim=-1, keepdim=True).clamp_min(1e-20)
            plastic *= (c.motor_learning_limit / length).clamp_max(1)
        else:
            plastic = plastic.clamp(-c.motor_learning_limit, c.motor_learning_limit)
    else:
        plastic = torch.zeros_like(state["motor_plastic"])
    features = torch.cat((hidden, torch.ones_like(hidden[:, :1])), -1)
    if c.motor_normalized:
        # Bound both norms so the learned motor correction cannot grow with
        # hidden-layer width or many changes combining in the same direction.
        features /= features.norm(dim=-1, keepdim=True).clamp_min(1)
    changed = logits.clone()
    history = {}
    credit_features = features
    if c.ecology_version >= 21:
        from .exploration import sample_motors

        sampled, score, credit_features, history = sample_motors(
            features,
            logits[:, :2],
            plastic,
            state,
            noise,
            sigma,
            elapsed,
            c.motor_noise_tau,
            exploration_enabled,
        )
        changed[:, :2] = sampled
    else:
        if not exploration_enabled:
            noise = torch.zeros_like(noise)
        changed[:, :2] += (plastic @ features[..., None]).squeeze(-1) + sigma[:, None] * noise
    actions = changed.sigmoid()
    if learning:
        # A preset may explicitly disable exploration. There is then no
        # likelihood score and no new eligibility, including for the bias.
        if c.ecology_version < 21:
            score = torch.where(sigma[:, None] > 0, noise / sigma[:, None].clamp_min(1e-20), 0)
        eligibility = score[:, :, None] * credit_features[:, None, :] * actions[:, 4, None, None]
        trace = state["motor_trace"] * torch.exp(-elapsed / c.motor_trace_tau)[:, None, None]
        trace = (trace + eligibility) * mask
    else:
        trace = torch.zeros_like(state["motor_trace"])
    return dict(
        motor_plastic=plastic, motor_trace=trace, motor_baseline=baseline, **history
    ), actions
