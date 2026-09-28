"""Acquired recurrent connections with local perturbation credit.

Gaussian noise enters hidden drive before tanh. Its conditional likelihood
score is epsilon/sigma outer previous activity, without an extra leak or tanh
derivative. Bounded, decaying, continuously updated offsets and clipped feedback
make this an online approximation, not an exact episodic gradient guarantee.
"""

import math

import torch

from .topology import effective_masks


def recurrent_shapes(hidden_size):
    return dict(
        recurrent_plastic=(hidden_size, hidden_size),
        recurrent_trace=(hidden_size, hidden_size),
        recurrent_baseline=(),
        recurrent_applied_noise=(hidden_size,),
    )


def recurrent_state(config, genomes):
    return {
        key: genomes.new_zeros((len(genomes), *shape))
        for key, shape in recurrent_shapes(config.hidden_size).items()
    }


def recurrent_policy(
    config,
    genomes,
    previous,
    state,
    noise,
    reward,
    elapsed,
    *,
    learning=True,
    exploration_enabled=True,
):
    """Credit the old trace, then record the current perturbation's score.

    Reward is integrated net energy, already normalized by body area and the
    feedback scale. Noise is standard normal, drawn by the world's separate
    checkpointed stream. No random numbers are drawn here. Returned offsets
    apply to the current transition; the returned trace credits future reward.
    """
    c = config
    if not c.recurrent_noise_sigma:
        return recurrent_state(c, genomes)
    nodes, _, mask, _ = effective_masks(c, genomes)
    previous = previous * nodes
    if not exploration_enabled:
        noise = torch.zeros_like(noise)
    applied = c.recurrent_noise_sigma * noise * nodes
    reward_rate = reward / elapsed.clamp_min(1e-9)
    advantage = ((reward_rate - state["recurrent_baseline"]) * elapsed).clamp(-1, 1)
    beta = 1 - torch.exp(-elapsed / c.recurrent_baseline_tau)
    baseline = state["recurrent_baseline"] + beta * (reward_rate - state["recurrent_baseline"])
    if learning:
        allocation = genomes[:, c.brain_parameter_count + 9].sigmoid()
        rate = c.recurrent_learning_rate * allocation
        decay = torch.exp(-math.log(2) * elapsed / c.recurrent_half_life)
        plastic = state["recurrent_plastic"] * decay[:, None, None]
        plastic += (rate * advantage)[:, None, None] * state["recurrent_trace"]
        plastic *= mask
        length = plastic.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        plastic *= (c.recurrent_learning_limit / length).clamp_max(1)
        score = (noise / c.recurrent_noise_sigma)[:, :, None] * previous[:, None, :]
        trace = (
            state["recurrent_trace"] * torch.exp(-elapsed / c.recurrent_trace_tau)[:, None, None]
        )
        trace = (trace + score) * mask
    else:
        plastic = torch.zeros_like(state["recurrent_plastic"])
        trace = torch.zeros_like(state["recurrent_trace"])
    return dict(
        recurrent_plastic=plastic,
        recurrent_trace=trace,
        recurrent_baseline=baseline,
        recurrent_applied_noise=applied,
    )


def recurrent_metrics(config, agents):
    """Summaries include expressed modules, neurons, and recurrent edges only."""
    a, c = agents, config
    nodes, _, edges, _ = effective_masks(c, a["genome"])
    modules = a["module_mask"]
    active = modules[:, :, None] & nodes[:, None, :]
    connections = modules[:, :, None, None] & edges[:, None]
    count = max(1, int(connections.sum()))
    plastic = a["module_recurrent_plastic"]
    lengths = (plastic * connections).norm(dim=-1)[active]
    perturbation = a["module_recurrent_applied_noise"][active].double()
    allocation = a["genome"][:, c.brain_parameter_count + 9].sigmoid()
    return dict(
        mean_recurrent_plastic_magnitude=plastic[connections].abs().double().sum().item() / count,
        recurrent_rows_saturated_fraction=(lengths >= 0.999 * c.recurrent_learning_limit)
        .float()
        .mean()
        .item()
        if len(lengths)
        else 0.0,
        recurrent_noise_rms=perturbation.square().mean().sqrt().item()
        if len(perturbation)
        else 0.0,
        mean_recurrent_learning_rate=(c.recurrent_learning_rate * allocation).mean().item()
        if len(allocation)
        else 0.0,
    )
