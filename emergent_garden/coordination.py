"""Local body coordinates and causal signaling between adjacent body modules.

Signals belong to a single body, are bounded, and cross one module boundary
per controller update. They never carry another creature's state or hidden
environment labels. Emission and reception are separate so a reception ablation
can retain the cost of maintaining a signal.
"""

import math

import torch

BODY_INPUTS = ("body_x", "body_y", "internal_a", "internal_b")


def body_inputs(agents, index):
    a = agents
    mask = a["module_mask"][index]
    coordinates = a["module_offset"][index] / (2 * a["core_radius"][index, None, None])
    slots = torch.arange(mask.shape[1], device=mask.device)
    adjacent = (slots[:, None] - slots[None, :]).abs() == 1
    links = adjacent[None] & mask[:, :, None] & mask[:, None, :]
    incoming = links.to(coordinates.dtype) @ a["module_internal"][index]
    incoming /= links.sum(-1, keepdim=True).clamp_min(1)
    return torch.cat((coordinates, incoming), -1) * mask[..., None]


def update_signals(config, agents, index):
    """Commit new signals after every controller has read the old signals."""
    a = agents
    beta = 1 - math.exp(-1 / (config.controller_hz * config.internal_tau))
    previous = a["module_internal"][index]
    target = 2 * a["module_actions"][index, :, 5:7] - 1
    a["module_internal"][index] = (previous + beta * (target - previous)) * a["module_mask"][
        index, :, None
    ]


def signal_cost(config, agents):
    # Charging the held, filtered signal also leaves a newly grown module free
    # of signaling costs until its first controller update.
    return config.internal_cost * agents["module_internal"].square().sum((1, 2)) * config.dt
