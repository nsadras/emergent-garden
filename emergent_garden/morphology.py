"""Bounded developmental grammar: repeat and place a shared sensory/motor module.

The same inherited neural template is instantiated separately in each module.
Modules share body energy; a circular membrane defines the exclusion geometry.
"""

import math

import torch

MAX_MODULES = 3


def develop_modules(config, agents, traits):
    a = agents
    core = a["radius"].clone()
    count = 1 + (traits[:, 6] * MAX_MODULES).long().clamp_max(MAX_MODULES - 1)
    slots = torch.arange(MAX_MODULES, device=core.device)[None]
    mask = slots < count[:, None]
    spacing = core * (1 + traits[:, 7])
    axis = traits[:, 8] * math.pi / 2
    along = (slots - (count[:, None] - 1) / 2) * spacing[:, None]
    offsets = along[..., None] * torch.stack((axis.cos(), axis.sin()), -1)[:, None]
    offsets *= mask[..., None]
    radius = core + offsets.norm(dim=-1).max(dim=1).values
    tissue = count * (core / config.body_radius).square()
    membrane = ((radius / config.body_radius).square() - tissue).clamp_min(0)
    a.update(
        core_radius=core,
        modules=count,
        module_mask=mask,
        module_offset=offsets,
        radius=radius,
        area=tissue + 0.15 * membrane,
        body_axis=axis,
    )


def module_centers(agents, index=None):
    if index is None:
        pos, heading, offsets = agents["pos"], agents["heading"], agents["module_offset"]
    else:
        pos = agents["pos"][index]
        heading, offsets = agents["heading"][index], agents["module_offset"][index]
    co, si = heading.cos()[:, None], heading.sin()[:, None]
    x = offsets[..., 0] * co - offsets[..., 1] * si
    y = offsets[..., 0] * si + offsets[..., 1] * co
    return pos[:, None] + torch.stack((x, y), -1)


def module_turn(agents):
    a = agents
    motors = a["module_actions"][..., :2]
    differential = motors[..., 1] - motors[..., 0]
    lever = a["module_offset"][..., 1] / a["core_radius"][:, None]
    torque = (differential - 2 * lever * motors.mean(dim=-1)) * a["module_mask"]
    inertia = 1 + (a["module_offset"] / a["core_radius"][:, None, None]).square().sum(-1)
    return (torque.sum(1) / (inertia * a["module_mask"]).sum(1)).clamp(-1, 1)
