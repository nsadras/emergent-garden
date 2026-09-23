"""Patch identities whose digestibility reverses without revealing it in scent."""

import math

import torch


class ReversalLandscape:
    def __init__(self, config, generator, device):
        self.config = config
        order = torch.randperm(config.patches, generator=generator, device=device)
        self.identities = order % 2
        self.favorable = int(torch.randint(2, (), generator=generator, device=device))
        self.next_tick = self.interval(generator)

    def interval(self, generator):
        c = self.config
        sample = float(torch.rand((), generator=generator, device=self.identities.device))
        seconds = c.quality_period * (1 + c.quality_jitter * (2 * sample - 1))
        return max(1, math.ceil(seconds * c.physics_hz))

    def update(self, tick, generator):
        changes = []
        while tick >= self.next_tick:
            self.favorable = 1 - self.favorable
            changes.append(dict(tick=self.next_tick, favorable=self.favorable))
            self.next_tick += self.interval(generator)
        return changes

    def quality(self, patches):
        values = torch.where(
            self.is_favorable(patches), self.config.high_quality, self.config.low_quality
        )
        return torch.where(patches < 0, 1.0, values)

    def is_favorable(self, patches):
        return (patches < 0) | (self.identities[patches.clamp_min(0)] == self.favorable)

    def state_dict(self):
        return dict(
            identities=self.identities.clone(), favorable=self.favorable, next_tick=self.next_tick
        )

    @classmethod
    def from_state(cls, config, state, device):
        self = cls.__new__(cls)
        self.config = config
        self.identities = state["identities"].to(device).clone()
        self.favorable, self.next_tick = state["favorable"], state["next_tick"]
        return self
