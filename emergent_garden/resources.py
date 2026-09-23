"""Moving food sources and a stationary, renewable budget for growing food.

The shape warp preserves disk area. Fertility is charged only for accepted fresh
food, independently of the edible-energy ledger. It is not a neural input.
"""

import math

import torch


class ResourceLandscape:
    def __init__(self, config, positions, generator):
        self.config = config
        self.device = positions.device
        self.origins = positions.clone()
        self.angles = torch.rand(config.patches, generator=generator, device=self.device) * math.tau
        self.bends = (
            2 * torch.rand(config.patches, generator=generator, device=self.device) - 1
        ) * config.patch_irregularity
        self.headings = (
            torch.rand(config.patches, generator=generator, device=self.device) * math.tau
        )
        self.last_tick = 0
        self.distance = 0.0
        n = config.fertility_grid_size
        self.cell = config.diameter / n
        centers = (torch.arange(n, device=self.device) + 0.5) * self.cell - config.diameter / 2
        self.mask = (centers[:, None].square() + centers[None, :].square()) <= (
            config.diameter / 2
        ) ** 2
        self.capacity = config.fertility_capacity_per_area * self.cell**2
        self.fertility = self.mask.to(torch.float64) * self.capacity
        self.initial_fertility = self.fertility.sum().item()
        self.spent = self.recovered = 0.0
        self.proposals = self.capacity_rejected = self.fertility_rejected = 0

    def offsets(self, offsets, patches):
        c = self.config
        if c.patch_radius == 0 or (c.patch_aspect_ratio == 1 and c.patch_irregularity == 0):
            return offsets
        x, y = offsets.unbind(-1)
        stretched = math.sqrt(c.patch_aspect_ratio) * x + (
            self.bends[patches] * c.patch_radius * (math.pi * y / c.patch_radius).sin()
        )
        compressed = y / math.sqrt(c.patch_aspect_ratio)
        angle = self.angles[patches]
        co, si = angle.cos(), angle.sin()
        return torch.stack((co * stretched - si * compressed, si * stretched + co * compressed), -1)

    def advance(self, tick, positions, generator):
        """Advance at the field cadence; checkpoints retain the precise update time."""
        c = self.config
        seconds = (tick - self.last_tick) / c.physics_hz
        if seconds <= 0:
            return
        self.last_tick = tick
        if c.patch_drift_speed:
            # Independent, unbiased heading diffusion: about one radian RMS over
            # turn_time. No shared orbit or preferred rotational direction.
            noise = torch.randn(c.patches, generator=generator, device=self.device)
            self.headings += noise * math.sqrt(seconds / c.patch_drift_turn_time)
            direction = torch.stack((self.headings.cos(), self.headings.sin()), -1)
            proposed = positions + direction * (c.patch_drift_speed * seconds)
            offset = proposed - c.diameter / 2
            distance = offset.norm(dim=-1)
            limit = c.diameter / 2 - c.patch_extent - c.food_radius
            normal = offset / distance.clamp_min(1e-12)[:, None]
            outside = distance >= limit
            projected = (
                c.diameter / 2 + offset * (limit / distance.clamp_min(1e-12)).clamp_max(1)[:, None]
            )
            self.distance += (projected - positions).norm(dim=-1).double().sum().item()
            positions.copy_(projected)
            reflected = direction - 2 * (direction * normal).sum(-1)[:, None] * normal
            self.headings = torch.where(
                outside, torch.atan2(reflected[:, 1], reflected[:, 0]), self.headings
            )
            self.headings = (self.headings + math.pi) % math.tau - math.pi
        if self.capacity:
            recovery = (
                (self.capacity - self.fertility)
                * self.mask
                * (-math.expm1(-seconds / c.fertility_recovery_time))
            )
            self.fertility += recovery
            self.recovered += recovery.sum().item()

    def boundaries(self, positions, samples=64):
        """Current source outlines, with no state changes or random draws."""
        angle = torch.arange(samples, device=self.device) * (math.tau / samples)
        offsets = torch.stack((angle.cos(), angle.sin()), -1) * self.config.patch_radius
        patches = torch.arange(self.config.patches, device=self.device)[:, None]
        return positions[:, None] + self.offsets(offsets, patches)

    def fund(self, positions, eligible):
        """Reserve full particles per cell, in proposal order, without overdraw.

        Patch stock is checked before fertility, so rejected stock proposals
        cannot consume soil. There is no retry/redistribution of rejected energy.
        """
        self.proposals += len(positions)
        self.capacity_rejected += int((~eligible).sum())
        if not self.capacity:
            return eligible
        rows = eligible.nonzero().flatten()
        if not len(rows):
            return eligible
        n = self.config.fertility_grid_size
        xy = (positions[rows] / self.cell).floor().long().clamp(0, n - 1)
        cells = xy[:, 1] * n + xy[:, 0]
        order = torch.argsort(cells, stable=True)
        cells, rows = cells[order], rows[order]
        rank = torch.arange(len(rows), device=self.device)
        first = torch.ones(len(rows), dtype=torch.bool, device=self.device)
        first[1:] = cells[1:] != cells[:-1]
        rank = rank - torch.where(first, rank, 0).cummax(0).values
        budget = self.fertility.flatten()
        accepted = rank < (budget[cells] / self.config.food_energy).floor().long()
        keep = torch.zeros_like(eligible)
        keep[rows[accepted]] = True
        used, counts = torch.unique(cells[accepted], return_counts=True)
        budget[used] -= counts.to(torch.float64) * self.config.food_energy
        self.spent += int(accepted.sum()) * self.config.food_energy
        self.fertility_rejected += int((~accepted).sum())
        return keep

    def metrics(self, positions):
        remaining = self.fertility.sum().item()
        values = self.fertility[self.mask]
        return dict(
            food_spawn_proposals=self.proposals,
            food_rejected_capacity=self.capacity_rejected,
            food_rejected_fertility=self.fertility_rejected,
            fertility_spent=self.spent,
            fertility_recovered=self.recovered,
            fertility_remaining=remaining,
            fertility_balance_error=self.initial_fertility
            + self.recovered
            - self.spent
            - remaining,
            fertility_mean_fraction=values.mean().item() / self.capacity if self.capacity else 1.0,
            fertility_depleted_fraction=(values < self.config.food_energy).double().mean().item()
            if self.capacity
            else 0.0,
            source_travel=self.distance,
            source_mean_displacement=(positions - self.origins).norm(dim=-1).mean().item(),
            patch_positions=positions.tolist(),
        )

    def state_dict(self):
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in vars(self).items()
            if key not in ("config", "device", "mask", "cell", "capacity")
        }

    @classmethod
    def from_state(cls, config, state, device):
        # Derive geometry without consuming any of the world's random streams.
        generator = torch.Generator(device=device).manual_seed(0)
        self = cls(config, state["origins"].to(device), generator)
        for key, value in state.items():
            setattr(
                self, key, value.to(device).clone() if isinstance(value, torch.Tensor) else value
            )
        return self
