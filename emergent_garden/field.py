"""Particle-derived, zero-padded smell field with a radially truncated kernel."""

import math

import torch
import torch.nn.functional as F


def fast_size(n):
    while True:
        m = n
        for factor in (2, 3, 5):
            while m % factor == 0:
                m //= factor
        if m == 1:
            return n
        n += 1


class SmellField:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.grid = torch.zeros((config.grid_size, config.grid_size), device=device)
        self.cell = config.diameter / config.grid_size
        self.radius = math.ceil(config.smell_cutoff / self.cell)
        r = self.radius
        x = torch.arange(-r, r + 1, device=device) * self.cell
        d2 = x[:, None].square() + x[None, :].square()
        kernel = torch.exp(-d2 / (2 * config.smell_sigma**2))
        kernel *= d2 <= config.smell_cutoff**2
        self.fft_size = fast_size(config.grid_size + 2 * r)
        self.kernel_fft = torch.fft.rfft2(kernel, s=(self.fft_size, self.fft_size))
        centers = (torch.arange(config.grid_size, device=device) + 0.5) * self.cell
        self.mask = (
            (centers[:, None] - config.diameter / 2).square()
            + (centers[None, :] - config.diameter / 2).square()
        ) <= (config.diameter / 2) ** 2

    def rebuild(self, positions, energy):
        n = self.config.grid_size
        source = torch.zeros(n * n, device=self.device)
        p = positions / self.cell - 0.5
        base = p.floor().long()
        frac = p - base
        for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
            xy = base + torch.tensor([dx, dy], device=self.device)
            # Clamp boundary deposition to preserve each source's total strength.
            xy = xy.clamp(0, n - 1)
            weight = (frac[:, 0] if dx else 1 - frac[:, 0]) * (frac[:, 1] if dy else 1 - frac[:, 1])
            source.index_add_(0, xy[:, 1] * n + xy[:, 0], weight * energy / self.config.food_energy)
        spectrum = torch.fft.rfft2(source.reshape(n, n), s=(self.fft_size, self.fft_size))
        full = torch.fft.irfft2(spectrum * self.kernel_fft, s=(self.fft_size, self.fft_size))
        r = self.radius
        self.grid = full[r : r + n, r : r + n].clamp_min(0) * self.mask

    def sample(self, positions):
        shape = positions.shape[:-1]
        if positions.numel() == 0:
            return torch.empty(shape, device=self.device)
        normalized = 2 * positions / self.config.diameter - 1
        sampled = F.grid_sample(
            self.grid[None, None],
            normalized.reshape(1, 1, -1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled.reshape(shape)


class TrailField(SmellField):
    """Persistent concentration with conservative diffusion in an impermeable dish.

    Deposits are amounts, divided by cell area. Fluxes cross only faces shared
    by two valid cells. Stable explicit substeps support different resolutions.
    """

    def __init__(self, config, device):
        super().__init__(config, device)
        self.edges_x = self.mask[:, 1:] & self.mask[:, :-1]
        self.edges_y = self.mask[1:] & self.mask[:-1]

    def mass(self):
        return self.grid.double().sum().item() * self.cell**2

    def deposit(self, positions, amounts):
        n = self.config.grid_size
        if not len(positions):
            return
        p = positions / self.cell - 0.5
        base, frac = p.floor().long(), p - p.floor()
        indices, weights = [], []
        for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
            xy = (base + torch.tensor([dx, dy], device=self.device)).clamp(0, n - 1)
            weight = (frac[:, 0] if dx else 1 - frac[:, 0]) * (frac[:, 1] if dy else 1 - frac[:, 1])
            weight *= self.mask[xy[:, 1], xy[:, 0]]
            indices.append(xy[:, 1] * n + xy[:, 0])
            weights.append(weight)
        weights = torch.stack(weights, 1)
        weights /= weights.sum(1, keepdim=True).clamp_min(1e-20)
        values = amounts[:, None] * weights / self.cell**2
        self.grid.flatten().index_add_(0, torch.stack(indices, 1).flatten(), values.flatten())

    def advance(self, seconds):
        coefficient = self.config.signal_diffusion * seconds / self.cell**2
        steps = max(1, math.ceil(coefficient / 0.24))
        rate = coefficient / steps
        before = self.mass()
        for _ in range(steps):
            flux_x = (self.grid[:, 1:] - self.grid[:, :-1]) * rate * self.edges_x
            flux_y = (self.grid[1:] - self.grid[:-1]) * rate * self.edges_y
            delta = torch.zeros_like(self.grid)
            delta[:, :-1] += flux_x
            delta[:, 1:] -= flux_x
            delta[:-1] += flux_y
            delta[1:] -= flux_y
            self.grid = (self.grid + delta).clamp_min(0)
        factor = float(
            torch.tensor(
                math.exp(-math.log(2) * seconds / self.config.signal_half_life),
                dtype=self.grid.dtype,
            )
        )
        self.grid *= factor
        return before * (1 - factor)
