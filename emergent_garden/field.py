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
