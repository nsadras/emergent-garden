"""A fixed interface between inherited controllers and versioned bodies."""

import math

import torch


def initial_brains(config, count, device, generator):
    h, inputs, outputs = config.hidden_size, config.input_size, config.output_size
    pieces = []
    for shape, fan in (
        ((h, inputs), inputs),
        ((h, h), h),
        ((h,), None),
        ((outputs, h), h),
        ((outputs,), None),
    ):
        size = (count, math.prod(shape))
        pieces.append(
            torch.zeros(size, device=device)
            if fan is None
            else torch.randn(size, device=device, generator=generator) / math.sqrt(fan)
        )
    return torch.cat(pieces, 1).clamp(-config.weight_limit, config.weight_limit)


def advance(config, genome, inputs, hidden):
    h, ni, no = config.hidden_size, config.input_size, config.output_size
    offset = 0

    def take(size, shape):
        nonlocal offset
        value = genome[:, offset : offset + size].reshape(len(genome), *shape)
        offset += size
        return value

    wi, wr = take(h * ni, (h, ni)), take(h * h, (h, h))
    bias, wo, bo = take(h, (h,)), take(no * h, (no, h)), take(no, (no,))
    drive = (wi @ inputs[..., None]).squeeze(-1)
    drive += (wr @ hidden[..., None]).squeeze(-1) + bias
    alpha = 1 - math.exp(-1 / (config.controller_hz * config.neural_tau))
    hidden = (1 - alpha) * hidden + alpha * drive.tanh()
    return hidden, ((wo @ hidden[..., None]).squeeze(-1) + bo).sigmoid()
