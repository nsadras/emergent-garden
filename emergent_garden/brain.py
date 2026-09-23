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


def advance(config, genome, inputs, hidden, tau=None, plastic=None):
    h, ni, no = config.hidden_size, config.input_size, config.output_size
    offset = 0

    def take(size, shape):
        nonlocal offset
        value = genome[:, offset : offset + size].reshape(len(genome), *shape)
        offset += size
        return value

    wi, wr = take(h * ni, (h, ni)), take(h * h, (h, h))
    if plastic is not None:
        wr = wr + plastic
    bias, wo, bo = take(h, (h,)), take(no * h, (no, h)), take(no, (no,))
    drive = (wi @ inputs[..., None]).squeeze(-1)
    drive += (wr @ hidden[..., None]).squeeze(-1) + bias
    alpha = (
        1 - math.exp(-1 / (config.controller_hz * config.neural_tau))
        if tau is None
        else 1 - torch.exp(-1 / (config.controller_hz * tau[:, None]))
    )
    hidden = (1 - alpha) * hidden + alpha * drive.tanh()
    return hidden, ((wo @ hidden[..., None]).squeeze(-1) + bo).sigmoid()


def initial_state(config, genomes):
    n, h = len(genomes), config.hidden_size
    state = dict(hidden=genomes.new_zeros((n, h)))
    if config.ecology_version >= 7:
        state.update(plastic=genomes.new_zeros((n, h, h)), trace=genomes.new_zeros((n, h, h)))
    return state


def controller_step(config, genome, inputs, state, tau=None, plasticity=True):
    """One circuit update; acquired synaptic offsets never modify the genome.

    A signed fifth output modulates a low-pass trace of post/pre activity.
    The update and decay follow simulated time. Frozen controls retain the
    controller's outputs while suppressing acquired offsets and traces.
    """
    plastic = state.get("plastic") if plasticity else None
    hidden, actions = advance(config, genome, inputs, state["hidden"], tau, plastic)
    result = dict(hidden=hidden)
    if config.ecology_version < 7:
        return result, actions
    if not plasticity:
        result.update(
            plastic=torch.zeros_like(state["plastic"]), trace=torch.zeros_like(state["trace"])
        )
        return result, actions
    c = config
    dt = 1 / c.controller_hz
    parameters = genome[:, c.brain_parameter_count + 9 : c.brain_parameter_count + 11].sigmoid()
    rate = c.plasticity_rate * parameters[:, 0]
    half_life = c.plasticity_half_life_min + parameters[:, 1] * (
        c.plasticity_half_life_max - c.plasticity_half_life_min
    )
    beta = 1 - math.exp(-dt / c.plasticity_trace_tau)
    correlation = hidden[:, :, None] * state["hidden"][:, None, :]
    trace = (1 - beta) * state["trace"] + beta * correlation
    modulation = 2 * actions[:, 4] - 1
    decay = torch.exp(-math.log(2) * dt / half_life)
    changed = state["plastic"] * decay[:, None, None]
    changed += (dt * rate * modulation)[:, None, None] * trace
    result.update(plastic=changed.clamp(-c.plasticity_limit, c.plasticity_limit), trace=trace)
    return result, actions
