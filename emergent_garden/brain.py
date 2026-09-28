"""A fixed interface between inherited controllers and versioned bodies."""

import math

import torch

from .neural_timing import integration_factors
from .topology import effective_masks


def initial_brains(config, count, device, generator, active_counts=None):
    h, inputs, outputs = config.hidden_size, config.input_size, config.output_size
    active = config.initial_neurons if config.ecology_version >= 8 else h
    pieces = []
    for block, (shape, fan) in enumerate(
        (
            ((h, inputs), inputs),
            ((h, h), active),
            ((h,), None),
            ((outputs, h), active),
            ((outputs,), None),
        )
    ):
        size = (count, math.prod(shape))
        if fan is None:
            values = torch.zeros(size, device=device)
        else:
            values = torch.randn(size, device=device, generator=generator)
            if active_counts is not None and block in (1, 3):
                # Keep expected drive variance comparable across founder widths
                # and recurrent densities. This changes initialization, not an
                # inherited circuit during its lifetime or at reproduction.
                density = config.initial_recurrent_density if block == 1 else 1.0
                denominator = (active_counts * density).clamp_min(1).sqrt()[:, None]
                values /= denominator
            else:
                values /= math.sqrt(fan)
        pieces.append(values)
    return torch.cat(pieces, 1).clamp(-config.weight_limit, config.weight_limit)


def advance(
    config,
    genome,
    inputs,
    hidden,
    tau=None,
    plastic=None,
    return_logits=False,
    *,
    hidden_noise=None,
):
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
    if config.ecology_version >= 8:
        nodes, mi, mr, mo = effective_masks(config, genome)
        wi, wr, wo = wi * mi, wr * mr, wo * mo
        hidden = hidden * nodes
    drive = (wi @ inputs[..., None]).squeeze(-1)
    drive += (wr @ hidden[..., None]).squeeze(-1) + bias
    if hidden_noise is not None:
        drive = drive + hidden_noise
    alpha = integration_factors(config, genome, tau)
    hidden = (1 - alpha) * hidden + alpha * drive.tanh()
    if config.ecology_version >= 8:
        hidden = hidden * nodes
    logits = (wo @ hidden[..., None]).squeeze(-1) + bo
    return hidden, logits if return_logits else logits.sigmoid()


def initial_state(config, genomes):
    n, h = len(genomes), config.hidden_size
    state = dict(hidden=genomes.new_zeros((n, h)))
    if config.ecology_version >= 7:
        state.update(plastic=genomes.new_zeros((n, h, h)), trace=genomes.new_zeros((n, h, h)))
    if config.ecology_version >= 12:
        from .learning import motor_state

        state.update(motor_state(config, genomes))
    if config.ecology_version >= 25:
        from .recurrent_learning import recurrent_state

        state.update(recurrent_state(config, genomes))
    return state


def controller_step(
    config,
    genome,
    inputs,
    state,
    tau=None,
    plasticity=True,
    *,
    motor_learning=True,
    noise=None,
    reward=None,
    elapsed=None,
    evolved_rule=True,
    exploration_enabled=True,
    value_learning=True,
    recurrent_learning=True,
    recurrent_noise=None,
    recurrent_reward=None,
):
    """One circuit update; acquired synaptic offsets never modify the genome.

    A signed fifth output modulates a low-pass trace of post/pre activity.
    The update and decay follow simulated time. Frozen controls retain the
    controller's outputs while suppressing acquired offsets and traces.
    V12 also adapts motor readouts using supplied energetic returns and
    exploration samples. Callers without those supplies test only the older
    recurrent mechanism; this function never draws random numbers itself.
    V15 evolves a bounded mixture of correlation, pre-only, post-only, and
    constant trace drives. The fixed-rule control retains the previous rule.
    V25 adds optional hidden perturbations and separately acquired recurrent
    offsets. Energetic feedback credits the previous score trace before the
    current transition. `recurrent_noise` is standard normal; `recurrent_reward`
    may differ from motor feedback in mechanism-specific controls.
    """
    plastic = state.get("plastic") if plasticity else None
    recurrent = {}
    hidden_noise = None
    if config.ecology_version >= 25:
        from .recurrent_learning import recurrent_policy

        received = reward if recurrent_reward is None else recurrent_reward
        recurrent = recurrent_policy(
            config,
            genome,
            state["hidden"],
            state,
            torch.zeros_like(state["hidden"]) if recurrent_noise is None else recurrent_noise,
            inputs.new_zeros(len(genome)) if received is None else received,
            inputs.new_full((len(genome),), 1 / config.controller_hz)
            if elapsed is None
            else elapsed,
            learning=plasticity and recurrent_learning,
            exploration_enabled=exploration_enabled,
        )
        if config.recurrent_noise_sigma:
            learned = recurrent["recurrent_plastic"]
            plastic = learned if plastic is None else plastic + learned
            hidden_noise = recurrent["recurrent_applied_noise"]
    hidden, actions = advance(
        config,
        genome,
        inputs,
        state["hidden"],
        tau,
        plastic,
        config.ecology_version >= 12,
        hidden_noise=hidden_noise,
    )
    result = dict(hidden=hidden, **recurrent)
    if config.ecology_version >= 12:
        from .learning import motor_policy

        acquired, actions = motor_policy(
            config,
            genome,
            hidden,
            actions,
            state,
            inputs.new_zeros((len(genome), 2)) if noise is None else noise,
            inputs.new_zeros(len(genome)) if reward is None else reward,
            inputs.new_full((len(genome),), 1 / config.controller_hz)
            if elapsed is None
            else elapsed,
            learning=plasticity and motor_learning,
            exploration_enabled=exploration_enabled,
            value_learning=value_learning,
            inputs=inputs,
        )
        result.update(acquired)
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
    if c.ecology_version >= 15 and evolved_rule:
        from .plasticity import rule_coefficients

        coefficients = rule_coefficients(c, genome)
        correlation = (
            coefficients[:, 0, None, None] * correlation
            + coefficients[:, 1, None, None] * state["hidden"][:, None, :]
            + coefficients[:, 2, None, None] * hidden[:, :, None]
            + coefficients[:, 3, None, None]
        )
    trace = (1 - beta) * state["trace"] + beta * correlation
    if config.ecology_version >= 8:
        mask = effective_masks(config, genome)[2]
        trace = trace * mask
    modulation = 2 * actions[:, 4] - 1
    decay = torch.exp(-math.log(2) * dt / half_life)
    changed = state["plastic"] * decay[:, None, None]
    changed += (dt * rate * modulation)[:, None, None] * trace
    changed = changed.clamp(-c.plasticity_limit, c.plasticity_limit)
    if config.ecology_version >= 8:
        changed = changed * mask
    result.update(plastic=changed, trace=trace)
    return result, actions
