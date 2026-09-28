"""Inherited per-neuron response times; no acquired state or random controller draws."""

import math

import torch


def timing_genes(config, genomes):
    start = config.brain_parameter_count + config.trait_count + config.structure_count
    return genomes[:, start : start + config.timing_count]


def initial_timing(config, count, device, generator):
    shape = (count, config.timing_count)
    if not config.initial_timing_sigma:
        return torch.zeros(shape, device=device)
    return (
        torch.randn(shape, device=device, generator=generator) * config.initial_timing_sigma
    ).clamp(-config.weight_limit, config.weight_limit)


def mutate_timing(config, genome, generator):
    """Mutate timing before structural duplication, on its own random stream."""
    if not config.timing_mutation_probability or not config.timing_mutation_sigma:
        return genome
    out = genome.clone()
    genes = timing_genes(config, out[None])[0]
    selected = (
        torch.rand(genes.shape, device=genes.device, generator=generator)
        < config.timing_mutation_probability
    )
    noise = torch.randn(genes.shape, device=genes.device, generator=generator)
    genes[:] = (genes + selected * noise * config.timing_mutation_sigma).clamp(
        -config.weight_limit, config.weight_limit
    )
    return out


def response_times(config, genomes, tau=None):
    """Seconds for each hidden slot; body modules share their inherited circuit."""
    base = genomes.new_full((len(genomes),), config.neural_tau) if tau is None else tau
    if config.neural_timing_range == 1:
        return base[:, None].expand(-1, config.hidden_size)
    factors = (math.log(config.neural_timing_range) * timing_genes(config, genomes).tanh()).exp()
    return base[:, None] * factors


def integration_factors(config, genomes, tau=None):
    if config.neural_timing_range == 1:
        # Preserve the precise legacy scalar/tensor arithmetic, including V0.
        return (
            1 - math.exp(-1 / (config.controller_hz * config.neural_tau))
            if tau is None
            else 1 - torch.exp(-1 / (config.controller_hz * tau[:, None]))
        )
    return 1 - torch.exp(-1 / (config.controller_hz * response_times(config, genomes, tau)))


def timing_metrics(config, genomes, tau, nodes):
    """Slot statistics count each active neuron once per body, not per module.

    The within-brain log standard deviation weights bodies equally and omits
    dormant slots. A common body time constant contributes no within-brain spread.
    """
    values = response_times(config, genomes, tau)[nodes].double()
    logs = math.log(config.neural_timing_range) * timing_genes(config, genomes).double().tanh()
    counts = nodes.sum(-1).clamp_min(1)
    means = (logs * nodes).sum(-1) / counts
    within = (((logs - means[:, None]).square() * nodes).sum(-1) / counts).sqrt()
    return dict(
        neural_timing_mean_seconds=values.mean().item() if len(values) else 0.0,
        neural_timing_std_seconds=values.std(correction=0).item() if len(values) else 0.0,
        neural_timing_min_seconds=values.min().item() if len(values) else 0.0,
        neural_timing_max_seconds=values.max().item() if len(values) else 0.0,
        neural_timing_mean_within_log_std=within.mean().item() if len(within) else 0.0,
    )
