"""Bounded inherited recurrent graphs and gradual structural mutations."""

import math

import torch

from .inheritance import brain_parts


def mask_parts(config, genomes):
    h, ni, no = config.hidden_size, config.input_size, config.output_size
    offset = config.brain_parameter_count + config.trait_count
    pieces = []
    for shape in ((h,), (h, ni), (h, h), (no, h)):
        size = math.prod(shape)
        pieces.append(genomes[:, offset : offset + size].reshape(len(genomes), *shape))
        offset += size
    return pieces


def effective_masks(config, genomes):
    nodes, wi, wr, wo = (part > 0.5 for part in mask_parts(config, genomes))
    return (
        nodes,
        wi & nodes[:, :, None],
        wr & nodes[:, :, None] & nodes[:, None, :],
        wo & nodes[:, None, :],
    )


def initial_structure(config, count, device, generator=None):
    # Use a full temporary genome so layout offsets have one definition.
    genome = torch.zeros((count, config.parameter_count), device=device)
    nodes, wi, wr, wo = mask_parts(config, genome)
    if config.initial_neuron_spread or config.initial_recurrent_density != 1:
        if generator is None:
            raise ValueError("Variable founder circuits require an explicit random generator")
        spread = config.initial_neuron_spread
        sizes = (
            torch.randint(
                config.initial_neurons - spread,
                config.initial_neurons + spread + 1,
                (count,),
                generator=generator,
                device=device,
            )
            if spread
            else torch.full((count,), config.initial_neurons, device=device)
        )
        active = torch.arange(config.hidden_size, device=device)[None] < sizes[:, None]
        nodes[:] = active
        wi[:] = active[:, :, None]
        wr[:] = active[:, :, None] & active[:, None, :]
        if config.initial_recurrent_density != 1:
            wr *= (
                torch.rand(wr.shape, generator=generator, device=device)
                < config.initial_recurrent_density
            )
        wo[:] = active[:, None, :]
        return genome[:, config.brain_parameter_count + config.trait_count :]
    n = config.initial_neurons
    nodes[:, :n] = 1
    wi[:, :n] = 1
    wr[:, :n, :n] = 1
    wo[:, :, :n] = 1
    return genome[:, config.brain_parameter_count + config.trait_count :]


def counts(config, genomes):
    nodes, wi, wr, wo = effective_masks(config, genomes)
    connections = sum(part.flatten(1).sum(1) for part in (wi, wr, wo))
    return nodes.sum(1), connections, wr.flatten(1).sum(1)


def duplicate_neuron(config, genome, donor, destination):
    """Copy one active unit into a dormant slot, splitting outgoing influence.

    This preserves fixed-weight recurrent dynamics when the two units start
    with matching activity, including all self/cross edges. Acquired plastic
    state is not copied: this mutation is used for newborn genomes only.
    """
    out = genome.clone()
    node, mi, mr, mo = (p[0] for p in mask_parts(config, out[None]))
    if donor == destination or not bool(node[donor] > 0.5) or bool(node[destination] > 0.5):
        raise ValueError("Duplication requires an active donor and a dormant destination")
    wi, wr, bias, wo, _ = (p[0] for p in brain_parts(config, out[None]))
    # Clear dormant incident values before reusing the destination, then copy
    # the incoming row and split the outgoing column (which includes self-edges).
    wr[:, destination] = 0
    mr[:, destination] = 0
    wi[destination], mi[destination] = wi[donor].clone(), mi[donor].clone()
    bias[destination] = bias[donor]
    wr[destination], mr[destination] = wr[donor].clone(), mr[donor].clone()
    wr[:, donor] *= 0.5
    wr[:, destination] = wr[:, donor].clone()
    mr[:, destination] = mr[:, donor].clone()
    wo[:, donor] *= 0.5
    wo[:, destination] = wo[:, donor].clone()
    mo[:, destination] = mo[:, donor].clone()
    node[destination] = 1
    return out


def delete_neuron(config, genome, neuron):
    out = genome.clone()
    nodes, wi, wr, wo = (p[0] for p in mask_parts(config, out[None]))
    if not bool(nodes[neuron] > 0.5) or int((nodes > 0.5).sum()) <= config.min_neurons:
        raise ValueError("Deletion requires an active neuron above the configured minimum")
    nodes[neuron] = 0
    wi[neuron] = 0
    wr[neuron] = 0
    wr[:, neuron] = 0
    wo[:, neuron] = 0
    return out


def mutate_structure(config, genome, generator):
    out = genome.clone()
    device = genome.device

    def chance(probability):
        return bool(torch.rand((), generator=generator, device=device) < probability)

    def pick(choices):
        return int(choices[torch.randint(len(choices), (), generator=generator, device=device)])

    if chance(config.node_mutation_probability):
        nodes = mask_parts(config, out[None])[0][0] > 0.5
        active, dormant = nodes.nonzero().flatten(), (~nodes).nonzero().flatten()
        if chance(0.5):
            if len(dormant):
                out = duplicate_neuron(config, out, pick(active), pick(dormant))
        elif len(active) > config.min_neurons:
            out = delete_neuron(config, out, pick(active))
    if chance(config.edge_mutation_probability):
        nodes, *masks = (p[0] for p in mask_parts(config, out[None]))
        active = nodes > 0.5
        possible = (
            active[:, None].expand(-1, config.input_size),
            active[:, None] & active[None, :],
            active[None, :].expand(config.output_size, -1),
        )
        enabled = torch.cat([m.flatten() > 0.5 for m in masks])
        allowed = torch.cat([p.flatten() for p in possible])
        adding = chance(0.5)
        candidates = (allowed & (~enabled if adding else enabled)).nonzero().flatten()
        if len(candidates):
            edge = pick(candidates)
            weights = brain_parts(config, out[None])
            for mask, weight in zip(
                masks, (weights[0][0], weights[1][0], weights[3][0]), strict=True
            ):
                if edge < mask.numel():
                    mask.flatten()[edge] = float(adding)
                    if adding:
                        # Express a connection neutrally; later weight mutation can use it.
                        weight.flatten()[edge] = 0
                    break
                edge -= mask.numel()
    return out
