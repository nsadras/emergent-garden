"""Explicit, provenance-recorded genome transfer between successive world versions."""

import math
from pathlib import Path

import torch

from .config import Config


def brain_parts(config, genomes):
    h, ni, no = config.hidden_size, config.input_size, config.output_size
    offset = 0
    pieces = []
    for shape in ((h, ni), (h, h), (h,), (no, h), (no,)):
        size = math.prod(shape)
        pieces.append(genomes[:, offset : offset + size].reshape(len(genomes), *shape))
        offset += size
    return pieces


def upgrade_genomes(source, target, genomes):
    if (
        source.ecology_version > target.ecology_version
        or source.hidden_size > target.hidden_size
        or (source.hidden_size != target.hidden_size and target.ecology_version < 8)
    ):
        raise ValueError(
            "Transfer requires a newer/equal ecology; only V8+ supports wider templates"
        )
    if genomes.shape[1] != source.parameter_count:
        raise ValueError("Source genome does not match its configuration")
    out = torch.zeros((len(genomes), target.parameter_count), device=genomes.device)
    old, new = brain_parts(source, genomes), brain_parts(target, out)
    h = source.hidden_size
    for index, name in enumerate(source.input_names):
        new[0][:, :h, target.input_names.index(name)] = old[0][:, :, index]
    new[1][:, :h, :h], new[2][:, :h] = old[1], old[2]
    new[3][:, : source.output_size, :h], new[4][:, : source.output_size] = old[3], old[4]
    new[4][:, source.output_size :] = -2.0
    if target.ecology_version >= 7 and source.ecology_version < 7:
        new[4][:, 4] = 0.0  # Signed plasticity modulation starts at zero.
    if target.ecology_version >= 14 and source.ecology_version < 14:
        new[4][:, 5:7] = 0.0  # Internal signals are signed around sigmoid(0).
    if target.trait_count:
        traits = out[
            :, target.brain_parameter_count : target.brain_parameter_count + target.trait_count
        ]
        if target.ecology_version >= 2 and source.ecology_version < 2:
            traits[:, 3:5] = -2.0  # Modest initial weapon/armor investment.
        if target.ecology_version >= 3 and source.ecology_version < 3:
            p = min(0.999, max(0.001, (source.neural_tau - 0.2) / 4.8))
            traits[:, 5] = math.log(p / (1 - p))
        if target.ecology_version >= 4 and source.ecology_version < 4:
            traits[:, 6] = -0.75  # One module, near a viable duplication mutation.
        if target.ecology_version >= 7 and source.ecology_version < 7:
            traits[:, 9] = -2.0  # Modest learning rate; decay starts at its midpoint.
        if target.ecology_version >= 12 and source.ecology_version < 12:
            traits[:, 11] = -2.0  # Modest initial motor-learning rate.
            traits[:, 12] = -1.0  # Small, nonzero motor exploration.
        traits[:, : source.trait_count] = genomes[
            :, source.brain_parameter_count : source.brain_parameter_count + source.trait_count
        ]
    if target.ecology_version >= 8:
        from .topology import mask_parts

        nodes, mi, mr, mo = mask_parts(target, out)
        if source.ecology_version >= 8:
            on, oi, ore, oo = mask_parts(source, genomes)
            nodes[:, :h] = on
            for index, name in enumerate(source.input_names):
                mi[:, :h, target.input_names.index(name)] = oi[:, :, index]
            mr[:, :h, :h] = ore
            mo[:, : source.output_size, :h] = oo
        else:
            nodes[:, :h] = 1
            mi[:, :h] = 1
            mr[:, :h, :h] = 1
            mo[:, :, :h] = 1
        if target.ecology_version >= 14 and source.ecology_version < 14:
            from .coordination import BODY_INPUTS

            # Neutral weights preserve the circuit's existing computations.
            # Enable the new paths so an ordinary weight mutation can use them;
            # their additional construction/maintenance costs apply immediately.
            for name in BODY_INPUTS:
                mi[:, :, target.input_names.index(name)] = nodes
            mo[:, 5:7] = nodes[:, None]
        if ((nodes > 0.5).sum(1) < target.min_neurons).any():
            raise ValueError("Source has fewer active neurons than the destination minimum")
    continuous = out[:, : target.brain_parameter_count + target.trait_count]
    if (continuous.abs() > target.weight_limit).any():
        raise ValueError(
            "Transfer exceeds target weight_limit; increase it to preserve the circuit"
        )
    return out


def seed_population(world, path):
    path = Path(path)
    source = Config.load(path / "config.toml")
    saved = torch.load(path / "population.pt", map_location="cpu", weights_only=True)
    pool = saved["genomes"]
    if not len(pool) or not world.config.ecology_version:
        raise ValueError("Seeding requires a living source population and a V1+ destination")
    rng = torch.Generator().manual_seed(world.seed + 15485863)
    selection = torch.randint(len(pool), (world.population,), generator=rng)
    genomes = upgrade_genomes(source, world.config, pool[selection]).to(world.device)
    world.agents["genome"] = genomes
    world.develop(world.agents)
    world.agents["energy"] = world.config.birth_energy * world.agents["area"]
    world.initial_energy = world.agents["energy"].double().sum().item()
    world.move()  # Repair placement if transplanted bodies are larger.
    world.agents["distance"].zero_()
    world.totals["distance"] = 0.0
    world.founders = genomes.clone()
    world.seeded_from = dict(
        path=str(path),
        ecology_version=source.ecology_version,
        source_ids=saved["ids"][selection].tolist(),
        source_generations=saved["generations"][selection].tolist(),
        interpretation="Sampled living genotypes initialize zero-age founders with fresh states. "
        "New sensory weights start at zero; new physical effector biases start at -2. "
        "Plasticity modulation starts at zero and learned synaptic changes are empty. "
        "New modular bodies start with one module near the duplication boundary. "
        "V8+ can pad into a wider template with all added neurons and edges dormant.",
    )
    world.rebuild_fields()
