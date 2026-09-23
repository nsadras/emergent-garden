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
    if source.hidden_size != target.hidden_size or source.ecology_version > target.ecology_version:
        raise ValueError("Transfer requires equal hidden sizes and the same or a newer ecology")
    if genomes.shape[1] != source.parameter_count:
        raise ValueError("Source genome does not match its configuration")
    out = torch.zeros((len(genomes), target.parameter_count), device=genomes.device)
    old, new = brain_parts(source, genomes), brain_parts(target, out)
    for index, name in enumerate(source.input_names):
        new[0][:, :, target.input_names.index(name)] = old[0][:, :, index]
    new[1][:], new[2][:] = old[1], old[2]
    new[3][:, : source.output_size], new[4][:, : source.output_size] = old[3], old[4]
    new[4][:, source.output_size :] = -2.0
    if target.trait_count:
        traits = out[:, target.brain_parameter_count :]
        if target.ecology_version >= 2 and source.ecology_version < 2:
            traits[:, 3:5] = -2.0  # Modest initial weapon/armor investment.
        if target.ecology_version >= 3 and source.ecology_version < 3:
            p = min(0.999, max(0.001, (source.neural_tau - 0.2) / 4.8))
            traits[:, 5] = math.log(p / (1 - p))
        if target.ecology_version >= 4 and source.ecology_version < 4:
            traits[:, 6] = -0.75  # One module, near a viable duplication mutation.
        traits[:, : source.trait_count] = genomes[:, source.brain_parameter_count :]
    if (out.abs() > target.weight_limit).any():
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
        "New sensory weights start at zero; new effector biases start at -2. "
        "New modular bodies start with one module near the duplication boundary.",
    )
    world.rebuild_fields()
