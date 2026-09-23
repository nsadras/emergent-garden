from dataclasses import replace

import pytest
import torch

from emergent_garden.ecology import EcologyWorld
from emergent_garden.storage import load_checkpoint, save_checkpoint
from scripts.assemble_communities import seed_assembly


@pytest.mark.parametrize(
    "condition, expected", [("mixed", [3, 3]), ("first", [6, 0]), ("second", [0, 6])]
)
def test_assembly_preserves_source_ancestry_and_resets_acquired_state(
    config, tmp_path, condition, expected
):
    sources = []
    source_config = replace(config, ecology_version=7, hidden_size=4, initial_population=3)
    for group, diet in enumerate((2.0, -2.0)):
        w = EcologyWorld(source_config, seed=group + 1)
        w.agents["genome"][:, source_config.brain_parameter_count + 2] = diet
        path = tmp_path / f"source-{group}"
        path.mkdir()
        source_config.save(path / "config.toml")
        torch.save(
            dict(
                genomes=w.agents["genome"], ids=w.agents["id"], generations=w.agents["generation"]
            ),
            path / "population.pt",
        )
        sources.append(path)
    target = replace(
        config,
        ecology_version=8,
        hidden_size=8,
        initial_neurons=4,
        min_neurons=2,
        initial_population=6,
    )
    w = EcologyWorld(target, seed=53)
    origins = seed_assembly(w, sources, condition)
    assert [int((origins == group).sum()) for group in (0, 1)] == expected
    assert origins.tolist() == w.seeded_from["founder_sources"]
    for group, diet in enumerate((2.0, -2.0)):
        assert (
            w.agents["genome"][origins == group, target.brain_parameter_count + 2] == diet
        ).all()
    assert (w.agents["neurons"] == 4).all()
    for key in ("age", "h", "module_h", "module_plastic", "module_trace", "distance"):
        assert w.agents[key].count_nonzero() == 0
    assert w.initial_energy == w.agents["energy"].double().sum().item()
    save_checkpoint(w, tmp_path / "assembled.pt")
    resumed = load_checkpoint(tmp_path / "assembled.pt")
    assert resumed.seeded_from == w.seeded_from
    torch.testing.assert_close(resumed.agents["genome"], w.agents["genome"], rtol=0, atol=0)
