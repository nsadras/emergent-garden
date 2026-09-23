from dataclasses import replace

import pytest
import torch

from emergent_garden.senses import receptor_values
from emergent_garden.world import World, create_world


def same(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("contrast", [0, 1])
def test_wider_sampling_preserves_mean_of_linear_field_and_increases_spatial_signal(
    config, contrast
):
    c = replace(config, ecology_version=20, sensory_contrast=contrast, initial_population=1)
    worlds = [create_world(replace(c, sensor_radius_scale=scale)) for scale in (1, 4)]
    observations = []
    for w in worlds:
        w.agents["pos"].fill_(64)
        w.agents["heading"].zero_()
        field = w.fields[0]
        n = c.grid_size
        field.grid = torch.arange(n).float()[:, None].expand(n, n).clone() / n
        values = w.module_sensors(torch.tensor([0]))[0, 0, :4]
        if contrast:
            raw = receptor_values(values) * (c.smell_scale / (1 - values[0]))
        else:
            raw = values * c.smell_scale / (1 - values)
        observations.append(raw)
        assert raw[:2].mean() < raw[2:].mean()  # Right-rich field, for both footprints.
    near, wide = observations
    torch.testing.assert_close(near.mean(), wide.mean(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(wide[2] - wide[1], 4 * (near[2] - near[1]), rtol=1e-5, atol=1e-7)
    same(worlds[0].agents, worlds[1].agents)


def test_radius_changes_only_observation_not_genomes_costs_or_uncontrolled_physics(config):
    c = replace(config, ecology_version=20, initial_food=24, food_rate=1, gut_capacity=200)
    a = create_world(c, controller="rest")
    b = create_world(replace(c, sensor_radius_scale=4), controller="rest")
    same(a.agents, b.agents)
    a.step(51)
    b.step(51)
    left, right = a.state_dict(), b.state_dict()
    left.pop("config")
    right.pop("config")
    # Cached receptor readings should differ; every other physical state must match.
    assert not torch.equal(left["agents"]["inputs"], right["agents"]["inputs"])
    left["agents"].pop("inputs")
    right["agents"].pop("inputs")
    same(left, right)


def test_neutral_radius_is_exactly_v19_and_larger_radius_replays(config):
    c = replace(config, ecology_version=19, initial_food=24, food_rate=1, gut_capacity=200)
    a, b = create_world(c), create_world(replace(c, ecology_version=20))
    a.step(83)
    b.step(83)
    left, right = a.state_dict(), b.state_dict()
    left.pop("config")
    right.pop("config")
    same(left, right)
    w = create_world(replace(c, ecology_version=20, sensor_radius_scale=4), ablation="shuffled")
    w.step(19)
    replay = World.from_state(w.state_dict())
    w.step(47)
    replay.step(47)
    same(w.state_dict(), replay.state_dict())


def test_radius_validation_and_empty_population(config):
    with pytest.raises(ValueError, match="sensor radius requires"):
        replace(config, ecology_version=19, sensor_radius_scale=4).validate()
    with pytest.raises(ValueError, match="sensor_radius_scale must be positive"):
        replace(config, ecology_version=20, sensor_radius_scale=0).validate()
    w = create_world(
        replace(config, ecology_version=20, sensor_radius_scale=4, initial_population=0)
    )
    assert w.module_sensors(torch.empty(0, dtype=torch.long)).shape == (0, 3, w.config.input_size)
