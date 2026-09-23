from dataclasses import replace

import pytest
import torch

from emergent_garden.brain import initial_brains
from emergent_garden.inheritance import brain_parts
from emergent_garden.topology import effective_masks, initial_structure, mask_parts
from emergent_garden.world import World, create_world


def varied(config, **changes):
    return replace(
        config,
        **{
            "ecology_version": 19,
            "hidden_size": 32,
            "initial_neuron_spread": 8,
            "initial_recurrent_density": 0.5,
            "gut_capacity": 200.0,
            **changes,
        },
    ).validate()


def same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            same(x, y)
    else:
        assert left == right


def test_founder_graphs_vary_without_changing_bodies_placement_or_other_random_streams(config):
    c = varied(config, initial_population=32, capacity=64)
    w = create_world(c, seed=43)
    control = create_world(
        replace(c, initial_neuron_spread=0, initial_recurrent_density=1.0), seed=43
    )
    counts = w.agents["neurons"]
    assert counts.unique().numel() > 8
    assert counts.min() >= 8 and counts.max() <= 24
    nodes, mi, mr, mo = effective_masks(c, w.founders)
    density = mr.sum().item() / counts.square().sum().item()
    assert 0.4 < density < 0.6
    assert (mi.sum((1, 2)) == counts * c.input_size).all()
    assert (mo.sum((1, 2)) == counts * c.output_size).all()
    assert not (mr & ~(nodes[:, :, None] & nodes[:, None, :])).any()
    for key in ("pos", "heading", "radius", "diet", "power", "module_mask", "energy"):
        torch.testing.assert_close(w.agents[key], control.agents[key], rtol=0, atol=0)
    torch.testing.assert_close(w.patch_positions, control.patch_positions, rtol=0, atol=0)
    for key in w.rng.keys() - {"founder_structure"}:
        assert torch.equal(w.rng[key].get_state(), control.rng[key].get_state())
    expected = c.neuron_maintenance * counts + c.synapse_maintenance * w.agents["connections"]
    torch.testing.assert_close(w.agents["brain_maintenance"], expected)


@pytest.mark.parametrize("density", [0.0, 0.5, 1.0])
def test_weight_initialization_scales_with_width_and_expected_recurrent_fan_in(config, density):
    c = varied(config, initial_recurrent_density=density, weight_limit=100.0)
    number = torch.tensor([8.0, 24.0])
    reference = initial_brains(c, 2, "cpu", torch.Generator().manual_seed(79))
    actual = initial_brains(c, 2, "cpu", torch.Generator().manual_seed(79), active_counts=number)
    a, b = brain_parts(c, reference), brain_parts(c, actual)
    for k in (0, 2, 4):
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    recurrent_scale = (c.initial_neurons / (number * density).clamp_min(1)).sqrt()
    output_scale = (c.initial_neurons / number).sqrt()
    torch.testing.assert_close(b[1], a[1] * recurrent_scale[:, None, None], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(b[3], a[3] * output_scale[:, None, None], rtol=1e-6, atol=1e-7)
    w = create_world(c)
    if density == 0:
        assert effective_masks(c, w.founders)[2].count_nonzero() == 0
    w.step(20)
    assert torch.isfinite(w.agents["module_h"]).all()


def test_variable_graph_is_inherited_without_redrawing_or_rescaling_at_birth(config):
    c = varied(
        config,
        initial_population=1,
        mutation_probability=0,
        trait_mutation_probability=0,
        node_mutation_probability=0,
        edge_mutation_probability=0,
        module_mutation_probability=0,
    )
    w = create_world(c)
    w.agents["genome"][:, c.brain_parameter_count + 6] = -3
    w.develop(w.agents)
    w.agents["pos"][:] = 64
    w.agents["energy"] = c.max_energy * w.agents["area"]
    w.agents["module_plastic"].fill_(0.05)
    w.agents["module_motor_plastic"].fill_(0.05)
    rng = w.rng["founder_structure"].get_state().clone()
    w.reproduce()
    assert w.population == 2
    torch.testing.assert_close(w.agents["genome"][0], w.agents["genome"][1], rtol=0, atol=0)
    assert w.agents["neurons"][1] == w.agents["neurons"][0]
    for key in (
        "module_h",
        "module_plastic",
        "module_trace",
        "module_motor_plastic",
        "module_motor_trace",
    ):
        assert w.agents[key][1].count_nonzero() == 0
    assert torch.equal(w.rng["founder_structure"].get_state(), rng)


def test_neutral_v19_is_exactly_v18_and_variable_learning_world_replays(config):
    c = replace(config, ecology_version=18, gut_capacity=200.0, initial_food=12, food_rate=1.0)
    old = create_world(c)
    neutral = create_world(replace(c, ecology_version=19))
    old.step(87)
    neutral.step(87)
    a, b = old.state_dict(), neutral.state_dict()
    a.pop("config")
    b.pop("config")
    b["rng"].pop("founder_structure")
    same(a, b)
    w = create_world(
        varied(
            config,
            initial_food=12,
            food_rate=1.0,
            exploration_min=0.05,
            exploration_max=0.15,
            motor_learning_rate=0.01,
            motor_normalized=1,
        )
    )
    w.step(19)
    replay = World.from_state(w.state_dict())
    w.step(47)
    replay.step(47)
    same(w.state_dict(), replay.state_dict())
    assert w.totals["motor_learning_changes"] > 0
    for part in mask_parts(w.config, w.agents["genome"]):
        assert ((part == 0) | (part == 1)).all()


def test_founder_configuration_requires_valid_ranges_and_explicit_randomness(config):
    with pytest.raises(ValueError, match="Variable founder circuits require ecology_version"):
        varied(config, ecology_version=18)
    with pytest.raises(ValueError, match="Founder neuron range"):
        varied(config, initial_neuron_spread=15)
    with pytest.raises(ValueError, match="initial_recurrent_density"):
        varied(config, initial_recurrent_density=1.1)
    with pytest.raises(ValueError, match="explicit random generator"):
        initial_structure(varied(config), 2, "cpu")
    empty = create_world(varied(config, initial_population=0))
    assert empty.founders.shape == (0, empty.config.parameter_count)
    assert empty.metrics()["population"] == 0
