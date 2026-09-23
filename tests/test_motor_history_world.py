from dataclasses import replace

import pytest
import torch

from emergent_garden.development import grow
from emergent_garden.exploration import history_shapes
from emergent_garden.world import World, create_world


def configuration(config, **changes):
    return replace(
        config,
        **{
            "ecology_version": 21,
            "hidden_size": 32,
            "initial_neuron_spread": 8,
            "initial_recurrent_density": 0.5,
            "motor_noise_tau": 2.0,
            "motor_learning_rate": 0.01,
            "motor_normalized": 1,
            "motor_learning_limit": 0.15,
            "exploration_min": 0.05,
            "exploration_max": 0.15,
            "initial_food": 24,
            "food_rate": 1,
            "gut_capacity": 200,
            **changes,
        },
    ).validate()


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


def test_neutral_correlation_preserves_legacy_physical_state_exactly(config):
    c = configuration(config, motor_noise_tau=0)
    old = create_world(replace(c, ecology_version=20))
    new = create_world(c)
    old.step(103)
    new.step(103)
    a, b = old.state_dict(), new.state_dict()
    a.pop("config")
    b.pop("config")
    for key in history_shapes(c.hidden_size):
        b["agents"].pop(f"module_{key}")
    same(a, b)


def test_history_replays_and_no_exploration_clears_a_warm_history(config):
    w = create_world(configuration(config))
    w.step(37)
    assert w.agents["module_motor_applied_noise"].abs().sum() > 0
    replay = World.from_state(w.state_dict())
    w.step(67)
    replay.step(67)
    same(w.state_dict(), replay.state_dict())
    assert w.totals["motor_learning_changes"] > 0
    w.ablation = "no_exploration"
    w.step(4)
    assert not w.agents["module_motor_history_ready"].any()
    assert w.agents["module_motor_applied_noise"].count_nonzero() == 0


def test_suppressing_learning_preserves_the_correlated_exploration_process(config):
    c = configuration(config, motor_learning_rate=0)
    normal = create_world(c)
    frozen = create_world(c, ablation="no_motor_learning")
    normal.step(93)
    frozen.step(93)
    for key in (
        "pos",
        "heading",
        "energy",
        "module_actions",
        *[f"module_{key}" for key in history_shapes(c.hidden_size)],
    ):
        torch.testing.assert_close(normal.agents[key], frozen.agents[key], rtol=0, atol=0)
    assert normal.totals == frozen.totals


def test_newborn_inherits_genes_but_not_motor_history(config):
    c = configuration(
        config,
        initial_population=1,
        initial_food=0,
        food_rate=0,
        mutation_probability=0,
        trait_mutation_probability=0,
        node_mutation_probability=0,
        edge_mutation_probability=0,
        module_mutation_probability=0,
    )
    w = create_world(c)
    w.agents["genome"][:, c.brain_parameter_count + 6] = -3
    w.develop(w.agents)
    w.agents["pos"].fill_(64)
    w.step(1)
    assert w.agents["module_motor_history_ready"][0, 0]
    w.agents["energy"] = c.max_energy * w.agents["area"]
    w.reproduce()
    assert w.population == 2
    torch.testing.assert_close(w.agents["genome"][0], w.agents["genome"][1], rtol=0, atol=0)
    for key in history_shapes(c.hidden_size):
        assert w.agents[f"module_{key}"][1].count_nonzero() == 0


def test_grown_module_begins_with_fresh_history(config):
    c = configuration(
        config, initial_population=1, initial_food=0, food_rate=0, growth_delay=1, growth_reserve=1
    )
    w = create_world(c)
    w.agents["genome"][:, c.brain_parameter_count + 6] = 2
    w.develop(w.agents)
    w.agents["pos"].fill_(64)
    w.step(1)
    w.agents["energy"] = c.max_energy * w.agents["area"]
    w.agents["growth_tick"].fill_(w.tick)
    grow(w)
    assert w.agents["modules"][0] == 2
    for key in history_shapes(c.hidden_size):
        assert w.agents[f"module_{key}"][0, 1].count_nonzero() == 0
    w.step(3)
    assert w.agents["module_motor_history_ready"][0, :2].all()
    assert not w.agents["module_motor_history_ready"][0, 2]


def test_correlated_exploration_version_validation_and_empty_world(config):
    with pytest.raises(ValueError, match="Correlated motor exploration requires"):
        configuration(config, ecology_version=20)
    w = create_world(configuration(config, initial_population=0))
    assert w.agents["module_motor_history_ready"].shape == (0, 3)
