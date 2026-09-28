from dataclasses import replace

import pytest
import torch

from emergent_garden.challenges import fork_challenge
from emergent_garden.development import grow
from emergent_garden.value import value_shapes
from emergent_garden.world import World, create_world


def configuration(config, **changes):
    return replace(
        config,
        **{
            "ecology_version": 22,
            "hidden_size": 32,
            "initial_neuron_spread": 8,
            "initial_recurrent_density": 0.5,
            "motor_learning_rate": 0.01,
            "motor_normalized": 1,
            "motor_learning_limit": 0.15,
            "motor_value_rate": 0.02,
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


@pytest.mark.parametrize("disabled", ["config", "ablation"])
def test_neutral_or_disabled_value_predictor_preserves_legacy_complete_state(config, disabled):
    c = configuration(config, motor_noise_tau=2)
    old = create_world(replace(c, ecology_version=21, motor_value_rate=0))
    new = create_world(
        replace(c, motor_value_rate=0) if disabled == "config" else c,
        ablation="no_motor_value" if disabled == "ablation" else "none",
    )
    old.step(103)
    new.step(103)
    a, b = old.state_dict(), new.state_dict()
    for state in (a, b):
        state.pop("config")
        state.pop("ablation")
    for key in value_shapes(c.hidden_size):
        assert not b["agents"].pop(f"module_{key}").count_nonzero()
    same(a, b)


def test_neutral_sensory_value_setting_preserves_complete_v22_state(config):
    c = configuration(config, motor_noise_tau=2, motor_value_centered=0)
    old = create_world(c)
    new = create_world(replace(c, ecology_version=23))
    old.step(103)
    new.step(103)
    a, b = old.state_dict(), new.state_dict()
    a.pop("config")
    b.pop("config")
    same(a, b)


@pytest.mark.parametrize("disabled", ["config", "ablation"])
def test_disabled_sensory_head_preserves_running_mean_actor(config, disabled):
    c = configuration(config, motor_value_rate=0, motor_noise_tau=2)
    old = create_world(c)
    new = create_world(
        replace(
            c, ecology_version=23, motor_value_inputs=1,
            motor_value_rate=0 if disabled == "config" else 0.02,
        ), ablation="no_motor_value" if disabled == "ablation" else "none",
    )
    old.step(103)
    new.step(103)
    a, b = old.state_dict(), new.state_dict()
    for state in (a, b):
        state.pop("config")
        state.pop("ablation")
        for key in value_shapes(c.hidden_size):
            assert not state["agents"].pop(f"module_{key}").count_nonzero()
    same(a, b)


def test_sensory_value_configuration_and_empty_world(config):
    with pytest.raises(ValueError, match="ecology_version >= 23"):
        configuration(config, motor_value_inputs=1)
    with pytest.raises(ValueError, match="motor_value_inputs must"):
        configuration(config, ecology_version=23, motor_value_inputs=2)
    w = create_world(configuration(config, ecology_version=23, motor_value_inputs=1,
                                   initial_population=0))
    assert w.agents["module_motor_value_weights"].shape == (0, 3, 75)
    assert w.metrics()["mean_motor_value_norm"] == 0


@pytest.mark.parametrize("ablation", ["none", "shuffled_motor_reward"])
@pytest.mark.parametrize("centered", [0, 1])
@pytest.mark.parametrize("inputs", [0, 1])
def test_value_learning_replays_and_remains_bounded(config, ablation, centered, inputs):
    w = create_world(
        configuration(
            config, motor_noise_tau=2, motor_value_centered=centered,
            ecology_version=23 if inputs else 22, motor_value_inputs=inputs,
        ), ablation=ablation,
    )
    w.step(37)
    assert w.agents["module_motor_value_weights"].abs().sum() > 0
    assert w.metrics()["motor_value_error_rms"] > 0
    replay = World.from_state(w.state_dict())
    w.step(67)
    replay.step(67)
    same(w.state_dict(), replay.state_dict())
    mask = w.agents["module_mask"]
    for key in value_shapes(w.config.hidden_size):
        assert not w.agents[f"module_{key}"][~mask].count_nonzero()
    assert w.agents["module_motor_value_weights"].norm(dim=-1).max() <= w.config.motor_value_limit


@pytest.mark.parametrize("ablation", ["no_motor_learning", "no_plasticity"])
@pytest.mark.parametrize("inputs", [0, 1])
def test_disabled_learning_has_no_acquired_predictor_but_retains_capacity_cost(
    config, ablation, inputs
):
    w = create_world(
        configuration(config, ecology_version=23 if inputs else 22, motor_value_inputs=inputs),
        ablation=ablation,
    )
    w.step(73)
    for key in value_shapes(w.config.hidden_size):
        assert not w.agents[f"module_{key}"].count_nonzero()
    assert w.totals["motor_learning_cost"] > 0
    assert w.agents["module_motor_applied_noise"].abs().sum() > 0


@pytest.mark.parametrize("event", ["birth", "growth"])
@pytest.mark.parametrize("inputs", [0, 1])
def test_new_neural_modules_start_without_acquired_predictions(config, event, inputs):
    c = configuration(
        config,
        ecology_version=23 if inputs else 22,
        motor_value_inputs=inputs,
        initial_population=1,
        initial_food=0,
        food_rate=0,
        growth_delay=1,
        growth_reserve=1,
        mutation_probability=0,
        trait_mutation_probability=0,
        node_mutation_probability=0,
        edge_mutation_probability=0,
        module_mutation_probability=0,
    )
    w = create_world(c)
    w.agents["genome"][:, c.brain_parameter_count + 6] = 2 if event == "growth" else -3
    w.develop(w.agents)
    w.agents["pos"].fill_(64)
    w.step(4)
    assert w.agents["module_motor_value_weights"].abs().sum() > 0
    w.agents["energy"] = c.max_energy * w.agents["area"]
    if event == "birth":
        w.reproduce()
        assert w.population == 2
        torch.testing.assert_close(w.agents["genome"][0], w.agents["genome"][1], rtol=0, atol=0)
        for key in value_shapes(c.hidden_size):
            assert not w.agents[f"module_{key}"][1].count_nonzero()
    else:
        w.agents["growth_tick"].fill_(w.tick)
        grow(w)
        assert w.agents["modules"][0] == 2
        for key in value_shapes(c.hidden_size):
            assert not w.agents[f"module_{key}"][0, 1].count_nonzero()
        w.step(3)
        assert w.agents["module_motor_value_ready"][0, :2].all()
        assert not w.agents["module_motor_value_weights"][0, 1].count_nonzero()


def test_value_configuration_guards_and_empty_world(config):
    with pytest.raises(ValueError, match="Motor value prediction requires ecology"):
        configuration(config, ecology_version=21)
    with pytest.raises(ValueError, match="normalized neural features"):
        configuration(config, motor_normalized=0)
    with pytest.raises(ValueError, match="at least both trace durations"):
        configuration(config, motor_value_horizon=1)
    with pytest.raises(ValueError, match="motor_value_centered must"):
        configuration(config, motor_value_centered=2)
    with pytest.raises(ValueError, match="version containing that feature"):
        create_world(
            replace(configuration(config), ecology_version=21, motor_value_rate=0),
            ablation="no_motor_value",
        )
    w = create_world(configuration(config, initial_population=0))
    assert w.agents["module_motor_value_weights"].shape == (0, 3, 33)
    assert w.metrics()["mean_motor_value_norm"] == 0


@pytest.mark.parametrize("mode", ["erase_plastic", "no_plasticity", "erase_activity"])
@pytest.mark.parametrize("inputs", [0, 1])
def test_acquired_state_challenges_include_the_value_predictor(config, mode, inputs):
    w = create_world(
        configuration(config, ecology_version=23 if inputs else 22, motor_value_inputs=inputs)
    )
    w.step(73)
    assert w.agents["module_motor_value_weights"].abs().sum() > 0
    changed = fork_challenge(w, mode, False, w.tick + 300)
    for key in ("pos", "energy", "genome", "module_motor_applied_noise"):
        torch.testing.assert_close(w.agents[key], changed.agents[key], rtol=0, atol=0)
    for key in value_shapes(w.config.hidden_size):
        if mode == "erase_activity":
            torch.testing.assert_close(
                w.agents[f"module_{key}"], changed.agents[f"module_{key}"], rtol=0, atol=0
            )
        else:
            assert not changed.agents[f"module_{key}"].count_nonzero()
