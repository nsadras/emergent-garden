from dataclasses import replace

import pytest
import torch

from emergent_garden.brain import advance
from emergent_garden.challenges import fork_challenge
from emergent_garden.coordination import BODY_INPUTS, body_inputs, signal_cost, update_signals
from emergent_garden.development import grow
from emergent_garden.ecology import EcologyWorld
from emergent_garden.inheritance import brain_parts, upgrade_genomes
from emergent_garden.topology import counts


def organism(config, modules=3, **changes):
    c = replace(
        config,
        **{
            "ecology_version": 14,
            "initial_population": 1,
            "exploration_min": 0.0,
            "exploration_max": 0.0,
            "motor_normalized": 1,
            "mutation_probability": 0.0,
            "trait_mutation_probability": 0.0,
            "node_mutation_probability": 0.0,
            "edge_mutation_probability": 0.0,
            "module_mutation_probability": 0.0,
            **changes,
        },
    )
    w = EcologyWorld(c)
    a = w.agents
    a["genome"][:, c.brain_parameter_count : c.brain_parameter_count + c.trait_count] = 0
    a["genome"][:, c.brain_parameter_count + 6] = {1: -2, 2: 0, 3: 2}[modules]
    a["development_stage"].fill_(modules)
    w.develop(a)
    a["pos"][:] = c.diameter / 2
    a["energy"] = c.birth_energy * a["area"]
    w.initial_energy = a["energy"].double().sum().item()
    return w


def test_messages_travel_only_between_adjacent_active_modules_of_the_same_body(config):
    w = organism(config, initial_population=2)
    a = w.agents
    a["module_internal"][0] = torch.tensor([[0.2, -0.4], [0.6, 0.8], [-0.8, 0.2]])
    a["module_internal"][1].fill_(0.9)
    received = body_inputs(a, torch.arange(2))[..., 2:]
    torch.testing.assert_close(received[0], torch.tensor([[0.6, 0.8], [-0.3, -0.1], [0.6, 0.8]]))
    assert received[1].eq(0.9).all()
    # A's endpoint cannot hear itself or the far endpoint. A separate body
    # colocated with it still cannot leak into its private channel.
    a["module_internal"][1].fill_(-0.9)
    torch.testing.assert_close(
        body_inputs(a, torch.tensor([0]))[0, :, 2:], received[0], rtol=0, atol=0
    )
    a["module_mask"][0, 2] = False
    received = body_inputs(a, torch.tensor([0]))[0, :, 2:]
    torch.testing.assert_close(received[:2], torch.tensor([[0.6, 0.8], [0.2, -0.4]]))
    assert received[2].count_nonzero() == 0


@pytest.mark.parametrize("modules", [1, 2, 3])
def test_body_position_is_local_bounded_and_tracks_expressed_geometry(config, modules):
    w = organism(config, modules)
    index = torch.tensor([0])
    before = body_inputs(w.agents, index)
    assert before[..., :2].abs().max() <= 1
    assert before[:, modules:].count_nonzero() == 0
    w.agents["pos"] += 23
    w.agents["heading"] += 1.2
    torch.testing.assert_close(body_inputs(w.agents, index), before, rtol=0, atol=0)
    torch.testing.assert_close(before[..., :2].sum(1), torch.zeros(1, 2), atol=1e-7, rtol=0)
    if modules == 1:
        assert before.count_nonzero() == 0


def test_new_emissions_are_read_only_at_the_next_controller_update(config):
    w = organism(config)
    a, c = w.agents, w.config
    a["genome"][:, : c.brain_parameter_count] = 0
    wi, _, _, wo, bias = brain_parts(c, a["genome"])
    wi[:, 0, c.input_names.index("internal_a")] = 1
    wo[:, 0, 0] = 1
    bias[:, 5] = 2
    w.update_controllers()
    assert a["module_h"].count_nonzero() == 0
    assert a["module_actions"][..., :2].eq(0.5).all()
    assert (a["module_internal"][..., 0] > 0).all()
    w.tick += c.physics_hz // c.controller_hz
    w.update_controllers()
    assert (a["module_h"][..., 0] > 0).all()
    assert (a["module_actions"][..., 0] > 0.5).all()


def test_reception_and_position_controls_are_separate_and_keep_emission_cost(config):
    w = organism(config)
    w.agents["module_internal"].fill_(0.4)
    start = w.config.input_names.index(BODY_INPUTS[0])
    reference = w.module_sensors(torch.tensor([0]))
    cost = signal_cost(w.config, w.agents)
    for mode, absent, present in (
        ("no_internal", slice(start + 2, start + 4), slice(start, start + 2)),
        ("no_body_sense", slice(start, start + 2), slice(start + 2, start + 4)),
        ("no_coordination", slice(start, start + 4), slice(None, start)),
    ):
        w.ablation = mode
        value = w.module_sensors(torch.tensor([0]))
        assert value[..., absent].count_nonzero() == 0
        torch.testing.assert_close(value[..., present], reference[..., present], rtol=0, atol=0)
        torch.testing.assert_close(signal_cost(w.config, w.agents), cost, rtol=0, atol=0)
    assert w.agents["module_internal"].eq(0.4).all()


def test_signal_cost_is_paid_once_and_limited_by_available_energy(config):
    w = organism(config)
    a, c = w.agents, w.config
    a["cold"][:] = False
    a["module_internal"].fill_(0.5)
    w.tick = 1  # No control update; test the held signal's physical energy cost.
    expected = signal_cost(c, a).item()
    before = a["energy"].item()
    w.step()
    assert w.totals["internal_signaling_cost"] == pytest.approx(expected)
    assert a["internal_spent"].item() == pytest.approx(expected)
    assert before - a["energy"].item() == pytest.approx(a["spent"].item(), abs=2e-5)
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4
    previous = a["energy"].item()
    a["energy"].fill_(0.00001)
    w.initial_energy -= previous - a["energy"].item()
    w.tick = 2
    w.step()
    assert w.population == 0
    assert w.totals["internal_signaling_cost"] - expected <= 0.00001
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4


def test_self_signal_control_preserves_local_memory_but_removes_neighbor_information(config):
    w = organism(config, 2)
    a = w.agents
    a["module_internal"][0, :2] = torch.tensor([[0.2, -0.3], [0.6, 0.7]])
    start = w.config.input_names.index("internal_a")
    normal = w.module_sensors(torch.tensor([0]))[..., start : start + 2]
    w.ablation = "self_internal"
    local = w.module_sensors(torch.tensor([0]))[..., start : start + 2]
    torch.testing.assert_close(normal[0, :2], a["module_internal"][0, :2].flip(0))
    torch.testing.assert_close(local[0, :2], a["module_internal"][0, :2])
    assert local[0, 2].count_nonzero() == 0
    one = organism(config, 1)
    one.agents["module_internal"][:, 0].fill_(0.5)
    one.ablation = "self_internal"
    assert one.module_sensors(torch.tensor([0]))[..., start : start + 2].count_nonzero() == 0


def test_newborn_and_newly_grown_modules_have_fresh_signal_state(config):
    w = organism(config, 2)
    a, c = w.agents, w.config
    a["module_internal"][:, :2] = 0.4
    a["energy"] = c.max_energy * a["area"]
    w.initial_energy = a["energy"].double().sum().item()
    w.reproduce()
    a = w.agents
    assert w.population == 2
    assert a["module_internal"][1].count_nonzero() == 0
    assert a["module_internal"][0, :2].eq(0.4).all()
    a["pos"][1] = torch.tensor([95.0, 64])
    a["energy"][1] = c.max_energy * a["area"][1]
    a["module_internal"][1, 0] = 0.3
    w.tick = int(a["growth_tick"][1])
    grow(w)
    assert a["modules"][1] == 2
    assert a["module_internal"][1, 0].eq(0.3).all()
    assert a["module_internal"][1, 1:].count_nonzero() == 0


def test_transfer_preserves_old_circuit_and_enables_neutral_coordination_paths(config):
    source = replace(config, ecology_version=13)
    target = replace(source, ecology_version=14)
    w = EcologyWorld(source)
    g = w.agents["genome"]
    expanded = upgrade_genomes(source, target, g)
    old_inputs = torch.randn(len(g), source.input_size)
    new_inputs = torch.randn(len(g), target.input_size)
    for i, name in enumerate(source.input_names):
        new_inputs[:, target.input_names.index(name)] = old_inputs[:, i]
    hidden = torch.randn(len(g), source.hidden_size)
    oh, oa = advance(source, g, old_inputs, hidden)
    nh, na = advance(target, expanded, new_inputs, hidden)
    torch.testing.assert_close(nh, oh, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(na[:, :5], oa, rtol=1e-6, atol=1e-7)
    assert na[:, 5:].eq(0.5).all()
    before, after = counts(source, g), counts(target, expanded)
    assert (after[1] - before[1] == 6 * before[0]).all()
    torch.testing.assert_close(
        expanded[
            :, target.brain_parameter_count : target.brain_parameter_count + target.trait_count
        ],
        g[:, source.brain_parameter_count : source.brain_parameter_count + source.trait_count],
        rtol=0,
        atol=0,
    )


def test_internal_channels_are_erased_with_neural_activity(config):
    w = organism(config)
    w.agents["module_internal"].fill_(0.3)
    erased = fork_challenge(w, "erase_activity", False, 100)
    assert erased.agents["module_internal"].count_nonzero() == 0
    assert w.agents["module_internal"].eq(0.3).all()
    w.ablation = "memory_reset"
    w.update_controllers()
    start = w.config.input_names.index("internal_a")
    assert w.agents["inputs"][:, start : start + 2].count_nonzero() == 0


def test_message_filter_is_bounded_and_inactive_modules_stay_zero(config):
    w = organism(config, 1)
    w.agents["module_actions"].fill_(1)
    for _ in range(50):
        update_signals(w.config, w.agents, torch.tensor([0]))
    values = w.agents["module_internal"]
    assert (values[:, 0] > 0.99).all() and values.max() <= 1
    assert values[:, 1:].count_nonzero() == 0
    assert body_inputs(w.agents, torch.tensor([0]))[..., 2:].count_nonzero() == 0


def test_coordination_sensor_channels_do_not_replace_energy_and_feedback(config):
    w = organism(config)
    a, c = w.agents, w.config
    a["food_feedback"].fill_(12)
    a["contact"].fill_(1)
    inputs = w.module_sensors(torch.tensor([0]))
    assert inputs.shape == (1, 3, c.input_size)
    assert inputs[..., -1].eq(1).all()
    assert inputs[..., -2].eq(c.birth_energy / c.max_energy).all()
    assert (inputs[..., c.input_names.index("food_feedback")] > 0).all()
