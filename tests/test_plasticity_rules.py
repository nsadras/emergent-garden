from dataclasses import replace

import pytest
import torch

from emergent_garden.brain import controller_step, initial_state
from emergent_garden.ecology import EcologyWorld
from emergent_garden.inheritance import brain_parts, upgrade_genomes
from emergent_garden.plasticity import rule_coefficients
from emergent_garden.probes import association_probe
from emergent_garden.topology import effective_masks


def circuit(config, **changes):
    c = replace(
        config,
        **{
            "ecology_version": 15,
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
    ).validate()
    w = EcologyWorld(c)
    return w, w.agents["genome"].clone()


def set_rule(c, genome, values):
    genome[:, c.brain_parameter_count + 13 : c.brain_parameter_count + 17] = torch.tensor(values)


def test_rule_budget_preserves_sign_and_small_coefficients(config):
    w, g = circuit(config, initial_population=3)
    set_rule(w.config, g, [[3.0, -3.0, 3.0, -3.0], [0.2, -0.1, 0.0, 0.0], [0.0] * 4])
    before = g.clone()
    values = rule_coefficients(w.config, g)
    torch.testing.assert_close(values[0], torch.tensor([0.25, -0.25, 0.25, -0.25]))
    torch.testing.assert_close(values[1], torch.tensor([0.2, -0.1, 0.0, 0.0]))
    assert values[2].count_nonzero() == 0
    assert (values.abs().sum(1) <= 1).all()
    torch.testing.assert_close(g, before, rtol=0, atol=0)


def test_transfer_preserves_every_step_of_the_existing_plastic_controller(config):
    w, _ = circuit(config, initial_population=2)
    old = replace(w.config, ecology_version=14)
    parent = EcologyWorld(old)
    g = parent.agents["genome"].clone()
    upgraded = upgrade_genomes(old, w.config, g)
    first, second = initial_state(old, g), initial_state(w.config, upgraded)
    rng = torch.Generator().manual_seed(971)
    for _ in range(100):
        inputs = torch.rand((len(g), old.input_size), generator=rng)
        first, a = controller_step(old, g, inputs, first)
        second, b = controller_step(w.config, upgraded, inputs, second)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert first["plastic"].count_nonzero() > 0


def test_constant_rule_can_change_silent_connections_but_fixed_rule_cannot(config):
    w, g = circuit(config, hidden_size=20, initial_neurons=8)
    c = w.config
    g[:, : c.brain_parameter_count] = 0
    brain_parts(c, g)[4][:, 4] = 2
    set_rule(c, g, [0.0, 0.0, 0.0, 1.0])
    before = g.clone()
    state = initial_state(c, g)
    inputs = torch.zeros((1, c.input_size))
    changed, _ = controller_step(c, g, inputs, state)
    fixed, _ = controller_step(c, g, inputs, state, evolved_rule=False)
    mask = effective_masks(c, g)[2].bool()
    assert changed["hidden"].count_nonzero() == 0
    assert (changed["plastic"][mask] > 0).all()
    assert changed["plastic"][~mask].count_nonzero() == 0
    assert changed["trace"][~mask].count_nonzero() == 0
    assert fixed["plastic"].count_nonzero() == 0
    torch.testing.assert_close(g, before, rtol=0, atol=0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_signed_rules_remain_bounded_and_freezing_removes_offsets(config, sign):
    w, g = circuit(config, plasticity_rate=1000.0, plasticity_limit=0.1)
    c = w.config
    set_rule(c, g, [0.0, 0.0, 0.0, sign * 3])
    brain_parts(c, g)[4][:, 4] = 3
    state = initial_state(c, g)
    inputs = torch.zeros((1, c.input_size))
    for _ in range(100):
        state, _ = controller_step(c, g, inputs, state)
    active = effective_masks(c, g)[2].bool()
    assert (state["plastic"][active] == sign * c.plasticity_limit).all()
    assert state["trace"].abs().max() <= 1
    frozen, _ = controller_step(c, g, inputs, state, plasticity=False)
    assert frozen["plastic"].count_nonzero() == frozen["trace"].count_nonzero() == 0


def test_fixed_rule_intervention_keeps_genomes_costs_and_other_channels(config):
    w, _ = circuit(config)
    set_rule(w.config, w.agents["genome"], [0.0, 0.0, 0.0, 1.0])
    other = EcologyWorld.from_state(w.state_dict())
    other.ablation = "fixed_rule"
    before = w.agents["genome"].clone()
    w.step()
    other.step()
    for key in ("energy", "pos", "module_actions", "module_internal"):
        torch.testing.assert_close(w.agents[key], other.agents[key], rtol=0, atol=0)
    assert w.totals["plasticity_cost"] == other.totals["plasticity_cost"] > 0
    assert w.agents["module_plastic"].abs().sum() > 0
    assert other.agents["module_plastic"].count_nonzero() == 0
    torch.testing.assert_close(w.agents["genome"], before, rtol=0, atol=0)
    assert other.agent_record(0)["learning_rule"] == [1, 0, 0, 0]
    assert other.agent_record(0)["encoded_learning_rule"] == [0, 0, 0, 1]


def test_children_inherit_rule_genes_with_fresh_neural_state(config):
    w, _ = circuit(config)
    a, c = w.agents, w.config
    a["genome"][:, c.brain_parameter_count + 6] = -2
    set_rule(c, a["genome"], [0.3, -0.2, 0.4, 0.1])
    w.develop(a)
    a["pos"][0] = 64
    a["energy"] = c.max_energy * a["area"]
    a["module_h"][:, 0] = 0.3
    a["module_plastic"][:, 0] = 0.1
    a["module_trace"][:, 0] = 0.5
    w.reproduce()
    assert w.population == 2
    torch.testing.assert_close(w.agents["genome"][0], w.agents["genome"][1], rtol=0, atol=0)
    for key in ("module_h", "module_plastic", "module_trace", "module_internal"):
        assert w.agents[key][1].count_nonzero() == 0
    other, g = circuit(config, trait_mutation_probability=1.0)
    changed = other.mutate(g[0])
    start = other.config.brain_parameter_count + 13
    assert (changed[start : start + 4] != g[0, start : start + 4]).all()
    assert rule_coefficients(other.config, changed[None]).abs().sum() <= 1 + 1e-6


def test_native_founders_preserve_v14_initial_physics_before_rule_mutation(config):
    w, _ = circuit(config, initial_population=2, initial_food=10, food_rate=2.0)
    old = EcologyWorld(replace(w.config, ecology_version=14))
    upgraded = upgrade_genomes(old.config, w.config, old.agents["genome"])
    torch.testing.assert_close(w.agents["genome"], upgraded, rtol=0, atol=0)
    w.step(120)
    old.step(120)
    assert w.totals == old.totals
    for key in w.agents:
        if key != "genome":
            torch.testing.assert_close(w.agents[key], old.agents[key], rtol=0, atol=0)
    for key in w.rng:
        assert torch.equal(w.rng[key].get_state(), old.rng[key].get_state())


def test_association_assay_exposes_fixed_rule_control_without_mutating_genomes(config):
    w, g = circuit(config)
    set_rule(w.config, g, [0.2, 0.2, -0.3, 0.1])
    before = g.clone()
    report = association_probe(w.config, g, rounds=1, mode="fixed_rule")
    assert report["mode"] == "fixed_rule"
    torch.testing.assert_close(g, before, rtol=0, atol=0)
