from dataclasses import replace

import pytest
import torch

from emergent_garden.brain import advance, controller_step, initial_state
from emergent_garden.ecology import EcologyWorld
from emergent_garden.inheritance import brain_parts, upgrade_genomes


def plastic_config(config, **changes):
    return replace(config, ecology_version=7, **changes).validate()


def tonic_circuit(config, modulation=2.0):
    genome = torch.zeros((1, config.parameter_count))
    parts = brain_parts(config, genome)
    parts[2].fill_(1.0)
    parts[4][:, 4] = modulation
    return genome


def test_plastic_offsets_change_without_changing_inherited_weights(config):
    c = plastic_config(config)
    genome = tonic_circuit(c)
    original = genome.clone()
    state = initial_state(c, genome)
    inputs = torch.zeros((1, c.input_size))
    for _ in range(200):
        state, _ = controller_step(c, genome, inputs, state)
    assert state["plastic"].abs().max() > 0.01
    assert state["trace"].abs().max() > 0.1
    torch.testing.assert_close(genome, original, rtol=0, atol=0)


def test_modulation_can_strengthen_or_weaken_and_offsets_are_bounded(config):
    c = plastic_config(config, plasticity_rate=1000.0)
    positive = tonic_circuit(c, 3.0)
    negative = tonic_circuit(c, -3.0)
    state = initial_state(c, positive)
    state["hidden"].fill_(1.0)
    state["trace"].fill_(1.0)
    inputs = torch.zeros((1, c.input_size))
    up, _ = controller_step(c, positive, inputs, state)
    down, _ = controller_step(c, negative, inputs, state)
    torch.testing.assert_close(up["plastic"], -down["plastic"], rtol=0, atol=0)
    assert (up["plastic"] == c.plasticity_limit).all()
    assert (state["plastic"] == 0).all()


@pytest.mark.parametrize("hz", [10, 20])
def test_synaptic_half_life_is_in_simulated_seconds(config, hz):
    c = plastic_config(
        config, controller_hz=hz, plasticity_half_life_min=0.5, plasticity_half_life_max=0.5
    )
    genome = torch.zeros((1, c.parameter_count))
    state = initial_state(c, genome)
    state["plastic"].fill_(1)
    inputs = torch.zeros((1, c.input_size))
    for _ in range(hz):
        state, _ = controller_step(c, genome, inputs, state)
    torch.testing.assert_close(state["plastic"], torch.full_like(state["plastic"], 0.25))


def test_frozen_control_uses_inherited_circuit_and_erases_acquired_state(config):
    c = plastic_config(config)
    genome = tonic_circuit(c)
    state = initial_state(c, genome)
    state["hidden"].fill_(0.2)
    state["plastic"].fill_(0.4)
    state["trace"].fill_(0.8)
    inputs = torch.ones((1, c.input_size))
    expected_h, expected_actions = advance(c, genome, inputs, state["hidden"])
    frozen, actions = controller_step(c, genome, inputs, state, plasticity=False)
    torch.testing.assert_close(frozen["hidden"], expected_h, rtol=0, atol=0)
    torch.testing.assert_close(actions, expected_actions, rtol=0, atol=0)
    assert frozen["plastic"].count_nonzero() == frozen["trace"].count_nonzero() == 0


def test_transfer_neutralizes_plasticity_while_preserving_existing_outputs(config):
    c = plastic_config(config)
    source = EcologyWorld(replace(c, ecology_version=6))
    old = source.agents["genome"]
    new = upgrade_genomes(source.config, c, old)
    old_state, new_state = initial_state(source.config, old), initial_state(c, new)
    for _ in range(40):
        inputs = torch.rand((len(old), c.input_size))
        old_state, a = controller_step(source.config, old, inputs, old_state)
        new_state, b = controller_step(c, new, inputs, new_state)
        torch.testing.assert_close(a, b[:, :4], rtol=1e-6, atol=1e-7)
        assert (b[:, 4] == 0.5).all()
        assert new_state["plastic"].count_nonzero() == 0


def test_children_inherit_learning_rules_but_not_acquired_synapses(config):
    c = plastic_config(
        config, initial_population=1, mutation_probability=0.0, trait_mutation_probability=0.0
    )
    w = EcologyWorld(c)
    w.agents["pos"][0] = 64
    w.agents["energy"][0] = c.reproduction_threshold * w.agents["area"][0]
    w.agents["module_h"].fill_(0.3)
    w.agents["module_plastic"].fill_(0.4)
    w.agents["module_trace"].fill_(0.5)
    w.reproduce()
    assert w.population == 2
    torch.testing.assert_close(w.agents["genome"][0], w.agents["genome"][1], rtol=0, atol=0)
    for key in ("module_h", "module_plastic", "module_trace"):
        assert w.agents[key][1].count_nonzero() == 0
        assert w.agents[key][0].count_nonzero() > 0


def test_shared_genome_allows_distinct_local_synaptic_states(config):
    c = plastic_config(config, initial_population=1)
    w = EcologyWorld(c)
    w.agents["genome"][:, c.brain_parameter_count + 6] = 0
    w.develop(w.agents)
    assert w.agents["modules"].item() == 2
    for field in w.fields:
        field.grid.zero_()
    w.agents["module_h"].fill_(0.2)
    w.agents["module_plastic"][0, 0] = 0.4
    w.update_controllers()
    assert not torch.equal(w.agents["module_h"][0, 0], w.agents["module_h"][0, 1])
    for key in ("module_h", "module_plastic", "module_trace"):
        assert w.agents[key][0, 2].count_nonzero() == 0


def test_frozen_and_active_circuits_pay_same_machinery_cost(config):
    c = plastic_config(config, initial_population=1)
    w = EcologyWorld(c, ablation="no_plasticity")
    w.step()
    expected = c.plasticity_cost * int(w.agents["modules"].sum()) * c.dt
    assert w.totals["plasticity_cost"] == pytest.approx(expected)
    assert w.totals["plasticity_changes"] == 0
    assert abs(w.metrics()["energy_balance_error"]) < 1e-3
    other = EcologyWorld(c)
    other.step()
    assert other.totals["plasticity_cost"] == w.totals["plasticity_cost"]


def test_plastic_association_probe_preserves_genomes_and_has_feedback_control(config):
    from emergent_garden.probes import association_probe

    w = EcologyWorld(plastic_config(config, initial_population=8))
    before = w.agents["genome"].clone()
    learned = association_probe(w.config, before, rounds=1)
    assert max(learned["plastic_magnitude_after_training"]) > 0
    frozen = association_probe(w.config, before, rounds=1, mode="no_plasticity")
    assert frozen["plastic_magnitude_after_training"] == [0.0] * len(before)
    control = association_probe(w.config, before, rounds=1, mode="no_feedback")
    assert control["association_alignment"] == [0.0] * len(before)
    assert control["reversal_alignment"] == [0.0] * len(before)
    torch.testing.assert_close(before, w.agents["genome"], rtol=0, atol=0)


def test_association_probe_detects_a_constructed_plastic_reference(config):
    """A measurement check only: this synthetic circuit never seeds the dish."""
    from emergent_garden.probes import association_probe

    c = plastic_config(config)
    genome = torch.zeros((1, c.parameter_count))
    wi, _, _, wo, _ = brain_parts(c, genome)
    for i, (identity, side) in enumerate(
        (("a", "left"), ("a", "right"), ("b", "left"), ("b", "right"))
    ):
        for angle in (-135, -45) if side == "left" else (45, 135):
            wi[0, i, c.input_names.index(f"identity_{identity}_{angle}")] = 1.5
        wo[0, 0 if side == "left" else 1, i] = -2
    wi[0, 4, c.input_names.index("food_feedback")] = 3
    wo[0, 4, 4] = 3
    genome[0, c.brain_parameter_count + 5] = -3
    genome[0, c.brain_parameter_count + 9] = 3
    genome[0, c.brain_parameter_count + 10] = -3
    for mode in ("none", "reset_h", "no_plasticity", "no_feedback"):
        result = association_probe(c, genome, rounds=3, mode=mode)
        for key in ("association_alignment", "reversal_alignment"):
            if mode in ("none", "reset_h"):
                assert result[key][0] > 1e-5
            else:
                assert result[key] == [0.0]
