from dataclasses import replace

import pytest
import torch

from emergent_garden.brain import advance, controller_step, initial_state
from emergent_garden.ecology import EcologyWorld
from emergent_garden.inheritance import brain_parts, upgrade_genomes
from emergent_garden.topology import (
    counts,
    delete_neuron,
    duplicate_neuron,
    effective_masks,
    mask_parts,
    mutate_structure,
)


def graph_config(config, **changes):
    return replace(
        config,
        **{
            "ecology_version": 8,
            "hidden_size": 8,
            "initial_neurons": 4,
            "min_neurons": 2,
            **changes,
        },
    ).validate()


def test_initial_graph_counts_only_expressed_nodes_and_connections(config):
    c = graph_config(config)
    w = EcologyWorld(c)
    assert w.agents["genome"].shape == (2, c.parameter_count)
    assert (w.agents["neurons"] == 4).all()
    assert (w.agents["connections"] == 4 * (c.input_size + 4 + c.output_size)).all()
    assert (w.agents["recurrent_connections"] == 16).all()


def test_dormant_graph_elements_cannot_affect_activity_actions_or_plasticity(config):
    c = graph_config(config)
    w = EcologyWorld(c)
    original = w.agents["genome"].clone()
    mask_parts(c, original)[2][:, 0, 1] = 0
    changed = original.clone()
    nodes, mi, mr, mo = effective_masks(c, original)
    wi, wr, bias, wo, _ = brain_parts(c, changed)
    for weight, mask in ((wi, mi), (wr, mr), (wo, mo)):
        weight[~mask] = 3
    bias[~nodes] = 3
    state = initial_state(c, original)
    state["hidden"].fill_(0.2)
    state["hidden"] *= nodes
    altered = {key: value.clone() for key, value in state.items()}
    altered["hidden"][~nodes] = 9
    altered["plastic"][~mr] = 1
    altered["trace"][~mr] = 1
    inputs = torch.ones((2, c.input_size))
    expected, actions = controller_step(c, original, inputs, state)
    actual, other_actions = controller_step(c, changed, inputs, altered)
    torch.testing.assert_close(actions, other_actions, rtol=0, atol=0)
    for key in state:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
    assert actual["hidden"][~nodes].count_nonzero() == 0
    assert actual["plastic"][~mr].count_nonzero() == 0


def test_neuron_duplication_preserves_recurrent_function_including_self_edges(config):
    c = graph_config(config, initial_population=1)
    w = EcologyWorld(c)
    original = w.agents["genome"].clone()
    expanded = duplicate_neuron(c, original[0], donor=1, destination=6)[None]
    assert counts(c, expanded)[0].item() == 5
    h = torch.zeros((1, c.hidden_size))
    h[0, :4] = torch.tensor([0.2, -0.3, 0.1, 0.4])
    enlarged = h.clone()
    enlarged[0, 6] = h[0, 1]
    rng = torch.Generator().manual_seed(97)
    for _ in range(100):
        inputs = torch.rand((1, c.input_size), generator=rng)
        h, actions = advance(c, original, inputs, h)
        enlarged, copied_actions = advance(c, expanded, inputs, enlarged)
        torch.testing.assert_close(actions, copied_actions, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(h[:, :4], enlarged[:, :4], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(enlarged[:, 1], enlarged[:, 6], rtol=0, atol=0)


def test_structural_mutation_respects_bounds_and_binary_inheritance(config):
    c = graph_config(config, node_mutation_probability=1.0, edge_mutation_probability=1.0)
    genome = EcologyWorld(c).agents["genome"][0]
    rng = torch.Generator().manual_seed(199)
    observed = set()
    for _ in range(300):
        genome = mutate_structure(c, genome, rng)
        number = int(counts(c, genome[None])[0][0])
        observed.add(number)
        assert c.min_neurons <= number <= c.hidden_size
        for part in mask_parts(c, genome[None]):
            assert ((part == 0) | (part == 1)).all()
    assert min(observed) == c.min_neurons
    assert max(observed) == c.hidden_size
    nodes = mask_parts(c, genome[None])[0][0]
    for neuron in (nodes > 0.5).nonzero().flatten().tolist()[c.min_neurons :]:
        genome = delete_neuron(c, genome, neuron)
    active = int((mask_parts(c, genome[None])[0][0] > 0.5).nonzero()[0])
    with pytest.raises(ValueError, match="minimum"):
        delete_neuron(c, genome, active)


def test_adding_a_connection_is_neutral_until_its_weight_mutates(config):
    c = graph_config(config, node_mutation_probability=0.0, edge_mutation_probability=1.0)
    genome = EcologyWorld(c).agents["genome"][0]
    for mask in mask_parts(c, genome[None])[1:]:
        mask.zero_()
    inputs = torch.ones((1, c.input_size))
    h = torch.full((1, c.hidden_size), 0.3)
    expected_h, expected_actions = advance(c, genome[None], inputs, h)
    rng = torch.Generator().manual_seed(41)
    for _ in range(100):
        changed = mutate_structure(c, genome, rng)
        if int(counts(c, changed[None])[1][0]) == 1:
            actual_h, actions = advance(c, changed[None], inputs, h)
            torch.testing.assert_close(expected_h, actual_h, rtol=0, atol=0)
            torch.testing.assert_close(expected_actions, actions, rtol=0, atol=0)
            break
    else:
        pytest.fail("No connection addition sampled")


def test_padding_v7_preserves_plastic_dynamics_and_existing_traits(config):
    from emergent_garden.probes import association_probe

    c = graph_config(config)
    source = replace(c, ecology_version=7, hidden_size=4)
    genome = EcologyWorld(source).agents["genome"]
    expanded = upgrade_genomes(source, c, genome)
    old, new = initial_state(source, genome), initial_state(c, expanded)
    rng = torch.Generator().manual_seed(127)
    for _ in range(100):
        inputs = torch.rand((2, c.input_size), generator=rng)
        old, a = controller_step(source, genome, inputs, old)
        new, b = controller_step(c, expanded, inputs, new)
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(old["plastic"], new["plastic"][:, :4, :4], rtol=1e-6, atol=1e-7)
    assert new["hidden"][:, 4:].count_nonzero() == 0
    before = association_probe(source, genome, rounds=1)
    after = association_probe(c, expanded, rounds=1)
    torch.testing.assert_close(
        torch.tensor(before["plastic_magnitude_after_training"]),
        torch.tensor(after["plastic_magnitude_after_training"]),
    )
    same = upgrade_genomes(c, c, expanded)
    torch.testing.assert_close(expanded, same, rtol=0, atol=0)
    with pytest.raises(ValueError, match="minimum"):
        upgrade_genomes(source, replace(c, min_neurons=5, initial_neurons=5), genome)


def test_brain_construction_is_paid_only_on_successful_birth(config):
    c = graph_config(
        config,
        initial_population=1,
        mutation_probability=0.0,
        trait_mutation_probability=0.0,
        node_mutation_probability=0.0,
        edge_mutation_probability=0.0,
    )
    w = EcologyWorld(c)
    w.agents["pos"][0] = 64
    w.agents["energy"] = c.reproduction_threshold * w.agents["area"]
    w.initial_energy = w.agents["energy"].double().sum().item()
    w.agents["module_plastic"].fill_(0.2)
    w.reproduce()
    assert w.population == 2
    build = float(w.agents["brain_construction"][1])
    assert w.totals["neural_construction"] == pytest.approx(build)
    assert w.totals["reproduction"] == pytest.approx(
        (c.reproduction_debit - c.birth_energy) * float(w.agents["area"][1]) + build
    )
    torch.testing.assert_close(w.agents["genome"][0], w.agents["genome"][1], rtol=0, atol=0)
    assert w.agents["module_plastic"][1].count_nonzero() == 0
    assert abs(w.metrics()["energy_balance_error"]) < 1e-3
    blocked = EcologyWorld(replace(c, capacity=1))
    blocked.agents["energy"] = c.reproduction_threshold * blocked.agents["area"]
    energy = blocked.agents["energy"].clone()
    blocked.reproduce()
    torch.testing.assert_close(energy, blocked.agents["energy"], rtol=0, atol=0)
    assert blocked.totals["neural_construction"] == 0


def test_neural_maintenance_is_charged_per_expressed_circuit(config):
    c = graph_config(config)
    w = EcologyWorld(c, controller="rest")
    expected = float(w.agents["brain_maintenance"].sum()) * c.dt
    w.step()
    assert w.totals["neural_maintenance"] == pytest.approx(expected)
    assert abs(w.metrics()["energy_balance_error"]) < 1e-3


def test_neural_weight_clipping_does_not_clip_binary_structure(config):
    c = graph_config(config, weight_limit=0.1)
    w = EcologyWorld(c)
    changed = w.mutate(w.agents["genome"][0])
    assert changed[: c.brain_parameter_count + c.trait_count].abs().max() <= c.weight_limit
    nodes = mask_parts(c, changed[None])[0]
    assert (nodes == 1).any()
    assert ((nodes == 0) | (nodes == 1)).all()
