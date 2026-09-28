import math
from dataclasses import replace

import pytest
import torch
from test_observation import assert_same

from emergent_garden.brain import advance, controller_step, initial_state
from emergent_garden.inheritance import brain_parts, upgrade_genomes
from emergent_garden.neural_timing import (
    integration_factors,
    response_times,
    timing_genes,
    timing_metrics,
)
from emergent_garden.topology import duplicate_neuron, effective_masks, initial_structure
from emergent_garden.world import World, create_world


def timed(config, **changes):
    return replace(
        config,
        **dict(
            ecology_version=24,
            hidden_size=8,
            initial_neurons=4,
            min_neurons=2,
            neural_timing_range=4.0,
            initial_timing_sigma=0.5,
            timing_mutation_probability=0.1,
            timing_mutation_sigma=0.15,
        )
        | changes,
    ).validate()


def common_state(world):
    state = world.state_dict()
    state.pop("config")
    state["rng"].pop("neural_timing", None)
    # A longer inherited genome necessarily has a different birth digest.
    state["events"] = [{k: v for k, v in e.items() if k != "genome_hash"} for e in state["events"]]
    if world.config.timing_count:
        state["founders"] = state["founders"][:, : -world.config.timing_count]
        state["agents"]["genome"] = state["agents"]["genome"][:, : -world.config.timing_count]
    return state


def test_step_and_decay_follow_each_neurons_analytical_time_constant(config):
    c = timed(config, hidden_size=4, initial_population=1)
    w = create_world(c)
    g = w.founders.double()
    g[:, : c.brain_parameter_count] = 0
    timing_genes(c, g)[0] = torch.tensor([-3.0, -0.5, 0.5, 3.0])
    parts = brain_parts(c, g)
    parts[2].fill_(math.atanh(0.5))
    body_tau = torch.tensor([2.0], dtype=torch.float64)
    times = response_times(c, g, body_tau)
    assert times.min() > 0.5 and times.max() < 8
    h = torch.zeros(1, 4, dtype=torch.float64)
    x = torch.zeros(1, c.input_size, dtype=torch.float64)
    for step in range(1, 101):
        h, _ = advance(c, g, x, h, body_tau)
        expected = 0.5 * (1 - torch.exp(-step / (c.controller_hz * times)))
        torch.testing.assert_close(h, expected, rtol=1e-12, atol=1e-14)
    initial = h.clone()
    parts[2].zero_()
    for step in range(1, 101):
        h, _ = advance(c, g, x, h, body_tau)
        expected = initial * torch.exp(-step / (c.controller_hz * times))
        torch.testing.assert_close(h, expected, rtol=1e-12, atol=1e-14)


def test_extreme_allowed_times_integrate_finitely_and_keep_activity_bounded(config):
    c = timed(config, neural_timing_range=100.0)
    w = create_world(c)
    g = w.founders
    timing_genes(c, g)[:, ::2] = -c.weight_limit
    timing_genes(c, g)[:, 1::2] = c.weight_limit
    tau = torch.tensor([0.2, 5.0])
    alpha = integration_factors(c, g, tau)
    assert torch.isfinite(alpha).all() and ((alpha > 0) & (alpha <= 1)).all()
    state = initial_state(c, g)
    for _ in range(100):
        state, actions = controller_step(c, g, torch.full((2, c.input_size), 1e5), state, tau)
        assert (state["hidden"].abs() <= 1).all()
        assert torch.isfinite(actions).all()


@pytest.mark.parametrize("sigma", [0.0, 0.5])
def test_neutral_expression_preserves_complete_common_world_through_births(config, sigma):
    c = timed(
        config,
        neural_timing_range=1.0,
        initial_timing_sigma=sigma,
        timing_mutation_probability=0.1,
        node_mutation_probability=1.0,
        edge_mutation_probability=1.0,
        initial_food=12,
        food_rate=1.0,
    )
    old_c = replace(
        c, ecology_version=22, initial_timing_sigma=0.0, timing_mutation_probability=0.0
    )
    old, new = create_world(old_c), create_world(c)
    assert new.founders.shape[1] == old.founders.shape[1] + c.hidden_size
    for w in (old, new):
        w.agents["genome"][:, c.brain_parameter_count + 6] = -3
        w.develop(w.agents)
        w.agents["energy"] = c.max_energy * w.agents["area"]
        w.step(91)
    assert new.totals["births"] > 0
    assert_same(common_state(old), common_state(new))
    old_metrics, new_metrics = old.metrics(), new.metrics()
    for key in old_metrics.keys() - {"ecology_version", "genome_variance"}:
        assert old_metrics[key] == new_metrics[key], key
    prefix_variance = new.agents["genome"][:, : old_c.parameter_count].var(0, correction=0).mean()
    assert prefix_variance.item() == old_metrics["genome_variance"]
    assert new_metrics["neural_timing_mean_within_log_std"] == 0


def test_timing_initialization_and_mutation_leave_other_draws_and_genes_unchanged(config):
    c = timed(config, timing_mutation_probability=1.0, timing_mutation_sigma=0.7)
    a = create_world(c, seed=39)
    b = create_world(replace(c, timing_mutation_probability=0.0), seed=39)
    assert_same(a.founders, b.founders)
    assert torch.count_nonzero(timing_genes(c, a.founders)) == a.population * c.hidden_size
    ga, gb = a.founders[0], b.founders[0]
    for _ in range(80):
        ga, gb = a.mutate(ga), b.mutate(gb)
        assert_same(ga[: -c.hidden_size], gb[: -c.hidden_size])
        assert timing_genes(c, ga[None]).abs().max() <= c.weight_limit
    assert not torch.equal(timing_genes(c, ga[None]), timing_genes(c, gb[None]))
    for key in a.rng.keys() - {"neural_timing"}:
        assert torch.equal(a.rng[key].get_state(), b.rng[key].get_state()), key


def test_duplication_copies_timing_and_preserves_heterogeneous_recurrent_dynamics(config):
    c = timed(config, initial_population=1)
    w = create_world(c)
    original = w.founders
    timing_genes(c, original)[0, 1] = -2
    timing_genes(c, original)[0, 6] = 2
    expanded = duplicate_neuron(c, original[0], 1, 6)[None]
    assert timing_genes(c, expanded)[0, 6] == -2
    h = torch.zeros(1, c.hidden_size)
    h[0, :4] = torch.tensor([0.2, -0.3, 0.1, 0.4])
    other = h.clone()
    other[0, 6] = h[0, 1]
    rng = torch.Generator().manual_seed(97)
    for _ in range(100):
        inputs = torch.rand((1, c.input_size), generator=rng)
        h, actions = advance(c, original, inputs, h, w.agents["memory_tau"])
        other, copied_actions = advance(c, expanded, inputs, other, w.agents["memory_tau"])
        torch.testing.assert_close(actions, copied_actions, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(h[:, :4], other[:, :4], rtol=1e-6, atol=1e-7)
        assert torch.equal(other[:, 1], other[:, 6])


def test_dormant_timing_genes_are_unexpressed_and_excluded_from_statistics(config):
    c = timed(config)
    w = create_world(c)
    original, altered = w.founders, w.founders.clone()
    nodes = effective_masks(c, original)[0]
    timing_genes(c, altered)[~nodes] = c.weight_limit
    state = initial_state(c, original)
    inputs = torch.ones(2, c.input_size)
    a, actions = controller_step(c, original, inputs, state, w.agents["memory_tau"])
    b, other = controller_step(c, altered, inputs, state, w.agents["memory_tau"])
    assert_same(a, b)
    assert_same(actions, other)
    assert timing_metrics(c, original, w.agents["memory_tau"], nodes) == timing_metrics(
        c, altered, w.agents["memory_tau"], nodes
    )


@pytest.mark.parametrize("mutation", [0.0, 1.0])
def test_birth_inherits_timing_but_resets_acquired_states_and_separates_structure(config, mutation):
    c = timed(
        config,
        initial_population=1,
        mutation_probability=0.0,
        trait_mutation_probability=0.0,
        node_mutation_probability=0.0,
        edge_mutation_probability=0.0,
        module_mutation_probability=0.0,
        timing_mutation_probability=mutation,
    )
    w = create_world(c)
    w.agents["genome"][:, c.brain_parameter_count + 6] = -3
    w.develop(w.agents)
    w.agents["pos"][:] = 64
    w.agents["energy"] = c.max_energy * w.agents["area"]
    states = ["module_h", "module_plastic", "module_trace"]
    states += [key for key in w.agents if key.startswith("module_motor_")]
    for key in states:
        w.agents[key].fill_(True if w.agents[key].dtype == torch.bool else 0.05)
    w.reproduce()
    assert w.population == 2
    assert w.totals["neural_structural_births"] == 0
    event = next(e for e in w.events if e["event"] == "birth")
    assert not event["neural_structure_changed"]
    parent, child = timing_genes(c, w.agents["genome"])
    assert torch.equal(parent, child) == (mutation == 0)
    for key in states:
        assert not w.agents[key][1].count_nonzero(), key


def test_old_transfer_adds_neutral_genes_and_wider_transfer_preserves_existing_times(config):
    c = timed(config)
    old = replace(
        c,
        ecology_version=23,
        neural_timing_range=1.0,
        initial_timing_sigma=0.0,
        timing_mutation_probability=0.0,
    )
    w = create_world(old)
    transferred = upgrade_genomes(old, c, w.founders)
    assert timing_genes(c, transferred).count_nonzero() == 0
    assert_same(transferred[:, : -c.hidden_size], w.founders)
    assert_same(
        integration_factors(c, transferred, w.agents["memory_tau"]),
        integration_factors(old, w.founders, w.agents["memory_tau"]).expand(-1, c.hidden_size),
    )
    newer = create_world(c)
    wider = replace(c, hidden_size=12)
    grown = upgrade_genomes(c, wider, newer.founders)
    assert_same(timing_genes(wider, grown)[:, : c.hidden_size], timing_genes(c, newer.founders))
    assert timing_genes(wider, grown)[:, c.hidden_size :].count_nonzero() == 0
    bad = newer.founders.clone()
    timing_genes(c, bad)[0, 0] = c.weight_limit + 1
    with pytest.raises(ValueError, match="weight_limit"):
        upgrade_genomes(c, wider, bad)


def test_resume_restores_timing_randomness_and_neural_dynamics(config):
    from emergent_garden.observation import ControllerObserver

    c = timed(
        config,
        initial_food=20,
        food_rate=2.0,
        gut_capacity=200.0,
        motor_learning_rate=0.01,
        motor_normalized=1,
    )
    w = create_world(c)
    observer = w.controller_observer = ControllerObserver()
    observer.select(0)
    w.step(29)
    assert observer.sample.predicted_return is None  # The reference disables its value head.
    replay = World.from_state(w.state_dict())
    for world in (w, replay):
        # Test the separate mutation stream after restoration, even without a natural birth.
        world.agents["genome"][0] = world.mutate(world.agents["genome"][0])
        world.develop(world.agents)
        world.step(61)
    assert_same(w.state_dict(), replay.state_dict())
    assert w.metrics()["neural_timing_mean_within_log_std"] > 0


def test_acquired_state_challenge_freezes_timing_as_well_as_other_inheritance(config):
    from emergent_garden.challenges import fork_challenge

    c = timed(
        config,
        timing_mutation_probability=1.0,
        mutation_probability=1.0,
        trait_mutation_probability=1.0,
        node_mutation_probability=1.0,
        edge_mutation_probability=1.0,
        module_mutation_probability=1.0,
    )
    source = create_world(c)
    fork = fork_challenge(source, "intact", False, 300)
    assert fork.config.timing_mutation_probability == 0
    genome = fork.agents["genome"][0]
    assert torch.equal(fork.mutate(genome), genome)
    assert source.config.timing_mutation_probability == 1


def test_configuration_layout_empty_world_and_timing_metric_weighting(config):
    c = timed(config, hidden_size=32, initial_neurons=16)
    assert c.parameter_count == 5304 and c.timing_count == 32
    assert initial_structure(c, 2, "cpu").shape == (2, c.structure_count)
    empty = create_world(replace(c, initial_population=0))
    assert empty.founders.shape == (0, c.parameter_count)
    assert all(v == 0 for k, v in empty.metrics().items() if k.startswith("neural_timing_"))
    for key, value in (
        ("neural_timing_range", 0.5),
        ("neural_timing_range", 101),
        ("timing_mutation_probability", 1.1),
        ("initial_timing_sigma", -1),
    ):
        with pytest.raises(ValueError, match=key):
            replace(c, **{key: value}).validate()
    with pytest.raises(ValueError, match="requires ecology_version >= 24"):
        replace(c, ecology_version=23).validate()
    g = create_world(c).founders
    timing_genes(c, g).zero_()
    timing_genes(c, g)[0, 0] = -1
    timing_genes(c, g)[0, 1] = 1
    nodes = torch.zeros(2, c.hidden_size, dtype=torch.bool)
    nodes[0, :2], nodes[1, :4] = True, True
    result = timing_metrics(c, g, torch.tensor([1.0, 2.0]), nodes)
    spread = math.log(4) * math.tanh(1)
    assert result["neural_timing_mean_within_log_std"] == pytest.approx(spread / 2)
    assert result["neural_timing_mean_seconds"] == pytest.approx(
        (math.exp(-spread) + math.exp(spread) + 8) / 6
    )
