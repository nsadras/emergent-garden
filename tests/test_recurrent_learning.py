import math
from dataclasses import replace

import pytest
import torch
from test_observation import assert_same

from emergent_garden.brain import advance, controller_step, initial_state
from emergent_garden.challenges import fork_challenge
from emergent_garden.development import grow
from emergent_garden.inheritance import brain_parts
from emergent_garden.neural_timing import integration_factors
from emergent_garden.recurrent_learning import recurrent_policy, recurrent_shapes
from emergent_garden.storage import load_checkpoint, save_checkpoint
from emergent_garden.topology import effective_masks, mask_parts
from emergent_garden.world import create_world


def configured(config, **changes):
    return replace(
        config,
        **dict(
            ecology_version=25,
            hidden_size=8,
            initial_neurons=4,
            min_neurons=2,
            neural_timing_range=4.0,
            initial_timing_sigma=0.5,
            recurrent_noise_sigma=0.15,
            recurrent_learning_rate=0.001,
            motor_normalized=1,
            motor_learning_rate=0.01,
            exploration_max=0.15,
        )
        | changes,
    ).validate()


def test_native_hidden_transition_has_the_derived_masked_likelihood_score(config):
    c = configured(config, hidden_size=4, initial_population=1)
    w = create_world(c)
    g = w.founders.double()
    nodes, _, edges, _ = mask_parts(c, g)
    nodes[0, 3] = 0
    edges[0, 0, 1] = 0
    active, mi, mr, _ = effective_masks(c, g)
    previous = torch.tensor([[0.2, -0.4, 0.1, 0.3]], dtype=torch.float64)
    inputs = torch.linspace(-0.2, 0.4, c.input_size, dtype=torch.float64)[None]
    epsilon = torch.tensor([[0.6, -1.1, 0.2, 0.7]], dtype=torch.float64)
    weights = torch.zeros(1, 4, 4, dtype=torch.float64, requires_grad=True)
    tau = w.agents["memory_tau"].double()
    alpha = integration_factors(c, g, tau)
    observed, _ = advance(
        c,
        g,
        inputs,
        previous,
        tau,
        weights,
        hidden_noise=c.recurrent_noise_sigma * epsilon,
    )
    transformed = (observed.detach() - (1 - alpha) * previous * active) / alpha
    latent = transformed.atanh()
    wi, wr, bias, _, _ = brain_parts(c, g)
    mean = ((wi * mi) @ inputs[..., None]).squeeze(-1)
    mean += (((wr + weights) * mr) @ (previous * active)[..., None]).squeeze(-1) + bias
    density = (
        -0.5 * ((latent - mean) / c.recurrent_noise_sigma).square()
        - math.log(c.recurrent_noise_sigma * math.sqrt(2 * math.pi))
        - alpha.log()
        - (1 - transformed.square()).log()
    )[active].sum()
    gradient = torch.autograd.grad(density, weights)[0]
    expected = (epsilon / c.recurrent_noise_sigma)[:, :, None] * previous[:, None, :] * mr
    torch.testing.assert_close(gradient, expected, rtol=1e-11, atol=1e-11)
    result = recurrent_policy(
        c,
        g,
        previous,
        initial_state(c, g),
        epsilon,
        g.new_zeros(1),
        g.new_full((1,), 1 / c.controller_hz),
    )
    torch.testing.assert_close(result["recurrent_trace"], gradient, rtol=1e-11, atol=1e-11)
    assert not result["recurrent_trace"][~mr].count_nonzero()


def test_feedback_credits_only_earlier_perturbations_and_not_the_current_draw(config):
    c = configured(config)
    g = create_world(c).founders
    state = initial_state(c, g)
    previous = torch.full_like(state["hidden"], 0.3)
    noise = torch.ones_like(previous)
    elapsed = g.new_full((len(g),), 0.1)
    first = recurrent_policy(c, g, previous, state, noise, g.new_full((len(g),), 0.2), elapsed)
    assert not first["recurrent_plastic"].count_nonzero()
    assert first["recurrent_trace"].count_nonzero()
    prepared = recurrent_policy(c, g, previous, state, noise, g.new_zeros(len(g)), elapsed)
    positive = recurrent_policy(
        c,
        g,
        previous,
        prepared,
        noise,
        g.new_full((len(g),), 0.2),
        elapsed,
    )
    changed_draw = recurrent_policy(
        c,
        g,
        previous,
        prepared,
        -9 * noise,
        g.new_full((len(g),), 0.2),
        elapsed,
    )
    negative = recurrent_policy(
        c,
        g,
        previous,
        prepared,
        noise,
        g.new_full((len(g),), -0.2),
        elapsed,
    )
    assert_same(positive["recurrent_plastic"], changed_draw["recurrent_plastic"])
    assert_same(positive["recurrent_plastic"], -negative["recurrent_plastic"])
    assert positive["recurrent_plastic"].count_nonzero()
    assert not torch.equal(positive["recurrent_trace"], changed_draw["recurrent_trace"])


def test_credit_is_bounded_masked_and_does_not_mutate_genomes_or_input_state(config):
    c = configured(config, initial_recurrent_density=0.5)
    g = create_world(c).founders
    state = initial_state(c, g)
    state["recurrent_trace"].fill_(1000)
    state["hidden"].fill_(0.5)
    snapshot = {key: value.clone() for key, value in state.items()}
    inherited = g.clone()
    result, actions = controller_step(
        c,
        g,
        g.new_zeros((len(g), c.input_size)),
        state,
        recurrent_noise=torch.ones_like(state["hidden"]),
        recurrent_reward=g.new_full((len(g),), 1e6),
    )
    nodes, _, edges, _ = effective_masks(c, g)
    assert result["recurrent_plastic"].norm(dim=-1).max() <= c.recurrent_learning_limit + 1e-7
    assert not result["recurrent_plastic"][~edges].count_nonzero()
    assert not result["recurrent_trace"][~edges].count_nonzero()
    assert not result["recurrent_applied_noise"][~nodes].count_nonzero()
    assert torch.isfinite(actions).all()
    assert_same(state, snapshot)
    assert_same(g, inherited)
    for mode in ("recurrent_learning", "plasticity"):
        frozen, _ = controller_step(
            c,
            g,
            g.new_zeros((len(g), c.input_size)),
            state,
            recurrent_noise=torch.ones_like(state["hidden"]),
            **{mode: False},
        )
        assert not frozen["recurrent_plastic"].count_nonzero()
        assert not frozen["recurrent_trace"].count_nonzero()
        assert frozen["recurrent_applied_noise"].count_nonzero()


def test_no_exploration_stops_new_scores_but_preserves_credit_for_past_activity(config):
    c = configured(config)
    g = create_world(c).founders
    state = initial_state(c, g)
    nodes, _, mask, _ = effective_masks(c, g)
    state["recurrent_trace"][:] = mask
    result = recurrent_policy(
        c,
        g,
        torch.ones_like(state["hidden"]),
        state,
        torch.ones_like(state["hidden"]),
        g.new_full((len(g),), 0.2),
        g.new_full((len(g),), 0.1),
        exploration_enabled=False,
    )
    assert not result["recurrent_applied_noise"].count_nonzero()
    torch.testing.assert_close(
        result["recurrent_trace"], state["recurrent_trace"] * math.exp(-0.1 / c.recurrent_trace_tau)
    )
    assert result["recurrent_plastic"].count_nonzero()
    assert not result["recurrent_applied_noise"][~nodes].count_nonzero()


def test_disabled_v25_preserves_every_legacy_state_and_measurement_through_births(config):
    c = configured(
        config,
        recurrent_noise_sigma=0,
        recurrent_learning_rate=0,
        initial_food=12,
        food_rate=1,
        node_mutation_probability=1,
        edge_mutation_probability=1,
        timing_mutation_probability=0.1,
    )
    old, new = create_world(replace(c, ecology_version=24)), create_world(c)
    initial_noise = new.rng["recurrent_exploration"].get_state().clone()
    for world in (old, new):
        world.agents["genome"][:, c.brain_parameter_count + 6] = -3
        world.develop(world.agents)
        world.agents["energy"] = c.max_energy * world.agents["area"]
        world.initial_energy = world.agents["energy"].double().sum().item()
        world.step(91)
    assert old.totals["births"] > 0
    a, b = old.state_dict(), new.state_dict()
    a.pop("config")
    b.pop("config")
    b["rng"].pop("recurrent_exploration")
    assert b["totals"].pop("recurrent_learning_changes") == 0
    for key in recurrent_shapes(c.hidden_size):
        assert not b["agents"].pop(f"module_{key}").count_nonzero()
    assert_same(a, b)
    assert_same(initial_noise, new.rng["recurrent_exploration"].get_state())
    before, after = old.metrics(), new.metrics()
    for key in before.keys() - {"ecology_version"}:
        assert before[key] == after[key], key


@pytest.mark.parametrize("ablation", ["none", "shuffled_recurrent_reward"])
def test_complete_replay_including_separate_noise_and_feedback_streams(config, tmp_path, ablation):
    c = configured(config, initial_population=4, initial_food=24, food_rate=1)
    w = create_world(c, ablation=ablation)
    w.step(37)
    save_checkpoint(w, tmp_path / "world.pt")
    replay = load_checkpoint(tmp_path / "world.pt")
    w.step(79)
    replay.step(79)
    assert_same(w.state_dict(), replay.state_dict())
    assert w.totals["recurrent_learning_changes"] > 0
    assert w.agents["module_recurrent_applied_noise"].count_nonzero()
    assert abs(w.metrics()["energy_balance_error"]) < 0.01
    for key in recurrent_shapes(c.hidden_size):
        assert not w.agents[f"module_{key}"][~w.agents["module_mask"]].count_nonzero()


@pytest.mark.parametrize(
    "ablation",
    [
        "shuffled_recurrent_reward",
        "shuffled_motor_reward",
        "no_motor_reward",
        "no_feedback",
    ],
)
def test_feedback_interventions_keep_motor_and_recurrent_signals_separate(config, ablation):
    c = configured(config, initial_population=4)
    w = create_world(c, ablation=ablation)
    reference = create_world(c)
    assert_same(w.founders, reference.founders)
    for name in reference.rng:
        assert_same(w.rng[name].get_state(), reference.rng[name].get_state())
    rates = torch.tensor([-2.0, 0.0, 3.0, 5.0])
    interval = 1 / c.controller_hz
    w.tick = c.physics_hz // c.controller_hz
    w.agents["motor_reward"] = rates * interval * c.feedback_scale * w.agents["area"]
    energy = w.agents["energy"].clone()
    w.update_controllers()
    for name, tau in (("motor", c.motor_baseline_tau), ("recurrent", c.recurrent_baseline_tau)):
        actual = w.agents[f"module_{name}_baseline"][:, 0] / (1 - math.exp(-interval / tau))
        if ablation == "no_feedback" or (ablation == "no_motor_reward" and name == "motor"):
            assert not actual.count_nonzero()
        else:
            torch.testing.assert_close(actual.sort().values, rates, rtol=1e-5, atol=1e-6)
            if ablation == f"shuffled_{name}_reward":
                assert (actual - rates).abs().min() > 1
            else:
                torch.testing.assert_close(actual, rates, rtol=1e-5, atol=1e-6)
    assert_same(w.agents["energy"], energy)
    assert not w.agents["motor_reward"].count_nonzero()


def test_newborn_inherits_genes_but_no_recurrent_acquired_state(config):
    c = configured(
        config,
        initial_population=1,
        mutation_probability=0,
        trait_mutation_probability=0,
        node_mutation_probability=0,
        edge_mutation_probability=0,
        module_mutation_probability=0,
        timing_mutation_probability=0,
    )
    w = create_world(c)
    w.agents["genome"][:, c.brain_parameter_count + 6] = -3
    w.develop(w.agents)
    w.agents["pos"].fill_(64)
    w.step(31)
    inherited = w.agents["genome"][0].clone()
    assert w.agents["module_recurrent_plastic"].count_nonzero()
    w.agents["energy"] = c.max_energy * w.agents["area"]
    w.reproduce()
    assert w.population == 2
    assert_same(w.agents["genome"][1], inherited)
    for key in recurrent_shapes(c.hidden_size):
        assert w.agents[f"module_{key}"][0].count_nonzero()
        assert not w.agents[f"module_{key}"][1].count_nonzero()


def test_developing_a_module_preserves_old_learning_and_starts_new_state_empty(config):
    c = configured(config, initial_population=1, growth_delay=1, growth_reserve=1)
    w = create_world(c)
    w.agents["genome"][:, c.brain_parameter_count + 6] = 2
    w.develop(w.agents)
    w.agents["pos"].fill_(64)
    w.step(31)
    keys = [f"module_{key}" for key in recurrent_shapes(c.hidden_size)]
    before = {key: w.agents[key][0, 0].clone() for key in keys}
    w.agents["energy"] = c.max_energy * w.agents["area"]
    w.agents["growth_tick"].fill_(w.tick)
    grow(w)
    assert int(w.agents["modules"][0]) == 2
    for key in keys:
        assert_same(w.agents[key][0, 0], before[key])
        assert not w.agents[key][0, 1:].count_nonzero()
    w.step(4)
    assert w.agents["module_recurrent_applied_noise"][0, 1].count_nonzero()
    for key in keys:
        assert not w.agents[key][0, 2].count_nonzero()


def test_state_challenge_erases_acquired_recurrent_state_without_inheriting_it(config):
    w = create_world(configured(config))
    w.step(31)
    snapshot = w.state_dict()
    erased = fork_challenge(w, "erase_plastic", False, 1000)
    activity = fork_challenge(w, "erase_activity", False, 1000)
    for key in recurrent_shapes(w.config.hidden_size):
        assert not erased.agents[f"module_{key}"].count_nonzero()
        assert_same(activity.agents[f"module_{key}"], w.agents[f"module_{key}"])
    assert_same(w.state_dict(), snapshot)


def test_metrics_exclude_dormant_nodes_edges_and_unexpressed_modules(config):
    w = create_world(configured(config, initial_population=1))
    a = w.agents
    nodes, _, edges, _ = effective_masks(w.config, a["genome"])
    active = a["module_mask"][:, :, None] & nodes[:, None]
    mask = a["module_mask"][:, :, None, None] & edges[:, None]
    a["module_recurrent_plastic"].fill_(100)
    a["module_recurrent_plastic"][mask] = 0.01
    a["module_recurrent_applied_noise"].fill_(100)
    a["module_recurrent_applied_noise"][active] = 0.15
    metrics = w.metrics()
    assert metrics["mean_recurrent_plastic_magnitude"] == pytest.approx(0.01)
    assert metrics["recurrent_rows_saturated_fraction"] == 0
    assert metrics["recurrent_noise_rms"] == pytest.approx(0.15)
    empty = create_world(configured(config, initial_population=0))
    for key, value in empty.metrics().items():
        if "recurrent_" in key:
            assert value == 0, key


@pytest.mark.parametrize(
    "changes",
    [
        dict(ecology_version=24),
        dict(recurrent_noise_sigma=0),
        dict(recurrent_noise_sigma=1e-5),
        dict(recurrent_noise_sigma=11),
        dict(recurrent_noise_sigma=float("nan")),
        dict(recurrent_learning_rate=1.1),
        dict(recurrent_learning_limit=0),
        dict(recurrent_trace_tau=0),
        dict(recurrent_baseline_tau=0),
        dict(recurrent_half_life=0),
    ],
)
def test_configuration_rejects_invalid_or_unexpressible_learning(config, changes):
    with pytest.raises(ValueError):
        configured(config, **changes)


def test_recurrent_controls_require_v25(config):
    for mode in ("no_recurrent_learning", "shuffled_recurrent_reward"):
        with pytest.raises(ValueError, match="version containing"):
            create_world(replace(config, ecology_version=24), ablation=mode)
