from dataclasses import replace

import pytest
import torch

from emergent_garden.brain import controller_step, initial_state
from emergent_garden.challenges import fork_challenge
from emergent_garden.ecology import EcologyWorld
from emergent_garden.inheritance import upgrade_genomes
from emergent_garden.learning import motor_policy, motor_state
from emergent_garden.topology import initial_structure, mask_parts


def circuit(config, n=1):
    c = replace(config, ecology_version=12, controller_hz=10)
    genomes = torch.zeros(n, c.parameter_count)
    genomes[:, c.brain_parameter_count + c.trait_count :] = initial_structure(c, n, "cpu")
    return c, genomes


def test_reward_credits_previous_exploration_not_current_noise(config):
    c, g = circuit(config)
    state = motor_state(c, g)
    h = torch.ones(1, c.hidden_size)
    logits = torch.zeros(1, 5)
    interval = torch.tensor([0.1])
    first, _ = motor_policy(c, g, h, logits, state, torch.ones(1, 2), torch.zeros(1), interval)
    assert first["motor_plastic"].count_nonzero() == 0
    for reward, sign in ((0.5, 1), (-0.5, -1)):
        second, _ = motor_policy(
            c, g, h, logits, first, -torch.ones(1, 2), torch.tensor([reward]), interval
        )
        assert (sign * second["motor_plastic"] > 0).all()


def test_motor_readouts_respect_inherited_edges_and_freeze_keeps_noise(config):
    c, g = circuit(config)
    mask_parts(c, g)[3][:, :2, 0] = 0
    h, logits = torch.ones(1, c.hidden_size), torch.zeros(1, 5)
    state = motor_state(c, g)
    state["motor_trace"].fill_(100)
    original = g.clone()
    arguments = (c, g, h, logits, state, torch.ones(1, 2), torch.ones(1), torch.tensor([0.1]))
    changed, actions = motor_policy(*arguments)
    assert changed["motor_plastic"][:, :, 0].count_nonzero() == 0
    assert changed["motor_trace"][:, :, 0].count_nonzero() == 0
    assert changed["motor_plastic"].abs().max() <= c.motor_learning_limit
    frozen, noisy = motor_policy(*arguments, learning=False)
    assert frozen["motor_plastic"].count_nonzero() == 0
    assert frozen["motor_trace"].count_nonzero() == 0
    assert (noisy[:, :2] > 0.5).all()
    torch.testing.assert_close(actions[:, 2:], noisy[:, 2:], rtol=0, atol=0)
    torch.testing.assert_close(g, original, rtol=0, atol=0)


@pytest.mark.parametrize("width", [16, 32])
def test_normalized_motor_correction_is_bounded_independently_of_width(config, width):
    c, g = circuit(replace(config, hidden_size=width, initial_neurons=width, motor_normalized=1))
    h = torch.ones(1, c.hidden_size)
    state = motor_state(c, g)
    state["motor_trace"].fill_(1000)
    changed, actions = motor_policy(
        c, g, h, torch.zeros(1, 5), state, torch.zeros(1, 2), torch.ones(1), torch.tensor([0.1])
    )
    assert changed["motor_plastic"].norm(dim=-1).max() <= c.motor_learning_limit + 1e-6
    assert actions[:, :2].logit().abs().max() <= c.motor_learning_limit + 1e-6


def test_motor_normalization_does_not_change_the_no_learning_control(config):
    c = replace(config, ecology_version=12, initial_food=20, food_rate=10.0)
    old = EcologyWorld(c, ablation="no_motor_learning")
    new = EcologyWorld(replace(c, motor_normalized=1), ablation="no_motor_learning")
    for world in (old, new):
        world.step(120)
    assert old.totals == new.totals
    assert old.events == new.events
    for key in old.agents:
        torch.testing.assert_close(old.agents[key], new.agents[key], rtol=0, atol=0)
    for key in old.rng:
        assert torch.equal(old.rng[key].get_state(), new.rng[key].get_state())


def test_zero_exploration_preset_matches_intervention_without_invalid_scores(config):
    c = replace(config, ecology_version=12, initial_food=20, food_rate=10.0)
    intervention = EcologyWorld(c, ablation="no_exploration")
    preset = EcologyWorld(replace(c, exploration_min=0.0, exploration_max=0.0))
    for world in (intervention, preset):
        world.step(120)
        assert world.agents["module_motor_trace"].count_nonzero() == 0
        assert world.agents["module_motor_plastic"].count_nonzero() == 0
    assert intervention.totals == preset.totals
    for key in intervention.agents:
        torch.testing.assert_close(intervention.agents[key], preset.agents[key], rtol=0, atol=0)
    for key in intervention.rng:
        assert torch.equal(intervention.rng[key].get_state(), preset.rng[key].get_state())


def test_motor_rule_can_learn_and_reverse_an_immediate_cue_action_association(config):
    # A controlled mechanism check, not a simulated creature or an evolved brain.
    # Each unchanged genome sees both cues in shuffled order and must relearn
    # their consequences after a reversal. Both controls receive identical noise.
    c, g = circuit(config, 64)
    endpoints = []
    for learning in (True, False):
        state = motor_state(c, g)
        rng = torch.Generator().manual_seed(19)
        reward = torch.zeros(len(g))
        elapsed = torch.full((len(g),), 0.1)
        losses = []
        for step in range(2400):
            cue = torch.where(torch.rand((len(g),), generator=rng) < 0.5, -1.0, 1.0)
            hidden = torch.zeros(len(g), c.hidden_size)
            hidden[:, 0] = cue
            state, actions = motor_policy(
                c,
                g,
                hidden,
                torch.zeros(len(g), 5),
                state,
                torch.randn((len(g), 2), generator=rng),
                reward,
                elapsed,
                learning,
            )
            target = (cue * (1 if step < 1200 else -1) + 1) / 2
            loss = (actions[:, 0] - target).square()
            reward = -loss * 0.1
            losses.append(loss.mean().item())
        endpoints.append([sum(losses[start : start + 100]) / 100 for start in (1100, 1200, 2300)])
    assert endpoints[0][0] < endpoints[1][0] - 0.07
    assert endpoints[0][1] > endpoints[1][1] + 0.03
    assert endpoints[0][2] < endpoints[1][2] - 0.07


def test_transfer_preserves_circuit_and_older_traits_with_new_learning_off(config):
    old = EcologyWorld(replace(config, ecology_version=11))
    c = replace(old.config, ecology_version=12)
    transferred = upgrade_genomes(old.config, c, old.agents["genome"])
    inputs = torch.randn(old.population, c.input_size)
    before, old_actions = controller_step(
        old.config, old.agents["genome"], inputs, initial_state(old.config, old.agents["genome"])
    )
    after, new_actions = controller_step(c, transferred, inputs, initial_state(c, transferred))
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    torch.testing.assert_close(old_actions, new_actions, rtol=0, atol=0)
    start = c.brain_parameter_count
    torch.testing.assert_close(
        old.agents["genome"][:, start : start + 11], transferred[:, start : start + 11]
    )
    assert (transferred[:, start + 11] == -2).all()
    assert (transferred[:, start + 12] == -1).all()


def test_world_feedback_tracks_physical_energy_but_excludes_birth_transfer(config):
    c = replace(config, ecology_version=12, initial_population=1, feeding_hz=5, capacity=4)
    w = EcologyWorld(c)
    w.agents["pos"][:] = 64
    w.agents["energy"] = c.max_energy * w.agents["area"]
    w.initial_energy = w.agents["energy"].double().sum().item()
    w.agents["module_motor_plastic"][:, 0].fill_(0.1)
    w.agents["module_motor_baseline"][:, 0].fill_(0.5)
    before = w.agents["energy"].clone()
    w.step()
    assert w.totals["births"] == 1
    physical_cost = w.totals["maintenance"] + w.totals["propulsion"]
    assert w.agents["motor_reward"][0] == pytest.approx(-physical_cost)
    assert w.agents["energy"][0] < before[0] - physical_cost - 1
    assert w.agents["module_motor_plastic"][0].count_nonzero() > 0
    for key in (
        "module_motor_plastic",
        "module_motor_trace",
        "module_motor_baseline",
        "motor_reward",
    ):
        assert w.agents[key][1].count_nonzero() == 0
    w.step(60)
    assert w.totals["motor_learning_changes"] > 0
    assert w.totals["motor_learning_cost"] > 0
    assert w.metrics()["mean_motor_plastic_magnitude"] > 0
    assert abs(w.metrics()["energy_balance_error"]) < 0.005
    clone = fork_challenge(w, "erase_plastic", False, w.tick + 180)
    for key in ("module_motor_plastic", "module_motor_trace", "module_motor_baseline"):
        assert clone.agents[key].count_nonzero() == 0


def test_food_and_bites_contribute_actual_net_energy_to_motor_feedback(config):
    c = replace(config, ecology_version=12, feeding_hz=5)
    w = EcologyWorld(c)
    w.agents["genome"][:, c.brain_parameter_count + 6] = -3
    w.develop(w.agents)
    w.agents["pos"][:] = torch.tensor([[60.0, 64.0], [66.0, 64.0]])
    w.agents["heading"][:] = 0
    w.agents["actions"][:, 2] = torch.tensor([1.0, 0.0])
    w.agents["energy"] = c.birth_energy * w.agents["area"]
    w.initial_energy = w.agents["energy"].double().sum().item()
    w.append_food(torch.tensor([[60.0, 64.0]]), torch.tensor([20.0]), 0)
    w.totals["food_spawned"] += 20
    before = w.agents["energy"].clone()
    w.feed()
    w.hunt()
    assert w.totals["food_absorbed"] > 0 and w.totals["predation_absorbed"] > 0
    torch.testing.assert_close(
        w.agents["motor_reward"], w.agents["energy"] - before, rtol=0, atol=2e-5
    )
    assert abs(w.metrics()["energy_balance_error"]) < 0.005


@pytest.mark.parametrize(
    "ablation", ["no_motor_learning", "no_motor_reward", "no_exploration", "no_plasticity"]
)
def test_motor_controls_have_no_acquired_readout_and_retain_cost(config, ablation):
    c = replace(config, ecology_version=12, initial_population=1, initial_food=20)
    w = EcologyWorld(c, ablation=ablation)
    w.step(60)
    assert w.agents["module_motor_plastic"].count_nonzero() == 0
    assert w.totals["motor_learning_changes"] == 0
    assert w.totals["motor_learning_cost"] > 0
    assert w.agents["module_actions"][..., :2].count_nonzero() > 0
