import math
from dataclasses import replace

import pytest
import torch

from emergent_garden.learning import shuffled_returns
from emergent_garden.world import World, create_world


def test_shuffling_preserves_rates_and_assigns_a_different_body_even_with_unequal_intervals():
    rates = torch.tensor([-2.0, 0, 3.0, 5.0])
    elapsed = torch.tensor([0.1, 0.03, 0.1, 0.02])
    returns = rates * elapsed
    before = returns.clone()
    generator = torch.Generator().manual_seed(43)
    for _ in range(10):
        changed = shuffled_returns(returns, elapsed, generator) / elapsed
        torch.testing.assert_close(changed.sort().values, rates, rtol=0, atol=1e-6)
        assert (changed != rates).all()
    torch.testing.assert_close(returns, before, rtol=0, atol=0)


def test_singleton_and_empty_batches_preserve_signals_and_do_not_consume_randomness():
    generator = torch.Generator().manual_seed(43)
    before = generator.get_state().clone()
    for reward in (torch.empty(0), torch.tensor([0.3])):
        actual = shuffled_returns(reward, torch.full_like(reward, 0.1), generator)
        torch.testing.assert_close(actual, reward, rtol=0, atol=0)
    assert torch.equal(before, generator.get_state())


def test_world_intervention_changes_learning_signal_without_moving_energy_or_founders(config):
    c = replace(config, ecology_version=19, initial_population=4)
    control = create_world(c)
    w = create_world(c, ablation="shuffled_motor_reward")
    torch.testing.assert_close(w.founders, control.founders, rtol=0, atol=0)
    for key in control.rng:
        assert torch.equal(w.rng[key].get_state(), control.rng[key].get_state())
    rates = torch.tensor([-2.0, 0, 3.0, 5.0])
    interval = 1 / c.controller_hz
    w.tick = c.physics_hz // c.controller_hz
    w.agents["motor_reward"] = rates * interval * c.feedback_scale * w.agents["area"]
    before_energy = w.agents["energy"].clone()
    w.update_controllers()
    observed = w.agents["module_motor_baseline"][:, 0] / (
        1 - math.exp(-interval / c.motor_baseline_tau)
    )
    torch.testing.assert_close(observed.sort().values, rates, rtol=1e-5, atol=1e-6)
    assert (observed - rates).abs().min() > 1
    torch.testing.assert_close(w.agents["energy"], before_energy, rtol=0, atol=0)
    assert w.agents["motor_reward"].count_nonzero() == 0
    with pytest.raises(ValueError, match="version containing"):
        create_world(replace(c, ecology_version=11), ablation="shuffled_motor_reward")


def test_shuffled_rewards_resume_with_the_same_random_assignment(config):
    c = replace(config, ecology_version=20, initial_population=4, initial_food=20, food_rate=1)
    w = create_world(c, ablation="shuffled_motor_reward")
    w.step(31)
    replay = World.from_state(w.state_dict())
    w.step(59)
    replay.step(59)
    assert w.totals == replay.totals
    assert w.events == replay.events
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], replay.agents[key], rtol=0, atol=0)
    for key in w.rng:
        assert torch.equal(w.rng[key].get_state(), replay.rng[key].get_state())
