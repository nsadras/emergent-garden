from dataclasses import replace

import pytest
import torch

from emergent_garden.ecology import EcologyWorld
from emergent_garden.trophic import guilds


def tracked(config, **changes):
    world = EcologyWorld(
        replace(config, **{"ecology_version": 9, "detritus_delay": 0.0, **changes})
    )
    # One digestive module makes the controlled contacts unambiguous.
    world.agents["genome"][:, world.config.brain_parameter_count + 6] = -3
    world.develop(world.agents)
    world.agents["energy"] = world.config.birth_energy * world.agents["area"]
    world.initial_energy = world.agents["energy"].double().sum().item()
    return world


def assert_conserved(world):
    torch.testing.assert_close(world.food_credit.sum(1), world.food_energy.double())
    assert max(map(abs, world.metrics()["trophic_detritus_balance_error"])) < 1e-9


def test_guild_bins_include_thresholds_in_generalists():
    assert guilds(torch.tensor([0.1, 0.35, 0.5, 0.65, 0.9])).tolist() == [2, 1, 1, 1, 0]


def test_shared_packet_preserves_producer_mixture_through_partial_uptake_and_expiry(config):
    w = tracked(config)
    a = w.agents
    a["pos"][:] = 64
    a["diet"][:] = torch.tensor([0.9, 0.5])
    w.append_food(torch.tensor([[64.0, 64.0]]), torch.tensor([20.0]), 0)
    w.feed()
    # Two producers leave one physical packet, carrying equal raw-energy shares.
    assert w.food_kind.tolist() == [1]
    torch.testing.assert_close(w.food_credit, torch.tensor([[3.5, 3.5, 0, 0]]).double())
    assert_conserved(w)
    a["pos"][1] = 100
    a["diet"][0] = 0.1  # Classification follows the present phenotype, not ancestry.
    a["energy"][0] = w.config.max_energy * a["area"][0] - 0.2
    w.feed()
    assert 0 < w.food_energy.item() < 7
    assert w.trophic.detritus_uptake[0, 2].item() == pytest.approx(0.1, abs=1e-5)
    assert w.trophic.detritus_uptake[1, 2].item() == pytest.approx(0.1, abs=1e-5)
    assert w.trophic.detritus_uptake[:, :2].count_nonzero() == 0
    assert_conserved(w)
    stock = w.food_credit.sum(0).clone()
    w.food_expiry[:] = w.tick
    w.step()
    torch.testing.assert_close(w.trophic.detritus_expired, stock, rtol=0, atol=0)
    assert_conserved(w)


def test_unattributed_detritus_and_disabled_recycling_are_accounted(config):
    w = tracked(config, initial_population=1)
    w.agents["pos"][:] = 64
    w.agents["diet"][:] = 0.9
    w.ablation = "no_recycling"
    w.append_food(torch.tensor([[64.0, 64.0]]), torch.tensor([20.0]), 0)
    w.feed()
    assert w.trophic.detritus_introduced.count_nonzero() == 0
    assert w.trophic.fresh_uptake[0] > 0
    assert_conserved(w)
    w.append_food(torch.tensor([[64.0, 64.0]]), torch.tensor([10.0]), 1)
    w.feed()
    assert w.trophic.detritus_introduced.tolist() == [0, 0, 0, 10]
    assert w.trophic.detritus_uptake[3, 0] > 0
    assert_conserved(w)


def test_predation_tracks_actual_assimilation_and_death_cause(config):
    w = tracked(config, initial_population=3)
    a = w.agents
    a["pos"] = torch.tensor([[60.0, 60.0], [60.0, 68.0], [68.0, 64.0]])
    a["radius"][:] = 4
    a["area"][:] = 1
    a["heading"][:] = 0
    a["attack"][:] = torch.tensor([1.0, 1.0, 0.0])
    a["armor"][:] = 0
    a["actions"][:, 2] = 1
    a["diet"][:] = torch.tensor([0.9, 0.1, 0.5])
    a["energy"][:] = torch.tensor([20.0, 20.0, 1.0])
    w.initial_energy = 41
    w.hunt()
    expected = torch.zeros((3, 3), dtype=torch.float64)
    expected[1, 0] = expected[1, 2] = 0.325
    torch.testing.assert_close(w.trophic.predation_uptake, expected, atol=1e-7, rtol=0)
    assert w.trophic.predation_kills.tolist() == [0, 1, 0]
    w.remove_dead("predation")
    assert w.events[-1]["cause"] == "predation"
    assert w.events[-1]["id"] == 2
    assert abs(w.metrics()["energy_balance_error"]) < 1e-5


def test_accounting_is_physically_identical_to_v8_through_birth_death_and_reversals(config):
    c = replace(
        config,
        ecology_version=8,
        initial_population=4,
        initial_food=40,
        food_rate=10.0,
        food_lifetime=0.2,
        quality_period=0.2,
    )
    worlds = [EcologyWorld(c, seed=199), EcologyWorld(replace(c, ecology_version=9), seed=199)]
    for w in worlds:
        w.agents["energy"][0] = c.reproduction_threshold * w.agents["area"][0] + 1
        w.agents["energy"][1] = 0.00001
        w.initial_energy = w.agents["energy"].double().sum().item()
        w.step(180)
    before, after = worlds
    assert before.totals == after.totals
    assert after.totals["births"] > 0
    assert after.totals["deaths"] > 0
    assert after.totals["quality_reversals"] > 0
    assert before.events == [{k: v for k, v in e.items() if k != "cause"} for e in after.events]
    assert all(
        e["cause"] in ("maintenance", "predation") for e in after.events if e["event"] == "death"
    )
    for key in before.agents:
        torch.testing.assert_close(before.agents[key], after.agents[key], rtol=0, atol=0)
    for key in ("food_pos", "food_energy", "food_expiry", "food_ready", "food_kind", "food_patch"):
        torch.testing.assert_close(getattr(before, key), getattr(after, key), rtol=0, atol=0)
    for f, g in zip(before.fields, after.fields, strict=True):
        torch.testing.assert_close(f.grid, g.grid, rtol=0, atol=0)
    for key in before.rng:
        assert torch.equal(before.rng[key].get_state(), after.rng[key].get_state())
    assert_conserved(after)
