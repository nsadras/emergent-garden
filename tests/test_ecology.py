from dataclasses import replace

import pytest
import torch

from emergent_garden.spatial import neighbors
from emergent_garden.world import World


def food(world, positions, energies):
    world.food_pos = torch.tensor(positions, dtype=torch.float32)
    world.food_energy = torch.tensor(energies, dtype=torch.float32)
    world.food_expiry = torch.full((len(energies),), 10000, dtype=torch.long)


def test_local_neighbors_include_all_real_contacts():
    gen = torch.Generator().manual_seed(37)
    p = torch.rand((250, 2), generator=gen) * 128
    q = torch.rand((130, 2), generator=gen) * 128
    i, j = neighbors(q, p, 8, 128)
    observed = torch.zeros((len(q), len(p)), dtype=torch.bool)
    observed[i, j] = True
    actual = torch.cdist(q, p) <= 8
    assert observed[actual].all()
    assert len(i) == len(torch.unique(torch.stack((i, j), 1), dim=0))
    assert len(i) < len(q) * len(p) / 8


def test_competing_feeders_conserve_food_and_respect_capacity(config):
    w = World(config)
    w.agents["pos"][:] = torch.tensor([[60, 64], [68, 64]])
    w.agents["energy"][:] = torch.tensor([249, 100])
    food(w, [[64, 64]], [20])
    before = w.agents["energy"].sum() + w.food_energy.sum()
    w.feed()
    assert torch.equal(w.agents["energy"], torch.tensor([250.0, 110.0]))
    assert w.food_energy.item() == 9
    assert w.agents["energy"].sum() + w.food_energy.sum() == before


def test_capacity_applies_across_multiple_particles(config):
    w = World(replace(config, initial_population=1))
    w.agents["pos"][0] = torch.tensor([64, 64])
    w.agents["energy"][0] = 249
    food(w, [[64, 64], [64, 65]], [20, 20])
    w.feed()
    assert w.agents["energy"].item() == 250
    assert w.food_energy.sum().item() == 39


def test_full_and_differential_propulsion(config):
    w = World(replace(config, initial_population=1))
    w.agents["pos"][0] = torch.tensor([64, 64])
    w.agents["heading"][0] = 0
    w.agents["motors"][0] = 1
    w.move()
    assert w.agents["pos"][0, 0].item() == pytest.approx(64.4)
    assert w.agents["heading"].item() == pytest.approx(0)
    w.agents["motors"][0] = torch.tensor([0.0, 1.0])
    w.move()
    assert w.agents["heading"].item() == pytest.approx(torch.pi / 120, abs=1e-6)


def test_passive_contacts_separate_without_drift(config):
    w = World(config)
    w.agents["pos"][:] = torch.tensor([64, 64])
    center = w.agents["pos"].mean(0).clone()
    w.move()
    torch.testing.assert_close(w.agents["pos"].mean(0), center)
    assert (w.agents["pos"][0] - w.agents["pos"][1]).norm() >= 7.99
    separated = w.agents["pos"].clone()
    w.move()
    torch.testing.assert_close(w.agents["pos"], separated)


def test_walls_keep_entire_body_inside(config):
    w = World(config)
    w.agents["pos"][:] = torch.tensor([[127, 64], [0, 64]])
    w.move()
    assert ((w.agents["pos"] - 64).norm(dim=1) <= 60.00001).all()
    assert w.agents["contact"].sum() == 2


def test_birth_debit_fresh_state_and_inheritance(config):
    w = World(replace(config, initial_population=1, mutation_probability=0))
    w.agents["pos"][0] = torch.tensor([64, 64])
    w.agents["energy"][0] = 220
    w.agents["h"][0] = 0.9
    genome = w.agents["genome"][0].clone()
    w.reproduce()
    assert w.population == 2
    torch.testing.assert_close(w.agents["energy"], torch.tensor([100.0, 100.0]))
    torch.testing.assert_close(w.agents["genome"][1], genome)
    assert w.agents["h"][1].count_nonzero() == 0
    assert w.agents["generation"][1] == 1
    assert w.agents["parent"][1] == 0
    assert w.agents["age"][1] == 0
    assert w.agents["cold"][1]
    assert w.totals["reproduction"] == 20


def test_failed_birth_does_not_charge_energy(config):
    w = World(replace(config, initial_population=1, capacity=1))
    w.agents["energy"][0] = 220
    w.reproduce()
    assert w.population == 1
    assert w.agents["energy"].item() == 220
    assert w.totals["blocked_births"] == 1
    assert w.agents["retry_tick"].item() == 60
    w.reproduce()
    assert w.totals["blocked_births"] == 1


def test_no_birth_space_does_not_charge_energy(config):
    c = replace(config, diameter=17.0, initial_population=1, patch_radius=1.0)
    w = World(c)
    w.agents["pos"][0] = torch.tensor([8.5, 8.5])
    w.agents["energy"][0] = 220
    w.reproduce()
    assert w.population == 1
    assert w.agents["energy"].item() == 220
    assert w.totals["blocked_births"] == 1


def test_mutation_changes_child_only_and_obeys_bounds(config):
    w = World(replace(config, initial_population=1, mutation_probability=1, mutation_sigma=5))
    w.agents["pos"][0] = torch.tensor([64, 64])
    w.agents["energy"][0] = 220
    original = w.agents["genome"][0].clone()
    w.reproduce()
    torch.testing.assert_close(w.agents["genome"][0], original)
    assert not torch.equal(w.agents["genome"][1], original)
    assert w.agents["genome"][1].abs().max() <= config.weight_limit


def test_starvation_precedes_feeding_and_ends_without_reseeding(config):
    w = World(
        replace(config, initial_population=1, birth_energy=1, basal_cost=120), controller="rest"
    )
    w.agents["pos"][0] = torch.tensor([64, 64])
    food(w, [[64, 64]], [20])
    w.step()
    assert w.population == 0
    assert w.food_energy.item() == 20
    assert w.totals["maintenance"] == 1
    assert w.totals["deaths"] == 1
    w.step(100)
    assert w.tick == 1


def test_energy_budget_and_spawn_rate_over_time(config):
    w = World(replace(config, initial_food=40, food_rate=7.5, food_lifetime=0.8), controller="rest")
    w.step(120)
    assert w.totals["food_spawned"] == (40 + 15) * 20
    assert w.totals["food_expired"] > 0
    assert abs(w.metrics()["energy_balance_error"]) < 0.05


def test_timestep_cost_is_time_based(config):
    w1 = World(replace(config, initial_population=1), controller="rest")
    w2 = World(replace(config, initial_population=1, physics_hz=120), controller="rest")
    w1.step(60)
    w2.step(120)
    assert w1.agents["energy"].item() == pytest.approx(w2.agents["energy"].item(), abs=0.001)


def test_recurrent_state_persists_between_updates(config):
    w = World(config)
    w.step()
    first = w.agents["h"].clone()
    assert first.count_nonzero() > 0
    w.step(2)
    torch.testing.assert_close(w.agents["h"], first)
    w.step()
    assert not torch.equal(w.agents["h"], first)
