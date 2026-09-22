from dataclasses import replace

import pytest
import torch

from emergent_garden.ecology import EcologyWorld
from emergent_garden.storage import load_checkpoint, save_checkpoint


def eco(config, **changes):
    return EcologyWorld(replace(config, ecology_version=1, **changes))


def place_food(w, energy=20.0, kind=0):
    w.append_food(w.agents["pos"][:1].clone(), torch.tensor([energy]), kind)
    w.totals["food_spawned"] += energy


def test_patch_cap_and_recovery(config):
    w = eco(config, patches=1, patch_fraction=1.0, patch_capacity=40.0)
    w.spawn_food(100)
    assert len(w.food_energy) == 2
    assert w.totals["food_spawned"] == 40
    w.filter_food(torch.tensor([True, False]))
    w.spawn_food(100)
    assert w.patch_stock().item() == 40
    assert w.totals["food_spawned"] == 60


def test_feeding_recycles_without_creating_energy(config):
    w = eco(config, initial_population=1)
    place_food(w)
    w.feed()
    assert w.food_kind.tolist() == [1]
    assert w.food_energy.item() == pytest.approx(7.0)
    assert w.totals["fresh_absorbed"] > 0
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4
    w.feed()
    assert len(w.food_energy) == 0
    assert w.totals["detritus_created"] == 7
    assert w.totals["detritus_absorbed"] > 0
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4


def test_simultaneous_food_claims_respect_storage(config):
    w = eco(config)
    w.agents["pos"][1] = w.agents["pos"][0]
    w.agents["energy"] = config.max_energy * w.agents["area"] - 0.1
    before = w.agents["energy"].sum().item()
    place_food(w, 1000)
    w.feed()
    assert float(w.agents["energy"].sum()) - before == pytest.approx(0.2, abs=1e-4)
    assert (w.agents["energy"] <= config.max_energy * w.agents["area"] + 1e-4).all()
    assert w.food_energy.sum() > 990


def test_no_recycling_preserves_primary_assimilation(config):
    w = eco(config, initial_population=1)
    place_food(w)
    other = EcologyWorld.from_state(w.state_dict())
    other.ablation = "no_recycling"
    w.feed()
    other.feed()
    torch.testing.assert_close(w.agents["energy"], other.agents["energy"])
    assert other.totals["detritus_created"] == 0
    assert abs(other.metrics()["energy_balance_error"]) < 1e-4


def test_detritus_maturation_is_delayed_and_checkpointed(config):
    w = eco(config, initial_population=1, detritus_delay=3.0)
    place_food(w)
    w.feed()
    before = w.agents["energy"].clone()
    w.feed()
    torch.testing.assert_close(before, w.agents["energy"])
    other = EcologyWorld.from_state(w.state_dict())
    torch.testing.assert_close(w.food_ready, other.food_ready)
    w.tick = 3 * config.physics_hz
    w.feed()
    assert (w.agents["energy"] > before).all()


def test_inherited_body_birth_investment_and_blocked_birth(config):
    w = eco(
        config,
        initial_population=1,
        capacity=2,
        mutation_probability=0.0,
        trait_mutation_probability=0.0,
    )
    w.agents["pos"][0] = 64
    w.agents["energy"][0] = config.reproduction_threshold * w.agents["area"][0]
    before = w.agents["energy"].sum().item()
    w.reproduce()
    assert w.population == 2
    torch.testing.assert_close(w.agents["genome"][0], w.agents["genome"][1])
    torch.testing.assert_close(w.agents["radius"][0], w.agents["radius"][1])
    assert before - w.agents["energy"].sum().item() == pytest.approx(
        w.totals["reproduction"], abs=1e-4
    )
    w.agents["energy"] = config.max_energy * w.agents["area"]
    energy = w.agents["energy"].clone()
    w.reproduce()
    torch.testing.assert_close(energy, w.agents["energy"])


def test_ecology_checkpoint_full_replay(config, tmp_path):
    w = eco(config, initial_food=50, food_rate=10.0)
    w.step(17)
    save_checkpoint(w, tmp_path / "state.pt")
    other = load_checkpoint(tmp_path / "state.pt")
    w.step(123)
    other.step(123)
    assert w.events == other.events
    assert w.totals == other.totals
    for k in w.agents:
        torch.testing.assert_close(w.agents[k], other.agents[k], rtol=0, atol=0)
    for k in ("food_pos", "food_energy", "food_kind", "food_patch", "food_expiry"):
        torch.testing.assert_close(getattr(w, k), getattr(other, k), rtol=0, atol=0)
    for f, g in zip(w.fields, other.fields, strict=True):
        torch.testing.assert_close(f.grid, g.grid, rtol=0, atol=0)
    for k in w.rng:
        assert torch.equal(w.rng[k].get_state(), other.rng[k].get_state())
    assert abs(w.metrics()["energy_balance_error"]) < 0.01


def test_variable_radius_collision_and_wall(config):
    w = eco(config)
    w.agents["pos"][:] = 64
    w.agents["motors"][:] = 0
    w.move()
    assert (w.agents["pos"][0] - w.agents["pos"][1]).norm() >= w.agents["radius"].sum() - 1e-4
    w.agents["pos"][:] = 0
    w.project_walls()
    assert ((w.agents["pos"] - 64).norm(dim=1) <= 64 - w.agents["radius"] + 1e-4).all()


def test_community_assay_preserves_phenotypes_and_disables_mutation(config, tmp_path):
    from emergent_garden.experiments import community_assay
    from emergent_garden.storage import RunStore

    w = eco(config, initial_population=1)
    w.agents["pos"][0] = 64
    w.agents["energy"] = config.max_energy * w.agents["area"]
    w.reproduce()
    with_path = tmp_path / "source"
    store = RunStore(with_path, w)
    store.checkpoint(w)
    store.close()
    output = tmp_path / "assay"
    rows = community_assay(with_path, output, [10001], 0.1, "cpu", ["none", "disabled"])
    assert len(rows) == 3
    resumed = load_checkpoint(output / "10001-descendants-none" / "latest.pt")
    assert resumed.config.mutation_probability == 0
    assert resumed.config.trait_mutation_probability == 0
    assert (resumed.agents["radius"] > 0).all()
