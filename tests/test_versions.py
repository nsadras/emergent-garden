from dataclasses import replace

import pytest
import torch

from emergent_garden.ecology import EcologyWorld
from emergent_garden.storage import load_checkpoint, save_checkpoint


def eco(config, **changes):
    return EcologyWorld(replace(config, **{"ecology_version": 1, **changes}))


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


@pytest.mark.parametrize("version", [1, 2, 3])
def test_ecology_checkpoint_full_replay(config, tmp_path, version):
    w = eco(config, ecology_version=version, initial_food=50, food_rate=10.0)
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


def attacking_world(config):
    w = eco(config, ecology_version=2, initial_population=3)
    a = w.agents
    a["pos"] = torch.tensor([[60.0, 60.0], [60.0, 68.0], [68.0, 64.0]])
    a["radius"][:] = 4
    a["area"][:] = 1
    a["heading"][:] = 0
    a["attack"][:] = torch.tensor([1.0, 1.0, 0.0])
    a["armor"][:] = 0
    a["actions"][:, 2] = 1
    a["energy"][:] = torch.tensor([20.0, 20.0, 1.0])
    w.initial_energy = 41.0
    return w


def test_simultaneous_predation_caps_shared_prey_and_conserves_energy(config):
    w = attacking_world(config)
    w.hunt()
    assert w.agents["energy"][2] == 0
    assert w.agents["meat_acquired"].sum().item() == pytest.approx(0.65)
    assert w.agents["bitten"].sum().item() == pytest.approx(1.0)
    assert w.totals["predation_kills"] == 1
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4
    w.remove_dead()
    assert w.population == 2
    assert w.events[-1]["bitten"] == 1


def test_attack_storage_caps_and_ablation(config):
    w = attacking_world(config)
    w.agents["energy"][:2] = config.max_energy
    w.initial_energy = 501.0
    w.hunt()
    assert (w.agents["energy"][:2] == config.max_energy).all()
    assert w.totals["predation_loss"] == 1.0
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4
    other = attacking_world(config)
    other.ablation = "no_attacks"
    before = other.agents["energy"].clone()
    other.hunt()
    torch.testing.assert_close(before, other.agents["energy"])


def test_armor_reduces_damage_and_has_maintenance_cost(config):
    unarmored = attacking_world(config)
    armored = attacking_world(config)
    for w in (unarmored, armored):
        w.agents["energy"][2] = 100.0
    armored.agents["armor"][2] = 1.0
    unarmored.hunt()
    armored.hunt()
    assert armored.agents["bitten"][2] < unarmored.agents["bitten"][2]
    assert armored.costs()[0][2] > unarmored.costs()[0][2]


def test_forecast_precedes_burst_and_disappears_before_it(config):
    w = eco(
        config,
        ecology_version=3,
        patches=1,
        patch_period=12.0,
        resource_burst=2.0,
        cue_lead=2.0,
        cue_duration=1.0,
    )
    w.patch_phases.zero_()
    w.tick = 9 * config.physics_hz
    assert w.patch_cues().item()
    assert w.patch_activity().item() == pytest.approx(w.config.resource_floor)
    w.tick = 11 * config.physics_hz
    assert not w.patch_cues().item()
    assert w.patch_activity().item() == pytest.approx(w.config.resource_floor)
    w.tick = 12 * config.physics_hz
    assert not w.patch_cues().item()
    assert w.patch_activity().item() > 1.0


def test_cue_ablation_preserves_other_senses(config):
    w = eco(config, ecology_version=3, initial_food=20)
    w.fields[3].grid.fill_(10.0)
    index = torch.arange(w.population)
    normal = w.sensors(index)
    w.ablation = "no_cue"
    ablated = w.sensors(index)
    assert ablated[:, 12:16].count_nonzero() == 0
    torch.testing.assert_close(ablated[:, :12], normal[:, :12])
    torch.testing.assert_close(ablated[:, 16:], normal[:, 16:])


def test_probe_separates_history_from_present_input(config):
    from emergent_garden.probes import cue_probe

    w = eco(config, ecology_version=3, initial_population=1)
    c = w.config
    g = torch.zeros_like(w.agents["genome"])
    g[0, 13], g[0, 14] = -1.0, 1.0
    output = c.hidden_size * c.input_size + c.hidden_size**2 + c.hidden_size
    g[0, output], g[0, output + c.hidden_size] = -2.0, 2.0
    normal = cue_probe(c, g, 3.0)
    reset = cue_probe(c, g, 3.0, reset=True)
    assert normal["cue_aligned_turn"][0] > 0.01
    assert reset["history_effect"] == [0.0]


def test_burst_schedule_preserves_mean_offered_supply(config):
    w = eco(
        config,
        ecology_version=3,
        initial_population=1,
        patches=1,
        patch_period=12.0,
        resource_burst=2.0,
        cue_lead=2.0,
        cue_duration=1.0,
        resource_floor=0.0,
        patch_capacity=10000.0,
        food_rate=5.0,
        basal_cost=0.0,
        propulsion_cost=0.0,
    )
    w.controller = "rest"
    w.patch_phases.zero_()
    w.agents["pos"][0] = 25
    w.patch_positions[0] = 90
    w.step(12 * config.physics_hz)
    assert w.totals["food_spawned"] == 5 * 12 * config.food_energy
    assert abs(w.metrics()["energy_balance_error"]) < 0.001
