from dataclasses import replace

import pytest
import torch

from emergent_garden.brain import controller_step, initial_state
from emergent_garden.ecology import EcologyWorld
from emergent_garden.field import ShelterField
from emergent_garden.inheritance import upgrade_genomes
from emergent_garden.topology import counts


def sheltered(config, **changes):
    return EcologyWorld(replace(config, **{"ecology_version": 10, **changes}))


def test_cover_is_bounded_local_and_clipped_to_dish(config):
    c = replace(config, shelter_radius=20.0)
    field = ShelterField(c, "cpu")
    centers = torch.tensor([[64.0, 64.0], [65.0, 64.0], [2.0, 2.0]])
    field.rebuild_shelter(centers)
    assert field.grid.min() >= 0
    assert field.grid.max() == 1
    assert field.grid[~field.mask].count_nonzero() == 0
    samples = field.sample(torch.tensor([[64.0, 64.0], [82.0, 64.0], [108.0, 64.0]]))
    assert samples[0] == 1
    assert 0 < samples[1] < 1
    assert samples[2] == 0
    field.rebuild_shelter(torch.empty((0, 2)))
    assert field.grid.count_nonzero() == 0


def attack_pair(config, protection):
    w = sheltered(config, shelter_protection=protection)
    a = w.agents
    a["pos"][:] = torch.tensor([[60.0, 64.0], [68.0, 64.0]])
    a["radius"][:] = 4
    a["area"][:] = 1
    a["modules"][:] = 1
    a["heading"][:] = 0
    a["attack"][:] = torch.tensor([1.0, 0.0])
    a["armor"][:] = 0
    a["actions"][:, 2] = 1
    a["energy"][:] = 100
    w.initial_energy = 200
    w.fields[7].grid.fill_(1)
    return w


@pytest.mark.parametrize("protection", [0.0, 0.5, 0.95, 1.0])
def test_cover_reduces_damage_and_uptake_without_creating_energy(config, protection):
    w = attack_pair(config, protection)
    w.hunt()
    raw = w.config.bite_rate * w.config.dt
    expected = raw * (1 - protection)
    assert w.agents["bitten"][1] == pytest.approx(expected, abs=1e-6)
    assert w.agents["meat_acquired"][0] == pytest.approx(expected * 0.65, abs=1e-6)
    assert w.totals["shelter_obstructed_demand"] == pytest.approx(raw * protection, abs=1e-6)
    assert abs(w.metrics()["energy_balance_error"]) < 1e-5


def test_either_endpoint_is_hidden_and_cover_does_not_enable_one_way_attacks(config):
    # A sharp test field protects either the source or destination of the same bite.
    for side in (slice(0, 16), slice(16, None)):
        w = attack_pair(config, 1.0)
        w.fields[7].grid.zero_()
        w.fields[7].grid[:, side] = 1
        w.hunt()
        assert w.agents["bitten"].sum() == 0


def test_protection_and_cover_sensing_can_be_ablated_separately(config):
    w = attack_pair(config, 0.95)
    physical = EcologyWorld.from_state(w.state_dict())
    physical.ablation = "no_shelter"
    sensory = EcologyWorld.from_state(w.state_dict())
    sensory.ablation = "no_shelter_cue"
    index = torch.arange(w.population)
    normal_inputs = w.sensors(index)
    torch.testing.assert_close(normal_inputs, physical.sensors(index), rtol=0, atol=0)
    start = w.config.input_names.index("shelter_-135")
    absent_inputs = sensory.sensors(index)
    assert normal_inputs[:, start : start + 4].sum() > 0
    assert absent_inputs[:, start : start + 4].count_nonzero() == 0
    absent_inputs[:, start : start + 4] = normal_inputs[:, start : start + 4]
    torch.testing.assert_close(normal_inputs, absent_inputs, rtol=0, atol=0)
    for world in (w, physical, sensory):
        world.hunt()
    torch.testing.assert_close(w.agents["energy"], sensory.agents["energy"], rtol=0, atol=0)
    assert physical.agents["bitten"][1] > w.agents["bitten"][1]
    assert physical.totals["shelter_obstructed_demand"] == 0


def test_shelter_selection_does_not_perturb_other_initial_conditions(config):
    empty = sheltered(config, initial_food=40, shelter_fraction=0.0)
    full = sheltered(config, initial_food=40, shelter_fraction=1.0)
    for key in empty.agents:
        torch.testing.assert_close(empty.agents[key], full.agents[key], rtol=0, atol=0)
    for key in ("food_pos", "food_energy", "food_kind", "patch_positions", "patch_phases"):
        torch.testing.assert_close(getattr(empty, key), getattr(full, key), rtol=0, atol=0)
    assert len(empty.shelter_indices) == 0
    assert len(full.shelter_indices) == config.patches
    for key in empty.rng:
        assert torch.equal(empty.rng[key].get_state(), full.rng[key].get_state())


def test_v9_controller_transfer_preserves_behavior_with_new_cover_inputs(config):
    source = replace(config, ecology_version=9)
    target = replace(source, ecology_version=10)
    genome = EcologyWorld(source).agents["genome"]
    padded = upgrade_genomes(source, target, genome)
    for old_count, new_count in zip(counts(source, genome), counts(target, padded), strict=True):
        torch.testing.assert_close(old_count, new_count, rtol=0, atol=0)
    old_state, new_state = initial_state(source, genome), initial_state(target, padded)
    rng = torch.Generator().manual_seed(17)
    for _ in range(40):
        x = torch.rand((2, source.input_size), generator=rng)
        y = torch.rand((2, target.input_size), generator=rng)
        for i, name in enumerate(source.input_names):
            y[:, target.input_names.index(name)] = x[:, i]
        old_state, old_actions = controller_step(source, genome, x, old_state)
        new_state, new_actions = controller_step(target, padded, y, new_state)
        torch.testing.assert_close(old_actions, new_actions, rtol=1e-6, atol=1e-7)
        for key in old_state:
            torch.testing.assert_close(old_state[key], new_state[key], rtol=1e-6, atol=1e-7)


def test_shelter_exposure_is_recorded_for_life_and_reset_at_birth(config):
    w = sheltered(config, initial_population=1, shelter_fraction=0.0)
    w.agents["pos"][:] = 64
    w.fields[7].grid.fill_(1)
    w.step(10)
    assert w.agents["shelter_time"].item() == pytest.approx(w.time)
    assert w.totals["shelter_agent_seconds"] == pytest.approx(w.time)
    w.agents["energy"] = w.config.max_energy * w.agents["area"]
    w.reproduce()
    assert w.population == 2
    assert w.agents["shelter_time"][1] == 0
    assert w.agent_record(0)["shelter_time"] > 0


def test_invalid_shelter_settings_and_incompatible_ablations(config):
    for settings in ({"shelter_fraction": 1.1}, {"shelter_protection": 1.1}, {"shelter_radius": 0}):
        with pytest.raises(ValueError):
            replace(config, ecology_version=10, **settings).validate()
    for ablation in ("no_shelter", "no_shelter_cue"):
        with pytest.raises(ValueError, match="requires a version"):
            EcologyWorld(replace(config, ecology_version=9), ablation=ablation)
