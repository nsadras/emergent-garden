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


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
def test_ecology_checkpoint_full_replay(config, tmp_path, version):
    w = eco(
        config,
        ecology_version=version,
        initial_food=50,
        food_rate=10.0,
        feeding_hz=5 if version >= 11 else 0,
        motor_normalized=1 if version >= 12 else 0,
    )
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
    if version >= 9:
        torch.testing.assert_close(w.food_credit, other.food_credit, rtol=0, atol=0)
        for key, value in w.trophic.state_dict().items():
            torch.testing.assert_close(value, other.trophic.state_dict()[key], rtol=0, atol=0)
        assert max(map(abs, w.metrics()["trophic_detritus_balance_error"])) < 1e-9
    if version >= 10:
        torch.testing.assert_close(w.shelter_indices, other.shelter_indices, rtol=0, atol=0)
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


@pytest.mark.parametrize("version", [1, 8])
def test_community_assay_preserves_phenotypes_and_disables_mutation(config, tmp_path, version):
    from emergent_garden.experiments import community_assay
    from emergent_garden.storage import RunStore

    w = eco(config, ecology_version=version, initial_population=1)
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
    assert resumed.config.node_mutation_probability == 0
    assert resumed.config.edge_mutation_probability == 0
    assert resumed.config.module_mutation_probability == 0
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


def test_development_repeats_bounded_modules_and_pays_for_tissue(config):
    w = eco(config, ecology_version=4, initial_population=3)
    a, c = w.agents, w.config
    a["genome"][:, c.brain_parameter_count :] = 0
    a["genome"][:, c.brain_parameter_count + 6] = torch.tensor([-3.0, 0.0, 3.0])
    w.develop(a)
    assert a["modules"].tolist() == [1, 2, 3]
    assert a["module_mask"].sum(1).tolist() == [1, 2, 3]
    assert (a["radius"] <= c.max_body_radius).all()
    assert a["area"][0] < a["area"][1] < a["area"][2]
    assert (
        a["module_offset"].norm(dim=-1) + a["core_radius"][:, None] <= a["radius"][:, None] + 1e-5
    ).all()


def test_modules_have_local_sensing_and_independent_state(config):
    w = eco(config, ecology_version=4, initial_population=1)
    w.agents["genome"][0, w.config.brain_parameter_count + 6] = 3.0
    w.develop(w.agents)
    # An external gradient gives each repeated circuit distinct observations.
    w.fields[0].grid[:] = torch.arange(config.grid_size)[None, :]
    values = w.module_sensors(torch.tensor([0]))
    assert not torch.equal(values[:, 0], values[:, 2])
    w.update_controllers()
    assert not torch.equal(w.agents["module_h"][:, 0], w.agents["module_h"][:, 2])
    w.ablation = "pooled"
    w.agents["module_h"].zero_()
    w.update_controllers()
    torch.testing.assert_close(w.agents["module_h"][:, 0], w.agents["module_h"][:, 2])


def test_rotated_senses_preserve_local_intensity(config):
    w = eco(config, ecology_version=4, initial_food=40)
    index = torch.arange(w.population)
    before = w.module_sensors(index)
    w.ablation = "rotated"
    after = w.module_sensors(index)
    for start in range(0, w.config.input_size - 2, 4):
        torch.testing.assert_close(
            after[..., start : start + 4], before[..., start : start + 4].roll(2, -1)
        )
        torch.testing.assert_close(
            after[..., start : start + 4].mean(-1), before[..., start : start + 4].mean(-1)
        )


def test_module_lever_arms_change_turning_and_inactive_parts_are_silent(config):
    from emergent_garden.morphology import module_turn

    w = eco(config, ecology_version=4, initial_population=1)
    a = w.agents
    a["genome"][:, w.config.brain_parameter_count :] = 0
    w.develop(a)
    a["module_actions"].zero_()
    a["module_actions"][0, 0, :2] = 1
    first = module_turn(a).item()
    a["module_actions"].zero_()
    a["module_actions"][0, 1, :2] = 1
    assert first * module_turn(a).item() < 0
    w.update_controllers()
    assert a["module_h"][0, 2].count_nonzero() == 0
    assert a["module_actions"][0, 2].count_nonzero() == 0


def test_food_requires_contact_with_a_digestive_module(config):
    w = eco(config, ecology_version=4, initial_population=1)
    a = w.agents
    a["genome"][:, w.config.brain_parameter_count :] = 0
    w.develop(a)
    a["pos"][0], a["heading"][0] = 64, 0
    offset = torch.tensor([[-1.0, 1.0]]) / (2**0.5) * 5.5
    w.append_food(a["pos"] + offset, torch.tensor([20.0]), 0)
    before = a["energy"].clone()
    w.feed()
    torch.testing.assert_close(before, a["energy"])
    assert w.food_energy.item() == 20


def test_developmental_birth_inherits_structure_and_resets_each_circuit(config):
    w = eco(
        config,
        ecology_version=4,
        initial_population=1,
        mutation_probability=0.0,
        trait_mutation_probability=0.0,
    )
    w.agents["pos"][0] = 64
    w.agents["energy"] = config.max_energy * w.agents["area"]
    w.agents["module_h"].fill_(0.4)
    w.reproduce()
    assert w.population == 2
    for key in ("modules", "module_mask", "module_offset", "core_radius", "area"):
        torch.testing.assert_close(w.agents[key][0], w.agents[key][1])
    assert w.agents["module_h"][1].count_nonzero() == 0
    assert w.agents["module_h"][0].count_nonzero() > 0


@pytest.mark.parametrize("version", [4, 7, 8, 9, 10, 11, 12])
def test_modular_rendering_is_read_only(config, version):
    from emergent_garden.viewer import Renderer

    w = eco(config, ecology_version=version, initial_food=20)
    other = EcologyWorld.from_state(w.state_dict())
    renderer = Renderer(256)
    renderer.zoom = 4
    renderer.show_brain = True
    renderer.center = w.agents["pos"][0].numpy()
    renderer.selected = 0
    for _ in range(3):
        w.step(4)
        renderer.draw(w)
        other.step(4)
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], other.agents[key], rtol=0, atol=0)
    if version >= 9:
        torch.testing.assert_close(w.food_credit, other.food_credit, rtol=0, atol=0)
        for key, value in w.trophic.state_dict().items():
            torch.testing.assert_close(value, other.trophic.state_dict()[key], rtol=0, atol=0)


def test_genome_transfer_preserves_existing_circuit_and_neutralizes_new_inputs(config):
    from emergent_garden.brain import advance
    from emergent_garden.inheritance import upgrade_genomes

    source = eco(config)
    target = replace(source.config, ecology_version=4)
    genome = upgrade_genomes(source.config, target, source.agents["genome"])
    inputs = torch.linspace(0.0, 1.0, source.config.input_size).expand(2, -1)
    expanded = torch.ones((2, target.input_size))  # New channels can be active immediately.
    expanded[:, : source.config.input_size - 2] = inputs[:, :-2]
    expanded[:, -2:] = inputs[:, -2:]
    hidden = torch.full((2, config.hidden_size), 0.1)
    old_h, old_actions = advance(source.config, source.agents["genome"], inputs, hidden)
    tau = 0.2 + 4.8 * genome[:, target.brain_parameter_count + 5].sigmoid()
    new_h, new_actions = advance(target, genome, expanded, hidden, tau)
    torch.testing.assert_close(new_h, old_h)
    torch.testing.assert_close(new_actions[:, :2], old_actions)
    torch.testing.assert_close(
        genome[:, target.brain_parameter_count : target.brain_parameter_count + 3],
        source.agents["genome"][:, source.config.brain_parameter_count :],
    )
    with pytest.raises(ValueError, match="weight_limit"):
        upgrade_genomes(source.config, replace(target, weight_limit=0.1), source.agents["genome"])


def test_legacy_api_cannot_silently_run_a_new_preset_as_v0(config):
    from emergent_garden.world import World, create_world

    c = replace(config, ecology_version=5)
    with pytest.raises(ValueError, match="create_world"):
        World(c)
    assert isinstance(create_world(c), EcologyWorld)


def test_seeded_population_records_origins_and_replays(config, tmp_path):
    from emergent_garden.inheritance import seed_population
    from emergent_garden.storage import RunStore

    old = eco(config, ecology_version=3)
    store = RunStore(tmp_path / "source", old)
    store.checkpoint(old)
    store.close()
    w = eco(config, ecology_version=4)
    seed_population(w, tmp_path / "source")
    assert w.agents["modules"].tolist() == [1, 1]
    assert w.agents["module_h"].count_nonzero() == 0
    assert len(w.seeded_from["source_ids"]) == w.population
    resumed = EcologyWorld.from_state(w.state_dict())
    assert resumed.seeded_from == w.seeded_from
    w.step(21)
    resumed.step(21)
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], resumed.agents[key], rtol=0, atol=0)
    assert abs(w.metrics()["energy_balance_error"]) < 0.01


def test_trail_diffusion_conserves_mass_inside_dish_and_decays(config):
    from emergent_garden.field import TrailField

    c = replace(config, signal_half_life=10.0)
    field = TrailField(c, torch.device("cpu"))
    field.deposit(torch.tensor([[64.0, 64.0], [123.0, 64.0]]), torch.tensor([10.0, 20.0]))
    assert field.mass() == pytest.approx(30.0, abs=1e-5)
    decayed = field.advance(10.0)
    assert decayed == pytest.approx(15.0, abs=1e-5)
    assert field.mass() == pytest.approx(15.0, abs=1e-4)
    assert field.grid[~field.mask].count_nonzero() == 0
    assert field.grid.min() >= 0


def test_emission_on_last_tick_is_paid_and_energy_conservative(config):
    w = eco(config, ecology_version=5, initial_population=1)
    w.agents["energy"][:] = 0.001
    w.initial_energy = float(w.agents["energy"].sum())
    w.agents["actions"][:, 3] = 1.0
    w.update_controllers = lambda: None
    w.step()
    assert w.population == 0
    assert 0 < w.totals["signaling_cost"] < 0.001
    assert w.totals["signal_emitted"] == pytest.approx(
        w.totals["signaling_cost"] * w.config.signal_rate / w.config.signal_cost, rel=1e-6
    )
    assert abs(w.metrics()["energy_balance_error"]) < 1e-8


def test_emission_and_reception_controls_keep_other_dynamics_matched(config):
    w = eco(config, ecology_version=5, initial_food=20)
    w.ablation = "no_signal"
    other = EcologyWorld.from_state(w.state_dict())
    other.ablation = "no_emission"
    w.step(120)
    other.step(120)
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], other.agents[key], rtol=0, atol=0)
    assert w.fields[4].mass() > 0
    assert other.fields[4].mass() == 0
    assert w.totals["signaling_cost"] == other.totals["signaling_cost"]
    assert abs(w.metrics()["signal_balance_error"]) < 0.01
    assert abs(w.metrics()["energy_balance_error"]) < 0.01


def test_periodic_archive_retains_multiple_replayable_states(config, tmp_path):
    from emergent_garden.storage import RunStore

    w = eco(config, ecology_version=5, archive_sim_seconds=0.1)
    store = RunStore(tmp_path / "run", w)
    for _ in range(3):
        w.step(6)
        store.measure(w)
    store.close()
    paths = sorted((tmp_path / "run" / "checkpoints").glob("*.pt"))
    assert len(paths) == 3
    first = load_checkpoint(paths[0])
    last = load_checkpoint(paths[-1])
    first.step(last.tick - first.tick)
    for key in w.agents:
        torch.testing.assert_close(first.agents[key], last.agents[key], rtol=0, atol=0)
    torch.testing.assert_close(first.fields[4].grid, last.fields[4].grid, rtol=0, atol=0)


def test_long_chemical_ledger_uses_the_fields_representable_decay(config):
    from emergent_garden.field import TrailField

    field = TrailField(config, torch.device("cpu"))
    decayed = 0.0
    for _ in range(400):
        field.deposit(torch.tensor([[64.0, 64.0]]), torch.tensor([20.0]))
        decayed += field.advance(0.2)
    assert abs(8000 - decayed - field.mass()) < 0.003
