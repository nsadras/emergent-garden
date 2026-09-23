from dataclasses import replace

import pytest
import torch

from emergent_garden import digestion
from emergent_garden.inheritance import brain_parts, upgrade_genomes
from emergent_garden.topology import effective_masks
from emergent_garden.world import World, create_world


def gut_world(config, **changes):
    c = replace(
        config,
        **{
            "ecology_version": 18,
            "gut_capacity": 200.0,
            "physics_hz": 30,
            "controller_hz": 10,
            "field_hz": 5,
            "feeding_hz": 5,
            "initial_population": 1,
            "detritus_delay": 3.0,
            **changes,
        },
    )
    w = create_world(c, controller="rest", ablation="no_attacks")
    a = w.agents
    a["genome"][:, c.brain_parameter_count] = 0  # Unit-sized digestive tissue.
    a["genome"][:, c.brain_parameter_count + 6] = -3  # One mature module.
    w.develop(a)
    a["pos"][:] = 64
    a["diet"][:] = 0.5
    a["energy"] = c.birth_energy * a["area"]
    w.initial_energy = a["energy"].double().sum().item()
    return w


def food(w, amount=100.0, kind=0, count=1):
    w.append_food(torch.full((count, 2), 64.0), torch.full((count,), amount), kind)
    w.totals["food_spawned"] += amount * count
    return torch.arange(len(w.food_energy) - count, len(w.food_energy))


def conserved(w):
    metrics = w.metrics()
    assert abs(metrics["energy_balance_error"]) < 0.005
    assert max(map(abs, metrics["trophic_detritus_balance_error"])) < 1e-8
    torch.testing.assert_close(w.food_credit.sum(1), w.food_energy, rtol=0, atol=1e-10)
    assert (w.food_energy >= 0).all()
    assert (digestion.loads(w) <= digestion.capacity(w) + 1e-8).all()


def same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            same(x, y)
    else:
        assert left == right


@pytest.mark.parametrize("kind", [0, 1])
def test_collection_shares_competition_without_assimilation_or_new_provenance(config, kind):
    w = gut_world(config, initial_population=2)
    packet = food(w, kind=kind)
    w.food_ready[packet] = 0
    energy = w.agents["energy"].clone()
    credits = w.food_credit.sum(0).clone()
    ledger = w.trophic.state_dict()
    digestion.capture(w, torch.tensor([0, 1]), packet.repeat(2))
    assert (digestion.loads(w)[:, kind] == 50).all()
    assert (w.food_owner >= 0).all()
    torch.testing.assert_close(w.agents["energy"], energy, rtol=0, atol=0)
    torch.testing.assert_close(w.food_credit.sum(0), credits, rtol=0, atol=0)
    same(w.trophic.state_dict(), ledger)
    assert w.totals["food_absorbed"] == w.agents["food_feedback"].sum() == 0
    assert w.totals["food_collected"] == 100
    conserved(w)


def test_full_pathway_does_not_block_others_and_both_paths_are_bounded(config):
    w = gut_world(config, initial_population=2, gut_capacity=100.0)
    w.agents["diet"][:] = torch.tensor([0.2, 0.8])
    j = food(w, amount=200)
    digestion.capture(w, torch.tensor([0, 1]), j.repeat(2))
    torch.testing.assert_close(digestion.loads(w)[:, 0], torch.tensor([20.0, 80.0]).double())
    assert w.food_energy[w.food_owner < 0].sum() == pytest.approx(100)
    # Vacate only the second body's fresh-food pathway; full first body cannot
    # reserve a share of the remaining resource on the next collection event.
    w.food_owner[w.food_owner == 1] = -1
    free = (w.food_owner < 0).nonzero().flatten()
    digestion.capture(w, torch.arange(2).repeat_interleave(len(free)), free.repeat(2))
    assert digestion.loads(w)[1, 0] == pytest.approx(80)
    j = food(w, amount=200, kind=1)
    w.food_ready[j] = 0
    digestion.capture(w, torch.tensor([0, 1]), j.repeat(2))
    torch.testing.assert_close(digestion.loads(w), digestion.capacity(w).double())
    conserved(w)


def test_collect_then_travel_and_digest_without_scent_from_the_gut(config):
    w = gut_world(config)
    food(w)
    w.rebuild_fields()
    assert w.fields[0].grid.sum() > 0
    w.feed()
    assert w.totals["fresh_processed"] == pytest.approx(60 / 5 * 0.5**2)
    carried = digestion.loads(w).sum().item()
    assert 90 < carried < 100
    before = w.agents["acquired"].item()
    w.agents["pos"][:] = torch.tensor([90.0, 90.0])
    digestion.follow(w)
    w.tick += 6
    w.rebuild_fields()
    assert w.fields[0].grid.count_nonzero() == 0
    w.feed()
    assert w.agents["acquired"].item() > before
    assert digestion.loads(w).sum().item() < carried
    torch.testing.assert_close(w.food_pos[w.food_owner >= 0], torch.tensor([[90.0, 90.0]]))
    # Recycling occurs at the carrier's current position, leaving a nutrient trail.
    assert (w.food_pos[w.food_kind == 1] == 90).all(1).any()
    conserved(w)


def test_gut_fullness_is_available_to_each_module_and_never_counts_as_stored_energy(config):
    w = gut_world(config)
    food(w, 50)
    digestion.capture(w, torch.tensor([0]), torch.tensor([0]))
    index = torch.arange(w.population)
    inputs = w.module_sensors(index)
    start = w.config.input_names.index("gut_fresh")
    assert inputs.shape == (1, 3, 42)
    assert (inputs[..., start] == 0.5).all()
    assert (inputs[..., start + 1] == 0).all()
    torch.testing.assert_close(inputs[..., -2], torch.full((1, 3), 0.4))
    w.ablation = "disabled"
    assert w.module_sensors(index)[..., start].sum() == 1.5
    w.ablation = "no_gut"
    assert w.module_sensors(index)[..., start : start + 2].count_nonzero() == 0
    conserved(w)


def test_carried_food_keeps_its_original_expiry_and_readiness(config):
    w = gut_world(config, food_lifetime=0.1, detritus_delay=1.0)
    fresh = food(w)
    detritus = food(w, kind=1)
    w.feed()
    assert digestion.loads(w)[0, 1] == 0  # Immature detritus cannot be collected.
    expiry = int(w.food_expiry[w.food_owner >= 0][0])
    w.step(expiry + 1)
    assert digestion.loads(w)[0, 0] == 0
    assert w.totals["food_expired"] > 90
    assert fresh.numel() == detritus.numel() == 1
    conserved(w)


def test_death_drops_food_at_last_position_without_reassigning_after_compaction(config):
    w = gut_world(config, initial_population=2)
    # IDs deliberately differ from array positions and aren't sorted.
    w.agents["id"][:] = torch.tensor([27, 10])
    food(w)
    digestion.capture(w, torch.tensor([0, 1]), torch.tensor([0, 0]))
    w.agents["pos"][:] = torch.tensor([[70.0, 60.0], [60.0, 70.0]])
    digestion.follow(w)
    w.initial_energy -= w.agents["energy"][0].item()
    w.agents["energy"][0] = 0
    w.remove_dead("fixture")
    assert w.agents["id"].tolist() == [10]
    assert w.food_owner.tolist() == [-1, 10]
    torch.testing.assert_close(digestion.loads(w), torch.tensor([[50.0, 0.0]]).double())
    torch.testing.assert_close(w.food_pos[0], torch.tensor([70.0, 60.0]))
    conserved(w)


def test_children_start_empty_and_carried_packets_do_not_inflate_source_vacancy(config):
    w = gut_world(config)
    food(w)
    w.food_patch[:] = 0
    before = w.patch_stock().clone()
    digestion.capture(w, torch.tensor([0]), torch.tensor([0]))
    torch.testing.assert_close(w.patch_stock(), before, rtol=0, atol=0)
    w.agents["energy"] = w.config.max_energy * w.agents["area"]
    w.initial_energy = w.agents["energy"].double().sum().item()
    w.reproduce()
    assert w.population == 2
    assert digestion.loads(w)[1].sum() == 0
    assert digestion.loads(w)[0].sum() == 100
    conserved(w)


@pytest.mark.parametrize(
    "mode", ["none", "no_recycling", "unlimited_handling", "unlimited_feeding", "no_gut"]
)
def test_food_inventory_replays_and_conserves_under_other_feeding_rules(config, mode):
    w = gut_world(config, initial_population=2, gut_capacity=20.0)
    w.ablation = mode
    food(w, 50, count=2)
    food(w, 50, kind=1)
    w.food_ready[:] = 0
    w.step(13)
    replay = World.from_state(w.state_dict())
    w.step(73)
    replay.step(73)
    same(w.state_dict(), replay.state_dict())
    conserved(w)
    assert torch.isfinite(digestion.fullness(w)).all()
    if mode == "no_recycling":
        assert w.totals["detritus_created"] == 0
    if mode == "no_gut":
        assert (w.food_owner == -1).all() and w.totals["food_collected"] == 0


def test_partial_collection_merges_equivalent_packets_without_resetting_timers(config):
    w = gut_world(config)
    food(w, 20, count=100)
    w.food_ready[:50] = -1
    w.food_expiry[50:] += 1
    old = w.food_credit.sum(0).clone()
    digestion.capture(w, torch.zeros(100, dtype=torch.long), torch.arange(100))
    assert (w.food_owner >= 0).sum() == 2
    torch.testing.assert_close(w.food_credit.sum(0), old, rtol=0, atol=1e-10)
    assert sorted(w.food_ready[w.food_owner >= 0].tolist()) == [-1, 0]
    assert w.food_expiry[w.food_owner >= 0].unique().numel() == 2
    conserved(w)


def test_v17_transfer_preserves_old_weights_and_new_founders_are_matched(config):
    old = replace(config, ecology_version=17)
    new = replace(old, ecology_version=18, gut_capacity=200.0)
    source = create_world(old)
    transferred = upgrade_genomes(old, new, source.founders)
    before, after = brain_parts(old, source.founders), brain_parts(new, transferred)
    for i, name in enumerate(old.input_names):
        torch.testing.assert_close(
            before[0][..., i], after[0][..., new.input_names.index(name)], rtol=0, atol=0
        )
    for name in digestion.GUT_INPUTS:
        k = new.input_names.index(name)
        assert after[0][..., k].count_nonzero() == 0
        assert effective_masks(new, transferred)[1][..., k].sum() > 0
    for a, b in zip(before[1:], after[1:], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    left = create_world(new)
    right = create_world(replace(new, gut_capacity=0.0))
    torch.testing.assert_close(left.founders, right.founders, rtol=0, atol=0)
    assert torch.equal(left.agents["pos"], right.agents["pos"])
    with pytest.raises(ValueError, match="Carried food requires"):
        replace(old, gut_capacity=20).validate()
    with pytest.raises(ValueError, match="requires a version"):
        create_world(old, ablation="no_gut")
