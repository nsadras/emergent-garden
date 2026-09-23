from dataclasses import replace

import pytest
import torch

from emergent_garden.ecology import EcologyWorld


def handling_world(config, **changes):
    c = replace(
        config,
        **{
            "ecology_version": 11,
            "initial_population": 1,
            "physics_hz": 30,
            "controller_hz": 10,
            "field_hz": 5,
            "detritus_delay": 3.0,
            **changes,
        },
    )
    w = EcologyWorld(c)
    w.agents["genome"][:, c.brain_parameter_count + 6] = -3
    w.develop(w.agents)
    w.agents["pos"][:] = 64
    w.agents["energy"] = c.birth_energy * w.agents["area"]
    w.initial_energy = w.agents["energy"].double().sum().item()
    return w


def add_food(w, kind, count=1):
    w.append_food(torch.full((count, 2), 64.0), torch.full((count,), 20.0), kind)
    # Controlled initial detritus is already mature; newly produced detritus isn't.
    w.food_ready[-count:] = w.tick
    w.totals["food_spawned"] += 20.0 * count


def conserved(w):
    assert abs(w.metrics()["energy_balance_error"]) < 0.005
    assert max(map(abs, w.metrics()["trophic_detritus_balance_error"])) < 1e-8
    torch.testing.assert_close(w.food_credit.sum(1), w.food_energy.double())


def test_handling_limits_both_food_paths_without_unwanted_food_clogging_them(config):
    w = handling_world(config)
    a, c = w.agents, w.config
    a["diet"][:] = 0.1
    add_food(w, 0, 5)
    add_food(w, 1, 5)
    tissue = (a["core_radius"][0] / c.body_radius).square().item()
    w.feed()
    assert a["fresh_processed"].item() == pytest.approx(60 * c.dt * tissue * 0.1**2)
    assert a["detritus_processed"].item() == pytest.approx(60 * c.dt * tissue * 0.9**2)
    assert a["detritus_processed"].item() > 80 * a["fresh_processed"].item()
    assert sum(w.totals[f"{k}_processed"] for k in ("fresh", "detritus")) <= 60 * c.dt * tissue
    conserved(w)


def test_simultaneous_competitors_keep_food_and_storage_limits(config):
    w = handling_world(config, initial_population=2)
    w.agents["diet"][:] = torch.tensor([0.9, 0.1])
    w.agents["energy"] = w.config.max_energy * w.agents["area"] - 0.1
    w.initial_energy = w.agents["energy"].double().sum().item()
    add_food(w, 0, 5)
    add_food(w, 1, 5)
    w.feed()
    assert (w.agents["energy"] <= w.config.max_energy * w.agents["area"] + 1e-4).all()
    assert w.agents["acquired"].sum().item() <= 0.2 + 1e-4
    assert w.food_energy.sum() > 190
    conserved(w)


@pytest.mark.parametrize("feeding_hz", [0, 5])
def test_processing_rate_scales_with_time_and_digestive_tissue(config, feeding_hz):
    worlds = []
    for hz in (30, 60):
        w = handling_world(config, physics_hz=hz, feeding_hz=feeding_hz, max_energy=5000.0)
        w.agents["diet"][:] = 0.9
        add_food(w, 0, 5)
        for _ in range(hz):
            w.feed()
            w.tick += 1
        tissue = (w.agents["core_radius"][0] / w.config.body_radius).square().item()
        assert w.totals["fresh_processed"] == pytest.approx(60 * tissue * 0.9**2, rel=1e-6)
        assert w.totals["detritus_processed"] == 0  # Maturation delay remains in force.
        conserved(w)
        worlds.append(w)
    assert worlds[0].totals["fresh_processed"] == pytest.approx(worlds[1].totals["fresh_processed"])
    multi = handling_world(config)
    multi.agents["genome"][:, multi.config.brain_parameter_count + 6] = 3
    multi.develop(multi.agents)
    multi.agents["area"][:] = 100  # Membrane volume cannot increase digestive throughput.
    multi.agents["diet"][:] = 0.9
    add_food(multi, 0, 5)
    multi.feed()
    expected = worlds[0].totals["fresh_processed"] * 3 * multi.config.dt
    assert multi.totals["fresh_processed"] == pytest.approx(expected, rel=1e-6)


def test_feeding_cadence_applies_to_both_capacity_treatments(config):
    for ablation in ("none", "unlimited_handling"):
        w = handling_world(config, feeding_hz=5, max_energy=5000.0)
        w.ablation = ablation
        w.agents["diet"][:] = 0.1
        add_food(w, 0, 5)
        add_food(w, 1, 5)
        for tick in range(1, 6):
            w.tick = tick
            w.feed()
        assert w.totals["food_absorbed"] == 0
        w.tick = 6
        w.feed()
        assert w.totals["food_absorbed"] > 0
        if ablation == "unlimited_handling":
            assert w.totals["fresh_processed"] == 100
            assert w.totals["detritus_processed"] == 100
        else:
            tissue = (w.agents["core_radius"][0] / w.config.body_radius).square().item()
            assert w.totals["fresh_processed"] == pytest.approx(60 / 5 * tissue * 0.1**2)
        conserved(w)


def test_many_tiny_meals_are_aggregated_before_body_energy_rounding(config):
    w = handling_world(config, feeding_hz=5)
    count = 10000
    amount = torch.full((count,), 1e-5)
    w.append_food(torch.full((count, 2), 64.0), amount, 0)
    w.totals["food_spawned"] += amount.double().sum().item()
    before = w.agents["energy"].item()
    w.feed()
    assert w.totals["food_absorbed"] > 0.005
    assert w.agents["energy"].item() - before == pytest.approx(w.totals["food_absorbed"], abs=1e-5)
    conserved(w)


def test_processing_history_belongs_to_the_individual_and_resets_at_birth(config):
    w = handling_world(config)
    add_food(w, 0)
    w.feed()
    assert w.agent_record(0)["fresh_processed"] > 0
    w.agents["energy"] = w.config.max_energy * w.agents["area"]
    w.reproduce()
    assert w.population == 2
    assert w.agents["fresh_processed"][1] == 0
    assert w.agents["detritus_processed"][1] == 0


def test_unlimited_processing_restores_v10_physics_exactly(config):
    c = replace(
        config,
        ecology_version=10,
        initial_population=4,
        initial_food=40,
        food_rate=10.0,
        food_lifetime=0.2,
        quality_period=0.2,
    )
    old = EcologyWorld(c, seed=199)
    new = EcologyWorld(
        replace(c, ecology_version=11, feeding_hz=5), seed=199, ablation="unlimited_feeding"
    )
    for w in (old, new):
        w.agents["energy"][0] = c.reproduction_threshold * w.agents["area"][0] + 1
        w.agents["energy"][1] = 0.00001
        w.initial_energy = w.agents["energy"].double().sum().item()
        w.step(180)
    assert old.totals == {k: new.totals[k] for k in old.totals}
    assert old.totals["births"] > 0 and old.totals["deaths"] > 0
    assert old.events == [
        {k: v for k, v in e.items() if k not in ("fresh_processed", "detritus_processed")}
        for e in new.events
    ]
    for key in old.agents:
        torch.testing.assert_close(old.agents[key], new.agents[key], rtol=0, atol=0)
    for key in (
        "food_pos",
        "food_energy",
        "food_expiry",
        "food_ready",
        "food_kind",
        "food_patch",
        "food_credit",
    ):
        torch.testing.assert_close(getattr(old, key), getattr(new, key), rtol=0, atol=0)
    for f, g in zip(old.fields, new.fields, strict=True):
        torch.testing.assert_close(f.grid, g.grid, rtol=0, atol=0)
    for key in old.rng:
        assert torch.equal(old.rng[key].get_state(), new.rng[key].get_state())
    conserved(new)


def test_invalid_handling_rate_and_version_are_rejected(config):
    with pytest.raises(ValueError, match="handling_rate"):
        handling_world(config, handling_rate=0)
    with pytest.raises(ValueError, match="requires a version"):
        EcologyWorld(replace(config, ecology_version=10), ablation="unlimited_feeding")
    with pytest.raises(ValueError, match="requires a version"):
        EcologyWorld(replace(config, ecology_version=10), ablation="unlimited_handling")
    with pytest.raises(ValueError, match="feeding_hz"):
        handling_world(config, feeding_hz=7)
