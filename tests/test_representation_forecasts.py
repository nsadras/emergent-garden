import importlib
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_observation import assert_same

from emergent_garden.world import World, create_world


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_learned_representations")


def test_forecast_excludes_already_received_energy_and_includes_fatal_partial_interval(probe):
    ledger = probe.ForecastLedger(30, horizon=2, window=0.2)
    prediction = dict(fixed=0.7)
    ledger.start(8, 3, 5, 5.01, prediction)
    prediction["fixed"] = 99
    ledger.receive(8, 0, 3, 100)
    ledger.receive(8, 3, 6, 0.1)
    ledger.receive(8, 6, 8, -0.05, terminal=True)
    row = ledger.rows[0]
    assert row["discounted_return"] == pytest.approx(0.1 - math.exp(-0.1 / 2) * 0.05)
    assert row["intervals"] == 2
    assert row["predictions"] == dict(fixed=0.7)
    assert row["complete"] and row["terminal_tick"] == 8
    assert not ledger.pending


def test_forecast_closes_at_horizon_and_does_not_reclassify_later_death(probe):
    ledger = probe.ForecastLedger(30, horizon=2, window=0.2)
    ledger.start(8, 3, 30, 30, {})
    ledger.receive(8, 3, 6, 1)
    ledger.receive(8, 6, 9, 2)
    ledger.receive(8, 9, 10, -100, terminal=True)
    row = ledger.rows[0]
    assert row["discounted_return"] == pytest.approx(1 + 2 * math.exp(-0.1 / 2))
    assert row["complete"] and row["terminal_tick"] is None


def test_partial_window_boundary_is_rejected_instead_of_prorating_unknown_returns(probe):
    ledger = probe.ForecastLedger(30, window=0.2)
    ledger.start(8, 3, 5, 5, {})
    with pytest.raises(AssertionError):
        ledger.receive(8, 6, 10, 1)


def test_received_returns_use_each_interval_current_area_including_death(probe, config):
    world = create_world(config, seed=1)
    observer = probe.RepresentationObserver(world)
    observer.ledger = probe.ForecastLedger(30, horizon=2, window=0.2)
    observer.ledger.start(0, 3, 5, 5, {})
    observer.receive(
        torch.tensor([0]), torch.tensor([3]), 6, torch.tensor([20.0]), torch.tensor([2.0])
    )
    observer.receive(
        torch.tensor([0]),
        torch.tensor([6]),
        8,
        torch.tensor([-30.0]),
        torch.tensor([3.0]),
        terminal=True,
    )
    row = observer.ledger.rows[0]
    scale = config.feedback_scale
    assert row["discounted_return"] == pytest.approx((10 - 10 * math.exp(-0.1 / 2)) / scale)
    assert torch.cat(observer.intervals["area"]).tolist() == [2, 3]


def test_observing_native_births_and_deaths_is_exactly_passive(probe, config):
    c = replace(
        config,
        ecology_version=25,
        hidden_size=8,
        initial_neurons=4,
        neural_timing_range=4,
        initial_timing_sigma=0.5,
        initial_population=4,
        motor_normalized=1,
        motor_learning_rate=0.01,
        exploration_max=0.15,
        max_energy=1000,
    ).validate()
    world = create_world(c, seed=7)
    # Give both copies the same engineering-fixture energy to exercise births.
    world.agents["energy"][:] = 900 * world.agents["area"]
    plain = World.from_state(world.state_dict())
    observer = probe.RepresentationObserver(world, forecast_until=6)
    world.controller_observer = observer
    original = world.remove_dead

    def capture(cause="unspecified"):
        observer.before_death(world)
        original(cause)

    world.remove_dead = capture
    for tick in range(27 * c.physics_hz):
        if tick == 6 * c.physics_hz:
            world.agents["energy"][0] = 0
            plain.agents["energy"][0] = 0
        start = len(world.events)
        world.step()
        observer.record_births(world, start)
        plain.step()
    assert_same(world.state_dict(), plain.state_dict())
    assert world.totals["births"] > 0
    assert observer.deaths
    assert len(observer.births) == c.initial_population + world.totals["births"]
    assert observer.ledger.rows and not observer.ledger.pending
    assert any(row["terminal_tick"] is not None for row in observer.ledger.rows)
    assert observer.changed["fixed"] == 0
    assert observer.changed["adaptive"] > 0
    assert observer.shuffled_count > 0
    assert all(
        abs(row["actual_age"] - row["cohort_age"]) < 1 / c.controller_hz
        for row in observer.ledger.rows
    )
