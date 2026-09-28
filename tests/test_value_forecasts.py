import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def fixture(monkeypatch, centered):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    observer = importlib.import_module("probe_value_forecasts").ForecastObserver
    a = {
        key: torch.tensor([value], dtype=torch.float64)
        for key, value in dict(age=30, area=2, acquired=0, spent=0, bitten=0).items()
    }
    a.update(
        id=torch.tensor([0]),
        modules=torch.tensor([1]),
        target_modules=torch.tensor([1]),
        module_motor_value_weights=torch.ones(1, 1, 2, dtype=torch.float64),
        module_motor_value_prediction=torch.ones(1, 1, dtype=torch.float64),
        module_motor_baseline=torch.full((1, 1), 0.3, dtype=torch.float64),
    )
    c = SimpleNamespace(
        dt=0.1, feedback_scale=10, motor_value_centered=centered, motor_value_horizon=1
    )
    w = SimpleNamespace(agents=a, config=c, time=0.0, events=[])
    probe = observer(w, 0.4)
    probe.before_controller(w, torch.tensor([0]), None)
    probe.after_controller(w)
    return w, probe


@pytest.mark.parametrize("centered", [False, True])
def test_forecast_uses_controller_discounts_and_does_not_count_later_deaths(monkeypatch, centered):
    w, probe = fixture(monkeypatch, centered)
    a = w.agents
    for tick in range(4):
        w.time = tick * 0.1
        if tick == 2:
            a["module_motor_baseline"].fill_(0.1)
            a["module_motor_value_prediction"].fill_(9)
            probe.after_controller(w)
        if tick < 2:
            a["acquired"] += 2
        else:
            a["spent"] += 1
        probe.record_step(w, w.time, 0)
    row = probe.rows[0]
    expected = (0.14 - 0.12 * math.exp(-0.2)) if centered else (0.2 - 0.1 * math.exp(-0.2))
    assert row["discounted_return"] == pytest.approx(expected, abs=1e-12)
    assert row["prediction"] == 1  # Subsequent predictions cannot replace the initial forecast.
    w.time = 0.4
    a["module_motor_value_prediction"].fill_(7)
    probe.after_controller(w)
    w.events.append(dict(event="death", id=0, time=0.4, acquired=4, spent=10, bitten=0))
    w.agents = {key: value[:0] for key, value in a.items()}
    probe.record_step(w, 0.4, 0)
    assert row["tail"] == 7
    assert row["died_at"] is None
    assert row["discounted_return"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("centered", [False, True])
def test_forecast_includes_the_fatal_partial_controller_interval(monkeypatch, centered):
    w, probe = fixture(monkeypatch, centered)
    w.events.append(dict(event="death", id=0, time=0.0, acquired=0, spent=2, bitten=0))
    w.agents = {key: value[:0] for key, value in w.agents.items()}
    probe.record_step(w, 0.0, 0)
    row = probe.rows[0]
    assert row["discounted_return"] == pytest.approx(-0.13 if centered else -0.1, abs=1e-12)
    assert row["tail"] == 0
    assert row["died_at"] == 0
