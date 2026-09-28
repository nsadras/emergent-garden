"""The passive critics must reproduce native updates on the same experience."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from emergent_garden.value import value_shapes
from emergent_garden.world import World, create_world


@pytest.mark.parametrize("inputs", [0, 1])
def test_shadow_critic_matches_native_core_and_preserves_full_world(config, inputs, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "probe_paired_values.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("paired_value_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = replace(
        config, ecology_version=23, hidden_size=32, capacity=8,
        motor_normalized=1, motor_value_rate=0.02, motor_value_centered=0,
        motor_value_inputs=inputs, growth_delay=0.1, growth_reserve=1,
        initial_food=10, gut_capacity=200,
    ).validate()
    w = create_world(c)
    w.agents["genome"][0, c.brain_parameter_count + 6] = -3
    w.agents["genome"][1, c.brain_parameter_count + 6] = 2
    w.develop(w.agents)
    w.agents["energy"] = c.max_energy * w.agents["area"]
    plain = World.from_state(w.state_dict())
    observer = w.controller_observer = module.PairedValueObserver(w)
    mode = "sensory" if inputs else "hidden"
    for step in range(60):
        if step == 20:
            observer.start_forecasts(w, 0.5)
        before, events = w.time, len(w.events)
        w.step()
        plain.step()
        observer.record_step(w, before, events)
        observer.reserve(w.agents["id"])
        for key in value_shapes(c.hidden_size):
            torch.testing.assert_close(
                observer.states[mode][key][w.agents["id"]],
                w.agents[f"module_{key}"][:, 0], rtol=0, atol=0,
            )
    module.same_state(w.state_dict(), plain.state_dict())
    assert w.totals["births"] > 0
    assert w.agents["modules"].max() > 1
    for forecast in observer.forecasts.values():
        assert forecast.rows and all(r["tail"] is not None for r in forecast.rows.values())
    # Storage for later permanent IDs starts fresh without losing earlier memories.
    retained = {k: v.clone() for k, v in observer.states[mode].items()}
    observer.reserve(torch.tensor([1000]))
    for key, previous in retained.items():
        torch.testing.assert_close(
            observer.states[mode][key][:len(previous)], previous, rtol=0, atol=0
        )
        assert not observer.states[mode][key][len(previous):].count_nonzero()
