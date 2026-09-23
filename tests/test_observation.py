from dataclasses import replace

import numpy as np
import pytest
import torch

from emergent_garden.observation import ControllerObserver, TrailHistory, reproduction_readiness
from emergent_garden.world import World, create_world


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            assert_same(x, y)
    else:
        assert left == right


@pytest.mark.parametrize("version", [0, 1, 8, 12, 15])
def test_readiness_matches_birth_energy_boundary(config, version):
    c = replace(
        config,
        ecology_version=version,
        initial_population=1,
        reproduction_threshold=150,
        reproduction_debit=110,
        mutation_probability=0,
        trait_mutation_probability=0,
        node_mutation_probability=0,
        edge_mutation_probability=0,
        module_mutation_probability=0,
    )
    w = create_world(c)
    a = w.agents
    a["pos"][:] = c.diameter / 2
    if version >= 4:
        a["genome"][:, c.brain_parameter_count + 6] = -3  # Mature one-module body.
        w.develop(a)
    threshold = c.reproduction_threshold * (a["area"][0] if version else 1)
    a["energy"][0] = threshold - 1
    readiness = reproduction_readiness(w, 0)
    assert 0 < readiness.energy_fraction < 1
    assert not readiness.energy_ready and not readiness.ready_to_try
    assert readiness.status == "Needs energy"
    w.reproduce()
    assert w.totals["births"] == 0

    a["energy"][0] = threshold
    readiness = reproduction_readiness(w, 0)
    assert readiness.threshold == float(threshold)
    assert readiness.energy_fraction == 1 and readiness.ready_to_try
    assert readiness.storage_fraction == pytest.approx(c.reproduction_threshold / c.max_energy)
    assert readiness.status == "Ready to try"
    w.reproduce()
    assert w.totals["births"] == 1
    assert reproduction_readiness(w, 0).energy_fraction < 1


def test_full_birth_bar_still_waits_for_maturity_retry_and_capacity(config):
    c = replace(config, ecology_version=15, initial_population=1, capacity=2)
    w = create_world(c)
    a = w.agents
    a["pos"][:] = c.diameter / 2
    a["genome"][:, c.brain_parameter_count + 6] = 3
    w.develop(a)
    a["energy"] = c.max_energy * a["area"]
    readiness = reproduction_readiness(w, 0)
    assert readiness.energy_fraction == 1
    assert readiness.storage_fraction == 1
    assert not readiness.mature and not readiness.ready_to_try
    assert readiness.status == "Growing 1/3 modules"
    w.reproduce()
    assert w.totals["births"] == 0

    a["development_stage"].fill_(3)
    w.develop(a)
    readiness = reproduction_readiness(w, 0)
    assert readiness.mature
    # The same energy buys a smaller fraction of the larger body's threshold.
    assert readiness.energy_fraction < 1
    assert readiness.storage_fraction < 1
    a["energy"] = c.reproduction_threshold * a["area"]
    a["retry_tick"].fill_(c.physics_hz // 2)
    readiness = reproduction_readiness(w, 0)
    assert readiness.energy_ready and readiness.mature and not readiness.ready_to_try
    assert readiness.retry_seconds == 0.5
    assert readiness.status == "Retry in 0.5s"
    w.reproduce()
    assert w.totals["births"] == 0
    w.tick = c.physics_hz // 2
    assert reproduction_readiness(w, 0).ready_to_try
    w.reproduce()
    assert w.totals["births"] == 1

    w.agents["energy"][0] = c.max_energy * w.agents["area"][0]
    w.tick = int(w.agents["retry_tick"][0])
    readiness = reproduction_readiness(w, 0)
    assert readiness.energy_ready and readiness.mature and readiness.retry_seconds == 0
    assert readiness.population_full and not readiness.ready_to_try
    assert readiness.status == "Population full"
    w.reproduce()
    assert w.totals["births"] == 1 and w.totals["blocked_births"] == 1


@pytest.mark.parametrize("version", [0, 3, 4, 7, 8, 12, 15])
def test_sample_reconstructs_actual_inputs_hidden_and_outputs(config, version, monkeypatch):
    c = replace(config, ecology_version=version, initial_food=8)
    w = create_world(c)
    a = w.agents
    if version >= 4:
        a["genome"][:, c.brain_parameter_count + 6] = 4
        if version >= 13:
            a["development_stage"].fill_(3)
        w.develop(a)
        a["module_h"].copy_(
            torch.linspace(-0.6, 0.8, a["module_h"].numel()).reshape_as(a["module_h"])
        )
    else:
        a["h"].fill_(0.2)
    if version >= 7:
        a["module_plastic"].fill_(0.15)
        a["module_trace"].fill_(0.3)
    if version >= 12:
        a["module_motor_plastic"].fill_(0.06)
        a["module_motor_trace"].fill_(0.1)
        a["motor_reward"].fill_(5)
    if version >= 6:
        a["food_feedback"].fill_(4)
        a["damage_feedback"].fill_(2)
    observer = w.controller_observer = ControllerObserver()
    observer.select(int(a["id"][1]))
    name = "module_sensors" if version >= 4 else "sensors"
    sense = getattr(w, name)
    captured = []

    def record_inputs(index):
        values = sense(index)
        captured.append(values.clone())
        return values

    monkeypatch.setattr(w, name, record_inputs)
    w.update_controllers()
    sample = observer.sample
    count = int(a["modules"][1]) if version >= 4 else 1
    expected = captured[0][1, :count] if version >= 4 else captured[0][1:2]
    np.testing.assert_array_equal(sample.inputs, expected.numpy())
    assert sample.identifier == 1
    assert sample.tick == w.tick
    if version >= 6:
        assert sample.inputs[:, -4].min() > 0
        assert a["food_feedback"].count_nonzero() == 0
    for module in range(count):
        input_drive, recurrent, bias = sample.drives(module)
        hidden = (
            (1 - sample.alpha) * sample.previous[module]
            + sample.alpha * np.tanh(input_drive + recurrent + bias)
        ) * sample.nodes
        np.testing.assert_allclose(hidden, sample.hidden[module], rtol=2e-6, atol=2e-7)
        _, _, _, wo, bo = sample.matrices(module)
        logits = wo @ sample.hidden[module] + bo
        if sample.noise is not None:
            logits[:2] += sample.noise[module]
        outputs = 1 / (1 + np.exp(-logits))
        np.testing.assert_allclose(outputs, sample.actions[module], rtol=2e-6, atol=2e-7)
    actual = a["module_actions"][1, :count] if version >= 4 else a.get("actions", a["motors"])[1:2]
    np.testing.assert_array_equal(sample.actions, actual.numpy())
    if version >= 7:
        assert not np.array_equal(sample.plastic, a["module_plastic"][1, :count].numpy())
    # Samples own their arrays; inspecting an old frame cannot alias live state.
    copied = sample.hidden.copy()
    a["module_h" if version >= 4 else "h"].zero_()
    np.testing.assert_array_equal(sample.hidden, copied)


@pytest.mark.parametrize(
    "version,ablation",
    [
        (0, "shuffled"),
        (4, "pooled"),
        (15, "shuffled"),
        (15, "memory_reset"),
        (15, "no_plasticity"),
        (15, "no_exploration"),
    ],
)
def test_inspection_and_trails_preserve_complete_trajectory(config, version, ablation):
    from emergent_garden.inspector import Inspector
    from emergent_garden.leaderboard import Leaderboard
    from emergent_garden.viewer import Renderer

    w = create_world(replace(config, ecology_version=version, initial_food=8), ablation=ablation)
    plain = World.from_state(w.state_dict())
    observer = w.controller_observer = ControllerObserver()
    observer.select(0)
    renderer, panel = Renderer(256), Inspector()
    leaderboard = Leaderboard()
    renderer.selected = 0
    renderer.observe(w)
    for j in range(25):
        w.step()
        plain.step()
        renderer.observe(w)
        if j % 5 == 0:
            renderer.draw(w)
            leaderboard.draw(w, 0, 800)
            for tab in ("brain", "body"):
                panel.tab = tab
                panel.draw(w, 0, observer, 800)
    assert observer.sample is not None
    assert "controller_observer" not in w.state_dict()
    assert_same(w.state_dict(), plain.state_dict())


@pytest.mark.parametrize("version", [0, 15])
@pytest.mark.parametrize("controller", ["rest", "random", "forager"])
def test_scripted_controller_samples_do_not_claim_neural_activity(config, version, controller):
    w = create_world(replace(config, ecology_version=version), controller=controller)
    observer = w.controller_observer = ControllerObserver()
    observer.select(0)
    w.step()
    assert observer.sample.controller == controller
    assert observer.sample.noise is None
    expected = w.agents.get("actions", w.agents["motors"])[0]
    np.testing.assert_array_equal(observer.sample.actions[0], expected.numpy())


def test_selection_uses_permanent_id_and_retains_death_sample(config):
    w = World(config)
    observer = w.controller_observer = ControllerObserver()
    observer.select(1)
    w.update_controllers()
    # Compaction changes an array index, not the selected organism's identity.
    w.agents = {key: value[1:] for key, value in w.agents.items()}
    w.tick += w.config.physics_hz // w.config.controller_hz
    w.update_controllers()
    assert observer.sample.identifier == 1 and observer.sample.tick == w.tick
    last = observer.sample
    w.agents = {key: value[:0] for key, value in w.agents.items()}
    w.update_controllers()
    assert observer.sample is last
    observer.select(99)
    assert observer.sample is None


def test_trails_freeze_on_pause_fade_on_simulated_time_and_bound_history(config):
    w = World(config)
    trails = TrailHistory(seconds=1, hz=5)
    trails.observe(w)
    interval = round(config.physics_hz / 5)
    for _ in range(5):
        trails.observe(w)
    assert len(trails.tracks[0].points) == 1
    assert trails.opacity(0, 0.5, 1) == 0.5
    assert trails.opacity(0, 2, 1) == 0
    w.tick += interval
    w.agents["pos"][0, 0] += 1
    trails.observe(w)
    assert len(trails.tracks[0].points) == 2
    # A newborn occupying the same array slot gets its own trail.
    w.agents["id"][0] = 99
    w.tick += interval
    trails.observe(w)
    assert len(trails.tracks[0].points) == 2
    assert len(trails.tracks[99].points) == 1
    for _ in range(20):
        w.tick += interval
        trails.observe(w)
    assert 0 not in trails.tracks
    assert len(trails.tracks[99].points) <= 6
    # Loading another world or rewinding must not join unrelated histories.
    w.tick = 0
    trails.observe(w)
    assert len(trails.tracks[99].points) == 1
    other = World(config)
    trails.observe(other)
    assert 99 not in trails.tracks


def test_viewer_module_controls_resize_and_detach(config, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from emergent_garden.viewer import Viewer

    w = create_world(replace(config, ecology_version=12))
    w.agents["genome"][:, w.config.brain_parameter_count + 6] = 4
    w.develop(w.agents)
    viewer = Viewer(256)
    viewer.selected = 0
    try:
        viewer.observe(w)
        w.step()
        viewer.present(w)
        for key in (
            pygame.K_SPACE,
            pygame.K_n,
            pygame.K_t,
            pygame.K_RIGHTBRACKET,
            pygame.K_m,
            pygame.K_g,
        ):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
        viewer.events(w)
        assert viewer.paused
        assert viewer.step_ticks == w.config.physics_hz // w.config.controller_hz
        assert viewer.trail_mode == "selected" and viewer.trail_seconds == 120
        assert viewer.inspector.module == 1
        viewer.present(w)
        np.testing.assert_array_equal(viewer.center, w.agents["pos"][0].numpy())
        # A small window scales the whole panel; module clicks still line up.
        panel = viewer.inspector
        module_button = next(rect for rect, action in panel.buttons if action == ("module", 2))
        panel.click(
            (
                panel.offset_x + module_button.centerx * panel.scale,
                module_button.centery * panel.scale,
            )
        )
        assert panel.module == 2
        # Sidebar clicks cannot accidentally deselect a creature in the dish.
        pygame.event.post(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(viewer.size + 40, 60))
        )
        viewer.events(w)
        assert viewer.selected == 0
        viewer.resize(1300, 700)
        for tab in ("brain", "body", "details"):
            panel.tab = tab
            viewer.present(w)
            assert panel.content_height <= panel.layout_height
            assert panel.surface.get_height() == viewer.size
    finally:
        viewer.close()
    assert not hasattr(w, "controller_observer")
