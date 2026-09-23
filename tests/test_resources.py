import math
from dataclasses import replace

import pytest
import torch

from emergent_garden.storage import load_checkpoint, save_checkpoint
from emergent_garden.world import create_world


def dynamic(config, **changes):
    return replace(
        config,
        **{
            "ecology_version": 16,
            "patches": 2,
            "patch_fraction": 1.0,
            "patch_aspect_ratio": 2.0,
            "patch_irregularity": 0.35,
            "patch_drift_speed": 0.5,
            "fertility_grid_size": 16,
            "fertility_capacity_per_area": 1.0,
            **changes,
        },
    ).validate()


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


def test_patch_warp_preserves_area_and_has_nonconstant_radius(config):
    w = create_world(dynamic(config))
    boundary = w.resources.boundaries(w.patch_positions, samples=2048).double()
    p = boundary - w.patch_positions[:, None]
    q = p.roll(-1, dims=1)
    area = (p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]).sum(1).abs() / 2
    expected = math.pi * w.config.patch_radius**2
    torch.testing.assert_close(area, torch.full_like(area, expected), rtol=1e-5, atol=0)
    assert (p.norm(dim=-1).std(dim=1) > 1).all()
    assert p.norm(dim=-1).max() <= w.config.patch_extent


def test_drifting_sources_stay_inside_without_carrying_food_or_shelter(config):
    w = create_world(dynamic(config, initial_food=40, patch_drift_speed=10.0))
    food, cover = w.food_pos.clone(), w.fields[7].grid.clone()
    shelter = w.shelter_positions.clone()
    old_identity = w.fields[5].grid.clone()
    limit = w.config.diameter / 2 - w.config.patch_extent - w.config.food_radius
    w.patch_positions[:] = torch.tensor([w.config.diameter / 2 + limit - 0.01, 64])
    w.resources.headings.zero_()  # Exercise outward motion at the wall.
    for tick in range(6, 1201, 6):
        w.resources.advance(tick, w.patch_positions, w.rng["resources"])
        boundary = w.resources.boundaries(w.patch_positions)
        assert (boundary - 64).norm(dim=-1).max() <= 63.0001
    assert w.resources.distance > 100
    torch.testing.assert_close(w.food_pos, food, rtol=0, atol=0)
    torch.testing.assert_close(w.shelter_positions, shelter, rtol=0, atol=0)
    w.rebuild_fields()
    torch.testing.assert_close(w.fields[7].grid, cover, rtol=0, atol=0)
    assert not torch.equal(w.fields[5].grid, old_identity)


def test_local_fertility_funds_duplicate_proposals_without_overdraw(config):
    w = create_world(dynamic(config))
    r = w.resources
    positions = torch.tensor([[64, 64]] * 5 + [[72, 64]] * 2, dtype=torch.float32)
    keep = r.fund(positions, torch.tensor([False, True, True, True, True, True, True]))
    assert keep.tolist() == [False, True, True, True, False, True, True]
    assert r.fertility[8, 8] == 4
    assert r.fertility[8, 9] == 24
    assert r.fertility[7, 7] == 64
    assert r.spent == 100
    assert r.capacity_rejected == r.fertility_rejected == 1
    assert r.fertility.min() == 0
    assert r.metrics(w.patch_positions)["fertility_balance_error"] == 0


def test_fertility_recovers_with_elapsed_time_and_preserves_its_budget(config):
    w = create_world(dynamic(config, patch_drift_speed=0))
    r = w.resources
    pos = torch.tensor([[64.0, 64.0]] * 4)
    assert r.fund(pos, torch.ones(4, dtype=torch.bool)).sum() == 3
    depleted = r.fertility.clone()
    r.advance(0, w.patch_positions, w.rng["resources"])
    torch.testing.assert_close(r.fertility, depleted, rtol=0, atol=0)
    tick = round(w.config.fertility_recovery_time * w.config.physics_hz)
    r.advance(tick, w.patch_positions, w.rng["resources"])
    assert r.fertility[8, 8].item() == pytest.approx(64 - 60 / math.e)
    assert r.metrics(w.patch_positions)["fertility_balance_error"] == pytest.approx(0, abs=1e-10)
    assert r.fund(pos[:1], torch.ones(1, dtype=torch.bool)).item()
    assert r.fertility.min() >= 0 and r.fertility.max() <= r.capacity


def test_patch_capacity_rejection_cannot_spend_fertility(config):
    w = create_world(
        dynamic(
            config,
            patch_capacity=40.0,
            fertility_capacity_per_area=100.0,
            initial_food=0,
            resource_floor=1.0,
        )
    )
    w.spawn_food(100)
    assert w.resources.capacity_rejected > 0
    assert w.resources.spent == w.totals["food_spawned"] == 80
    before = w.resources.fertility.clone()
    w.spawn_food(100)
    torch.testing.assert_close(w.resources.fertility, before, rtol=0, atol=0)
    assert (w.patch_stock() <= 40).all()
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4


def test_fertility_rejection_creates_no_food_or_energy(config):
    w = create_world(dynamic(config, initial_food=0))
    # Remove the remaining production reserve as an explicit test intervention.
    w.resources.spent += w.resources.fertility.sum().item()
    w.resources.fertility.zero_()
    w.spawn_food(100)
    assert len(w.food_pos) == 0 and w.totals["food_spawned"] == 0
    assert w.resources.fertility_rejected == 100
    assert w.metrics()["fertility_balance_error"] == 0
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4


def test_resource_replay_and_rendering_preserve_complete_state(config, tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from emergent_garden.inspector import Inspector
    from emergent_garden.observation import ControllerObserver
    from emergent_garden.viewer import Renderer

    w = create_world(dynamic(config, initial_food=80, food_rate=10.0), ablation="shuffled")
    w.step(13)  # Save between resource/controller updates.
    path = tmp_path / "world.pt"
    save_checkpoint(w, path)
    replay = load_checkpoint(path)
    observer = w.controller_observer = ControllerObserver()
    observer.select(0)
    renderer, inspector = Renderer(256), Inspector()
    for tick in range(120):
        w.step()
        replay.step()
        if tick % 15 == 0:
            renderer.field_index = tick // 15 + 1
            renderer.draw(w)
            inspector.draw(w, 0, observer, 640)
    assert_same(w.state_dict(), replay.state_dict())
    assert w.resources.fertility_rejected > 0
    assert w.resources.distance > 0 and w.resources.recovered > 0
    assert abs(w.metrics()["energy_balance_error"]) < 0.05
    assert abs(w.metrics()["fertility_balance_error"]) < 1e-8


def test_disabled_new_mechanisms_match_v15_trajectory(config):
    old = create_world(replace(config, ecology_version=15, initial_food=40, food_rate=5))
    new = create_world(replace(old.config, ecology_version=16))
    assert old.config.input_names == new.config.input_names
    assert old.config.parameter_count == new.config.parameter_count
    old.step(180)
    new.step(180)
    before, after = old.state_dict(), new.state_dict()
    before.pop("config")
    after.pop("config")
    after.pop("resources")
    after["rng"].pop("resources")
    assert_same(before, after)


def test_fertility_is_not_an_extra_sensory_input(config):
    w = create_world(dynamic(config, initial_food=20))
    before = w.module_sensors(torch.arange(w.population))
    w.resources.fertility.zero_()
    after = w.module_sensors(torch.arange(w.population))
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert before.shape[-1] == 40


def test_viewer_cycles_fertility_and_toggles_sources_without_stepping(config, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from emergent_garden.viewer import Viewer

    w = create_world(dynamic(config))
    before = w.state_dict()
    viewer = Viewer(256)
    try:
        viewer.field_index = 7
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
        viewer.events(w)
        assert viewer.field_index == 8 and not viewer.show_sources
        viewer.present(w)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
        viewer.events(w)
        assert viewer.field_index == 0
        assert_same(before, w.state_dict())
    finally:
        viewer.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"patch_aspect_ratio": 0.5},
        {"patch_irregularity": 1.1},
        {"patch_drift_turn_time": 0},
        {"patch_drift_speed": -1},
        {"patch_radius": 50},
        {"fertility_grid_size": 4},
        {"fertility_capacity_per_area": 0.01},
        {"fertility_capacity_per_area": 20 / 64},
        {"fertility_recovery_time": 0},
        {"ecology_version": 15},
    ],
)
def test_invalid_resource_settings_are_rejected(config, changes):
    with pytest.raises(ValueError):
        dynamic(config, **changes)
