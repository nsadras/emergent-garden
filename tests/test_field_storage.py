import os
from dataclasses import asdict, replace

import pytest
import torch

from emergent_garden.config import Config
from emergent_garden.field import SmellField
from emergent_garden.storage import load_checkpoint, save_checkpoint
from emergent_garden.world import World


def test_field_symmetry_cutoff_and_removal(config):
    field = SmellField(config, torch.device("cpu"))
    field.rebuild(torch.tensor([[64.0, 64.0]]), torch.tensor([20.0]))
    samples = field.sample(torch.tensor([[60.0, 64.0], [68.0, 64.0], [64.0, 60.0], [64.0, 68.0]]))
    torch.testing.assert_close(samples, samples[0].expand(4))
    assert field.sample(torch.tensor([[64.0, 64.0]])).item() > samples[0]
    assert field.sample(torch.tensor([[120.0, 64.0]])).item() < 1e-6
    field.rebuild(torch.empty((0, 2)), torch.empty(0))
    assert field.grid.count_nonzero() == 0


def test_grid_refinement_preserves_field_strength(config):
    values = []
    for size in (64, 128):
        field = SmellField(replace(config, grid_size=size), torch.device("cpu"))
        field.rebuild(torch.tensor([[64.0, 64.0]]), torch.tensor([20.0]))
        values.append(field.sample(torch.tensor([[68.0, 64.0]])).item())
    assert values[0] == pytest.approx(values[1], rel=0.04)


def test_checkpoint_preserves_trajectory_and_random_streams(config, tmp_path):
    w = World(replace(config, initial_food=40, food_rate=10, patch_period=2))
    w.agents["energy"][:] = 240
    w.step(13)  # Deliberately save between field and controller updates.
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(w, path)
    resumed = load_checkpoint(path)
    w.step(180)
    resumed.step(180)
    assert w.tick == resumed.tick
    assert w.events == resumed.events
    assert w.totals == resumed.totals
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], resumed.agents[key], rtol=0, atol=0)
    for key in ("food_pos", "food_energy", "food_expiry", "patch_positions"):
        torch.testing.assert_close(getattr(w, key), getattr(resumed, key), rtol=0, atol=0)
    torch.testing.assert_close(w.field.grid, resumed.field.grid, rtol=0, atol=0)
    for key in w.rng:
        assert torch.equal(w.rng[key].get_state(), resumed.rng[key].get_state())


def test_renderer_and_recording_do_not_change_trajectory(config, tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from emergent_garden.viewer import Recorder, Renderer

    w = World(replace(config, initial_food=20, food_rate=3))
    plain = World.from_state(w.state_dict())
    renderer = Renderer(256)
    recorder = Recorder(tmp_path / "test.mp4", 256, fps=10, speed=1)
    for _ in range(3):
        w.step(6)
        renderer.draw(w)
        recorder.observe(w)
        plain.step(6)
    recorder.close()
    assert recorder.frames == 3
    assert (tmp_path / "test.mp4").stat().st_size > 500
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], plain.agents[key], rtol=0, atol=0)
    for key in w.rng:
        assert torch.equal(w.rng[key].get_state(), plain.rng[key].get_state())


def test_sensor_ablations_only_remove_or_relocate_smell(config):
    w = World(replace(config, initial_food=30))
    index = torch.arange(w.population)
    normal = w.sensors(index)
    w.ablation = "disabled"
    disabled = w.sensors(index)
    assert disabled[:, :4].count_nonzero() == 0
    torch.testing.assert_close(disabled[:, 4:], normal[:, 4:])
    w.ablation = "shuffled"
    shuffled = w.sensors(index)
    torch.testing.assert_close(shuffled[:, 4:], normal[:, 4:])
    assert not torch.equal(shuffled[:, :4], normal[:, :4])


def test_default_file_matches_code_defaults():
    assert asdict(Config.load("configs/v0.toml")) == asdict(Config())


@pytest.mark.parametrize(
    "overrides",
    [
        dict(controller_hz=19),
        dict(capacity=1),
        dict(birth_energy=121),
        dict(smell_sigma=0),
        dict(initial_population=2.5),
        dict(food_rate=float("nan")),
    ],
)
def test_invalid_configuration_is_rejected(config, overrides):
    with pytest.raises(ValueError):
        replace(config, **overrides).validate()
