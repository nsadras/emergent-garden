import json
import os
import signal
import subprocess
import sys
import time

import torch

from emergent_garden.config import Config
from emergent_garden.storage import load_checkpoint
from emergent_garden.world import World


def test_graceful_termination_checkpoint_is_a_complete_tick(config, tmp_path):
    config_path = tmp_path / "config.toml"
    config.save(config_path)
    output = tmp_path / "run"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "emergent_garden",
            "run",
            "--device",
            "cpu",
            "--config",
            str(config_path),
            "--seconds",
            "0",
            "--output",
            str(output),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"},
    )
    try:
        deadline = time.monotonic() + 15
        metrics = output / "metrics.jsonl"
        while time.monotonic() < deadline:
            if metrics.exists() and len(metrics.read_text().splitlines()) > 1:
                break
            if process.poll() is not None:
                raise AssertionError(process.stderr.read())
            time.sleep(0.02)
        else:
            raise AssertionError("Run did not start")
        process.send_signal(signal.SIGTERM)
        _, error = process.communicate(timeout=15)
        assert process.returncode == 0, error
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert json.loads((output / "summary.json").read_text())["stop_reason"] == "interrupted"
    resumed = load_checkpoint(output / "latest.pt")
    expected = World(config)
    expected.step(resumed.tick)
    for key in expected.agents:
        torch.testing.assert_close(expected.agents[key], resumed.agents[key], rtol=0, atol=0)


def test_viewer_controls_and_selection_do_not_change_world(config):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    from emergent_garden.viewer import Viewer

    w = World(config)
    viewer = Viewer(256)
    before = w.agents["pos"].clone()
    try:
        pos, _ = viewer.transform(w.agents["pos"][0].numpy(), config.diameter)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, y=3))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        assert viewer.events(w)
        assert viewer.selected == 0
        assert viewer.paused
        assert viewer.zoom > 1
        assert not viewer.show_field
        viewer.center = w.agents["pos"][0].numpy()
        viewer.zoom = 4
        viewer.present(w)
        torch.testing.assert_close(w.agents["pos"], before, rtol=0, atol=0)
    finally:
        viewer.close()


def test_unknown_config_key_has_a_clear_error():
    import pytest

    with pytest.raises(ValueError, match="food_rtae"):
        Config.from_dict({"food_rtae": 20})
