from dataclasses import replace

import numpy as np
import pytest
import torch

from emergent_garden.world import World, create_world


def panel_point(panel, action):
    rect = next(rect for rect, value in panel.buttons if value == action)
    return (
        panel.offset_x + rect.centerx * panel.scale,
        rect.centery * panel.scale,
    )


@pytest.mark.parametrize("version", [0, 16])
@pytest.mark.parametrize(
    "key", ["age", "generation", "acquired", "offspring", "energy", "distance"]
)
def test_headers_sort_numeric_lifetime_totals_both_ways(config, version, key):
    import pygame

    from emergent_garden.leaderboard import Leaderboard

    pygame.font.init()
    w = create_world(replace(config, ecology_version=version, initial_population=4))
    w.agents["id"][:] = torch.tensor([40, 10, 30, 20])
    w.agents[key][:] = torch.tensor([2, 10, 10, 1])
    # A resumed world's full lifetime counters are visible immediately.
    w = World.from_state(w.state_dict())
    panel = Leaderboard()
    panel.sort_key = "id"
    panel.draw(w, None, 640)
    panel.click(panel_point(panel, ("sort", key)))
    panel.draw(w, None, 640)
    assert [row["id"] for row in panel.rows] == [10, 30, 40, 20]
    assert panel.rows[0][key] == 10
    panel.click(panel_point(panel, ("sort", key)))
    panel.draw(w, None, 640)
    assert [row["id"] for row in panel.rows] == [20, 40, 10, 30]
    panel.click(panel_point(panel, ("sort", "id")))
    panel.draw(w, None, 640)
    assert [row["id"] for row in panel.rows] == [10, 20, 30, 40]


@pytest.mark.parametrize("height", [256, 640, 984])
def test_row_selection_survives_compaction_and_ignores_dead_rows(config, monkeypatch, height):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from emergent_garden.viewer import Viewer

    w = create_world(replace(config, ecology_version=16, initial_population=3))
    w.agents["age"][:] = torch.tensor([5, 20, 10])
    viewer = Viewer(256)

    def click(panel, action):
        x, y = panel_point(panel, action)
        pygame.event.post(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(viewer.size + x, y))
        )
        viewer.events(w)

    try:
        viewer.resize(height + viewer.inspector.width, height)
        viewer.present(w)
        click(viewer.inspector, ("view", "leaderboard"))
        assert viewer.sidebar_view == "leaderboard"
        viewer.present(w)
        assert viewer.leaderboard.content_height <= viewer.leaderboard.layout_height
        # ID 2 moves from array index 2 to index 1 after this displayed frame.
        keep = w.agents["id"] != 1
        w.agents = {key: value[keep] for key, value in w.agents.items()}
        viewer.inspector.module = 2
        click(viewer.leaderboard, ("select", 2))
        assert viewer.selected == viewer.observer.selected == 2
        assert viewer.inspector.module == 0 and viewer.sidebar_view == "inspector"
        np.testing.assert_array_equal(viewer.center, w.agents["pos"][1].numpy())
        w.step()
        viewer.present(w)
        assert viewer.observer.sample.identifier == 2

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l))
        viewer.events(w)
        viewer.present(w)
        # A displayed row can die before the event is handled. Its click must
        # not select whichever creature now occupies the old array index.
        keep = w.agents["id"] != 0
        w.agents = {key: value[keep] for key, value in w.agents.items()}
        click(viewer.leaderboard, ("select", 0))
        assert viewer.selected == 2 and viewer.sidebar_view == "leaderboard"
        viewer.present(w)
        assert [row["id"] for row in viewer.leaderboard.rows] == [2]
        w.agents = {key: value[:0] for key, value in w.agents.items()}
        viewer.present(w)
        assert viewer.leaderboard.rows == [] and viewer.leaderboard.page == 0
        assert not any(action[0] == "select" for _, action in viewer.leaderboard.buttons)
    finally:
        viewer.close()


def test_pagination_sidebar_toggle_and_wheel_routing(config, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from emergent_garden.viewer import Viewer

    w = World(replace(config, initial_population=30, capacity=32))
    viewer = Viewer(256)

    def keypress(key):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
        viewer.events(w)

    try:
        viewer.resize(1260, 640)
        keypress(pygame.K_l)
        panel = viewer.leaderboard
        seen = []
        for page in range(3):
            viewer.present(w)
            assert panel.page == page
            seen.extend(value for _, (kind, value) in panel.buttons if kind == "select")
            keypress(pygame.K_PAGEDOWN)
        assert seen == list(range(30))  # All tied lifetimes resolve by permanent ID.
        assert panel.page == panel.page_count - 1
        keypress(pygame.K_PAGEUP)
        assert panel.page == 1
        zoom = viewer.zoom
        monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (viewer.size + 40, 200))
        pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, y=1))
        viewer.events(w)
        assert panel.page == 0 and viewer.zoom == zoom
        monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (40, 200))
        pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, y=1))
        viewer.events(w)
        assert panel.page == 0 and viewer.zoom > zoom
        keypress(pygame.K_PAGEDOWN)
        panel.click(panel_point(panel, ("sort", "energy")))
        assert panel.page == 0
        keypress(pygame.K_b)
        assert not viewer.show_brain
        keypress(pygame.K_l)
        assert viewer.show_brain and viewer.sidebar_view == "leaderboard"
        keypress(pygame.K_l)
        assert viewer.sidebar_view == "inspector"
        viewer.show_sidebar("leaderboard")
        viewer.zoom, viewer.center = 1, None
        pos, _ = viewer.transform(w.agents["pos"][0].numpy(), w.config.diameter)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
        viewer.events(w)
        assert viewer.selected == 0 and viewer.sidebar_view == "inspector"
        viewer.show_sidebar("leaderboard")
        panel.page = 2
        w.agents = {key: value[:1] for key, value in w.agents.items()}
        viewer.present(w)
        assert panel.page == 0 and panel.page_count == 1
    finally:
        viewer.close()
