"""Pygame observer and streaming MP4 recorder; neither changes the world state."""

import colorsys
import math
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio_ffmpeg
import numpy as np
import pygame

from .inspector import Inspector
from .observation import ControllerObserver, TrailHistory, array


def lineage_color(identifier):
    return tuple(int(v * 255) for v in colorsys.hsv_to_rgb((identifier * 0.618034) % 1, 0.55, 0.96))


class Renderer:
    def __init__(self, size=1024):
        pygame.font.init()
        self.size = size
        self.surface = pygame.Surface((size, size))
        self.font = pygame.font.Font(None, 23)
        self.small = pygame.font.Font(None, 18)
        self.title = pygame.font.Font(None, 32)
        self.zoom = 1.0
        self.center = None
        self.selected = None
        self.selected_module = 0
        self.show_field = True
        self.show_brain = False
        self.field_index = 0
        self.color_mode = "diet"
        self.trails = TrailHistory()
        self.trail_mode = "all"
        self.trail_seconds = 30
        self.trail_layer = pygame.Surface((size, size), pygame.SRCALPHA)

    def observe(self, world):
        self.trails.observe(world)

    def draw_trails(self, world):
        if self.trail_mode == "off":
            return
        self.trail_layer.fill((0, 0, 0, 0))
        for identifier, track in self.trails.tracks.items():
            selected = identifier == self.selected
            if self.trail_mode == "selected" and not selected:
                continue
            points = np.asarray(track.points)
            if len(points) < 2:
                continue
            points = points[points[:, 0] >= world.time - self.trail_seconds]
            if len(points) < 2:
                continue
            positions, _ = self.transform(points[:, 1:], world.config.diameter)
            color = lineage_color(track.lineage)
            if track.diet is not None and self.color_mode == "diet":
                d = track.diet
                color = (int(235 - 130 * d), int(156 + 76 * d), int(89 + 46 * d))
            if selected:
                color = (251, 240, 192)
            # Batch adjacent segments into twelve fade levels instead of one
            # draw call per physics sample. Gaps are never bridged.
            levels = np.ceil(
                self.trails.opacity(points[1:, 0], world.time, self.trail_seconds) * 12
            )
            valid = np.diff(points[:, 0]) <= self.trails.period * 1.5
            breaks = np.flatnonzero((np.diff(levels) != 0) | (np.diff(valid) != 0)) + 1
            for group in np.split(np.arange(len(levels)), breaks):
                start, end = group[0], group[-1] + 1
                if not valid[start]:
                    continue
                alpha = round(float(levels[start]) / 12 * (220 if selected else 125))
                pygame.draw.lines(
                    self.trail_layer,
                    (*color, alpha),
                    False,
                    positions[start : end + 1],
                    2 if selected else 1,
                )
        self.surface.blit(self.trail_layer, (0, 0))

    def transform(self, points, diameter):
        points = np.asarray(points, dtype=np.float64)
        center = self.center if self.center is not None else np.array([diameter / 2] * 2)
        scale = (self.size - 70) / diameter * self.zoom
        return (points - center) * scale + self.size / 2, scale

    def text(self, value, point, color=(210, 229, 230), small=False):
        self.surface.blit((self.small if small else self.font).render(value, True, color), point)

    def draw(self, world, status=""):
        self.observe(world)
        c, a = world.config, world.agents
        surface = self.surface
        surface.fill((8, 17, 23))
        dish_center, scale = self.transform(np.array([c.diameter / 2] * 2), c.diameter)
        radius = int(c.diameter / 2 * scale)
        pygame.draw.circle(surface, (13, 31, 37), dish_center.astype(int), radius)
        if self.show_field:
            fields = getattr(world, "fields", [world.field])
            field = fields[self.field_index % len(fields)].grid.detach().cpu().numpy()
            norm = c.signal_scale if self.field_index == 4 else c.smell_scale
            strength = (field if self.field_index == 7 else field / (field + norm)) * 100
            tints = (
                (0.2, 0.5, 0.25),
                (0.8, 0.4, 0.05),
                (0.3, 0.5, 0.8),
                (0.7, 0.25, 0.7),
                (0.15, 0.8, 0.9),
                (0.9, 0.25, 0.3),
                (0.3, 0.35, 0.95),
                (0.3, 0.65, 0.65),
            )
            rgb = (
                np.array([13, 31, 37]) + strength[..., None] * np.array(tints[self.field_index])
            ).astype(np.uint8)
            rgb[~world.field.mask.cpu().numpy()] = (8, 17, 23)
            layer = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
            side = max(1, int(c.diameter * scale))
            # Rendering zooms above 2x uses a crop to keep temporary surfaces bounded.
            n = c.grid_size
            top_left, _ = self.transform(np.array([0.0, 0.0]), c.diameter)
            x0 = max(0, int(-top_left[0] / side * n))
            y0 = max(0, int(-top_left[1] / side * n))
            x1 = min(n, math.ceil((self.size - top_left[0]) / side * n))
            y1 = min(n, math.ceil((self.size - top_left[1]) / side * n))
            if x1 > x0 and y1 > y0:
                cropped = layer.subsurface((x0, y0, x1 - x0, y1 - y0))
                resized = pygame.transform.smoothscale(
                    cropped,
                    (max(1, round((x1 - x0) * side / n)), max(1, round((y1 - y0) * side / n))),
                )
                surface.blit(resized, top_left + np.array([x0, y0]) * side / n)
        pygame.draw.circle(surface, (74, 133, 137), dish_center.astype(int), radius, 2)
        self.draw_trails(world)
        if c.ecology_version >= 10:
            centers, _ = self.transform(
                world.patch_positions[world.shelter_indices].cpu().numpy(), c.diameter
            )
            color = (65, 125, 126) if world.ablation != "no_shelter" else (63, 71, 77)
            for pos in centers:
                pygame.draw.circle(
                    surface, color, pos.astype(int), round(c.shelter_radius * scale), 1
                )
        food, _ = self.transform(world.food_pos.detach().cpu().numpy(), c.diameter)
        food_radius = max(1, round(c.food_radius * scale))
        kinds = (
            world.food_kind.cpu().numpy() if hasattr(world, "food_kind") else np.zeros(len(food))
        )
        if c.ecology_version >= 11:
            # Partial meals leave many overlapping crumbs. Display their summed
            # local energy instead of drawing each one as a full food particle.
            keep = (food >= -food_radius).all(1) & (food < self.size + food_radius).all(1)
            pixels, inverse = np.unique(food[keep].astype(int), axis=0, return_inverse=True)
            amounts = world.food_energy.detach().cpu().numpy()[keep]
            fresh = np.bincount(inverse, weights=amounts * (kinds[keep] == 0))
            detritus = np.bincount(inverse, weights=amounts * (kinds[keep] == 1))
            total = fresh + detritus
            strength = np.sqrt(np.minimum(total / c.food_energy, 1))
            colors = (
                fresh[:, None] * np.array((131, 192, 104))
                + detritus[:, None] * np.array((221, 158, 83))
            ) / total.clip(1e-300)[:, None]
            colors *= (0.25 + 0.75 * strength)[:, None]
            for pos, color, fraction in zip(pixels, colors, strength, strict=True):
                pygame.draw.circle(
                    surface, tuple(color.astype(int)), pos, max(1, round(food_radius * fraction))
                )
        else:
            for pos, kind in zip(food, kinds, strict=True):
                if (pos >= -food_radius).all() and (pos < self.size + food_radius).all():
                    color = (131, 192, 104) if kind == 0 else (221, 158, 83)
                    pygame.draw.circle(surface, color, pos.astype(int), food_radius)
        visible = (
            "pos",
            "id",
            "lineage",
            "heading",
            "motors",
            "generation",
            "energy",
            "age",
            "offspring",
            "h",
        )
        data = {key: a[key].detach().cpu().numpy() for key in visible}
        for key in (
            "radius",
            "power",
            "diet",
            "attack",
            "armor",
            "actions",
            "neurons",
            "connections",
            "target_modules",
            "module_internal",
        ):
            if key in a:
                data[key] = a[key].detach().cpu().numpy()
        if "modules" in a:
            from .morphology import module_centers

            for key in ("core_radius", "modules", "module_actions"):
                data[key] = a[key].detach().cpu().numpy()
            data["module_positions"], _ = self.transform(
                module_centers(a).cpu().numpy(), c.diameter
            )
        positions, _ = self.transform(data["pos"], c.diameter)
        r = max(2, round(c.body_radius * scale))
        for i, pos in enumerate(positions):
            if "radius" in data:
                r = max(2, round(float(data["radius"][i]) * scale))
            if (pos < -r).any() or (pos > self.size + r).any():
                continue
            color = lineage_color(int(data["lineage"][i]))
            if "diet" in data and self.color_mode == "diet":
                d = float(data["diet"][i])
                color = (int(235 - 130 * d), int(156 + 76 * d), int(89 + 46 * d))
            if "modules" in data:
                pygame.draw.circle(surface, tuple(int(v * 0.2) for v in color), pos.astype(int), r)
                pygame.draw.circle(
                    surface, tuple(int(v * 0.55) for v in color), pos.astype(int), r, 1
                )
                core_r = max(2, round(float(data["core_radius"][i]) * scale))
                for part in data["module_positions"][i, : data["modules"][i]]:
                    pygame.draw.circle(surface, color, part.astype(int), core_r)
                if "module_internal" in data and self.zoom >= 2:
                    for k in range(int(data["modules"][i]) - 1):
                        strength = np.abs(data["module_internal"][i, k : k + 2]).mean()
                        tint = (40, int(85 + 135 * strength), int(105 + 135 * strength))
                        pygame.draw.line(
                            surface,
                            tint,
                            data["module_positions"][i, k],
                            data["module_positions"][i, k + 1],
                            max(1, core_r // 8),
                        )
            else:
                pygame.draw.circle(surface, color, pos.astype(int), r)
            angle = data["heading"][i]
            forward = np.array([math.cos(angle), math.sin(angle)])
            pygame.draw.line(surface, (14, 32, 36), pos, pos + forward * r, max(1, r // 5))
            if "attack" in data and data["attack"][i] * data["actions"][i, 2] > 0.2:
                pygame.draw.line(
                    surface,
                    (243, 100, 107),
                    pos + forward * r * 0.6,
                    pos + forward * (r + 3),
                    max(1, r // 4),
                )
            if "armor" in data and data["armor"][i] > 0.65 and self.zoom >= 2:
                pygame.draw.circle(surface, (167, 191, 210), pos.astype(int), r, 2)
            if self.zoom >= 2:
                components = [(pos, r, data["motors"][i])]
                if "modules" in data:
                    components = [
                        (data["module_positions"][i, k], core_r, data["module_actions"][i, k, :2])
                        for k in range(data["modules"][i])
                    ]
                for center, part_r, motors in components:
                    for sensor_angle in (-135, -45, 45, 135):
                        theta = angle + math.radians(sensor_angle)
                        sensor = center + part_r * np.array([math.cos(theta), math.sin(theta)])
                        pygame.draw.circle(surface, (213, 254, 192), sensor.astype(int), 2)
                    for side, activation in zip((1, -1), motors, strict=True):
                        lateral = np.array([-forward[1], forward[0]]) * side
                        base = center + lateral * part_r * 0.6 - forward * part_r * 0.4
                        pygame.draw.line(
                            surface,
                            (252, 186, 106),
                            base,
                            base - forward * (3 + activation * part_r),
                            2,
                        )
            if int(data["id"][i]) == self.selected:
                pygame.draw.circle(surface, (251, 240, 192), pos.astype(int), r + 5, 2)
                if "module_positions" in data:
                    k = min(self.selected_module, int(data["modules"][i]) - 1)
                    part = data["module_positions"][i, k]
                    pygame.draw.circle(surface, (255, 255, 240), part, core_r + 2, 1)
                    self.text(str(k + 1), part + (core_r + 4, -core_r), small=True)
        header = pygame.Surface((self.size, 74), pygame.SRCALPHA)
        header.fill((8, 17, 23, 232))
        surface.blit(header, (0, 0))
        surface.blit(self.title.render("EMERGENT GARDEN", True, (205, 239, 221)), (22, 14))
        if c.ecology_version and self.size >= 640:
            names = (
                "fresh food",
                "detritus",
                "organisms",
                "forecast",
                "secretions",
                "identity A",
                "identity B",
                "shelter",
            )
            self.text(
                f"V{c.ecology_version} | {names[self.field_index]} field | {self.color_mode}",
                (self.size - 320, 22),
                small=True,
            )
        generation = int(data["generation"].max()) if world.population else 0
        self.text(
            f"{world.time:,.1f}s   |   {world.population} creatures   |   "
            f"{len(food)} food   |   generation {generation}",
            (22, 47),
            small=True,
        )
        panel = pygame.Surface((self.size, 62), pygame.SRCALPHA)
        panel.fill((8, 17, 23, 232))
        surface.blit(panel, (0, self.size - 62))
        description = "Particle scents / inherited recurrent brains / continuous life"
        if c.ecology_version >= 6:
            favorable = "A" if world.landscape.favorable == 0 else "B"
            description = (
                f"High-quality patches: {favorable} | Tab: field | C: body colors | B: brain"
            )
        self.text(
            status or description,
            (20, self.size - 51),
            small=True,
        )
        self.text(
            f"Trails: {self.trail_mode} / {self.trail_seconds}s   |   T mode   [ / ] duration"
            "   |   B inspector   G follow   M module",
            (20, self.size - 28),
            small=True,
        )
        return surface

    def save(self, world, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.draw(world), str(path))


class Viewer(Renderer):
    def __init__(self, size=1024):
        pygame.display.init()
        desktop = pygame.display.get_desktop_sizes()[0]
        # Fit the initial window to the desktop, including the inspector.
        size = min(size, max(256, desktop[1] - 96), max(256, desktop[0] - Inspector.width - 48))
        super().__init__(size)
        self.inspector = Inspector()
        self.observer = ControllerObserver()
        self.observed_world = None
        self.show_brain = True
        self.window = pygame.display.set_mode((size + Inspector.width, size), pygame.RESIZABLE)
        pygame.display.set_caption("Emergent Garden / living controllers")
        self.paused = False
        self.speed = 1.0
        self.running = True
        self.drag = None
        self.following = False
        self.step_ticks = 0

    def observe(self, world):
        super().observe(world)
        if world is not self.observed_world:
            self.detach()
            self.observed_world = world
            world.controller_observer = self.observer
            self.observer.sample = None
        self.observer.select(self.selected)

    def detach(self):
        if self.observed_world is not None:
            if getattr(self.observed_world, "controller_observer", None) is self.observer:
                del self.observed_world.controller_observer
            self.observed_world = None

    def resize(self, width, height):
        sidebar = Inspector.width if self.show_brain else 0
        self.size = max(128, min(height, width - sidebar))
        self.surface = pygame.Surface((self.size, self.size))
        self.trail_layer = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.window = pygame.display.set_mode((self.size + sidebar, self.size), pygame.RESIZABLE)

    def cycle_module(self, world):
        match = (
            (world.agents["id"] == self.selected).nonzero().flatten()
            if self.selected is not None
            else []
        )
        if len(match) and "modules" in world.agents:
            self.inspector.module = (self.inspector.module + 1) % int(
                world.agents["modules"][match[0]]
            )

    def events(self, world):
        self.observe(world)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.resize(event.w, event.h)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    if not self.paused:
                        self.step_ticks = 0
                elif event.key == pygame.K_n and self.paused:
                    self.step_ticks += world.config.physics_hz // world.config.controller_hz
                elif event.key == pygame.K_b:
                    self.show_brain = not self.show_brain
                    self.resize(self.size + (Inspector.width if self.show_brain else 0), self.size)
                elif event.key == pygame.K_t:
                    modes = ("all", "selected", "off")
                    self.trail_mode = modes[(modes.index(self.trail_mode) + 1) % len(modes)]
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    durations = (10, 30, 120)
                    direction = 1 if event.key == pygame.K_RIGHTBRACKET else -1
                    self.trail_seconds = durations[
                        (durations.index(self.trail_seconds) + direction) % 3
                    ]
                elif event.key == pygame.K_g:
                    self.following = not self.following
                elif event.key == pygame.K_m:
                    self.cycle_module(world)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.speed = min(1024, self.speed * 2)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed = max(0.125, self.speed / 2)
                elif event.key == pygame.K_f:
                    self.show_field = not self.show_field
                elif event.key == pygame.K_TAB:
                    self.field_index = (self.field_index + 1) % len(
                        getattr(world, "fields", [world.field])
                    )
                elif event.key == pygame.K_c:
                    self.color_mode = "lineage" if self.color_mode == "diet" else "diet"
                elif event.key == pygame.K_r:
                    self.zoom, self.center, self.following = 1, None, False
            elif event.type == pygame.MOUSEWHEEL:
                if self.show_brain and pygame.mouse.get_pos()[0] >= self.size:
                    self.inspector.wheel(event.y, self.size)
                else:
                    self.zoom = min(8, max(1, self.zoom * 1.25**event.y))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.show_brain and event.pos[0] >= self.size:
                    if event.button == 1:
                        self.inspector.click((event.pos[0] - self.size, event.pos[1]))
                elif event.button == 1:
                    self.selected = None
                    if world.population:
                        positions, scale = self.transform(
                            world.agents["pos"].cpu().numpy(), world.config.diameter
                        )
                        distances = np.linalg.norm(positions - event.pos, axis=1)
                        radii = world.agents.get("radius")
                        thresholds = (
                            np.maximum(12, array(radii) * scale)
                            if radii is not None
                            else np.full(
                                world.population, max(12, world.config.body_radius * scale)
                            )
                        )
                        candidates = np.flatnonzero(distances <= thresholds)
                        if len(candidates):
                            index = candidates[np.argmin(distances[candidates])]
                            self.selected = int(world.agents["id"][index])
                    if self.selected != self.inspector.identifier:
                        self.inspector.module, self.inspector.scroll = 0, 0
                        self.inspector.identifier = self.selected
                    self.observer.select(self.selected)
                elif event.button == 3:
                    self.drag = event.pos
                    self.following = False
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                self.drag = None
            elif event.type == pygame.MOUSEMOTION and self.drag is not None:
                if self.center is None:
                    self.center = np.array([world.config.diameter / 2] * 2)
                scale = (self.size - 70) / world.config.diameter * self.zoom
                self.center -= np.array(event.rel) / scale
        return self.running

    def present(self, world):
        if self.following and self.selected is not None:
            match = (world.agents["id"] == self.selected).nonzero().flatten()
            if len(match):
                self.center = array(world.agents["pos"][match[0]])
        self.selected_module = self.inspector.module
        state = "PAUSED" if self.paused else f"{self.speed:g}x"
        status = f"{state} | Space pause | N step | +/- speed | Wheel zoom"
        if self.size >= 800:
            status += " | Right-drag pan | F field | Tab channel | C color | R reset"
        self.draw(world, status)
        self.window.blit(self.surface, (0, 0))
        if self.show_brain:
            panel = self.inspector.draw(
                world, self.selected, self.observer, self.size, self.following
            )
            self.window.blit(panel, (self.size, 0))
        pygame.display.flip()

    def close(self):
        self.detach()
        pygame.display.quit()


class Recorder:
    def __init__(self, path, size, fps=30, speed=100, start_time=0):
        self.renderer = Renderer(size)
        self.period = speed / fps
        self.next_time = start_time
        self.frames = 0
        self.path = Path(path)
        self.writer = imageio_ffmpeg.write_frames(
            str(path),
            (size, size),
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=2,
            ffmpeg_log_level="error",
        )
        self.writer.send(None)

    def observe(self, world):
        self.renderer.observe(world)
        if world.time + 1e-9 >= self.next_time:
            surface = self.renderer.draw(world)
            self.writer.send(pygame.image.tobytes(surface, "RGB"))
            self.frames += 1
            self.next_time += self.period

    def close(self):
        self.writer.close()
