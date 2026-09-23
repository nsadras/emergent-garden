"""Pygame observer and streaming MP4 recorder; neither changes the world state."""

import colorsys
import math
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio_ffmpeg
import numpy as np
import pygame


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
        self.show_field = True
        self.show_brain = False
        self.field_index = 0
        self.color_mode = "diet"

    def transform(self, points, diameter):
        points = np.asarray(points, dtype=np.float64)
        center = self.center if self.center is not None else np.array([diameter / 2] * 2)
        scale = (self.size - 70) / diameter * self.zoom
        return (points - center) * scale + self.size / 2, scale

    def text(self, value, point, color=(210, 229, 230), small=False):
        self.surface.blit((self.small if small else self.font).render(value, True, color), point)

    def draw(self, world, status=""):
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
        panel = pygame.Surface((self.size, 42), pygame.SRCALPHA)
        panel.fill((8, 17, 23, 232))
        surface.blit(panel, (0, self.size - 42))
        description = "Particle scents / inherited recurrent brains / continuous life"
        if c.ecology_version >= 6:
            favorable = "A" if world.landscape.favorable == 0 else "B"
            description = (
                f"High-quality patches: {favorable} | Tab: field | C: body colors | B: brain"
            )
        self.text(
            status or description,
            (20, self.size - 31),
            small=True,
        )
        chosen = np.flatnonzero(data["id"] == self.selected) if self.selected is not None else []
        if len(chosen):
            i = chosen[0]
            height = 270 if "modules" in data else 248 if "attack" in data else 222
            if "module_plastic" in a:
                height += 48
            if "neurons" in data:
                height += 22
            if c.ecology_version >= 11:
                height += 42
            if c.ecology_version >= 12:
                height += 42
            if self.show_brain:
                height += 150
                if c.ecology_version >= 12:
                    height += 74
            panel = pygame.Surface((285, height), pygame.SRCALPHA)
            panel.fill((6, 14, 21, 230))
            surface.blit(panel, (18, 92))
            self.text(f"Creature {self.selected} / lineage {data['lineage'][i]}", (30, 103))
            self.text(f"Energy {data['energy'][i]:.1f}   Age {data['age'][i]:.1f}s", (30, 133))
            self.text(
                f"Generation {data['generation'][i]}   Children {data['offspring'][i]}", (30, 159)
            )
            self.text("Neural activity", (30, 189), small=True)
            for j, activity in enumerate(data["h"][i]):
                color = (100, 200, 160) if activity >= 0 else (120, 133, 230)
                x = 30 + j * min(13, 224 / len(data["h"][i]))
                pygame.draw.line(surface, color, (x, 239), (x, 239 - float(activity) * 25), 6)
            if "diet" in data:
                self.text(
                    f"Radius {data['radius'][i]:.1f}  Power {data['power'][i]:.2f}",
                    (30, 264),
                    small=True,
                )
                self.text(f"Fresh-food allocation {data['diet'][i]:.0%}", (30, 285), small=True)
            if "attack" in data:
                self.text(
                    f"Weapon {data['attack'][i]:.0%}  Armor {data['armor'][i]:.0%}",
                    (30, 306),
                    small=True,
                )
            if "modules" in data:
                neurons = data["neurons"][i] if "neurons" in data else len(data["h"][i])
                self.text(
                    f"{data['modules'][i]} modules / {neurons} neurons each",
                    (30, 327),
                    small=True,
                )
            if "module_plastic" in a:
                count = int(data["modules"][i])
                connections = (
                    int(a["recurrent_connections"][i])
                    if "recurrent_connections" in a
                    else c.hidden_size**2
                )
                magnitude = a["module_plastic"][i, :count].abs().sum().item() / max(
                    1, count * connections
                )
                modulation = 2 * data["module_actions"][i, :count, 4].mean() - 1
                self.text(f"Synaptic change {magnitude:.4f}", (30, 348), small=True)
                self.text(f"Plasticity gate {modulation:+.2f}", (30, 369), small=True)
            if "connections" in data:
                self.text(f"{data['connections'][i]} connections per module", (30, 390), small=True)
            if c.ecology_version >= 11:
                self.text("Raw processing capacity / second", (30, 411), small=True)
                if world.ablation in ("unlimited_feeding", "unlimited_handling"):
                    label = "Fresh: unlimited   Detritus: unlimited"
                else:
                    tissue = data["modules"][i] * (data["core_radius"][i] / c.body_radius) ** 2
                    diet = data["diet"][i]
                    fresh = c.handling_rate * tissue * diet**2
                    detritus = c.handling_rate * tissue * (1 - diet) ** 2
                    label = f"Fresh {fresh:.1f}   Detritus {detritus:.1f}"
                self.text(label, (30, 432), small=True)
            if c.ecology_version >= 12:
                from .topology import effective_masks

                motor_mask = effective_masks(c, a["genome"][i : i + 1])[3][:, :2]
                count = int(data["modules"][i])
                magnitude = a["module_motor_plastic"][i, :count].abs().sum().item() / (
                    count * (int(motor_mask.sum()) + 2)
                )
                traits = a["genome"][i, c.brain_parameter_count + 11 : c.brain_parameter_count + 13]
                rate, exploration = traits.sigmoid().tolist()
                rate *= c.motor_learning_rate
                exploration = (
                    c.exploration_min + (c.exploration_max - c.exploration_min) * exploration
                )
                self.text(f"Motor offsets {magnitude:.4f}", (30, 453), small=True)
                self.text(f"Exploration {exploration:.2f}   Rate {rate:.3f}", (30, 474), small=True)
            if self.show_brain:
                self.draw_brain(world, i, 92 + height - (220 if c.ecology_version >= 12 else 146))
        return surface

    def draw_brain(self, world, index, top):
        from .inheritance import brain_parts
        from .topology import effective_masks

        c, a = world.config, world.agents
        genome = a["genome"][index : index + 1]
        base = brain_parts(c, genome)[1][0].detach().cpu().numpy()
        mask = (
            effective_masks(c, genome)[2][0].cpu().numpy()
            if c.ecology_version >= 8
            else np.ones_like(base, dtype=bool)
        )
        plastic = (
            a["module_plastic"][index, 0].detach().cpu().numpy()
            if "module_plastic" in a
            else np.zeros_like(base)
        )
        panels = (
            (base, 0.5, "Base ±0.5", 30),
            (plastic, c.plasticity_limit, f"Module 1 ±{c.plasticity_limit:g}", 166),
        )
        for matrix, limit, label, x in panels:
            strength = np.clip(np.abs(matrix) / limit, 0, 1)[..., None]
            tint = np.where((matrix >= 0)[..., None], (100, 210, 161), (153, 133, 235))
            rgb = (np.array((20, 35, 38)) + strength * (tint - (20, 35, 38))).astype(np.uint8)
            rgb[~mask] = (10, 19, 23)
            layer = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
            self.text(label, (x, top), small=True)
            self.surface.blit(pygame.transform.scale(layer, (112, 112)), (x, top + 20))
        if c.ecology_version >= 12:
            matrix = a["module_motor_plastic"][index, 0].detach().cpu().numpy()
            mask = effective_masks(c, genome)[3][0, :2].cpu().numpy()
            mask = np.concatenate((mask, np.ones((2, 1), dtype=bool)), 1)
            strength = np.clip(np.abs(matrix) / c.motor_learning_limit, 0, 1)[..., None]
            tint = np.where((matrix >= 0)[..., None], (100, 210, 161), (153, 133, 235))
            rgb = (np.array((20, 35, 38)) + strength * (tint - (20, 35, 38))).astype(np.uint8)
            rgb[~mask] = (10, 19, 23)
            layer = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
            self.text(
                f"Motor offsets ±{c.motor_learning_limit:g} (L/R)", (30, top + 140), small=True
            )
            self.surface.blit(pygame.transform.scale(layer, (248, 28)), (30, top + 160))

    def save(self, world, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.draw(world), str(path))


class Viewer(Renderer):
    def __init__(self, size=1024):
        super().__init__(size)
        pygame.display.init()
        self.window = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Emergent Garden")
        self.paused = False
        self.speed = 1.0
        self.running = True
        self.drag = None

    def events(self, world):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_b:
                    self.show_brain = not self.show_brain
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
                    self.zoom, self.center = 1, None
            elif event.type == pygame.MOUSEWHEEL:
                self.zoom = min(8, max(1, self.zoom * 1.25**event.y))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and world.population:
                    positions, scale = self.transform(
                        world.agents["pos"].cpu().numpy(), world.config.diameter
                    )
                    distances = np.linalg.norm(positions - event.pos, axis=1)
                    index = int(distances.argmin())
                    self.selected = (
                        int(world.agents["id"][index])
                        if distances[index] < max(12, world.config.body_radius * scale)
                        else None
                    )
                elif event.button == 3:
                    self.drag = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                self.drag = None
            elif event.type == pygame.MOUSEMOTION and self.drag is not None:
                if self.center is None:
                    self.center = np.array([world.config.diameter / 2] * 2)
                scale = (self.size - 70) / world.config.diameter * self.zoom
                self.center -= np.array(event.rel) / scale
        return self.running

    def present(self, world):
        state = "PAUSED" if self.paused else f"{self.speed:g}x"
        self.draw(
            world,
            f"{state} | Space pause | +/- speed | Scroll zoom | Right-drag pan | "
            "F field | Tab channel | C color | R reset",
        )
        self.window.blit(self.surface, (0, 0))
        pygame.display.flip()

    def close(self):
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
        if world.time + 1e-9 >= self.next_time:
            surface = self.renderer.draw(world)
            self.writer.send(pygame.image.tobytes(surface, "RGB"))
            self.frames += 1
            self.next_time += self.period

    def close(self):
        self.writer.close()
