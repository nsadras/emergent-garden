"""A scrollable Pygame panel for one body's actual controller samples."""

import math

import numpy as np
import pygame

from .observation import array

BG = (11, 23, 31)
TEXT = (215, 232, 235)
MUTED = (132, 163, 174)
POSITIVE = (111, 225, 181)
NEGATIVE = (173, 153, 244)
GOLD = (250, 218, 150)
OUTPUTS = ("Motor 0", "Motor 1", "Attack", "Secretion", "Learn gate", "Internal A", "Internal B")


def activity_color(value):
    color = POSITIVE if value >= 0 else NEGATIVE
    strength = min(1, abs(float(value)))
    return tuple(int(36 + (v - 36) * strength) for v in color)


class Inspector:
    width = 620

    def __init__(self):
        self.font = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 18)
        self.title = pygame.font.Font(None, 30)
        self.surface = None
        self.scroll = 0
        self.content_height = 0
        self.module = 0
        self.neuron = 0
        self.tab = "brain"
        self.buttons = []
        self.nodes = []
        self.identifier = None

    def text(self, value, xy, color=TEXT, font=None):
        self.surface.blit((font or self.small).render(value, True, color), xy)

    def button(self, label, rect, action, active=False):
        rect = pygame.Rect(rect)
        pygame.draw.rect(
            self.surface, (40, 69, 76) if active else (23, 40, 50), rect, border_radius=5
        )
        self.text(label, (rect.x + 10, rect.y + 7), POSITIVE if active else MUTED)
        self.buttons.append((rect, action))

    def click(self, point):
        point = (point[0], point[1] + self.scroll)
        for rect, action in self.buttons:
            if rect.collidepoint(point):
                kind, value = action
                setattr(self, kind, value)
                if kind == "tab":
                    self.scroll = 0
                return
        for identifier, pos in self.nodes:
            if math.dist(point, pos) < 12:
                self.neuron = identifier
                return

    def wheel(self, amount, height):
        self.scroll = min(max(0, self.content_height - height), max(0, self.scroll - amount * 48))

    def draw(self, world, selected, observer, height, following=False):
        if selected != self.identifier:
            self.identifier, self.module, self.scroll = selected, 0, 0
        capacity = 480 + max(world.config.input_size, world.config.hidden_size) * 16
        self.surface = pygame.Surface((self.width, max(height, capacity)))
        self.surface.fill(BG)
        self.buttons, self.nodes = [], []
        pygame.draw.line(self.surface, (45, 74, 84), (0, 0), (0, self.surface.get_height()), 2)
        self.text("CONTROLLER OBSERVATORY", (22, 20), POSITIVE, self.title)
        if selected is None:
            self.text("Click a creature in the dish to inspect it.", (22, 74), TEXT, self.font)
            lines = (
                "Each body module has its own recurrent state and learned weights.",
                "The inspector reads the actual controller update, including its inputs.",
                "Select a hidden neuron to see its incoming and outgoing links.",
                "",
                "T   trails: all / selected / off         [ / ]   trail duration",
                "G   follow the selected creature      M   next body module",
                "Space   pause / resume                  N   advance one controller interval",
                "B   hide / show this panel               Mouse wheel here   scroll",
                "",
                "Trails fade in simulated seconds. Pausing freezes them.",
                "New selections show their first sample on the next controller update.",
            )
            for j, line in enumerate(lines):
                self.text(line, (22, 118 + 27 * j), MUTED)
            self.content_height = 440
        else:
            self.content_height = self.draw_selected(world, selected, observer, following)
        self.scroll = min(self.scroll, max(0, self.content_height - height))
        visible = pygame.Surface((self.width, height))
        visible.blit(self.surface, (0, -self.scroll))
        if self.content_height > height:
            length = max(24, int(height * height / self.content_height))
            top = round((height - length) * self.scroll / (self.content_height - height))
            pygame.draw.rect(
                visible, (74, 113, 121), (self.width - 6, top, 4, length), border_radius=2
            )
        return visible

    def draw_selected(self, world, selected, observer, following):
        a, c = world.agents, world.config
        found = (a["id"] == selected).nonzero().flatten()
        i = int(found[0]) if len(found) else None
        sample = observer.sample
        if sample is not None and sample.identifier != selected:
            sample = None
        count = int(a["modules"][i]) if i is not None and "modules" in a else 1
        if i is None and sample is not None:
            count = len(sample.inputs)
        self.module = min(self.module, count - 1)
        state = (
            "following" if following and i is not None else "selected" if i is not None else "died"
        )
        self.text(f"Creature {selected}  /  {state}", (22, 56), GOLD, self.font)
        if i is not None:
            self.text(
                f"Lineage {int(a['lineage'][i])}    Generation {int(a['generation'][i])}    "
                f"Children {int(a['offspring'][i])}",
                (22, 83),
                MUTED,
            )
            self.text(
                f"Energy {float(a['energy'][i]):.1f}    Age {float(a['age'][i]):.1f}s", (22, 105)
            )
        else:
            self.text(
                "Last controller sample retained. Select another creature to continue.",
                (22, 88),
                MUTED,
            )
        self.button("Brain", (22, 137, 86, 28), ("tab", "brain"), self.tab == "brain")
        self.button("Body / learning", (116, 137, 130, 28), ("tab", "body"), self.tab == "body")
        self.text("Module", (290, 145), MUTED)
        for k in range(count):
            self.button(str(k + 1), (354 + 48 * k, 137, 40, 28), ("module", k), k == self.module)
        if self.tab == "body":
            return self.draw_body(world, i, sample)
        if sample is None or self.module >= len(sample.inputs):
            self.text(
                "Waiting for this module's first controller sample.", (22, 203), TEXT, self.font
            )
            self.text(
                "Resume, or press N while paused to advance one controller interval.",
                (22, 238),
                MUTED,
            )
            return 282
        self.text(
            f"Sample t={sample.time:,.3f}s  /  tick {sample.tick}  /  {c.controller_hz} Hz",
            (22, 179),
            MUTED,
        )
        if sample.controller != "neural":
            self.text(
                f"Scripted '{sample.controller}' controller; no neural graph is active.",
                (22, 218),
                GOLD,
            )
            for j, value in enumerate(sample.actions[self.module]):
                self.text(f"{OUTPUTS[j]}   {value:.4f}", (22, 253 + j * 25))
            return 290 + 25 * c.output_size
        return self.draw_brain(sample)

    def draw_brain(self, sample):
        k = self.module
        active = np.flatnonzero(sample.nodes)
        if self.neuron not in active:
            self.neuron = int(active[0]) if len(active) else 0
        focus = self.neuron
        c = sample.config
        extent = max(310, (max(c.input_size, len(active)) - 1) * 16)
        y0 = 244
        input_pos = np.column_stack(
            (np.full(c.input_size, 222), np.linspace(y0, y0 + extent, c.input_size))
        )
        hidden_pos = dict(
            zip(
                active,
                zip(
                    np.full(len(active), 348),
                    np.linspace(y0, y0 + extent, len(active)),
                    strict=True,
                ),
                strict=True,
            )
        )
        output_pos = np.column_stack(
            (np.full(c.output_size, 474), np.linspace(y0 + 10, y0 + extent - 10, c.output_size))
        )
        wi, wr, _, wo, _ = sample.matrices(k)
        self.text("INPUTS", (22, 207), POSITIVE)
        self.text(f"RECURRENT  {len(active)}/{c.hidden_size}", (290, 207), POSITIVE)
        self.text("OUTPUTS", (485, 207), POSITIVE)
        self.text("Sensor angle relative to heading", (22, 224), MUTED)
        self.text(f"Links touching h{focus:02d}  /  click a neuron", (290, 224), MUTED)

        def link(start, end, weight, bend=False):
            if abs(weight) < 1e-6:
                return
            tint = POSITIVE if weight >= 0 else NEGATIVE
            strength = 0.18 + 0.52 * min(1, abs(float(weight)))
            color = tuple(int(BG[j] + strength * (tint[j] - BG[j])) for j in range(3))
            if bend:
                mid = (
                    (start[0] + end[0]) / 2 - 22 - abs(start[1] - end[1]) * 0.05,
                    (start[1] + end[1]) / 2,
                )
                if start == end:
                    pygame.draw.circle(
                        self.surface, color, (int(start[0] - 9), int(start[1] - 5)), 10, 1
                    )
                else:
                    pygame.draw.aalines(self.surface, color, False, (start, mid, end))
            else:
                pygame.draw.aaline(self.surface, color, start, end)

        if focus in hidden_pos:
            target = hidden_pos[focus]
            for j, pos in enumerate(input_pos):
                link(tuple(pos), target, wi[focus, j])
            for j, pos in hidden_pos.items():
                link(pos, target, wr[focus, j], True)
            for j, pos in enumerate(output_pos):
                link(target, tuple(pos), wo[j, focus])
        for j, (name, pos) in enumerate(zip(c.input_names, input_pos, strict=True)):
            value = sample.inputs[k, j]
            label = name.replace("identity_", "id ").replace("_feedback", " fb").replace("_", " ")
            self.text(label, (22, pos[1] - 5))
            self.text(f"{value:+.4f}", (124, pos[1] - 5), MUTED)
            pygame.draw.rect(self.surface, (29, 47, 57), (177, pos[1] - 3, 30, 6))
            pygame.draw.rect(
                self.surface,
                activity_color(value),
                (177, pos[1] - 3, round(30 * min(abs(float(value)), 1)), 6),
            )
            pygame.draw.circle(self.surface, activity_color(value), pos, 4)
        for j, pos in hidden_pos.items():
            value = sample.hidden[k, j]
            pygame.draw.circle(self.surface, activity_color(value), pos, 6)
            if j == focus:
                pygame.draw.circle(self.surface, GOLD, pos, 10, 1)
            label = self.small.render(f"{j:02d} {value:+.2f}", True, TEXT, BG)
            self.surface.blit(label, (pos[0] + 13, pos[1] - 5))
            self.nodes.append((int(j), pos))
        for j, pos in enumerate(output_pos):
            value = sample.actions[k, j]
            pygame.draw.circle(self.surface, activity_color(value), pos, 6)
            self.text(OUTPUTS[j], (pos[0] + 14, pos[1] - 12))
            self.text(f"{value:.4f}", (pos[0] + 14, pos[1] + 5), GOLD)
        top = y0 + extent + 30
        incoming, recurrent, bias = sample.drives(k)
        self.text(
            f"h{focus:02d}: {sample.previous[k, focus]:+.3f} -> {sample.hidden[k, focus]:+.3f}",
            (22, top),
            GOLD,
        )
        self.text(
            f"Drive = input {incoming[focus]:+.3f}  +  recurrent {recurrent[focus]:+.3f}"
            f"  +  bias {bias[focus]:+.3f}",
            (22, top + 21),
        )
        common, contrast = sample.sensory_drives(k)

        def rms(values):
            return np.sqrt(np.mean(values[active] ** 2)) if len(active) else 0

        self.text(
            f"Drive RMS: field mean {rms(common):.3f}  /  field contrast {rms(contrast):.3f}"
            f"  /  recurrent {rms(recurrent):.3f}",
            (22, top + 42),
            MUTED,
        )
        self.text(
            "Green + / violet - ; links = effective weights, node fill = activity.",
            (22, top + 63),
            MUTED,
        )
        self.text(
            "Inputs/hidden use unit-scale bars; outputs are sigmoid values in [0, 1].",
            (22, top + 84),
            MUTED,
        )
        if sample.noise is not None:
            self.text(
                f"Motor noise (logits): {sample.noise[k, 0]:+.3f} / {sample.noise[k, 1]:+.3f}"
                "   |   learned readouts included",
                (22, top + 105),
                MUTED,
            )
        if c.output_size >= 5:
            self.text(
                "Gate and internal signals are used as signed values: 2 x output - 1.",
                (22, top + 126),
                MUTED,
            )
        return int(top + 156)

    def draw_body(self, world, i, sample):
        a, c = world.agents, world.config
        top = 192
        if i is not None:
            rows = []
            if "radius" in a:
                rows += [
                    f"Radius {float(a['radius'][i]):.2f}    Power {float(a['power'][i]):.2f}",
                    f"Fresh-food allocation {float(a['diet'][i]):.1%}",
                ]
            if "armor" in a:
                rows.append(
                    f"Weapon {float(a['attack'][i]):.1%}    Armor {float(a['armor'][i]):.1%}"
                )
            if "modules" in a:
                count = int(a["modules"][i])
                target = int(a.get("target_modules", a["modules"])[i])
                rows.append(
                    f"Developed modules {count} / {target}    Viewing module {self.module + 1}"
                )
                from .morphology import module_turn

                turn = float(module_turn(a)[i])
            else:
                turn = float(a["motors"][i, 1] - a["motors"][i, 0])
            rows.append(
                f"Body turn command {turn * c.max_turn_degrees:+.2f} deg/s  (+ = clockwise)"
            )
            if "connections" in a:
                rows.append(
                    f"{int(a['neurons'][i])} active neurons / "
                    f"{int(a['connections'][i])} connections per module"
                )
            if c.ecology_version >= 11:
                tissue = int(a["modules"][i]) * (float(a["core_radius"][i]) / c.body_radius) ** 2
                diet = float(a["diet"][i])
                rows.append(
                    "Raw processing per second: "
                    + (
                        "unlimited"
                        if world.ablation in ("unlimited_feeding", "unlimited_handling")
                        else f"fresh {c.handling_rate * tissue * diet**2:.1f} / "
                        f"detritus {c.handling_rate * tissue * (1 - diet) ** 2:.1f}"
                    )
                )
            if "module_internal" in a:
                signal = a["module_internal"][i, self.module]
                rows.append(
                    f"Internal signals A / B: {float(signal[0]):+.3f} / {float(signal[1]):+.3f}"
                )
            if c.ecology_version >= 15:
                from .plasticity import rule_coefficients

                rule = array(
                    rule_coefficients(
                        c, a["genome"][i : i + 1], evolved=world.ablation != "fixed_rule"
                    )[0]
                )
                rows.append("Learning rule A / B / C / D: " + " / ".join(f"{v:+.2f}" for v in rule))
            for row in rows:
                self.text(row, (22, top), TEXT, self.font)
                top += 28
        if sample is None or self.module >= len(sample.inputs) or sample.controller != "neural":
            self.text("Weight maps appear after a neural controller sample.", (22, top + 22), MUTED)
            return top + 70
        self.text(
            f"Recurrent weights used at t={sample.time:.3f}s (row = target, column = source)",
            (22, top + 15),
            MUTED,
        )
        top += 47
        base = sample.parts[1] * sample.masks[1]
        plastic = sample.plastic[self.module] * sample.masks[1]
        for x, values, label in (
            (22, base, "Inherited"),
            (221, plastic, "Acquired"),
            (420, base + plastic, "Effective"),
        ):
            self.text(label, (x, top), POSITIVE)
            self.heatmap(values, sample.masks[1], (x, top + 25), (176, 176), 0.5)
        top += 229
        self.text(
            "All three maps share a fixed +/-0.5 color scale; darker = near zero.", (22, top), MUTED
        )
        self.text(
            "Inactive connections are black. These are sampled weights, before the next update.",
            (22, top + 22),
            MUTED,
        )
        if sample.motor_plastic is not None:
            top += 62
            self.text(
                f"Acquired motor readouts (weights + bias), +/-{c.motor_learning_limit:g}",
                (22, top),
                POSITIVE,
            )
            mask = np.concatenate((sample.masks[2][:2], np.ones((2, 1), dtype=bool)), axis=1)
            self.heatmap(
                sample.motor_plastic[self.module],
                mask,
                (22, top + 26),
                (574, 40),
                c.motor_learning_limit,
            )
            top += 80
        return top + 65

    def heatmap(self, values, mask, pos, size, limit):
        strength = np.clip(np.abs(values) / max(limit, 1e-9), 0, 1)[..., None]
        tint = np.where((values >= 0)[..., None], POSITIVE, NEGATIVE)
        rgb = (np.array(BG) + strength * (tint - BG)).astype(np.uint8)
        rgb[~mask] = (5, 12, 17)
        image = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
        self.surface.blit(pygame.transform.scale(image, size), pos)
