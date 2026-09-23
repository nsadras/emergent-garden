"""A sortable, paginated view of living creatures and their lifetime counters."""

import math

import pygame

from .inspector import BG, GOLD, MUTED, POSITIVE, TEXT, Inspector
from .observation import array

# Widths fill the same 574-pixel content area as the controller inspector.
COLUMNS = (
    ("id", "Creature", 76),
    ("age", "Lifetime (s)", 94),
    ("generation", "Gen.", 54),
    ("acquired", "Food", 94),
    ("offspring", "Children", 78),
    ("energy", "Energy", 86),
    ("distance", "Distance", 92),
)
INTEGER_COLUMNS = {"id", "generation", "offspring"}


class Leaderboard:
    width = Inspector.width
    row_height = 30

    def __init__(self):
        self.font = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 18)
        self.title = pygame.font.Font(None, 30)
        self.sort_key = "age"
        self.descending = True
        self.page = 0
        self.page_count = 1
        self.page_size = 1
        self.rows = []
        self.buttons = []
        self.layout_height = 640
        self.content_height = 0
        self.scale = 1.0
        self.offset_x = 0
        self.surface = None

    def text(self, value, xy, color=TEXT, font=None):
        self.surface.blit((font or self.small).render(value, True, color), xy)

    def button(self, label, rect, action, enabled=True):
        rect = pygame.Rect(rect)
        pygame.draw.rect(self.surface, (23, 40, 50), rect, border_radius=5)
        self.text(label, (rect.x + 10, rect.y + 7), POSITIVE if enabled else MUTED)
        if enabled:
            self.buttons.append((rect, action))

    def turn_page(self, direction):
        self.page = min(self.page_count - 1, max(0, self.page + direction))

    def click(self, point):
        point = ((point[0] - self.offset_x) / self.scale, point[1] / self.scale)
        for rect, action in self.buttons:
            if rect.collidepoint(point):
                kind, value = action
                if kind == "sort":
                    self.descending = (
                        not self.descending if self.sort_key == value else value != "id"
                    )
                    self.sort_key, self.page = value, 0
                elif kind == "page":
                    self.turn_page(value)
                else:
                    # Row actions retain the ID that was actually displayed,
                    # even if the population changes before this click arrives.
                    return action
                return None
        return None

    def draw(self, world, selected, height):
        self.layout_height = max(640, height)
        self.scale = height / self.layout_height
        self.offset_x = (self.width - round(self.width * self.scale)) // 2
        self.surface = pygame.Surface((self.width, self.layout_height))
        self.surface.fill(BG)
        self.buttons = []
        pygame.draw.line(self.surface, (45, 74, 84), (0, 0), (0, self.layout_height), 2)
        self.text("CREATURE LEADERBOARD", (22, 20), POSITIVE, self.title)
        self.button("Inspector  L", (466, 18, 130, 28), ("view", "inspector"))
        self.text("Living creatures / click a row to locate it and open its inspector.", (22, 62))
        self.text(
            "Click a column to sort; click again to reverse. Space pauses the world.",
            (22, 84),
            MUTED,
        )

        # Use existing lifetime totals, including when resuming a checkpoint.
        # Read only scalar statistics; never evaluate sensors or controllers.
        data = {key: array(world.agents[key]) for key, _, _ in COLUMNS}
        self.rows = [
            {key: values[i].item() for key, values in data.items()} for i in range(world.population)
        ]
        self.rows.sort(
            key=lambda row: (
                -row[self.sort_key] if self.descending else row[self.sort_key],
                row["id"],
            )
        )
        self.page_size = max(1, (self.layout_height - 174 - 94) // self.row_height)
        self.page_count = max(1, math.ceil(len(self.rows) / self.page_size))
        self.page = min(self.page, self.page_count - 1)
        label = next(label for key, label, _ in COLUMNS if key == self.sort_key)
        direction = "highest first" if self.descending else "lowest first"
        self.text(
            f"{len(self.rows)} alive  |  {world.time:,.1f}s  |  {label}: {direction}",
            (22, 111),
            GOLD,
        )

        x = 22
        for key, label, width in COLUMNS:
            rect = pygame.Rect(x, 138, width - 2, 28)
            active = key == self.sort_key
            pygame.draw.rect(self.surface, (40, 69, 76) if active else (23, 40, 50), rect)
            self.text(label, (x + 5, 145), POSITIVE if active else MUTED)
            self.buttons.append((rect, ("sort", key)))
            if active:
                cx, cy = rect.right - 8, rect.centery
                dy = 3 if self.descending else -3
                pygame.draw.polygon(
                    self.surface, POSITIVE, ((cx - 3, cy - dy), (cx + 3, cy - dy), (cx, cy + dy))
                )
            x += width

        start = self.page * self.page_size
        for j, row in enumerate(self.rows[start : start + self.page_size]):
            rect = pygame.Rect(22, 174 + j * self.row_height, 574, self.row_height - 2)
            active = row["id"] == selected
            color = (40, 69, 76) if active else (17, 32, 41) if j % 2 == 0 else BG
            pygame.draw.rect(self.surface, color, rect, border_radius=3)
            if active:
                pygame.draw.rect(self.surface, GOLD, (rect.x, rect.y, 3, rect.height))
            self.buttons.append((rect, ("select", row["id"])))
            x = rect.x
            for key, _, width in COLUMNS:
                value = row[key]
                label = f"{value:,}" if key in INTEGER_COLUMNS else f"{value:,.1f}"
                # Long-running worlds can accumulate large counts. Keep them
                # readable without letting a number spill into its neighbor.
                if self.small.size(label)[0] > width - 12:
                    label = f"{value:.3g}"
                tx = x + 8 if key == "id" else x + width - 8 - self.small.size(label)[0]
                self.text(label, (tx, rect.y + 8), GOLD if active else TEXT)
                x += width
        if not self.rows:
            self.text("No living creatures.", (22, 193), MUTED, self.font)

        bottom = self.layout_height
        self.text(
            "Food = lifetime energy absorbed from fresh food, detritus, and prey.",
            (22, bottom - 87),
            MUTED,
        )
        self.text(
            "Distance = total movement in world units. Wheel / PgUp / PgDn change pages.",
            (22, bottom - 68),
            MUTED,
        )
        self.button("Previous", (22, bottom - 42, 92, 28), ("page", -1), self.page > 0)
        self.button(
            "Next", (504, bottom - 42, 92, 28), ("page", 1), self.page < self.page_count - 1
        )
        end = min(len(self.rows), start + self.page_size)
        label = (
            f"{start + 1 if self.rows else 0}-{end} of {len(self.rows)}"
            f"  |  Page {self.page + 1}/{self.page_count}"
        )
        self.text(label, ((self.width - self.small.size(label)[0]) // 2, bottom - 34))
        self.content_height = bottom - 14
        if self.scale == 1:
            return self.surface
        visible = pygame.Surface((self.width, height))
        visible.fill(BG)
        scaled = pygame.transform.smoothscale(
            self.surface, (round(self.width * self.scale), height)
        )
        visible.blit(scaled, (self.offset_x, 0))
        return visible
