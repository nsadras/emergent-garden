"""Validated, versioned experiment configuration."""

import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import tomli_w


@dataclass
class Config:
    schema_version: int = 1
    diameter: float = 1024.0
    body_radius: float = 4.0
    initial_population: int = 256
    capacity: int = 2048
    max_speed: float = 24.0
    max_turn_degrees: float = 90.0
    collision_iterations: int = 4
    initial_food: int = 1024
    food_radius: float = 1.0
    food_energy: float = 20.0
    food_rate: float = 20.0
    food_lifetime: float = 180.0
    patch_fraction: float = 0.8
    patches: int = 8
    patch_radius: float = 48.0
    patch_period: float = 120.0
    grid_size: int = 512
    smell_sigma: float = 24.0
    smell_cutoff: float = 72.0
    smell_scale: float = 8.0
    birth_energy: float = 100.0
    max_energy: float = 250.0
    basal_cost: float = 1.0
    propulsion_cost: float = 0.5
    reproduction_threshold: float = 220.0
    reproduction_debit: float = 120.0
    birth_attempts: int = 16
    birth_gap: float = 0.1
    birth_retry: float = 1.0
    hidden_size: int = 16
    neural_tau: float = 0.5
    mutation_probability: float = 0.02
    mutation_sigma: float = 0.05
    weight_limit: float = 3.0
    physics_hz: int = 60
    controller_hz: int = 20
    field_hz: int = 10
    metrics_period: float = 1.0
    checkpoint_wall_seconds: float = 300.0
    viewer_size: int = 1024
    viewer_fps: int = 60
    video_fps: int = 30
    video_speed: float = 100.0

    def validate(self):
        defaults = Config()
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{f.name} must be numeric")
            if isinstance(getattr(defaults, f.name), int) and not isinstance(value, int):
                raise ValueError(f"{f.name} must be an integer")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{f.name} must be finite and nonnegative")
        positive = (
            "diameter body_radius capacity collision_iterations food_energy food_lifetime "
            "patches patch_period grid_size smell_sigma smell_cutoff smell_scale birth_energy "
            "max_energy reproduction_threshold reproduction_debit birth_attempts birth_gap "
            "birth_retry hidden_size neural_tau weight_limit physics_hz controller_hz field_hz "
            "metrics_period checkpoint_wall_seconds viewer_size viewer_fps video_fps video_speed"
        )
        for key in positive.split():
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.schema_version != 1:
            raise ValueError("Unsupported configuration schema")
        if self.initial_population > self.capacity:
            raise ValueError("initial_population must not exceed capacity")
        if self.body_radius >= self.diameter / 2:
            raise ValueError("Body must fit in the dish")
        if self.patch_radius + self.food_radius >= self.diameter / 2:
            raise ValueError("Food patches must fit in the dish")
        if self.grid_size < 8 or self.viewer_size < 128 or self.viewer_size % 2:
            raise ValueError("grid_size >= 8; viewer_size must be even and >= 128")
        if not 0 <= self.patch_fraction <= 1 or not 0 <= self.mutation_probability <= 1:
            raise ValueError("Probabilities must be in [0, 1]")
        if not self.birth_energy <= self.reproduction_debit <= self.reproduction_threshold:
            raise ValueError("Require birth_energy <= reproduction_debit <= threshold")
        if self.reproduction_threshold > self.max_energy:
            raise ValueError("Reproduction threshold must not exceed storage capacity")
        for hz in (self.controller_hz, self.field_hz):
            if self.physics_hz % hz:
                raise ValueError("Controller and field rates must divide physics_hz")
        return self

    @property
    def dt(self):
        return 1 / self.physics_hz

    @property
    def parameter_count(self):
        h = self.hidden_size
        return 6 * h + h * h + h + 2 * h + 2

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        return cls(**data).validate()

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls().validate()
        with Path(path).open("rb") as stream:
            return cls.from_dict(tomllib.load(stream))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomli_w.dumps(asdict(self)))
