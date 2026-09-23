"""Validated, versioned experiment configuration."""

import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import tomli_w


@dataclass
class Config:
    schema_version: int = 1
    ecology_version: int = 0
    patch_capacity: float = 2400.0
    detritus_fraction: float = 0.35
    detritus_lifetime: float = 240.0
    detritus_delay: float = 0.0
    trait_mutation_probability: float = 0.15
    trait_mutation_sigma: float = 0.15
    bite_rate: float = 60.0
    attack_reach: float = 3.0
    predation_efficiency: float = 0.65
    armor_protection: float = 0.8
    attack_cost: float = 0.25
    resource_burst: float = 10.0
    resource_floor: float = 0.15
    cue_lead: float = 3.0
    cue_duration: float = 2.0
    cue_strength: float = 20.0
    signal_rate: float = 200.0
    signal_cost: float = 0.12
    signal_diffusion: float = 12.0
    signal_half_life: float = 30.0
    signal_scale: float = 1.0
    archive_sim_seconds: float = 0.0
    quality_period: float = 240.0
    quality_jitter: float = 0.25
    low_quality: float = 0.25
    high_quality: float = 1.0
    identity_strength: float = 20.0
    feedback_scale: float = 10.0
    plasticity_rate: float = 0.2
    plasticity_limit: float = 1.0
    plasticity_trace_tau: float = 2.0
    plasticity_half_life_min: float = 30.0
    plasticity_half_life_max: float = 600.0
    plasticity_cost: float = 0.02
    initial_neurons: int = 16
    min_neurons: int = 4
    node_mutation_probability: float = 0.05
    edge_mutation_probability: float = 0.1
    neuron_maintenance: float = 0.0005
    synapse_maintenance: float = 0.00001
    neuron_construction: float = 0.05
    synapse_construction: float = 0.001
    shelter_fraction: float = 0.5
    shelter_radius: float = 36.0
    shelter_protection: float = 0.95
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
            "metrics_period checkpoint_wall_seconds viewer_size viewer_fps video_fps video_speed "
            "signal_half_life signal_scale quality_period feedback_scale identity_strength "
            "plasticity_limit plasticity_trace_tau "
            "plasticity_half_life_min plasticity_half_life_max shelter_radius"
        )
        for key in positive.split():
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.schema_version != 1:
            raise ValueError("Unsupported configuration schema")
        if self.ecology_version not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            raise ValueError("Unsupported ecology version")
        if self.patch_capacity < self.food_energy or self.detritus_lifetime <= 0:
            raise ValueError("Patch capacity must hold food; detritus lifetime must be positive")
        if not 0 <= self.detritus_fraction < 1 or not 0 <= self.trait_mutation_probability <= 1:
            raise ValueError("Invalid ecology probability")
        if not 0 <= self.predation_efficiency <= 1 or not 0 <= self.armor_protection <= 1:
            raise ValueError("Predation efficiency and armor protection must be in [0, 1]")
        if not 0 <= self.resource_floor <= 1:
            raise ValueError("resource_floor must be in [0, 1]")
        if not 0 <= self.shelter_fraction <= 1 or not 0 <= self.shelter_protection <= 1:
            raise ValueError("Shelter fraction and protection must be in [0, 1]")
        if not 0 <= self.low_quality <= self.high_quality <= 1 or not 0 <= self.quality_jitter < 1:
            raise ValueError("Require 0 <= low_quality <= high_quality <= 1 and jitter in [0, 1)")
        if self.ecology_version >= 6 and self.patches < 2:
            raise ValueError("Reversal ecology requires at least two patches")
        if self.plasticity_half_life_max < self.plasticity_half_life_min:
            raise ValueError("Plasticity half-life bounds must be ordered")
        if self.ecology_version >= 8 and not (
            1 <= self.min_neurons <= self.initial_neurons <= self.hidden_size
        ):
            raise ValueError("Require 1 <= min_neurons <= initial_neurons <= hidden_size")
        if (
            not 0 <= self.node_mutation_probability <= 1
            or not 0 <= self.edge_mutation_probability <= 1
        ):
            raise ValueError("Structural mutation probabilities must be in [0, 1]")
        if self.ecology_version >= 3 and (
            self.resource_burst <= 0
            or self.cue_duration <= 0
            or self.resource_burst + self.cue_duration + self.cue_lead >= self.patch_period
        ):
            raise ValueError("Burst and cue timing must fit within patch_period")
        if self.initial_population > self.capacity:
            raise ValueError("initial_population must not exceed capacity")
        if self.body_radius >= self.diameter / 2:
            raise ValueError("Body must fit in the dish")
        if self.ecology_version and self.max_body_radius >= self.diameter / 2:
            raise ValueError("Largest inherited body must fit in the dish")
        if self.detritus_delay >= self.detritus_lifetime:
            raise ValueError("Detritus must mature before it expires")
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
        return self.brain_parameter_count + self.trait_count + self.structure_count

    @property
    def structure_count(self):
        h = self.hidden_size
        return (
            h + h * self.input_size + h * h + self.output_size * h
            if self.ecology_version >= 8
            else 0
        )

    @property
    def input_size(self):
        return len(self.input_names)

    @property
    def input_names(self):
        channels = ["fresh"]
        if self.ecology_version >= 1:
            channels.append("detritus")
        if self.ecology_version >= 2:
            channels.append("organisms")
        if self.ecology_version >= 3:
            channels.append("forecast")
        if self.ecology_version >= 5:
            channels.append("secretions")
        if self.ecology_version >= 6:
            channels.extend(("identity_a", "identity_b"))
        if self.ecology_version >= 10:
            channels.append("shelter")
        names = [f"{channel}_{angle}" for channel in channels for angle in (-135, -45, 45, 135)]
        if self.ecology_version >= 6:
            names.extend(("food_feedback", "damage_feedback"))
        return (*names, "energy", "contact")

    @property
    def output_size(self):
        if self.ecology_version >= 7:
            return 5
        if self.ecology_version >= 5:
            return 4
        return 3 if self.ecology_version >= 2 else 2

    @property
    def trait_count(self):
        if self.ecology_version >= 7:
            return 11
        if self.ecology_version >= 4:
            return 9
        if self.ecology_version >= 3:
            return 6
        if self.ecology_version >= 2:
            return 5
        return 3 if self.ecology_version else 0

    @property
    def brain_parameter_count(self):
        h = self.hidden_size
        return self.input_size * h + h * h + h + self.output_size * h + self.output_size

    @property
    def max_body_radius(self):
        return self.body_radius * (3.9 if self.ecology_version >= 4 else 1.3)

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
