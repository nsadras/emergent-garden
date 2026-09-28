"""Bounded, disposable viewer state; never serialized into a simulation.

Controller samples copy the values at the actual update boundary. In particular,
rendering must not call sensors again: feedback has already been cleared and the
shuffled-sensing intervention would consume another random draw.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


def array(tensor):
    return tensor.detach().cpu().numpy().copy()


@dataclass(frozen=True)
class ReproductionReadiness:
    energy: float
    threshold: float
    capacity: float
    modules: int
    target_modules: int
    retry_seconds: float
    population_full: bool

    @property
    def energy_fraction(self):
        return min(1.0, max(0.0, self.energy / self.threshold))

    @property
    def storage_fraction(self):
        return min(1.0, max(0.0, self.energy / self.capacity))

    @property
    def energy_ready(self):
        return self.energy >= self.threshold

    @property
    def mature(self):
        return self.modules >= self.target_modules

    @property
    def ready_to_try(self):
        return (
            self.energy_ready
            and self.mature
            and not self.retry_seconds
            and not self.population_full
        )

    @property
    def status(self):
        if not self.mature:
            return f"Growing {self.modules}/{self.target_modules} modules"
        if self.population_full:
            return "Population full"
        if self.retry_seconds:
            return f"Retry in {math.ceil(self.retry_seconds * 10) / 10:.1f}s"
        return "Ready to try" if self.energy_ready else "Needs energy"


def reproduction_readiness(world, index):
    """Read known birth gates without mutating a child or sampling placement.

    A full energy bar is not a promise of birth: development, retry timing, and
    capacity also matter. Actual child cost and free space are checked by the
    simulation after mutation, and cannot be predicted exactly by an observer.
    """
    a, c = world.agents, world.config
    threshold, capacity = c.reproduction_threshold, c.max_energy
    if c.ecology_version:
        # Match the simulation's tensor precision at the eligibility boundary.
        threshold = float(threshold * a["area"][index])
        capacity = float(capacity * a["area"][index])
    modules = int(a["modules"][index]) if "modules" in a else 1
    target = int(a["target_modules"][index]) if c.ecology_version >= 13 else modules
    return ReproductionReadiness(
        energy=float(a["energy"][index]),
        threshold=threshold,
        capacity=capacity,
        modules=modules,
        target_modules=target,
        retry_seconds=max(0, int(a["retry_tick"][index]) - world.tick) / c.physics_hz,
        population_full=world.population >= c.capacity,
    )


@dataclass
class BrainSample:
    identifier: int
    tick: int
    time: float
    controller: str
    config: object
    inputs: np.ndarray
    previous: np.ndarray
    hidden: np.ndarray
    actions: np.ndarray
    parts: list
    nodes: np.ndarray
    masks: list
    plastic: np.ndarray
    alpha: float | np.ndarray
    response_times: np.ndarray | None = None
    motor_plastic: np.ndarray | None = None
    noise: np.ndarray | None = None
    predicted_return: np.ndarray | None = None
    prediction_error: np.ndarray | None = None
    recurrent_plastic: np.ndarray | None = None
    hidden_noise: np.ndarray | None = None

    def matrices(self, module):
        """Effective weights used for this sample, including bounded motor learning."""
        wi, wr, bias, wo, bo = (part.copy() for part in self.parts)
        mi, mr, mo = self.masks
        wi *= mi
        offset = self.plastic[module]
        if self.recurrent_plastic is not None:
            # Match the native grouping: base + (local offset + reward offset).
            offset = offset + self.recurrent_plastic[module]
        wr = (wr + offset) * mr
        wo *= mo
        if self.motor_plastic is not None:
            norm = (
                max(1, math.sqrt(float(self.hidden[module] @ self.hidden[module]) + 1))
                if self.config.motor_normalized
                else 1
            )
            wo[:2] += self.motor_plastic[module, :, :-1] / norm
            bo[:2] += self.motor_plastic[module, :, -1] / norm
        return wi, wr, bias, wo, bo

    def drives(self, module):
        wi, wr, bias, _, _ = self.matrices(module)
        return wi @ self.inputs[module], wr @ self.previous[module], bias

    def sensory_drives(self, module):
        """Split each four-sensor field into its mean and directional differences."""
        count = 4 * len(self.config.field_names)
        senses = self.inputs[module, :count]
        if self.config.sensory_contrast:
            common = np.zeros_like(senses)
            common[::4] = senses[::4]
        else:
            common = np.repeat(senses.reshape(-1, 4).mean(1), 4)
        wi = self.matrices(module)[0][:, :count]
        return wi @ common, wi @ (senses - common)


class ControllerObserver:
    """Copy only the selected body's circuits; do no work when nothing is selected."""

    def __init__(self):
        self.selected = None
        self.sample = None
        self._pending = None

    def select(self, identifier):
        if self.selected != identifier:
            self.selected = identifier
            self.sample = None
            self._pending = None

    def before_controller(self, world, index, inputs, module_inputs=None):
        self._pending = None
        if self.selected is None:
            return
        a, c = world.agents, world.config
        match = (a["id"][index] == self.selected).nonzero().flatten()
        if not len(match):
            return
        local = int(match[0])
        i = int(index[local])
        count = int(a["modules"][i]) if module_inputs is not None else 1
        x = array(
            module_inputs[local, :count] if module_inputs is not None else inputs[local : local + 1]
        )
        previous = array(
            a["module_h"][i, :count] if module_inputs is not None else a["h"][i : i + 1]
        )
        from .inheritance import brain_parts
        from .neural_timing import integration_factors, response_times
        from .topology import effective_masks

        genome = a["genome"][i : i + 1]
        parts = [array(part[0]) for part in brain_parts(c, genome)]
        if c.ecology_version >= 8:
            nodes, *masks = (array(part[0]) for part in effective_masks(c, genome))
        else:
            nodes = np.ones(c.hidden_size, dtype=bool)
            masks = [np.ones_like(parts[k], dtype=bool) for k in (0, 1, 3)]
        previous *= nodes
        plastic = np.zeros((count, c.hidden_size, c.hidden_size), dtype=np.float32)
        if "module_plastic" in a and world.ablation != "no_plasticity":
            plastic = array(a["module_plastic"][i, :count])
        tau = a["memory_tau"][i : i + 1] if "memory_tau" in a else None
        factors = integration_factors(c, genome, tau)
        sample = BrainSample(
            self.selected,
            world.tick,
            world.time,
            world.controller,
            c,
            x,
            previous,
            previous.copy(),
            np.zeros((count, c.output_size)),
            parts,
            nodes,
            masks,
            plastic,
            factors if isinstance(factors, float) else array(factors[0]),
            response_times=array(response_times(c, genome, tau)[0]),
        )
        self._pending = (i, local, sample)

    def after_controller(self, world, motor_noise=None):
        if self._pending is None:
            return
        i, local, sample = self._pending
        self._pending = None
        a, c = world.agents, world.config
        count = len(sample.inputs)
        sample.hidden = array(a["module_h"][i, :count] if "module_h" in a else a["h"][i : i + 1])
        if "module_actions" in a:
            sample.actions = array(a["module_actions"][i, :count])
        else:
            sample.actions = array(a.get("actions", a["motors"])[i : i + 1])
        if "module_motor_plastic" in a and world.controller == "neural":
            # Motor learning is applied before action selection; the older
            # local recurrent update is applied afterwards (captured above).
            sample.motor_plastic = array(a["module_motor_plastic"][i, :count])
            if c.ecology_version >= 21:
                sample.noise = array(a["module_motor_applied_noise"][i, :count])
            else:
                trait = a["genome"][i, c.brain_parameter_count + 12].sigmoid().item()
                sigma = c.exploration_min + (c.exploration_max - c.exploration_min) * trait
                sample.noise = array(motor_noise[local, :count]) * sigma
            if (
                c.ecology_version >= 22
                and c.motor_value_rate
                and world.ablation not in ("no_motor_value", "no_motor_learning", "no_plasticity")
            ):
                sample.predicted_return = array(a["module_motor_value_prediction"][i, :count])
                sample.prediction_error = array(a["module_motor_value_error"][i, :count])
        if c.ecology_version >= 25 and c.recurrent_noise_sigma and world.controller == "neural":
            # Recurrent energetic credit is applied before the transition,
            # unlike local plasticity captured by before_controller above.
            sample.recurrent_plastic = array(a["module_recurrent_plastic"][i, :count])
            sample.hidden_noise = array(a["module_recurrent_applied_noise"][i, :count])
        self.sample = sample


@dataclass
class Trail:
    lineage: int
    diet: float | None
    points: deque = field(default_factory=deque)


class TrailHistory:
    """Centroid paths sampled in simulated time, keyed by permanent creature ID."""

    def __init__(self, seconds=120, hz=5):
        self.seconds = seconds
        self.hz = hz
        self.tracks = {}
        self.next_tick = 0
        self.last_tick = -1
        self.world = None
        self.period = 1 / hz

    def observe(self, world):
        if world is not self.world or world.tick < self.last_tick:
            self.tracks.clear()
            self.next_tick = 0
            self.world = world
        self.last_tick = world.tick
        if world.tick < self.next_tick:
            return
        interval = max(1, round(world.config.physics_hz / self.hz))
        self.period = interval / world.config.physics_hz
        self.next_tick = world.tick + interval
        now = world.time
        for identifier, track in list(self.tracks.items()):
            while track.points and track.points[0][0] < now - self.seconds:
                track.points.popleft()
            if not track.points:
                del self.tracks[identifier]
        a = world.agents
        ids, positions, lineages = (array(a[key]) for key in ("id", "pos", "lineage"))
        diets = array(a["diet"]) if "diet" in a else [None] * len(ids)
        for identifier, pos, lineage, diet in zip(ids, positions, lineages, diets, strict=True):
            track = self.tracks.setdefault(int(identifier), Trail(int(lineage), diet))
            track.points.append((now, float(pos[0]), float(pos[1])))

    @staticmethod
    def opacity(sample_time, now, duration):
        return np.clip(1 - (now - sample_time) / duration, 0, 1)
