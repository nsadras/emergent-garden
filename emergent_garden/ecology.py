"""Versioned resource ecology, sharing V0's spatial lookup and observation API.

All randomness is explicit; observers never advance the simulation. Energy
transfers are capped simultaneously before either side of a transaction changes.
"""

import math
from dataclasses import asdict

import torch

from .brain import advance, initial_brains
from .config import Config
from .field import SmellField
from .morphology import MAX_MODULES, develop_modules, module_centers, module_turn
from .spatial import neighbors
from .world import World, genome_hash


class EcologyWorld(World):
    def __init__(self, config, seed=1, device="cpu", controller="neural", ablation="none"):
        self.config = config.validate()
        self.device = torch.device(device)
        self.seed, self.controller, self.ablation = seed, controller, ablation
        if controller not in ("neural", "forager", "rest", "random"):
            raise ValueError(f"Unknown controller: {controller}")
        if ablation not in (
            "none",
            "disabled",
            "shuffled",
            "memory_reset",
            "no_recycling",
            "no_attacks",
            "no_cue",
            "pooled",
            "rotated",
        ):
            raise ValueError(f"Unknown ablation: {ablation}")
        if (
            (ablation == "no_cue" and config.ecology_version < 3)
            or (ablation == "no_attacks" and config.ecology_version < 2)
            or (ablation == "pooled" and config.ecology_version < 4)
        ):
            raise ValueError("Ablation requires a version containing that feature")
        self.rng = {
            name: torch.Generator(device=self.device).manual_seed(seed + 104729 * i)
            for i, name in enumerate(("world", "initial", "mutation", "birth", "evaluation"))
        }
        self.tick, self.next_id, self.spawn_accumulator = 0, 0, 0.0
        self.events = []
        self.totals = dict(
            births=0,
            deaths=0,
            blocked_births=0,
            food_spawned=0.0,
            food_expired=0.0,
            food_absorbed=0.0,
            maintenance=0.0,
            propulsion=0.0,
            reproduction=0.0,
            death_energy=0.0,
            organism_steps=0,
            distance=0.0,
            digestion_loss=0.0,
            detritus_created=0.0,
            fresh_absorbed=0.0,
            detritus_absorbed=0.0,
            predation_absorbed=0.0,
            predation_loss=0.0,
            predation_kills=0,
        )
        c = self.config
        self.sensor_angles = torch.tensor([-135, -45, 45, 135], device=self.device) * math.pi / 180
        self.fields = [SmellField(c, self.device) for _ in range(2 + (c.ecology_version >= 2))]
        if c.ecology_version >= 3:
            self.fields.append(SmellField(c, self.device))
        self.field = self.fields[0]
        self.patch_positions = self.disk(c.patches, c.diameter / 2 - c.patch_radius - c.food_radius)
        self.patch_phases = (
            self.rand((c.patches,)) * c.patch_period
            if c.ecology_version >= 3
            else torch.zeros(c.patches, device=self.device)
        )
        self.food_pos = torch.empty((0, 2), device=self.device)
        self.food_energy = torch.empty(0, device=self.device)
        self.food_expiry = torch.empty(0, dtype=torch.long, device=self.device)
        self.food_ready = torch.empty(0, dtype=torch.long, device=self.device)
        self.food_kind = torch.empty(0, dtype=torch.long, device=self.device)
        self.food_patch = torch.empty(0, dtype=torch.long, device=self.device)
        self.spawn_food(c.initial_food)
        n = c.initial_population
        self.agents = self.empty_agents(n)
        a = self.agents
        a["genome"] = self.initial_genomes(n)
        self.develop(a)
        a["energy"] = c.birth_energy * a["area"]
        for i in range(n):
            for _ in range(10000):
                p = self.disk(1, c.diameter / 2 - float(a["radius"][i]), "initial")[0]
                if ((a["pos"][:i] - p).norm(dim=1) >= a["radius"][:i] + a["radius"][i]).all():
                    a["pos"][i] = p
                    break
            else:
                raise ValueError("Could not place founders; reduce population or body size")
        a["heading"] = self.rand((n,), "initial") * 2 * math.pi - math.pi
        a["id"] = torch.arange(n, device=self.device)
        a["lineage"] = a["id"].clone()
        self.next_id = n
        self.founders = a["genome"].clone()
        self.initial_energy = a["energy"].double().sum().item()
        self.rebuild_fields()

    def empty_agents(self, n):
        a = super().empty_agents(n)
        c = self.config
        a["inputs"] = torch.zeros((n, c.input_size), device=self.device)
        a["actions"] = torch.zeros((n, c.output_size), device=self.device)
        for key in ("radius", "area", "power", "diet", "fresh_acquired", "detritus_acquired"):
            a[key] = torch.zeros(n, device=self.device)
        if c.ecology_version >= 2:
            for key in ("attack", "armor", "meat_acquired", "bitten"):
                a[key] = torch.zeros(n, device=self.device)
        if c.ecology_version >= 3:
            a["memory_tau"] = torch.zeros(n, device=self.device)
        if c.ecology_version >= 4:
            a["module_h"] = torch.zeros((n, MAX_MODULES, c.hidden_size), device=self.device)
            a["module_actions"] = torch.zeros((n, MAX_MODULES, c.output_size), device=self.device)
        return a

    def initial_genomes(self, n):
        brain = initial_brains(self.config, n, self.device, self.rng["initial"])
        traits = (self.rand((n, self.config.trait_count), "initial") * 4 - 2).clamp(
            -self.config.weight_limit, self.config.weight_limit
        )
        return torch.cat((brain, traits), 1)

    def develop(self, a):
        traits = a["genome"][:, self.config.brain_parameter_count :].sigmoid()
        a["radius"] = self.config.body_radius * (0.7 + 0.6 * traits[:, 0])
        a["area"] = (a["radius"] / self.config.body_radius).square()
        a["power"] = 0.5 + traits[:, 1]
        a["diet"] = traits[:, 2]
        if self.config.ecology_version >= 2:
            a["attack"], a["armor"] = traits[:, 3], traits[:, 4]
        if self.config.ecology_version >= 3:
            a["memory_tau"] = 0.2 + 4.8 * traits[:, 5]
        if self.config.ecology_version >= 4:
            develop_modules(self.config, a, traits)

    def patch_stock(self):
        stock = torch.zeros(self.config.patches, device=self.device)
        assigned = self.food_patch >= 0
        stock.index_add_(0, self.food_patch[assigned], self.food_energy[assigned])
        return stock

    def append_food(self, positions, energy, kind, patches=None):
        n, c = len(energy), self.config
        if not n:
            return
        self.food_pos = torch.cat((self.food_pos, positions))
        self.food_energy = torch.cat((self.food_energy, energy))
        self.food_kind = torch.cat((self.food_kind, torch.full((n,), kind, device=self.device)))
        if patches is None:
            patches = torch.full((n,), -1, device=self.device)
        self.food_patch = torch.cat((self.food_patch, patches))
        lifetime = c.food_lifetime if kind == 0 else c.detritus_lifetime
        expiry = self.tick + math.ceil(lifetime * c.physics_hz)
        self.food_expiry = torch.cat(
            (self.food_expiry, torch.full((n,), expiry, device=self.device, dtype=torch.long))
        )
        ready = self.tick + (math.ceil(c.detritus_delay * c.physics_hz) if kind else 0)
        self.food_ready = torch.cat((self.food_ready, torch.full((n,), ready, device=self.device)))

    def filter_food(self, keep):
        for key in (
            "food_pos",
            "food_energy",
            "food_expiry",
            "food_ready",
            "food_kind",
            "food_patch",
        ):
            setattr(self, key, getattr(self, key)[keep])

    def spawn_food(self, n):
        if not n:
            return
        c = self.config
        pos = self.disk(n, c.diameter / 2 - c.food_radius)
        patch = torch.randint(c.patches, (n,), generator=self.rng["world"], device=self.device)
        if c.ecology_version >= 3 and hasattr(self, "agents"):
            activity = self.patch_activity()
            if not bool(activity.sum() > 0):
                return
            patch = torch.multinomial(activity, n, replacement=True, generator=self.rng["world"])
        offsets = self.disk(n, c.patch_radius) - c.diameter / 2
        clustered = self.rand((n,)) < c.patch_fraction
        stock = self.patch_stock()
        # Rank proposals within each patch so a batch cannot overfill a vacancy.
        keep = torch.ones(n, device=self.device, dtype=torch.bool)
        for k in range(c.patches):
            rows = ((patch == k) & clustered).nonzero().flatten()
            slots = max(0, int((c.patch_capacity - float(stock[k])) / c.food_energy))
            keep[rows[slots:]] = False
        pos = torch.where(clustered[:, None], self.patch_positions[patch] + offsets, pos)
        patch = torch.where(clustered, patch, -1)
        energy = torch.full((int(keep.sum()),), c.food_energy, device=self.device)
        self.append_food(pos[keep], energy, 0, patch[keep])
        self.totals["food_spawned"] += energy.double().sum().item()

    def patch_activity(self):
        c = self.config
        phase = (self.time + self.patch_phases) % c.patch_period
        duty = c.resource_burst / c.patch_period
        return c.resource_floor + (phase < c.resource_burst).float() * (1 - c.resource_floor) / duty

    def patch_cues(self):
        c = self.config
        phase = (self.time + self.patch_phases) % c.patch_period
        end = c.patch_period - c.cue_lead
        return (phase >= end - c.cue_duration) & (phase < end)

    def rebuild_fields(self):
        for kind, field in enumerate(self.fields[:2]):
            mask = (self.food_kind == kind) & (self.food_ready <= self.tick)
            field.rebuild(self.food_pos[mask], self.food_energy[mask])
        if self.config.ecology_version >= 2:
            self.fields[2].rebuild(
                self.agents["pos"], self.config.food_energy * self.agents["area"]
            )
        if self.config.ecology_version >= 3:
            cues = self.patch_cues()
            self.fields[3].rebuild(
                self.patch_positions[cues],
                torch.full(
                    (int(cues.sum()),),
                    self.config.food_energy * self.config.cue_strength,
                    device=self.device,
                ),
            )

    def sensors(self, index):
        if self.config.ecology_version >= 4:
            values = self.module_sensors(index)
            mask = self.agents["module_mask"][index]
            return (values * mask[..., None]).sum(1) / mask.sum(1)[:, None]
        a = self.agents
        angle = a["heading"][index, None] + self.sensor_angles
        pos = a["pos"][index, None] + a["radius"][index, None, None] * torch.stack(
            (angle.cos(), angle.sin()), -1
        )
        return self.sense_positions(index, pos)

    def module_sensors(self, index):
        a = self.agents
        centers = module_centers(a, index)
        angle = a["heading"][index, None] + self.sensor_angles
        around = a["core_radius"][index, None, None] * torch.stack((angle.cos(), angle.sin()), -1)
        return self.sense_positions(index, centers[:, :, None] + around[:, None])

    def sense_positions(self, index, pos):
        a, c = self.agents, self.config
        prefix = pos.shape[:-2]
        if self.ablation == "shuffled":
            centers = self.disk(len(index), c.diameter / 2 - c.max_body_radius, "evaluation")
            shift = (centers - a["pos"][index]).reshape(len(index), *([1] * (pos.ndim - 2)), 2)
            pos = pos + shift
        channels = []
        for channel, field in enumerate(self.fields):
            smell = field.sample(pos)
            smell = smell / (smell + c.smell_scale)
            if self.ablation == "rotated":
                smell = smell.roll(2, dims=-1)
            if self.ablation == "disabled" or (self.ablation == "no_cue" and channel == 3):
                smell.zero_()
            channels.append(smell)
        for values in (a["energy"][index] / (c.max_energy * a["area"][index]), a["contact"][index]):
            channels.append(values.reshape(len(index), *([1] * len(prefix))).expand(*prefix, 1))
        return torch.cat(channels, -1)

    def update_controllers(self):
        a, c = self.agents, self.config
        index = (
            torch.arange(self.population, device=self.device)
            if self.tick % (c.physics_hz // c.controller_hz) == 0
            else a["cold"].nonzero().flatten()
        )
        if not len(index):
            return
        module_inputs = self.module_sensors(index) if c.ecology_version >= 4 else None
        if module_inputs is not None:
            mask = a["module_mask"][index]
            inputs = (module_inputs * mask[..., None]).sum(1) / mask.sum(1)[:, None]
            if self.ablation == "pooled":
                module_inputs = inputs[:, None].expand(-1, MAX_MODULES, -1)
        else:
            inputs = self.sensors(index)
        a["inputs"][index] = inputs
        a["contact"][index] = 0
        a["cold"][index] = False
        if self.ablation == "memory_reset":
            a["h"][index] = 0
            if c.ecology_version >= 4:
                a["module_h"][index] = 0
        if self.controller == "neural":
            tau = a["memory_tau"][index] if c.ecology_version >= 3 else None
            if c.ecology_version >= 4:
                count = len(index)
                genomes = a["genome"][index, None].expand(-1, MAX_MODULES, -1)
                times = tau[:, None].expand(-1, MAX_MODULES).reshape(-1)
                hidden, actions = advance(
                    c,
                    genomes.reshape(-1, c.parameter_count),
                    module_inputs.reshape(-1, c.input_size),
                    a["module_h"][index].reshape(-1, c.hidden_size),
                    times,
                )
                hidden = hidden.reshape(count, MAX_MODULES, c.hidden_size) * mask[..., None]
                actions = actions.reshape(count, MAX_MODULES, c.output_size) * mask[..., None]
                a["module_h"][index], a["module_actions"][index] = hidden, actions
                a["h"][index] = hidden.sum(1) / mask.sum(1)[:, None]
                a["actions"][index] = actions.sum(1) / mask.sum(1)[:, None]
            else:
                hidden, actions = advance(c, a["genome"][index], inputs, a["h"][index], tau)
                a["h"][index], a["actions"][index] = hidden, actions
        elif self.controller == "random":
            a["actions"][index] = self.rand((len(index), c.output_size), "evaluation")
        elif self.controller == "rest":
            a["actions"][index] = 0
        else:
            diet = a["diet"][index, None]
            smell = inputs[:, :4] * diet + inputs[:, 4:8] * (1 - diet)
            dx = (smell * self.sensor_angles.cos()).sum(1)
            dy = (smell * self.sensor_angles.sin()).sum(1)
            a["h"][index, 0] += 1 / c.controller_hz
            wander = 0.3 * (a["h"][index, 0] * 0.7).sin()
            turn = torch.where(smell.max(1).values > 0.001, torch.atan2(dy, dx) / math.pi, wander)
            turn = (2 * turn).clamp(-1, 1)
            a["actions"][index, :2] = torch.stack((0.7 - turn, 0.7 + turn), 1).clamp(0, 1)
        a["motors"][index] = a["actions"][index, :2]
        if c.ecology_version >= 4 and self.controller != "neural":
            a["module_actions"][index] = (
                a["actions"][index, None] * a["module_mask"][index, :, None]
            )

    def project_walls(self):
        a, c = self.agents, self.config
        offset = a["pos"] - c.diameter / 2
        radius = offset.norm(dim=1)
        limit = c.diameter / 2 - a["radius"]
        a["contact"][radius >= limit] = 1
        a["pos"] = c.diameter / 2 + offset * (limit / radius.clamp_min(1e-10)).clamp_max(1)[:, None]

    def overlap_pairs(self):
        a, c = self.agents, self.config
        i, j = neighbors(a["pos"], a["pos"], 2 * c.max_body_radius, c.diameter)
        valid = (i < j) & (
            (a["pos"][i] - a["pos"][j]).norm(dim=1) < a["radius"][i] + a["radius"][j]
        )
        return i[valid], j[valid]

    def move(self):
        a, c = self.agents, self.config
        speed = c.max_speed * a["power"] * a["motors"].mean(1) / a["area"].sqrt()
        if c.ecology_version >= 4:
            speed = (
                c.max_speed * a["power"] * a["motors"].mean(1) / (a["core_radius"] / c.body_radius)
            )
        if c.ecology_version >= 2:
            speed /= 1 + 0.5 * a["armor"]
        turn = math.radians(c.max_turn_degrees) * (a["motors"][:, 1] - a["motors"][:, 0])
        if c.ecology_version >= 4:
            turn = math.radians(c.max_turn_degrees) * module_turn(a)
        middle = a["heading"] + turn * c.dt / 2
        old = a["pos"].clone()
        a["pos"] += speed[:, None] * torch.stack((middle.cos(), middle.sin()), 1) * c.dt
        a["heading"] = (a["heading"] + turn * c.dt + math.pi) % (2 * math.pi) - math.pi
        self.project_walls()
        for _ in range(c.collision_iterations):
            i, j = self.overlap_pairs()
            if not len(i):
                break
            delta = a["pos"][i] - a["pos"][j]
            distance = delta.norm(dim=1)
            angle = ((a["id"][i] * 37 + a["id"][j] * 17) % 360) * math.pi / 180
            fallback = torch.stack((angle.cos(), angle.sin()), 1)
            normal = torch.where(
                (distance > 1e-8)[:, None], delta / distance.clamp_min(1e-8)[:, None], fallback
            )
            correction = 0.5 * (a["radius"][i] + a["radius"][j] - distance)[:, None] * normal
            displacement = torch.zeros_like(a["pos"])
            displacement.index_add_(0, i, correction)
            displacement.index_add_(0, j, -correction)
            displacement *= (a["radius"] / displacement.norm(dim=1).clamp_min(1e-8)).clamp_max(1)[
                :, None
            ]
            a["pos"] += displacement
            a["contact"][i], a["contact"][j] = 1, 1
            self.project_walls()
        traveled = (a["pos"] - old).norm(dim=1)
        a["distance"] += traveled
        self.totals["distance"] += traveled.double().sum().item()

    def feed(self):
        a, c = self.agents, self.config
        i, j = neighbors(a["pos"], self.food_pos, c.max_body_radius + c.food_radius, c.diameter)
        hit = (a["pos"][i] - self.food_pos[j]).norm(dim=1) <= a["radius"][i] + c.food_radius
        if c.ecology_version >= 4:
            centers = module_centers(a)
            distance = (centers[i] - self.food_pos[j, None]).norm(dim=-1)
            hit &= (
                (distance <= a["core_radius"][i, None] + c.food_radius) & a["module_mask"][i]
            ).any(1)
        hit &= self.food_ready[j] <= self.tick
        i, j = i[hit], j[hit]
        if not len(i):
            return
        fresh = self.food_kind[j] == 0
        allocation = torch.where(fresh, a["diet"][i], 1 - a["diet"][i])
        efficiency = 0.08 + 0.85 * allocation.square()
        recycle_fraction = c.detritus_fraction if self.ablation != "no_recycling" else 0.0
        recycled = fresh.float() * recycle_fraction
        # Ablating recycling dissipates that same fraction, preserving assimilation.
        assimilated = efficiency * (1 - fresh.float() * c.detritus_fraction)
        if c.ecology_version >= 2:
            assimilated *= 1 - 0.55 * a["attack"][i]
        claims = torch.bincount(j, minlength=len(self.food_energy)).clamp_min(1)
        shares = self.food_energy[j] / claims[j]
        requested = torch.zeros_like(a["energy"]).index_add_(0, i, shares * assimilated)
        room = (c.max_energy * a["area"] - a["energy"]).clamp_min(0)
        shares *= (room / requested.clamp_min(1e-20)).clamp_max(1)[i]
        received = shares * assimilated
        a["energy"].index_add_(0, i, received)
        a["acquired"].index_add_(0, i, received)
        for kind, label in ((True, "fresh"), (False, "detritus")):
            mask = fresh == kind
            a[f"{label}_acquired"].index_add_(0, i[mask], received[mask])
            self.totals[f"{label}_absorbed"] += received[mask].double().sum().item()
        consumed = torch.zeros_like(self.food_energy).index_add_(0, j, shares)
        remains = torch.zeros_like(self.food_energy).index_add_(0, j, shares * recycled)
        recycle_mask = remains > 0
        det_pos, det_energy = self.food_pos[recycle_mask].clone(), remains[recycle_mask]
        self.food_energy = (self.food_energy - consumed).clamp_min(0)
        self.totals["food_absorbed"] += received.double().sum().item()
        self.totals["detritus_created"] += det_energy.double().sum().item()
        self.totals["digestion_loss"] += (
            (shares - received - shares * recycled).double().sum().item()
        )
        self.filter_food(self.food_energy > 0)
        self.append_food(det_pos, det_energy, 1)

    def remove_dead(self):
        # Base death recording includes genome hashes and individual life histories.
        super().remove_dead()

    def hunt(self):
        a, c = self.agents, self.config
        if c.ecology_version < 2 or self.ablation == "no_attacks" or not self.population:
            return
        reach = 2 * c.max_body_radius + c.attack_reach
        i, j = neighbors(a["pos"], a["pos"], reach, c.diameter)
        delta = a["pos"][j] - a["pos"][i]
        distance = delta.norm(dim=1)
        forward = torch.stack((a["heading"].cos(), a["heading"].sin()), 1)
        facing = (delta * forward[i]).sum(1) >= 0.5 * distance
        hit = (i != j) & facing & (distance <= a["radius"][i] + a["radius"][j] + c.attack_reach)
        i, j = i[hit], j[hit]
        if not len(i):
            return
        targets = torch.bincount(i, minlength=self.population).clamp_min(1)
        bites = c.bite_rate * c.dt * a["attack"][i] * a["actions"][i, 2] / targets[i]
        if c.ecology_version >= 4:
            bites *= a["modules"][i]
        bites *= 1 - c.armor_protection * a["armor"][j]
        demanded = torch.zeros_like(a["energy"]).index_add_(0, j, bites)
        bites *= (a["energy"] / demanded.clamp_min(1e-20)).clamp_max(1)[j]
        lost = torch.zeros_like(a["energy"]).index_add_(0, j, bites)
        available = (a["energy"] - lost).clamp_min(0)
        meals = bites * c.predation_efficiency
        requested = torch.zeros_like(a["energy"]).index_add_(0, i, meals)
        room = (c.max_energy * a["area"] - available).clamp_min(0)
        meals *= (room / requested.clamp_min(1e-20)).clamp_max(1)[i]
        received = torch.zeros_like(a["energy"]).index_add_(0, i, meals)
        a["energy"] = available + received
        a["acquired"] += received
        a["meat_acquired"] += received
        a["bitten"] += lost
        self.totals["predation_absorbed"] += received.double().sum().item()
        self.totals["predation_loss"] += lost.double().sum().item() - received.double().sum().item()
        self.totals["predation_kills"] += int((a["energy"] <= 0).sum())

    def agent_record(self, index):
        result = super().agent_record(index)
        for key in ("radius", "power", "diet", "fresh_acquired", "detritus_acquired"):
            result[key] = self.agents[key][index].item()
        for key in ("attack", "armor", "meat_acquired", "bitten"):
            if key in self.agents:
                result[key] = self.agents[key][index].item()
        if "memory_tau" in self.agents:
            result["memory_tau"] = self.agents["memory_tau"][index].item()
        for key in ("modules", "body_axis", "core_radius"):
            if key in self.agents:
                result[key] = self.agents[key][index].item()
        return result

    def mutate(self, genome):
        c = self.config
        mask = self.rand(genome.shape, "mutation") < c.mutation_probability
        scale = torch.full_like(genome, c.mutation_sigma)
        start = c.brain_parameter_count
        mask[start:] = self.rand((c.trait_count,), "mutation") < c.trait_mutation_probability
        scale[start:] = c.trait_mutation_sigma
        noise = torch.randn(genome.shape, device=self.device, generator=self.rng["mutation"])
        return (genome + mask * noise * scale).clamp(-c.weight_limit, c.weight_limit)

    def reproduce(self):
        a, c = self.agents, self.config
        eligible = (
            ((a["energy"] >= c.reproduction_threshold * a["area"]) & (a["retry_tick"] <= self.tick))
            .nonzero()
            .flatten()
        )
        if not len(eligible):
            return
        order = torch.randperm(len(eligible), generator=self.rng["birth"], device=self.device)
        children = []
        for i in eligible[order].tolist():
            a["retry_tick"][i] = self.tick + math.ceil(c.birth_retry * c.physics_hz)
            if self.population + len(children) >= c.capacity:
                self.totals["blocked_births"] += 1
                continue
            child = self.empty_agents(1)
            child["genome"][0] = self.mutate(a["genome"][i])
            self.develop(child)
            child["energy"] = c.birth_energy * child["area"]
            overhead = (c.reproduction_debit - c.birth_energy) * child["area"][0]
            debit = child["energy"][0] + overhead
            if a["energy"][i] <= debit:
                self.totals["blocked_births"] += 1
                continue
            angles = self.rand((c.birth_attempts,), "birth") * 2 * math.pi
            directions = torch.stack((angles.cos(), angles.sin()), 1)
            positions = (
                a["pos"][i] + (a["radius"][i] + child["radius"][0] + c.birth_gap) * directions
            )
            occupied, radii = a["pos"], a["radius"]
            if children:
                occupied = torch.cat([occupied] + [x["pos"] for x in children])
                radii = torch.cat([radii] + [x["radius"] for x in children])
            valid = (positions - c.diameter / 2).norm(dim=1) <= c.diameter / 2 - child["radius"][0]
            q, p = neighbors(positions, occupied, 2 * c.max_body_radius, c.diameter)
            collide = (positions[q] - occupied[p]).norm(dim=1) < child["radius"][0] + radii[p]
            valid[q[collide]] = False
            options = valid.nonzero().flatten()
            if not len(options):
                self.totals["blocked_births"] += 1
                continue
            child["pos"][0] = positions[options[0]]
            child["heading"] = self.rand((1,), "birth") * 2 * math.pi - math.pi
            child["id"][0], child["parent"][0] = self.next_id, a["id"][i]
            child["lineage"][0], child["generation"][0] = a["lineage"][i], a["generation"][i] + 1
            a["energy"][i] -= debit
            a["offspring"][i] += 1
            self.totals["births"] += 1
            if c.ecology_version >= 4:
                key = f"module_births_{int(child['modules'][0])}"
                self.totals[key] = self.totals.get(key, 0) + 1
                if child["modules"][0] != a["modules"][i]:
                    self.totals["structural_births"] = self.totals.get("structural_births", 0) + 1
            self.totals["reproduction"] += float(overhead)
            self.events.append(
                dict(
                    event="birth",
                    time=(self.tick + 1) * c.dt,
                    id=self.next_id,
                    parent=int(a["id"][i]),
                    lineage=int(a["lineage"][i]),
                    generation=int(child["generation"][0]),
                    genome_hash=genome_hash(child["genome"][0]),
                    radius=float(child["radius"][0]),
                    diet=float(child["diet"][0]),
                    **({"modules": int(child["modules"][0])} if c.ecology_version >= 4 else {}),
                )
            )
            self.next_id += 1
            children.append(child)
        if children:
            self.agents = {
                key: torch.cat([value] + [x[key] for x in children]) for key, value in a.items()
            }

    def costs(self):
        a, c = self.agents, self.config
        propulsion = c.propulsion_cost * a["power"].square() * a["motors"].square().sum(1) * c.dt
        maintenance = c.basal_cost * (0.25 + 0.75 * a["area"]) * c.dt
        copies = a["modules"] if c.ecology_version >= 4 else 1.0
        if c.ecology_version >= 4:
            propulsion = (
                c.propulsion_cost
                * a["power"].square()
                * a["module_actions"][..., :2].square().sum((1, 2))
                * c.dt
            )
            maintenance += 0.04 * (copies - 1) * c.dt
        if c.ecology_version >= 2:
            maintenance += (
                (0.15 * a["attack"].square() + 0.25 * a["armor"].square()) * c.dt * copies
            )
            maintenance += c.attack_cost * a["actions"][:, 2].square() * c.dt * copies
        return maintenance, propulsion

    @torch.no_grad()
    def step(self, steps=1):
        for _ in range(steps):
            if not self.population:
                return
            c = self.config
            expired = self.food_expiry <= self.tick
            self.totals["food_expired"] += self.food_energy[expired].double().sum().item()
            self.filter_food(~expired)
            activity = self.patch_activity().mean().item() if c.ecology_version >= 3 else 1.0
            self.spawn_accumulator += c.food_rate * c.dt * activity
            n = int(self.spawn_accumulator + 1e-9)
            self.spawn_accumulator -= n
            self.spawn_food(n)
            if self.tick % (c.physics_hz // c.field_hz) == 0:
                self.rebuild_fields()
            self.update_controllers()
            self.move()
            a = self.agents
            maintenance, propulsion = self.costs()
            scale = (a["energy"] / (maintenance + propulsion).clamp_min(1e-20)).clamp_max(1)
            maintenance, propulsion = maintenance * scale, propulsion * scale
            cost = maintenance + propulsion
            a["energy"] = (a["energy"] - cost).clamp_min(0)
            a["spent"] += cost
            a["age"] += c.dt
            self.totals["maintenance"] += maintenance.double().sum().item()
            self.totals["propulsion"] += propulsion.double().sum().item()
            self.totals["organism_steps"] += self.population
            self.remove_dead()
            self.feed()
            if c.ecology_version >= 2:
                self.hunt()
                self.remove_dead()
            self.reproduce()
            self.tick += 1

    def metrics(self):
        result = super().metrics()
        a, c, n = self.agents, self.config, self.population
        result["ecology_version"] = c.ecology_version
        for key in ("radius", "power", "diet"):
            result[f"mean_{key}"] = a[key].mean().item() if n else 0
            result[f"std_{key}"] = a[key].std(correction=0).item() if n else 0
        result["grazers"] = int((a["diet"] > 0.65).sum())
        result["scavengers"] = int((a["diet"] < 0.35).sum())
        result["generalists"] = n - result["grazers"] - result["scavengers"]
        if c.ecology_version >= 2:
            result["mean_attack"] = a["attack"].mean().item() if n else 0
            result["mean_armor"] = a["armor"].mean().item() if n else 0
            fraction = a["meat_acquired"] / a["acquired"].clamp_min(1e-20)
            result["meat_eaters"] = int(((fraction > 0.2) & (a["acquired"] > 20)).sum())
        if c.ecology_version >= 3:
            result["mean_memory_tau"] = a["memory_tau"].mean().item() if n else 0
            result["cue_patches"] = int(self.patch_cues().sum())
        if c.ecology_version >= 4:
            result["module_histogram"] = [
                int((a["modules"] == k).sum()) for k in range(1, MAX_MODULES + 1)
            ]
            result["mean_modules"] = a["modules"].float().mean().item() if n else 0
        result["patch_stock"] = self.patch_stock().tolist()
        result["detritus_energy"] = self.food_energy[self.food_kind == 1].double().sum().item()
        i, j = self.overlap_pairs()
        overlap = a["radius"][i] + a["radius"][j] - (a["pos"][i] - a["pos"][j]).norm(dim=1)
        result["overlaps"] = int((overlap > 0.01).sum())
        result["max_overlap"] = float(overlap.max()) if len(overlap) else 0
        result["living_energy"] = a["energy"].double().sum().item()
        result["food_energy"] = self.food_energy.double().sum().item()
        result["energy_balance_error"] = (
            self.initial_energy
            + result["food_spawned"]
            - result["food_expired"]
            - result["maintenance"]
            - result["propulsion"]
            - result["reproduction"]
            - result["digestion_loss"]
            - result.get("predation_loss", 0.0)
            - result["death_energy"]
            - result["living_energy"]
            - result["food_energy"]
        )
        return result

    def state_dict(self):
        result = dict(
            version=2,
            config=asdict(self.config),
            seed=self.seed,
            controller=self.controller,
            ablation=self.ablation,
            tick=self.tick,
            next_id=self.next_id,
            spawn_accumulator=self.spawn_accumulator,
            initial_energy=self.initial_energy,
            seeded_from=getattr(self, "seeded_from", None),
            totals=self.totals.copy(),
            events=self.events.copy(),
            agents={k: v.clone() for k, v in self.agents.items()},
            fields=[f.grid.clone() for f in self.fields],
            rng={k: g.get_state() for k, g in self.rng.items()},
        )
        for key in (
            "food_pos",
            "food_energy",
            "food_expiry",
            "food_ready",
            "food_kind",
            "food_patch",
            "patch_positions",
            "patch_phases",
            "founders",
        ):
            result[key] = getattr(self, key).clone()
        return result

    @classmethod
    def from_state(cls, state, device="cpu"):
        if state["version"] != 2:
            raise ValueError("Unsupported ecology checkpoint version")
        self = cls.__new__(cls)
        self.config, self.device = Config.from_dict(state["config"]), torch.device(device)
        for key in (
            "seed",
            "controller",
            "ablation",
            "tick",
            "next_id",
            "spawn_accumulator",
            "initial_energy",
        ):
            setattr(self, key, state[key])
        self.totals, self.events = state["totals"].copy(), state["events"].copy()
        self.seeded_from = state.get("seeded_from")
        self.agents = {k: v.to(device).clone() for k, v in state["agents"].items()}
        for key in (
            "food_pos",
            "food_energy",
            "food_expiry",
            "food_ready",
            "food_kind",
            "food_patch",
            "patch_positions",
            "patch_phases",
            "founders",
        ):
            if key == "food_ready" and key not in state:
                value = torch.zeros_like(state["food_expiry"])
            elif key == "patch_phases" and key not in state:
                value = torch.zeros(self.config.patches)
            else:
                value = state[key]
            setattr(self, key, value.to(device).clone())
        self.sensor_angles = torch.tensor([-135, -45, 45, 135], device=self.device) * math.pi / 180
        self.fields = [SmellField(self.config, self.device) for _ in state["fields"]]
        for field, grid in zip(self.fields, state["fields"], strict=True):
            field.grid = grid.to(device).clone()
        self.field = self.fields[0]
        self.rng = {}
        for name, rng_state in state["rng"].items():
            generator = torch.Generator(device=self.device)
            try:
                generator.set_state(rng_state.cpu())
            except RuntimeError as exc:
                raise ValueError(
                    "Resume on the original CPU/CUDA device type for RNG continuity"
                ) from exc
            self.rng[name] = generator
        return self
