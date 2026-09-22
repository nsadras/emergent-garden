"""Batched continuous ecology. No rendering or wall-clock time enters this module."""

import hashlib
import math
from dataclasses import asdict

import torch

from .config import Config
from .field import SmellField
from .spatial import neighbors


def genome_hash(genome):
    return hashlib.sha256(genome.detach().cpu().numpy().tobytes()).hexdigest()[:16]


class World:
    def __init__(self, config=None, seed=1, device="cpu", controller="neural", ablation="none"):
        self.config = (config or Config()).validate()
        self.device = torch.device(device)
        self.seed = seed
        self.controller = controller
        self.ablation = ablation
        if controller not in ("neural", "forager", "rest", "random"):
            raise ValueError(f"Unknown controller: {controller}")
        if ablation not in ("none", "disabled", "shuffled"):
            raise ValueError(f"Unknown sensory ablation: {ablation}")
        self.rng = {
            name: torch.Generator(device=self.device).manual_seed(seed + 104729 * i)
            for i, name in enumerate(("world", "initial", "mutation", "birth", "evaluation"))
        }
        self.tick = 0
        self.next_id = 0
        self.spawn_accumulator = 0.0
        self.next_patch = 0
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
        )
        c = self.config
        self.field = SmellField(c, self.device)
        self.sensor_angles = torch.tensor([-135, -45, 45, 135], device=self.device) * math.pi / 180
        self.patch_positions = self.disk(c.patches, c.diameter / 2 - c.patch_radius - c.food_radius)
        self.next_patch_tick = round(c.patch_period / c.patches * c.physics_hz)
        self.food_pos = torch.empty((0, 2), device=self.device)
        self.food_energy = torch.empty(0, device=self.device)
        self.food_expiry = torch.empty(0, device=self.device, dtype=torch.long)
        self.spawn_food(c.initial_food)
        self.agents = self.empty_agents(0)
        n = c.initial_population
        positions = []
        for _ in range(n):
            for _attempt in range(10000):
                p = self.disk(1, c.diameter / 2 - c.body_radius, "initial")[0]
                if (
                    not positions
                    or (
                        (torch.stack(positions) - p).square().sum(1) >= (2 * c.body_radius) ** 2
                    ).all()
                ):
                    positions.append(p)
                    break
            else:
                raise ValueError("Could not place founders; reduce population or body size")
        if n:
            self.agents = self.empty_agents(n)
            self.agents["pos"] = torch.stack(positions)
            self.agents["heading"] = self.rand((n,), "initial") * (2 * math.pi) - math.pi
            self.agents["id"] = torch.arange(n, device=self.device)
            self.agents["lineage"] = self.agents["id"].clone()
            self.agents["genome"] = self.initial_genomes(n)
        self.next_id = n
        self.founders = self.agents["genome"].clone()
        self.field.rebuild(self.food_pos, self.food_energy)

    @property
    def time(self):
        return self.tick / self.config.physics_hz

    @property
    def population(self):
        return len(self.agents["id"])

    def rand(self, shape, stream="world"):
        return torch.rand(shape, generator=self.rng[stream], device=self.device)

    def disk(self, n, radius, stream="world"):
        values = self.rand((n, 2), stream)
        angle = values[:, 0] * 2 * math.pi
        r = values[:, 1].sqrt() * radius
        return torch.stack((angle.cos() * r, angle.sin() * r), 1) + self.config.diameter / 2

    def empty_agents(self, n):
        c = self.config

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(shape, device=self.device, dtype=dtype)

        result = {
            key: zeros(n)
            for key in ("heading", "energy", "age", "contact", "acquired", "spent", "distance")
        }
        result.update(
            {
                key: zeros(n, dtype=torch.long)
                for key in ("id", "parent", "lineage", "generation", "offspring", "retry_tick")
            }
        )
        result.update(
            pos=zeros(n, 2),
            genome=zeros(n, c.parameter_count),
            h=zeros(n, c.hidden_size),
            motors=zeros(n, 2),
            inputs=zeros(n, 6),
            cold=torch.ones(n, device=self.device, dtype=torch.bool),
        )
        result["energy"].fill_(c.birth_energy)
        result["parent"].fill_(-1)
        return result

    def initial_genomes(self, n):
        h = self.config.hidden_size
        pieces = []
        for shape, fan in (((h, 6), 6), ((h, h), h), ((h,), None), ((2, h), h), ((2,), None)):
            count = math.prod(shape)
            if fan is None:
                pieces.append(torch.zeros((n, count), device=self.device))
            else:
                pieces.append(
                    torch.randn((n, count), generator=self.rng["initial"], device=self.device)
                    / math.sqrt(fan)
                )
        return torch.cat(pieces, 1).clamp(-self.config.weight_limit, self.config.weight_limit)

    def spawn_food(self, n):
        if not n:
            return
        c = self.config
        p = self.disk(n, c.diameter / 2 - c.food_radius)
        patch_idx = torch.randint(c.patches, (n,), generator=self.rng["world"], device=self.device)
        offsets = self.disk(n, c.patch_radius) - c.diameter / 2
        clustered = self.rand((n,)) < c.patch_fraction
        p = torch.where(clustered[:, None], self.patch_positions[patch_idx] + offsets, p)
        self.food_pos = torch.cat((self.food_pos, p))
        self.food_energy = torch.cat(
            (self.food_energy, torch.full((n,), c.food_energy, device=self.device))
        )
        expiry = self.tick + math.ceil(c.food_lifetime * c.physics_hz)
        self.food_expiry = torch.cat(
            (self.food_expiry, torch.full((n,), expiry, device=self.device, dtype=torch.long))
        )
        self.totals["food_spawned"] += n * c.food_energy

    def sensors(self, index):
        a, c = self.agents, self.config
        angle = a["heading"][index, None] + self.sensor_angles
        pos = a["pos"][index, None, :] + c.body_radius * torch.stack((angle.cos(), angle.sin()), -1)
        if self.ablation == "shuffled":
            centers = self.disk(len(index), c.diameter / 2 - c.body_radius, "evaluation")
            pos = pos - a["pos"][index, None, :] + centers[:, None, :]
        smell = self.field.sample(pos)
        smell = smell / (smell + c.smell_scale)
        if self.ablation == "disabled":
            smell.zero_()
        return torch.cat(
            (smell, (a["energy"][index] / c.max_energy)[:, None], a["contact"][index, None]), 1
        )

    def update_controllers(self):
        a, c = self.agents, self.config
        if self.tick % (c.physics_hz // c.controller_hz) == 0:
            index = torch.arange(self.population, device=self.device)
        else:
            index = a["cold"].nonzero().flatten()
        if not len(index):
            return
        inputs = self.sensors(index)
        a["inputs"][index] = inputs
        a["contact"][index] = 0
        a["cold"][index] = False
        if self.controller == "rest":
            a["motors"][index] = 0
            return
        if self.controller == "random":
            a["motors"][index] = self.rand((len(index), 2), "evaluation")
            return
        if self.controller == "forager":
            smell = inputs[:, :4]
            dx = (smell * self.sensor_angles.cos()).sum(1)
            dy = (smell * self.sensor_angles.sin()).sum(1)
            direction = torch.atan2(dy, dx)
            a["h"][index, 0] += 1 / c.controller_hz
            wander = 0.3 * (a["h"][index, 0] * 0.7).sin()
            turn = torch.where(smell.max(1).values > 0.001, direction / math.pi, wander)
            turn = (2 * turn).clamp(-1, 1)
            motors = torch.stack((0.7 - turn, 0.7 + turn), 1).clamp(0, 1)
            a["motors"][index] = motors
            return
        g, h = a["genome"][index], c.hidden_size
        offset = 0

        def take(size, shape):
            nonlocal offset
            value = g[:, offset : offset + size].reshape(len(index), *shape)
            offset += size
            return value

        w_in = take(h * 6, (h, 6))
        w_rec = take(h * h, (h, h))
        bias = take(h, (h,))
        w_out = take(2 * h, (2, h))
        out_bias = take(2, (2,))
        hidden = a["h"][index]
        drive = (w_in @ inputs[..., None]).squeeze(-1)
        drive += (w_rec @ hidden[..., None]).squeeze(-1) + bias
        alpha = 1 - math.exp(-1 / (c.controller_hz * c.neural_tau))
        hidden = (1 - alpha) * hidden + alpha * drive.tanh()
        a["h"][index] = hidden
        a["motors"][index] = ((w_out @ hidden[..., None]).squeeze(-1) + out_bias).sigmoid()

    def project_walls(self):
        a, c = self.agents, self.config
        offset = a["pos"] - c.diameter / 2
        radius = offset.norm(dim=1)
        hit = radius >= c.diameter / 2 - c.body_radius
        a["contact"][hit] = 1
        factor = ((c.diameter / 2 - c.body_radius) / radius.clamp_min(1e-10)).clamp_max(1)
        a["pos"] = c.diameter / 2 + offset * factor[:, None]

    def overlap_pairs(self):
        c = self.config
        p = self.agents["pos"]
        i, j = neighbors(p, p, 2 * c.body_radius, c.diameter)
        valid = (i < j) & ((p[i] - p[j]).square().sum(1) < (2 * c.body_radius) ** 2)
        return i[valid], j[valid]

    def move(self):
        a, c = self.agents, self.config
        speed = c.max_speed * a["motors"].mean(1)
        turn = math.radians(c.max_turn_degrees) * (a["motors"][:, 1] - a["motors"][:, 0])
        middle = a["heading"] + turn * c.dt / 2
        old_pos = a["pos"].clone()
        a["pos"] += speed[:, None] * torch.stack((middle.cos(), middle.sin()), 1) * c.dt
        a["heading"] = (a["heading"] + turn * c.dt + math.pi) % (2 * math.pi) - math.pi
        self.project_walls()
        for _ in range(c.collision_iterations):
            i, j = self.overlap_pairs()
            if not len(i):
                break
            delta = a["pos"][i] - a["pos"][j]
            distance = delta.norm(dim=1)
            # Stable antisymmetric direction for exactly coincident centers.
            angle = ((a["id"][i] * 37 + a["id"][j] * 17) % 360) * math.pi / 180
            fallback = torch.stack((angle.cos(), angle.sin()), 1)
            normal = torch.where(
                (distance > 1e-8)[:, None], delta / distance.clamp_min(1e-8)[:, None], fallback
            )
            correction = 0.5 * (2 * c.body_radius - distance)[:, None] * normal
            displacement = torch.zeros_like(a["pos"])
            displacement.index_add_(0, i, correction)
            displacement.index_add_(0, j, -correction)
            length = displacement.norm(dim=1).clamp_min(1e-8)
            displacement *= (c.body_radius / length).clamp_max(1)[:, None]
            a["pos"] += displacement
            a["contact"][i] = 1
            a["contact"][j] = 1
            self.project_walls()
        traveled = (a["pos"] - old_pos).norm(dim=1)
        a["distance"] += traveled
        self.totals["distance"] += traveled.sum().item()

    def feed(self):
        a, c = self.agents, self.config
        radius = c.body_radius + c.food_radius
        i, j = neighbors(a["pos"], self.food_pos, radius, c.diameter)
        hit = (a["pos"][i] - self.food_pos[j]).square().sum(1) <= radius**2
        i, j = i[hit], j[hit]
        if not len(i):
            return
        claims = torch.bincount(j, minlength=len(self.food_energy)).clamp_min(1)
        shares = self.food_energy[j] / claims[j]
        requested = torch.zeros(self.population, device=self.device).index_add_(0, i, shares)
        factor = ((c.max_energy - a["energy"]).clamp_min(0) / requested.clamp_min(1e-20)).clamp_max(
            1
        )
        shares *= factor[i]
        received = torch.zeros_like(a["energy"]).index_add_(0, i, shares)
        consumed = torch.zeros_like(self.food_energy).index_add_(0, j, shares)
        a["energy"] += received
        a["acquired"] += received
        self.food_energy = (self.food_energy - consumed).clamp_min(0)
        self.totals["food_absorbed"] += received.sum().item()
        keep = self.food_energy > 0
        self.food_pos, self.food_energy, self.food_expiry = (
            t[keep] for t in (self.food_pos, self.food_energy, self.food_expiry)
        )

    def remove_dead(self):
        a = self.agents
        dead = a["energy"] <= 0
        for index in dead.nonzero().flatten().tolist():
            record = self.agent_record(index)
            self.events.append(dict(event="death", time=self.time, **record))
        self.totals["deaths"] += int(dead.sum())
        self.totals["death_energy"] += a["energy"][dead].clamp_min(0).sum().item()
        if dead.any():
            self.agents = {key: value[~dead] for key, value in a.items()}

    def agent_record(self, index):
        a = self.agents
        result = {
            key: a[key][index].item()
            for key in (
                "id",
                "parent",
                "lineage",
                "generation",
                "age",
                "energy",
                "offspring",
                "acquired",
                "spent",
                "distance",
            )
        }
        result["genome_hash"] = genome_hash(a["genome"][index])
        return result

    def reproduce(self):
        a, c = self.agents, self.config
        eligible = (
            ((a["energy"] >= c.reproduction_threshold) & (a["retry_tick"] <= self.tick))
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
            angles = self.rand((c.birth_attempts,), "birth") * 2 * math.pi
            directions = torch.stack((angles.cos(), angles.sin()), 1)
            positions = a["pos"][i] + (2 * c.body_radius + c.birth_gap) * directions
            occupied = a["pos"]
            if children:
                occupied = torch.cat((occupied, torch.cat([child["pos"] for child in children])))
            valid = (positions - c.diameter / 2).norm(dim=1) <= c.diameter / 2 - c.body_radius
            q, p = neighbors(positions, occupied, 2 * c.body_radius, c.diameter)
            collide = (positions[q] - occupied[p]).norm(dim=1) < 2 * c.body_radius
            valid[q[collide]] = False
            options = valid.nonzero().flatten()
            if not len(options):
                self.totals["blocked_births"] += 1
                continue
            child = self.empty_agents(1)
            child["pos"][0] = positions[options[0]]
            child["heading"] = self.rand((1,), "birth") * 2 * math.pi - math.pi
            genome = a["genome"][i].clone()
            mask = self.rand(genome.shape, "mutation") < c.mutation_probability
            noise = (
                torch.randn(genome.shape, generator=self.rng["mutation"], device=self.device)
                * c.mutation_sigma
            )
            child["genome"][0] = (genome + mask * noise).clamp(-c.weight_limit, c.weight_limit)
            child["id"][0] = self.next_id
            child["parent"][0] = a["id"][i]
            child["lineage"][0] = a["lineage"][i]
            child["generation"][0] = a["generation"][i] + 1
            a["energy"][i] -= c.reproduction_debit
            a["offspring"][i] += 1
            self.totals["births"] += 1
            self.totals["reproduction"] += c.reproduction_debit - c.birth_energy
            self.events.append(
                dict(
                    event="birth",
                    time=(self.tick + 1) * c.dt,
                    id=self.next_id,
                    parent=int(a["id"][i]),
                    lineage=int(a["lineage"][i]),
                    generation=int(child["generation"][0]),
                    genome_hash=genome_hash(child["genome"][0]),
                )
            )
            self.next_id += 1
            children.append(child)
        if children:
            self.agents = {
                key: torch.cat([value] + [child[key] for child in children])
                for key, value in a.items()
            }

    @torch.no_grad()
    def step(self, steps=1):
        for _ in range(steps):
            if not self.population:
                return
            c = self.config
            expired = self.food_expiry <= self.tick
            self.totals["food_expired"] += self.food_energy[expired].sum().item()
            keep = ~expired
            self.food_pos, self.food_energy, self.food_expiry = (
                t[keep] for t in (self.food_pos, self.food_energy, self.food_expiry)
            )
            if self.tick >= self.next_patch_tick:
                self.patch_positions[self.next_patch] = self.disk(
                    1, c.diameter / 2 - c.patch_radius - c.food_radius
                )[0]
                self.next_patch = (self.next_patch + 1) % c.patches
                self.next_patch_tick += max(1, round(c.patch_period / c.patches * c.physics_hz))
            self.spawn_accumulator += c.food_rate * c.dt
            n = int(self.spawn_accumulator + 1e-9)
            self.spawn_accumulator -= n
            self.spawn_food(n)
            if self.tick % (c.physics_hz // c.field_hz) == 0:
                self.field.rebuild(self.food_pos, self.food_energy)
            self.update_controllers()
            self.move()
            a = self.agents
            propulsion = c.propulsion_cost * a["motors"].square().sum(1) * c.dt
            maintenance = torch.full_like(propulsion, c.basal_cost * c.dt)
            # Charge only energy actually present on the final starvation tick.
            scale = (a["energy"] / (propulsion + maintenance).clamp_min(1e-20)).clamp_max(1)
            propulsion *= scale
            maintenance *= scale
            cost = propulsion + maintenance
            a["energy"] = (a["energy"] - cost).clamp_min(0)
            a["spent"] += cost
            a["age"] += c.dt
            self.totals["organism_steps"] += self.population
            self.totals["maintenance"] += maintenance.sum().item()
            self.totals["propulsion"] += propulsion.sum().item()
            self.remove_dead()
            self.feed()
            self.reproduce()
            self.tick += 1

    def metrics(self):
        a = self.agents
        n = self.population
        i, j = self.overlap_pairs()
        overlap = 2 * self.config.body_radius - (a["pos"][i] - a["pos"][j]).norm(dim=1)
        finite = all(
            torch.isfinite(a[key]).all().item() for key in ("pos", "h", "energy", "genome")
        )
        if not finite:
            raise FloatingPointError("Nonfinite organism state")
        result = dict(
            time=self.time,
            tick=self.tick,
            population=n,
            food_count=len(self.food_energy),
            food_energy=self.food_energy.sum().item(),
            living_energy=a["energy"].sum().item(),
            mean_age=a["age"].mean().item() if n else 0,
            generation_max=int(a["generation"].max()) if n else 0,
            generation_mean=a["generation"].float().mean().item() if n else 0,
            lineages=len(a["lineage"].unique()),
            genome_variance=a["genome"].var(0, correction=0).mean().item() if n else 0,
            contacts=int((a["contact"] > 0).sum()),
            overlaps=int((overlap > 0.01).sum()),
            max_overlap=overlap.max().item() if len(overlap) else 0,
            **self.totals,
        )
        result["energy_balance_error"] = (
            self.config.initial_population * self.config.birth_energy
            + result["food_spawned"]
            - result["food_expired"]
            - result["maintenance"]
            - result["propulsion"]
            - result["reproduction"]
            - result["living_energy"]
            - result["food_energy"]
        )
        return result

    def state_dict(self):
        return dict(
            version=1,
            config=asdict(self.config),
            seed=self.seed,
            controller=self.controller,
            ablation=self.ablation,
            tick=self.tick,
            next_id=self.next_id,
            next_patch=self.next_patch,
            next_patch_tick=self.next_patch_tick,
            spawn_accumulator=self.spawn_accumulator,
            totals=self.totals.copy(),
            agents={k: v.clone() for k, v in self.agents.items()},
            food_pos=self.food_pos,
            food_energy=self.food_energy,
            food_expiry=self.food_expiry,
            patch_positions=self.patch_positions,
            field=self.field.grid,
            founders=self.founders,
            rng={k: g.get_state() for k, g in self.rng.items()},
            events=self.events.copy(),
        )

    @classmethod
    def from_state(cls, state, device="cpu"):
        if state["version"] != 1:
            raise ValueError("Unsupported checkpoint version")
        # Skip initialization entirely: resuming must not consume any random numbers.
        self = cls.__new__(cls)
        self.config = Config.from_dict(state["config"])
        self.device = torch.device(device)
        for key in (
            "seed",
            "controller",
            "ablation",
            "tick",
            "next_id",
            "next_patch",
            "next_patch_tick",
            "spawn_accumulator",
            "totals",
            "events",
        ):
            setattr(self, key, state[key])
        self.agents = {k: v.to(self.device).clone() for k, v in state["agents"].items()}
        for key in ("food_pos", "food_energy", "food_expiry", "patch_positions", "founders"):
            setattr(self, key, state[key].to(self.device).clone())
        self.field = SmellField(self.config, self.device)
        self.field.grid = state["field"].to(self.device).clone()
        self.sensor_angles = torch.tensor([-135, -45, 45, 135], device=self.device) * math.pi / 180
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
