"""Atomic checkpoints and append-only experiment records."""

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from .world import World


def atomic_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_checkpoint(world, path):
    atomic_save(world.state_dict(), path)


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return World.from_state(state, device)


def runtime_metadata(world):
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    try:
        revision = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or "uncommitted"
        )
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        )
    except (OSError, subprocess.TimeoutExpired):
        revision, dirty = "unknown", True
    info = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=str(torch.__version__),
        cuda_build=torch.version.cuda,
        device=str(world.device),
        seed=world.seed,
        revision=revision,
        dirty=dirty,
        source_sha256=digest.hexdigest(),
        controller=world.controller,
        ablation=world.ablation,
        seeded_from=getattr(world, "seeded_from", None),
        created=time.time(),
    )
    if world.device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(world.device)
    try:
        info["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        info["driver"] = "unavailable"
    return info


class RunStore:
    def __init__(self, path, world, resumed_from=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        world.config.save(self.path / "config.toml")
        metadata = runtime_metadata(world)
        metadata["resumed_from"] = str(resumed_from) if resumed_from else None
        (self.path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        atomic_save(dict(version=1, genomes=world.founders.cpu()), self.path / "founders.pt")
        self.metrics_file = (self.path / "metrics.jsonl").open("w")
        self.events_file = (self.path / "events.jsonl").open("w")
        self.started = time.perf_counter()
        self.initial_time = world.time
        self.initial_steps = world.totals["organism_steps"]

    def write_events(self, world):
        for event in world.events:
            self.events_file.write(json.dumps(event) + "\n")
        world.events.clear()
        self.events_file.flush()

    def measure(self, world):
        metric = world.metrics()
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        metric["wall_seconds"] = elapsed
        metric["speed"] = (world.time - self.initial_time) / elapsed
        metric["organism_steps_per_second"] = (
            world.totals["organism_steps"] - self.initial_steps
        ) / elapsed
        if world.device.type == "cuda":
            metric["gpu_allocated_bytes"] = torch.cuda.memory_allocated(world.device)
        self.metrics_file.write(json.dumps(metric, allow_nan=False) + "\n")
        self.metrics_file.flush()
        self.write_events(world)
        return metric

    def checkpoint(self, world):
        self.write_events(world)
        save_checkpoint(world, self.path / "latest.pt")
        atomic_save(
            dict(
                version=1,
                genomes=world.agents["genome"].cpu(),
                generations=world.agents["generation"].cpu(),
                ids=world.agents["id"].cpu(),
                acquired=world.agents["acquired"].cpu(),
            ),
            self.path / "population.pt",
        )

    def close(self):
        self.metrics_file.close()
        self.events_file.close()
