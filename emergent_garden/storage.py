"""Atomic checkpoints and append-only experiment records."""

import hashlib
import io
import json
import os
import platform
import subprocess
import time
import zipfile
from pathlib import Path

import torch

from .world import World


def capture_sources():
    """Capture once at process import, before repeated experiments can span edits."""
    package = Path(__file__).resolve().parent
    sources = {p.name: p.read_bytes() for p in sorted(package.glob("*.py"))}
    digest = hashlib.sha256()
    for name, data in sources.items():
        digest.update(name.encode())
        digest.update(data)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, data in sources.items():
            output.writestr(f"emergent_garden/{name}", data)
        for name in ("pyproject.toml", "uv.lock"):
            path = package.parent / name
            if path.exists():
                output.writestr(name, path.read_bytes())
    try:
        revision = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=package, capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or "uncommitted"
        )
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=package,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.TimeoutExpired):
        revision, dirty = "unknown", True
    return digest.hexdigest(), archive.getvalue(), revision, dirty


SOURCE_SHA256, SOURCE_ARCHIVE, SOURCE_REVISION, SOURCE_DIRTY = capture_sources()


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
    info = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=str(torch.__version__),
        cuda_build=torch.version.cuda,
        device=str(world.device),
        seed=world.seed,
        revision=SOURCE_REVISION,
        dirty=SOURCE_DIRTY,
        source_sha256=SOURCE_SHA256,
        source_capture="process_import",
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
        (self.path / "source.zip").write_bytes(SOURCE_ARCHIVE)
        atomic_save(dict(version=1, genomes=world.founders.cpu()), self.path / "founders.pt")
        self.metrics_file = (self.path / "metrics.jsonl").open("w")
        self.events_file = (self.path / "events.jsonl").open("w")
        self.started = time.perf_counter()
        self.initial_time = world.time
        self.initial_steps = world.totals["organism_steps"]
        period = world.config.archive_sim_seconds
        self.archive_period = max(1, round(period * world.config.physics_hz)) if period else None
        self.next_archive_tick = (
            (world.tick // self.archive_period + 1) * self.archive_period
            if self.archive_period
            else None
        )

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
        if self.next_archive_tick is not None and world.tick >= self.next_archive_tick:
            save_checkpoint(world, self.path / "checkpoints" / f"tick-{world.tick:012d}.pt")
            self.next_archive_tick = (world.tick // self.archive_period + 1) * self.archive_period
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
