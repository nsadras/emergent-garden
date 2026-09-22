"""Run, watch, resume, record, and evaluate the same ecology."""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from .config import Config
from .experiments import calibration, evaluate
from .runtime import StopFlag
from .storage import RunStore, load_checkpoint, runtime_metadata
from .world import World


def choose_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable. Use --device cpu or check the NVIDIA driver/runtime."
        )
    return requested


def run(args, stop):
    device = choose_device(args.device)
    world = (
        load_checkpoint(args.resume, device)
        if args.resume
        else World(Config.load(args.config), args.seed, device, args.controller)
    )
    output = args.output or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    store = RunStore(output, world, args.resume)
    viewer = recorder = None
    from .viewer import Recorder, Renderer, Viewer

    target = (
        world.tick + math.ceil(args.seconds * world.config.physics_hz) if args.seconds else None
    )
    start = time.monotonic()
    last_checkpoint = last_report = start
    next_metric = world.time + world.config.metrics_period
    last_frame = start
    accumulator = 0.0
    reason = "duration"
    failure = None
    try:
        if args.view:
            viewer = Viewer(world.config.viewer_size)
        if args.record:
            video_speed = args.video_speed or world.config.video_speed
            if video_speed / world.config.video_fps < world.config.dt:
                raise ValueError("Video sampling interval must be at least one physics tick")
            recorder = Recorder(
                store.path / "timelapse.mp4",
                world.config.viewer_size,
                world.config.video_fps,
                video_speed,
                world.time,
            )
            recorder.observe(world)
        store.measure(world)
        print(
            f"Run: {store.path}\nDevice: {device}; seed: {world.seed}; "
            f"population: {world.population}",
            flush=True,
        )
        while world.population and not stop.requested and (target is None or world.tick < target):
            if viewer:
                if not viewer.events(world):
                    reason = "window_closed"
                    break
                now = time.monotonic()
                elapsed = min(now - last_frame, 0.25)
                last_frame = now
                accumulator = (
                    accumulator + elapsed * viewer.speed * world.config.physics_hz
                    if not viewer.paused
                    else 0
                )
                # Keep input responsive even if the requested speed exceeds throughput.
                steps = min(int(accumulator), 120)
                accumulator -= steps
                if viewer.paused:
                    steps = 0
            else:
                steps = world.config.physics_hz
            if target is not None:
                steps = min(steps, target - world.tick)
            for _ in range(steps):
                if not world.population or stop.requested:
                    break
                world.step()
                if recorder:
                    recorder.observe(world)
                if world.time + 1e-9 >= next_metric:
                    store.measure(world)
                    next_metric += world.config.metrics_period
            if viewer:
                viewer.present(world)
                frame_remaining = 1 / world.config.viewer_fps - (time.monotonic() - last_frame)
                if frame_remaining > 0:
                    time.sleep(frame_remaining)
            now = time.monotonic()
            if now - last_checkpoint >= world.config.checkpoint_wall_seconds:
                store.checkpoint(world)
                last_checkpoint = now
            if now - last_report >= 10:
                print(
                    f"t={world.time:.1f}s population={world.population} "
                    f"births={world.totals['births']} deaths={world.totals['deaths']}",
                    flush=True,
                )
                last_report = now
        if stop.requested:
            reason = "interrupted"
        elif not world.population:
            reason = "extinction"
    except KeyboardInterrupt:
        reason = "interrupted"
        print("\nSaving checkpoint...", flush=True)
    except Exception as exc:
        reason = "error"
        failure = repr(exc)
        raise
    finally:
        store.checkpoint(world)
        metric = store.measure(world)
        metric["stop_reason"] = reason
        if failure:
            metric["error"] = failure
        if recorder:
            recorder.close()
            metric["video_frames"] = recorder.frames
            metric["video_dropped_frames"] = 0
        if viewer:
            viewer.close()
        Renderer(world.config.viewer_size).save(world, store.path / "preview.png")
        (store.path / "summary.json").write_text(json.dumps(metric, indent=2))
        store.close()
    print(
        f"Finished ({reason}): {world.time:.1f}s, {world.population} creatures, "
        f"{metric['births']} births, generation {metric['generation_max']}, "
        f"{metric['speed']:.1f}x simulated speed\nCheckpoint: {store.path / 'latest.pt'}"
    )


def parser():
    root = argparse.ArgumentParser(description="Emergent Garden artificial-life simulator")
    root.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    sub = root.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Evolve a shared dish; optionally view or record it")
    run_parser.add_argument("--config", type=Path)
    run_parser.add_argument("--seed", type=int, default=1)
    run_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run_parser.add_argument(
        "--seconds",
        type=float,
        default=600,
        help="Additional simulated seconds; 0 runs until stopped or extinct",
    )
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--resume", type=Path)
    run_parser.add_argument("--view", action="store_true")
    run_parser.add_argument("--record", action="store_true")
    run_parser.add_argument("--video-speed", type=float)
    run_parser.add_argument(
        "--controller",
        choices=["neural", "forager", "rest", "random"],
        default="neural",
        help="Other controllers are diagnostics, not evolution",
    )
    check = sub.add_parser("doctor", help="Check Python, PyTorch, CUDA, and the device")
    check.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    init = sub.add_parser("config", help="Write the default TOML configuration")
    init.add_argument("path", type=Path)
    calibrate = sub.add_parser("calibrate", help="Run independent short ecology experiments")
    calibrate.add_argument("--config", type=Path)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    calibrate.add_argument("--seconds", type=float, default=600)
    calibrate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    calibrate.add_argument(
        "--controller", choices=["neural", "forager", "rest", "random"], default="neural"
    )
    evaluation = sub.add_parser(
        "evaluate", help="Compare founders, descendants, and smell ablations"
    )
    evaluation.add_argument("run", type=Path)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--genomes", type=int, default=32)
    evaluation.add_argument("--seeds", type=int, nargs="+", default=list(range(10001, 10009)))
    evaluation.add_argument("--seconds", type=float, default=120)
    evaluation.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return root


def main():
    args = parser().parse_args()
    stop = StopFlag()
    try:
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(args.threads)
        if hasattr(args, "seconds") and (not math.isfinite(args.seconds) or args.seconds < 0):
            raise ValueError("--seconds must be finite and nonnegative")
        if args.command in ("calibrate", "evaluate") and args.seconds <= 0:
            raise ValueError("Experiments require a positive duration")
        if args.command == "config":
            if args.path.exists():
                raise ValueError(f"Refusing to overwrite existing config: {args.path}")
            Config().save(args.path)
            print(args.path)
        elif args.command == "doctor":
            device = choose_device(args.device)
            world = World(Config(initial_population=2, initial_food=4, grid_size=32), device=device)
            world.step(6)
            print(json.dumps(runtime_metadata(world), indent=2))
        elif args.command == "run":
            if args.resume and args.config:
                raise ValueError("A resumed run uses the checkpoint configuration; omit --config")
            if args.video_speed is not None and (
                not math.isfinite(args.video_speed) or args.video_speed <= 0
            ):
                raise ValueError("--video-speed must be finite and positive")
            run(args, stop)
        elif args.command == "calibrate":
            calibration(
                Config.load(args.config),
                args.output,
                args.seeds,
                args.seconds,
                choose_device(args.device),
                args.controller,
                stop=stop,
            )
        elif args.command == "evaluate":
            if args.genomes < 1:
                raise ValueError("--genomes must be positive")
            report = evaluate(
                args.run,
                args.output,
                args.genomes,
                args.seeds,
                args.seconds,
                choose_device(args.device),
                stop=stop,
            )
            print(json.dumps(report["groups"], indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        stop.close()
    if return_code:
        raise SystemExit(return_code)
