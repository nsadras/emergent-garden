"""Decode every frame and check a run's recording against its saved frame count."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    video = args.run / "timelapse.mp4"
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(frames)
    width, height = metadata["size"]
    count, first, last = 0, None, None
    for frame in frames:
        assert len(frame) == width * height * 3
        if first is None:
            first = hashlib.sha256(frame).hexdigest()
        count += 1
        last = frame
    assert count == summary["video_frames"] and count > 0
    report = dict(
        path=str(video),
        metadata=metadata,
        decoded_frames=count,
        declared_frames=summary["video_frames"],
        first_frame_sha256=first,
        last_frame_sha256=hashlib.sha256(last).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Decoded all {count} frames: {width}×{height}, {metadata['fps']:g} FPS")


if __name__ == "__main__":
    main()
