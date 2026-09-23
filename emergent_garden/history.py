"""Read complete recorded histories, checking continuity across saved resumes."""

import json
from pathlib import Path

PERFORMANCE_KEYS = {
    "wall_seconds",
    "speed",
    "organism_steps_per_second",
    "gpu_allocated_bytes",
}


def read_history(path, follow_resumes=False):
    """Return chronological segments, measurements, and events from local records.

    Resume paths use the working-directory-relative convention recorded by
    RunStore. A missing ancestor, changed configuration, mismatched boundary,
    or resume cycle is an error; incomplete histories are never silently joined.
    Source revisions may change, but recorded physical state must match exactly
    at the boundary. This checks recorded evidence, not the code between samples.
    """
    segments, seen = [], set()
    path = Path(path)
    while True:
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"Resume history contains a cycle: {path}")
        seen.add(resolved)
        meta = json.loads((path / "metadata.json").read_text())
        rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        if not rows or any(a["tick"] > b["tick"] for a, b in zip(rows, rows[1:], strict=False)):
            raise ValueError(f"Missing or out-of-order measurements: {path}")
        segments.append(dict(path=path, metadata=meta, metrics=rows))
        if rows[0]["tick"] == 0 or not follow_resumes:
            break
        if not meta.get("resumed_from"):
            raise ValueError(f"Missing initialization history or resume provenance: {path}")
        path = Path(meta["resumed_from"]).parent
    segments.reverse()
    for parent, child in zip(segments, segments[1:], strict=False):
        a, b = parent["metrics"][-1], child["metrics"][0]
        physical_a = {k: v for k, v in a.items() if k not in PERFORMANCE_KEYS}
        physical_b = {k: v for k, v in b.items() if k not in PERFORMANCE_KEYS}
        if physical_a != physical_b:
            raise ValueError(f"Physical measurements differ at resume boundary: {child['path']}")
        for key in ("seed", "controller", "ablation", "seeded_from"):
            if parent["metadata"].get(key) != child["metadata"].get(key):
                raise ValueError(f"Resume changed {key}: {child['path']}")
        if (parent["path"] / "config.toml").read_bytes() != (
            child["path"] / "config.toml"
        ).read_bytes():
            raise ValueError(f"Resume changed configuration: {child['path']}")
    metrics, events = [], []
    for segment in segments:
        rows = segment["metrics"]
        metrics.extend(rows)
        for line in (segment["path"] / "events.jsonl").read_text().splitlines():
            event = json.loads(line)
            if not rows[0]["time"] <= event["time"] <= rows[-1]["time"]:
                raise ValueError(f"Event outside measured segment: {segment['path']}")
            events.append(event)
    return segments, metrics, events
