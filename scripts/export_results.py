"""Commit compact numerical evidence while keeping large simulation artifacts local.

Run with uv run python scripts/export_results.py OUTPUT.json RUN_DIRECTORY ...
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    entries = []
    for root in args.runs:
        candidates = [root] if (root / "metrics.jsonl").exists() else sorted(root.glob("*/"))
        for path in candidates:
            metrics = path / "metrics.jsonl"
            if not metrics.exists():
                continue
            rows = [json.loads(line) for line in metrics.read_text().splitlines()]
            meta = json.loads((path / "metadata.json").read_text())
            entry = dict(
                path=str(path),
                metadata=meta,
                final=rows[-1],
                checkpoints=[r for r in rows if r["tick"] == 0 or r["time"] % 120 == 0],
            )
            origins = path / "origins.jsonl"
            if origins.exists():
                history = [json.loads(line) for line in origins.read_text().splitlines()]
                entry["ancestry"] = dict(
                    final=history[-1],
                    checkpoints=[r for r in history if r["time"] % 120 == 0],
                    first_absent=[
                        next(
                            (r["time"] for r in history if r["groups"][k]["population"] == 0), None
                        )
                        for k in range(len(history[0]["groups"]))
                    ],
                )
            entries.append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Exported {len(entries)} runs to {args.output}")


if __name__ == "__main__":
    main()
