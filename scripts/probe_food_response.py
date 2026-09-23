"""Measure inherited turning responses to food intensity and side differences.

All stimuli are synthetic bounded inputs. This is a circuit diagnostic; it does
not measure successful searching, whole-body steering, or learned navigation.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from emergent_garden.config import Config
from emergent_garden.probes import food_response_probe
from emergent_garden.storage import SOURCE_ARCHIVE, SOURCE_SHA256, atomic_save
from emergent_garden.world import genome_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--genomes", type=int, default=64)
    args = parser.parse_args()
    if args.genomes < 1:
        parser.error("Genome count must be positive")
    torch.set_num_threads(1)
    c = Config.load(args.run / "config.toml")
    founders = torch.load(args.run / "founders.pt", map_location="cpu", weights_only=True)[
        "genomes"
    ]
    final = torch.load(args.run / "population.pt", map_location="cpu", weights_only=True)
    descendants = final["genomes"][final["generations"] > 0]
    if not len(descendants):
        parser.error("A living descendant population is required")
    args.output.mkdir(parents=True, exist_ok=False)
    experiment = Path(__file__).read_bytes()
    (args.output / "source.zip").write_bytes(SOURCE_ARCHIVE)
    (args.output / "experiment.py").write_bytes(experiment)
    groups, selected = {}, {}
    for group, pool in (("founders", founders), ("descendants", descendants)):
        rng = torch.Generator().manual_seed(982451653)
        chosen = pool[torch.randperm(len(pool), generator=rng)[: args.genomes]]
        selected[group] = chosen
        groups[group] = dict(
            genomes=[genome_hash(g) for g in chosen],
            trials=[
                food_response_probe(c, chosen, field, mean)
                for field in ("fresh", "detritus")
                for mean in (0.1, 0.4, 0.8)
            ],
        )
    atomic_save(selected, args.output / "genomes.pt")
    report = dict(
        source=str(args.run),
        source_metadata=json.loads((args.run / "metadata.json").read_text()),
        config=asdict(c),
        source_sha256=SOURCE_SHA256,
        experiment_sha256=hashlib.sha256(experiment).hexdigest(),
        interpretation=__doc__,
        groups=groups,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
