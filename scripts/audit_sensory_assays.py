"""Audit the matched V16 fast-feeding community transplants."""

import hashlib
import json
from pathlib import Path

import torch
from audit_foraging import audit_run

from emergent_garden.config import Config


def main():
    torch.set_num_threads(1)
    comparisons = []
    for source in (1, 2):
        root = Path(f"runs/v16-fast-feeding-assay-{source}")
        summary = json.loads((root / "summary.json").read_text())
        assert summary["completed"] and len(summary["trials"]) == 8
        pools = torch.load(root / "source-genomes.pt", weights_only=True)
        for seed in (501, 502):
            for group in ("founders", "descendants"):
                pool = pools[group]
                rng = torch.Generator().manual_seed(seed + 1299709)
                chosen = pool[torch.randint(len(pool), (192,), generator=rng)]
                modes = ("none",) if group == "founders" else ("none", "disabled", "rotated")
                for mode in modes:
                    path = root / f"{seed}-{group}-{mode}"
                    actual = torch.load(path / "founders.pt", weights_only=True)["genomes"]
                    assert torch.equal(actual, chosen), path
                    c = Config.load(path / "config.toml")
                    for name in (
                        "mutation_probability",
                        "trait_mutation_probability",
                        "node_mutation_probability",
                        "edge_mutation_probability",
                        "module_mutation_probability",
                    ):
                        assert getattr(c, name) == 0, (path, name)
                    result = audit_run(path, 360)
                    result.update(
                        source=source,
                        environment=seed,
                        group=group,
                        mode=mode,
                        chosen_genomes_sha256=hashlib.sha256(chosen.numpy().tobytes()).hexdigest(),
                    )
                    comparisons.append(result)
    report = dict(
        interpretation="Two selected evolved communities, each transplanted into two environments. "
        "Descendant genomes are exactly matched across interventions; mutation is disabled. "
        "Removing all fields also removes intensity and non-food signals. Rotating all fields "
        "tests directional dependence, but does not by itself establish food tracking or learning.",
        trials=comparisons,
    )
    output = Path("docs/results/foraging-sensory-assays.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {len(comparisons)} community transplants: {output}")


if __name__ == "__main__":
    main()
