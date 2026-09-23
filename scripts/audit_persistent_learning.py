"""Audit same-genotype V21 learning tests, including shuffled-return controls."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from audit_circuits import same_state
from audit_learning_assays import check_pool

from emergent_garden.config import Config


def main():
    torch.set_num_threads(1)
    trials, pairs = [], []
    for source in (1, 2, 3):
        original = Path(f"runs/v21-correlated-learning-pilot/seed-{source}")
        root = Path(f"runs/v21-learning-assay-{source}")
        report = json.loads((root / "summary.json").read_text())
        assert report["completed"] and len(report["trials"]) == 8, root
        population = torch.load(original / "population.pt", weights_only=True)
        expected = dict(
            founders=torch.load(original / "founders.pt", weights_only=True)["genomes"],
            descendants=population["genomes"][population["generations"] > 0],
        )
        same_state(expected, torch.load(root / "source-genomes.pt", weights_only=True))
        config = replace(
            Config.load(original / "config.toml"),
            mutation_probability=0,
            trait_mutation_probability=0,
            node_mutation_probability=0,
            edge_mutation_probability=0,
            module_mutation_probability=0,
        )
        assert config.motor_noise_tau == 2
        assert asdict(Config.load(root / "config.toml")) == asdict(config)
        for environment in (801, 802):
            descendants = {}
            for group, modes in (
                ("founders", ("none",)),
                ("descendants", ("none", "no_motor_learning", "shuffled_motor_reward")),
            ):
                for mode in modes:
                    path = root / f"{environment}-{group}-{mode}"
                    assert asdict(Config.load(path / "config.toml")) == asdict(config), path
                    result = check_pool(root, environment, group, mode, source)
                    assert result["segments"][0]["metadata"]["ablation"] == mode, path
                    trials.append(result)
                    if group == "descendants":
                        descendants[mode] = result
            normal = descendants["none"]
            for mode in ("no_motor_learning", "shuffled_motor_reward"):
                control = descendants[mode]
                assert normal["sampled_genomes_sha256"] == control["sampled_genomes_sha256"]
                pairs.append(
                    dict(
                        source=source,
                        environment=environment,
                        comparison=mode,
                        differences={
                            key: normal["final"][key] - control["final"][key]
                            for key in (
                                "births",
                                "population",
                                "fresh_absorbed",
                                "detritus_absorbed",
                                "predation_absorbed",
                            )
                        },
                    )
                )
    output = Path("docs/results/v21-learning-assays.json")
    result = dict(
        interpretation="Three persistent-exploration populations at 600 seconds, each sampled "
        "into two new environments for 360 seconds. Within each environment, the same 192 "
        "sampled descendant genotypes have normal, disabled, or shuffled motor learning. "
        "Mutation is disabled and lifetime neural state starts empty; recurrent plasticity "
        "and learning-capacity costs remain. Founder references have normal learning only. "
        "These are community interventions, not independent per-creature fitness samples. "
        "Shuffled rates retain population-wide signals; singleton batches cannot be shuffled.",
        shuffled_included=True,
        trials=trials,
        paired_differences=pairs,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Audited {len(trials)} community transplants and {len(pairs)} paired controls: {output}")


if __name__ == "__main__":
    main()
