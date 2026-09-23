"""Audit V19 descendant motor-learning transplants into two fresh environments."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from audit_circuits import circuit_run, physical, same_state

from emergent_garden.config import Config
from emergent_garden.history import read_history


def check_pool(root, environment, group, mode, source):
    pools = torch.load(root / "source-genomes.pt", weights_only=True)
    rng = torch.Generator().manual_seed(environment + 1299709)
    pool = pools[group]
    chosen = pool[torch.randint(len(pool), (192,), generator=rng)]
    path = root / f"{environment}-{group}-{mode}"
    actual = torch.load(path / "founders.pt", weights_only=True)["genomes"]
    assert torch.equal(actual, chosen), path
    config = Config.load(path / "config.toml")
    for name in (
        "mutation_probability",
        "trait_mutation_probability",
        "node_mutation_probability",
        "edge_mutation_probability",
        "module_mutation_probability",
    ):
        assert getattr(config, name) == 0, (path, name)
    result = circuit_run(path, 360)
    result.update(
        source=source,
        environment=environment,
        group=group,
        mode=mode,
        sampled_genomes_sha256=hashlib.sha256(chosen.numpy().tobytes()).hexdigest(),
    )
    if mode == "no_motor_learning":
        assert result["final"]["motor_learning_changes"] == 0, path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-shuffled", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    trials, repeated_founders = [], []
    for source in (1, 2, 3):
        root = Path(f"runs/v19-learning-assay-{source}")
        report = json.loads((root / "summary.json").read_text())
        assert report["completed"] and len(report["trials"]) == 6, root
        for environment in (601, 602):
            for group, modes in (
                ("founders", ("none",)),
                ("descendants", ("none", "no_motor_learning")),
            ):
                for mode in modes:
                    trials.append(check_pool(root, environment, group, mode, source))
        if args.include_shuffled:
            shuffled = Path(f"runs/v19-learning-shuffled-assay-{source}")
            report = json.loads((shuffled / "summary.json").read_text())
            assert report["completed"] and len(report["trials"]) == 4, shuffled
            original = torch.load(root / "source-genomes.pt", weights_only=True)
            copied = torch.load(shuffled / "source-genomes.pt", weights_only=True)
            same_state(original, copied)
            for environment in (601, 602):
                trials.append(
                    check_pool(
                        shuffled, environment, "descendants", "shuffled_motor_reward", source
                    )
                )
                # These rerun founders are reproducibility checks, not extra independent samples.
                a = root / f"{environment}-founders-none"
                b = shuffled / f"{environment}-founders-none"
                assert physical(read_history(a)[1]) == physical(read_history(b)[1])
                left = torch.load(a / "latest.pt", weights_only=True)
                right = torch.load(b / "latest.pt", weights_only=True)
                left.pop("config")
                right.pop("config")
                same_state(left, right)
                repeated_founders.append(
                    dict(source=source, environment=environment, exact_replay=True)
                )
    pairs = []
    for source in (1, 2, 3):
        for environment in (601, 602):
            rows = {
                t["mode"]: t
                for t in trials
                if t["source"] == source
                and t["environment"] == environment
                and t["group"] == "descendants"
            }
            for mode in rows.keys() - {"none"}:
                a, b = rows["none"], rows[mode]
                assert a["sampled_genomes_sha256"] == b["sampled_genomes_sha256"]
                pairs.append(
                    dict(
                        source=source,
                        environment=environment,
                        comparison=mode,
                        birth_difference=a["final"]["births"] - b["final"]["births"],
                        population_difference=a["final"]["population"] - b["final"]["population"],
                        fresh_absorbed_difference=a["final"]["fresh_absorbed"]
                        - b["final"]["fresh_absorbed"],
                    )
                )
    output = Path("docs/results/v19-learning-assays.json")
    result = dict(
        interpretation="Three 600-second evolutionary sources, each tested in two fresh "
        "environments. Descendant genotypes, initial placement, exploration, and capacity "
        "costs match within each "
        "pair; mutation is disabled and neural state starts empty. Controls remove motor updates "
        "or, when included, assign another body's normalized energetic return rate "
        "to each learner. "
        "Singleton update batches cannot be shuffled. Recurrent plasticity remains enabled. "
        "These are community effects, not independent per-creature fitness estimates or evidence "
        "of general intelligence. Shared environmental signals may remain informative "
        "when shuffled.",
        shuffled_included=args.include_shuffled,
        trials=trials,
        paired_differences=pairs,
        repeated_founder_parity=repeated_founders,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Audited {len(trials)} community transplants and {len(pairs)} paired differences: {output}"
    )


if __name__ == "__main__":
    main()
