from dataclasses import replace

import pytest
import torch

from emergent_garden.challenges import challenge_run, fork_challenge
from emergent_garden.ecology import EcologyWorld
from emergent_garden.storage import load_checkpoint, save_checkpoint


@pytest.mark.parametrize("version", [7, 8])
def test_challenge_forks_preserve_everything_except_named_interventions(config, version):
    w = EcologyWorld(replace(config, ecology_version=version))
    w.step(7)
    w.agents["module_plastic"].fill_(0.3)
    w.agents["module_trace"].fill_(0.2)
    for mode in ("intact", "erase_plastic", "erase_activity", "no_plasticity"):
        for reverse in (False, True):
            branch = fork_challenge(w, mode, reverse, 1000)
            cleared = set()
            if mode in ("erase_plastic", "no_plasticity"):
                cleared = {"module_plastic", "module_trace"}
            elif mode == "erase_activity":
                cleared = {"h", "module_h"}
            for key in w.agents:
                if key in cleared:
                    assert branch.agents[key].count_nonzero() == 0
                else:
                    torch.testing.assert_close(w.agents[key], branch.agents[key], rtol=0, atol=0)
            for key in w.rng:
                assert torch.equal(w.rng[key].get_state(), branch.rng[key].get_state())
            assert branch.landscape.favorable == (w.landscape.favorable + reverse) % 2
            assert branch.landscape.next_tick == 1001
            assert branch.config.mutation_probability == 0
            assert branch.config.trait_mutation_probability == 0
            assert branch.config.node_mutation_probability == 0
            assert branch.config.edge_mutation_probability == 0
            for a, b in zip(w.fields, branch.fields, strict=True):
                torch.testing.assert_close(a.grid, b.grid, rtol=0, atol=0)
    assert w.agents["module_plastic"].count_nonzero() > 0
    assert w.config.mutation_probability > 0


def test_challenge_keeps_dead_cohort_records_and_replayable_start_states(config, tmp_path):
    w = EcologyWorld(replace(config, ecology_version=7, initial_population=1))
    w.initial_energy += 0.0001 - w.agents["energy"].double().sum().item()
    w.agents["energy"].fill_(0.0001)
    checkpoint = tmp_path / "source.pt"
    save_checkpoint(w, checkpoint)
    output = tmp_path / "challenge"
    report = challenge_run(checkpoint, output, seconds=0.2)
    assert report["completed"]
    assert len(report["trials"]) == 8
    for row in report["trials"]:
        assert row["completed"]
        assert row["cohort_survivors"] == 0
        assert row["cohort"][0]["id"] == 0
        assert row["cohort_totals"]["acquired"] == 0
        assert row["cohort_totals"]["spent"] > 0
    resumed = load_checkpoint(output / "unchanged-intact/start.pt")
    torch.testing.assert_close(w.agents["genome"], resumed.agents["genome"], rtol=0, atol=0)
    assert (
        abs(
            load_checkpoint(output / "unchanged-intact/latest.pt").metrics()["energy_balance_error"]
        )
        < 1e-4
    )
