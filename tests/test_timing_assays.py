import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_observation import assert_same

from emergent_garden.neural_timing import timing_genes
from emergent_garden.storage import load_checkpoint, save_checkpoint
from emergent_garden.topology import effective_masks
from emergent_garden.world import create_world


@pytest.fixture
def assay():
    path = Path(__file__).resolve().parents[1] / "scripts" / "assay_neural_timing.py"
    spec = importlib.util.spec_from_file_location("timing_assay_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configured(config, **changes):
    return replace(
        config,
        **dict(
            ecology_version=24,
            hidden_size=8,
            initial_neurons=4,
            min_neurons=2,
            initial_population=4,
            neural_timing_range=4.0,
            initial_timing_sigma=0.5,
            timing_mutation_probability=0.1,
            motor_normalized=1,
            motor_learning_rate=0.01,
            exploration_max=0.15,
        )
        | changes,
    ).validate()


@pytest.mark.parametrize("mode", ["native", "mean_tau", "permuted"])
def test_timing_interventions_preserve_other_genes_and_the_promised_times(config, assay, mode):
    c = configured(config)
    original = create_world(c, seed=14).founders
    original[1] = original[0]  # Identical clones must not get different timing permutations.
    nodes = effective_masks(c, original)[0]
    # The homogeneous transform must also handle genes at either inherited bound.
    timing_genes(c, original)[2, :4] = -c.weight_limit
    timing_genes(c, original)[3, :4] = c.weight_limit
    snapshot = original.clone()
    actual = assay.transform_genomes(c, original, mode)
    assert_same(original, snapshot)
    assert actual.data_ptr() != original.data_ptr()
    assert_same(actual[:, : -c.hidden_size], original[:, : -c.hidden_size])
    old, new = timing_genes(c, original), timing_genes(c, actual)
    assert_same(old[~nodes], new[~nodes])
    assert_same(actual[0], actual[1])
    assert new.abs().max() <= c.weight_limit
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        assay.mean_times(c, actual), assay.mean_times(c, original), rtol=1e-6, atol=1e-6
    )
    if mode == "native":
        assert_same(actual, original)
    elif mode == "mean_tau":
        for row, mask in zip(new, nodes, strict=True):
            assert row[mask].unique().numel() == 1
    else:
        for before, after, mask in zip(old, new, nodes, strict=True):
            assert_same(before[mask].sort().values, after[mask].sort().values)
        # The intervention is keyed to genotype, independently of pool ordering.
        assert_same(assay.transform_genomes(c, original.flip(0), mode).flip(0), actual)


def test_transplanted_timing_controls_share_complete_initial_physical_state(config, assay):
    c = configured(config)
    genomes = create_world(c, seed=14).founders
    reference = None
    for mode in ("native", "mean_tau", "permuted"):
        changed = assay.transform_genomes(c, genomes, mode)
        w = assay.prepare_world(c, changed, 901, "none")
        assert all(getattr(w.config, key) == 0 for key in assay.MUTATION_KEYS)
        assert_same(w.mutate(w.agents["genome"][0]), w.agents["genome"][0])
        assert w.agents["age"].count_nonzero() == 0
        for key in ("module_h", "module_plastic", "module_trace", "module_motor_plastic"):
            assert not w.agents[key].count_nonzero(), key
        state = w.state_dict()
        state.pop("founders")
        state["agents"].pop("genome")
        if reference is None:
            reference = state
        else:
            assert_same(state, reference)


def test_small_complete_assay_records_all_interventions_and_preserves_source(
    config, assay, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    w = create_world(configured(config))
    # This fixture supplies a descendant pool; it is not an evolutionary result.
    w.agents["generation"].fill_(1)
    save_checkpoint(w, source / "latest.pt")
    source_hash = assay.digest(source / "latest.pt")
    output = tmp_path / "assay"
    report = assay.run_assay(source, output, [901], 2)
    assert report["completed"] and len(report["trials"]) == 5
    assert report["source_checkpoint_sha256"] == source_hash
    assert assay.digest(source / "latest.pt") == source_hash
    chosen = torch.load(output / "selected-901.pt", weights_only=True)["genomes"]
    for row in report["trials"]:
        transform, ablation = assay.TREATMENTS[row["treatment"]]
        path = Path(row["path"])
        expected = assay.transform_genomes(w.config, chosen, transform)
        recorded = torch.load(path / "founders.pt", weights_only=True)["genomes"]
        assert_same(recorded, expected)
        initial = load_checkpoint(path / "initial.pt")
        assert initial.time == 0 and initial.agents["age"].count_nonzero() == 0
        assert not initial.agents["module_h"].count_nonzero()
        assert not initial.agents["module_motor_plastic"].count_nonzero()
        resumed = load_checkpoint(path / "latest.pt")
        assert resumed.ablation == ablation and resumed.time == 2
        assert all(getattr(resumed.config, key) == 0 for key in assay.MUTATION_KEYS)
        assert resumed.seeded_from["intervention"] == row["treatment"]
        if ablation == "no_motor_learning":
            assert not resumed.agents["module_motor_plastic"].count_nonzero()


def test_requested_stop_leaves_a_reviewable_incomplete_assay(config, assay, tmp_path):
    from types import SimpleNamespace

    source = tmp_path / "source"
    source.mkdir()
    w = create_world(configured(config))
    w.agents["generation"].fill_(1)
    save_checkpoint(w, source / "latest.pt")
    output = tmp_path / "stopped"
    report = assay.run_assay(source, output, [901], 2, stop=SimpleNamespace(requested=True))
    assert not report["completed"] and not report["trials"]
    assert (output / "summary.json").exists()
    assert (output / "source-genomes.pt").exists()
