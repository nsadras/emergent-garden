from dataclasses import replace

import pytest
import torch

from emergent_garden.brain import advance
from emergent_garden.ecology import EcologyWorld
from emergent_garden.inheritance import upgrade_genomes
from emergent_garden.storage import load_checkpoint, save_checkpoint


def reversal(config, **kwargs):
    return EcologyWorld(replace(config, ecology_version=6, **kwargs))


def food_at_agent(world, patch, amount=20.0):
    world.append_food(
        world.agents["pos"][:1].clone(), torch.tensor([amount]), 0, torch.tensor([patch])
    )
    world.totals["food_spawned"] += amount


def test_quality_is_hidden_from_all_current_observations(config):
    w = reversal(config, initial_population=1, initial_food=20)
    before = w.module_sensors(torch.tensor([0]))
    grids = [f.grid.clone() for f in w.fields]
    w.landscape.favorable = 1 - w.landscape.favorable
    w.rebuild_fields()
    torch.testing.assert_close(before, w.module_sensors(torch.tensor([0])), rtol=0, atol=0)
    for field, grid in zip(w.fields, grids, strict=True):
        torch.testing.assert_close(field.grid, grid, rtol=0, atol=0)
    assert before.shape == (1, 3, 32)
    assert w.fields[5].grid.sum() > 0 and w.fields[6].grid.sum() > 0


def test_quality_changes_assimilation_and_dissipation_conservatively(config):
    w = reversal(config, initial_population=1, detritus_delay=3.0)
    patch = int((w.landscape.identities == w.landscape.favorable).nonzero()[0])
    food_at_agent(w, patch)
    other = EcologyWorld.from_state(w.state_dict())
    other.landscape.favorable = 1 - w.landscape.favorable
    w.feed()
    other.feed()
    assert other.totals["fresh_absorbed"] == pytest.approx(
        w.totals["fresh_absorbed"] * w.config.low_quality
    )
    assert w.totals["high_quality_absorbed"] == w.totals["fresh_absorbed"]
    assert other.totals["low_quality_absorbed"] == other.totals["fresh_absorbed"]
    assert w.totals["detritus_created"] == other.totals["detritus_created"]
    assert other.totals["digestion_loss"] > w.totals["digestion_loss"]
    for world in (w, other):
        assert abs(world.metrics()["energy_balance_error"]) < 1e-4


def test_unclustered_food_and_detritus_keep_their_digestibility(config):
    w = reversal(config, initial_population=1)
    food_at_agent(w, -1)
    other = EcologyWorld.from_state(w.state_dict())
    other.landscape.favorable = 1 - w.landscape.favorable
    for _ in range(2):
        w.feed()
        other.feed()
        torch.testing.assert_close(w.agents["energy"], other.agents["energy"], rtol=0, atol=0)
    assert w.totals["detritus_absorbed"] > 0


def test_uniform_quality_control_keeps_nutrition_equal_across_reversal(config):
    w = reversal(config, initial_population=1, low_quality=0.625, high_quality=0.625)
    food_at_agent(w, 0)
    other = EcologyWorld.from_state(w.state_dict())
    other.landscape.favorable = 1 - w.landscape.favorable
    w.feed()
    other.feed()
    torch.testing.assert_close(w.agents["energy"], other.agents["energy"], rtol=0, atol=0)
    assert w.totals["fresh_absorbed"] == other.totals["fresh_absorbed"]
    assert abs(w.metrics()["energy_balance_error"]) < 1e-4


def test_run_source_archive_matches_metadata_hash(config, tmp_path):
    import hashlib
    import json
    import zipfile

    from emergent_garden.storage import RunStore

    w = reversal(config)
    store = RunStore(tmp_path / "run", w)
    store.close()
    digest = hashlib.sha256()
    with zipfile.ZipFile(tmp_path / "run/source.zip") as archive:
        for name in sorted(n for n in archive.namelist() if n.startswith("emergent_garden/")):
            digest.update(name.removeprefix("emergent_garden/").encode())
            digest.update(archive.read(name))
    metadata = json.loads((tmp_path / "run/metadata.json").read_text())
    assert metadata["source_sha256"] == digest.hexdigest()
    assert metadata["source_capture"] == "process_import"


def test_intake_feedback_is_consumed_once_per_controller_update(config):
    w = reversal(config, initial_population=1)
    food_at_agent(w, -1)
    w.feed()
    received = w.agents["food_feedback"].clone()
    assert received.item() > 0
    value = received / (w.config.feedback_scale * w.agents["area"])
    expected = value / (1 + value)
    w.update_controllers()
    torch.testing.assert_close(w.agents["inputs"][:, -4], expected)
    assert w.agents["food_feedback"].item() == 0
    w.tick += w.config.physics_hz // w.config.controller_hz
    w.update_controllers()
    assert w.agents["inputs"][0, -4] == 0


def test_new_controls_remove_only_their_own_observations(config):
    w = reversal(config, initial_population=1)
    w.agents["food_feedback"].fill_(5)
    w.agents["damage_feedback"].fill_(3)
    index = torch.tensor([0])
    normal = w.module_sensors(index)
    w.ablation = "no_identity"
    removed = w.module_sensors(index)
    assert removed[..., 20:28].count_nonzero() == 0
    torch.testing.assert_close(removed[..., :20], normal[..., :20])
    torch.testing.assert_close(removed[..., 28:], normal[..., 28:])
    w.ablation = "no_feedback"
    removed = w.module_sensors(index)
    assert removed[..., 28:30].count_nonzero() == 0
    torch.testing.assert_close(removed[..., :28], normal[..., :28])
    torch.testing.assert_close(removed[..., 30:], normal[..., 30:])
    with pytest.raises(ValueError, match="requires a version"):
        EcologyWorld(replace(config, ecology_version=5), ablation="no_identity")


def test_checkpoint_replays_irregular_reversals_and_feedback(config, tmp_path):
    w = reversal(config, initial_food=30, quality_period=0.15)
    w.step(7)
    save_checkpoint(w, tmp_path / "state.pt")
    other = load_checkpoint(tmp_path / "state.pt")
    w.step(89)
    other.step(89)
    assert w.totals["quality_reversals"] > 3
    assert w.landscape.favorable == other.landscape.favorable
    assert w.landscape.next_tick == other.landscape.next_tick
    assert w.events == other.events
    assert w.totals == other.totals
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], other.agents[key], rtol=0, atol=0)
    for key in w.rng:
        assert torch.equal(w.rng[key].get_state(), other.rng[key].get_state())


def test_transfer_preserves_circuit_with_named_input_extension(config):
    old = EcologyWorld(replace(config, ecology_version=5))
    new_config = replace(config, ecology_version=6)
    genomes = upgrade_genomes(old.config, new_config, old.agents["genome"])
    inputs = torch.rand((old.population, old.config.input_size))
    new_inputs = torch.rand((old.population, new_config.input_size))
    for i, name in enumerate(old.config.input_names):
        new_inputs[:, new_config.input_names.index(name)] = inputs[:, i]
    state = torch.rand((old.population, old.config.hidden_size))
    h1, a1 = advance(old.config, old.agents["genome"], inputs, state)
    h2, a2 = advance(new_config, genomes, new_inputs, state)
    torch.testing.assert_close(h1, h2, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(a1, a2, rtol=1e-6, atol=1e-7)


def test_reversal_fields_render_without_mutating_the_world(config):
    from emergent_garden.viewer import Renderer

    w = reversal(config, initial_food=20)
    before = w.state_dict()
    renderer = Renderer(640)
    for identity in (5, 6):
        renderer.field_index = identity
        renderer.draw(w)
    for key in w.agents:
        torch.testing.assert_close(w.agents[key], before["agents"][key], rtol=0, atol=0)
    assert w.landscape.favorable == before["landscape"]["favorable"]
    for key in w.rng:
        assert torch.equal(w.rng[key].get_state(), before["rng"][key])


def test_association_probe_has_matched_feedback_and_state_reset_controls(config):
    from emergent_garden.probes import association_probe

    w = reversal(config, initial_population=8)
    genomes = w.agents["genome"].clone()
    ordinary = association_probe(w.config, genomes, rounds=1)
    assert max(abs(v) for v in ordinary["association_alignment"]) > 1e-7
    for mode in ("no_feedback", "reset_h"):
        control = association_probe(w.config, genomes, rounds=1, mode=mode)
        assert control["association_alignment"] == [0.0] * len(genomes)
        assert control["reversal_alignment"] == [0.0] * len(genomes)
    torch.testing.assert_close(genomes, w.agents["genome"], rtol=0, atol=0)
