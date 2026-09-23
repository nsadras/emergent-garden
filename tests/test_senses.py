from dataclasses import replace

import pytest
import torch

from emergent_garden.inheritance import upgrade_genomes
from emergent_garden.senses import directional_basis, encode_field, receptor_values
from emergent_garden.world import World, create_world


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            assert_same(x, y)
    else:
        assert left == right


def test_spatial_basis_preserves_information_and_has_body_relative_signs():
    samples = torch.tensor([1.0, 2.0, 7.0, 4.0])
    features = directional_basis(samples)
    torch.testing.assert_close(features, torch.tensor([3.5, 2.0, 1.0, 0.5]))
    torch.testing.assert_close(receptor_values(features), samples)
    # Mirror swaps left/right. Half a turn reverses both directional axes.
    mirror = directional_basis(samples.flip(-1))
    torch.testing.assert_close(mirror, features * torch.tensor([1, -1, 1, -1]))
    rotated = directional_basis(samples.roll(2, -1))
    torch.testing.assert_close(rotated, features * torch.tensor([1, -1, -1, 1]))


def test_contrasts_are_bounded_and_do_not_disappear_in_strong_fields(config):
    c = replace(config, ecology_version=17, sensory_contrast=1)
    raw = torch.tensor([[0.0] * 4, [10.0] * 4, [1000.0, 2000, 7000, 4000], [1e30, 0, 0, 0]])
    features = encode_field(c, raw, 8.0)
    assert torch.isfinite(features).all()
    assert (features[:, 0] >= 0).all() and (features[:, 0] <= 1).all()
    assert (features[:, 1:].abs() <= 1).all()
    assert features[:2, 1:].count_nonzero() == 0
    assert features[0].count_nonzero() == 0
    torch.testing.assert_close(
        features[2, 1:], encode_field(c, 100 * raw[2], 8.0)[1:], rtol=0.005, atol=1e-5
    )
    assert encode_field(c, torch.empty(0, 3, 4), 8.0).shape == (0, 3, 4)
    bounded = torch.tensor([[0.0, 0.2, 0.8, 1.0]])
    torch.testing.assert_close(receptor_values(encode_field(c, bounded)), bounded)


def test_direction_ablation_keeps_strength_and_internal_inputs(config):
    w = create_world(replace(config, ecology_version=17, sensory_contrast=1))
    n = w.config.grid_size
    for field in w.fields:
        field.grid = torch.arange(n).float()[None].expand(n, -1).clone() / n
    index = torch.arange(w.population)
    normal = w.module_sensors(index)
    w.ablation = "no_direction"
    mean_only = w.module_sensors(index)
    count = 4 * len(w.config.field_names)
    expected = normal.clone()
    fields = expected[..., :count].reshape(w.population, 3, -1, 4)
    assert fields[..., 1:].abs().sum() > 0
    fields[..., 1:] = 0
    torch.testing.assert_close(mean_only, expected, rtol=0, atol=0)
    w.ablation = "rotated"
    expected = normal.clone()
    expected[..., :count].reshape(w.population, 3, -1, 4)[..., 1:3] *= -1
    torch.testing.assert_close(w.module_sensors(index), expected, rtol=0, atol=0)
    w.ablation = "disabled"
    assert w.module_sensors(index)[..., :count].count_nonzero() == 0


def test_disabled_encoding_is_exactly_legacy_and_new_mode_replays(config):
    c = replace(config, ecology_version=16, initial_food=20, food_rate=2)
    old = create_world(c)
    neutral = create_world(replace(c, ecology_version=17))
    old.step(37)
    neutral.step(37)
    before, after = old.state_dict(), neutral.state_dict()
    before.pop("config")
    after.pop("config")
    assert_same(before, after)
    w = create_world(replace(c, ecology_version=17, sensory_contrast=1), ablation="shuffled")
    w.step(13)
    replay = World.from_state(w.state_dict())
    w.step(47)
    replay.step(47)
    assert_same(w.state_dict(), replay.state_dict())


def test_encoding_transfer_is_explicit_and_dimensions_stay_fixed(config):
    old = replace(config, ecology_version=16)
    new = replace(config, ecology_version=17, sensory_contrast=1)
    assert old.input_size == new.input_size == 40
    assert old.output_size == new.output_size == 7
    assert old.parameter_count == new.parameter_count
    w = create_world(old)
    with pytest.raises(ValueError, match="input meanings differ"):
        upgrade_genomes(old, new, w.agents["genome"])
    fresh = create_world(new)
    torch.testing.assert_close(w.founders, fresh.founders, rtol=0, atol=0)
    torch.testing.assert_close(
        upgrade_genomes(new, new, fresh.founders), fresh.founders, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="Contrast sensing requires"):
        replace(old, sensory_contrast=1).validate()
    with pytest.raises(ValueError, match="sensory_contrast must"):
        replace(new, sensory_contrast=2).validate()
