from dataclasses import replace

import pytest
import torch

from emergent_garden.inheritance import brain_parts
from emergent_garden.probes import food_response_probe
from emergent_garden.topology import initial_structure


@pytest.mark.parametrize("contrast", [0, 1])
def test_food_probe_distinguishes_directional_response_from_constant_turning(config, contrast):
    c = replace(config, ecology_version=19, sensory_contrast=contrast)
    g = torch.zeros(2, c.parameter_count)
    g[:, c.brain_parameter_count + c.trait_count :] = initial_structure(c, 2, "cpu")
    wi, _, _, wo, bias = brain_parts(c, g)
    # Constructed diagnostic circuits are never introduced into a population.
    if contrast:
        wi[0, 0, 1] = 2  # Spatial side contrast.
    else:
        wi[0, 0, 1], wi[0, 0, 2] = -1, 1  # Right-front minus left-front.
    wo[0, 1, 0] = 1
    bias[1, 1] = 1  # A constant turn, irrespective of the food field.
    before = g.clone()
    rng = torch.get_rng_state()
    for mean in (0.1, 0.4, 0.8):
        result = food_response_probe(c, g, mean=mean)
        assert abs(result["uniform_turn"][0]) < 1e-6
        assert result["left_turn"][0] < 0 < result["right_turn"][0]
        assert result["aligned_response"][0] > 0.009
        assert result["uniform_turn"][1] > 0.2
        assert result["aligned_response"][1] == 0
    torch.testing.assert_close(g, before, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(ValueError, match="intensities"):
        food_response_probe(c, g, mean=0.99, contrast=0.1)
