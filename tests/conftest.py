import pytest
import torch

from emergent_garden.config import Config


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


@pytest.fixture
def config():
    return Config(
        diameter=128.0,
        initial_population=2,
        capacity=16,
        initial_food=0,
        food_rate=0.0,
        patch_radius=12.0,
        grid_size=32,
        smell_sigma=8.0,
        smell_cutoff=24.0,
        viewer_size=256,
    )
