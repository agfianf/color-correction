import numpy as np
import pytest


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.zeros((640, 640, 3), dtype=np.uint8)
