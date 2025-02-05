import numpy as np
import pytest
from color_correction.utils.correction import (
    preprocessing_compute,
    postprocessing_compute,
)

@pytest.mark.parametrize("input_image, expected_shape", [
    # When shape is exactly (24, 3) and should be cast without reshaping
    (np.random.rand(24, 3) * 255, (24, 3)),
    # When shape is different; for example a 6x6 image with 3 channels should be reshaped to (36, 3)
    (np.random.rand(6, 6, 3) * 255, (36, 3)),
])
def test_preprocessing_compute(input_image, expected_shape):
    output = preprocessing_compute(input_image)
    assert output.dtype == np.float32
    assert output.shape == expected_shape

@pytest.mark.parametrize("original_shape, predict_image, expected_shape", [
    # Grid image patches (original_shape length 2); no reshape needed
    ((24, 3), np.array([[300, -10, 100]] * 24), (24, 3)),
    # Regular colored image: reshape required from flat list to (h, w, c)
    ((4, 4, 3), np.array([[300, -10, 100]] * 16), (4, 4, 3)),
])
def test_postprocessing_compute(original_shape, predict_image, expected_shape):
    output = postprocessing_compute(original_shape, predict_image)
    # Output must be of type uint8 after clipping
    assert output.dtype == np.uint8
    assert output.shape == expected_shape
    # Validate that all values are clipped between 0 and 255
    assert output.min() >= 0
    assert output.max() <= 255
