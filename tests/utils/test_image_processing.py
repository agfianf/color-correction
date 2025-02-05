import numpy as np
import pytest
from color_correction.utils.image_processing import crop_region_with_margin, calc_mean_color_patch

@pytest.fixture
def known_image() -> np.ndarray:
    # Create an image with a known pattern using np.arange,
    # reshape to (100, 100, 3) and wrap values with modulo 255
    img = np.arange(100 * 100 * 3, dtype=np.uint8) % 255
    return img.reshape((100, 100, 3))

@pytest.mark.parametrize(
    "coordinates, margin_ratio, expected_slice",
    [
        (
            (10, 20, 90, 80),
            0.2,
            (slice(32, 68), slice(26, 74))
        ),
        (
            (10, 20, 90, 80),
            0.0,
            (slice(20, 80), slice(10, 90))
        ),
    ],
)
def test_crop_region_with_margin(known_image: np.ndarray, coordinates, margin_ratio, expected_slice) -> None:
    result = crop_region_with_margin(known_image, coordinates, margin_ratio)
    expected = known_image[expected_slice[0], expected_slice[1]]
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize(
    "img, expected_mean",
    [
        (
            np.full((10, 10, 3), fill_value=50, dtype=np.uint8),
            np.array([50, 50, 50], dtype=np.uint8)
        ),
        (
            # Create an image where channel 0 is 0, channel 1 is 100, channel 2 is 200.
            np.stack([
                np.zeros((20, 20), dtype=np.uint8),
                np.full((20, 20), 100, dtype=np.uint8),
                np.full((20, 20), 200, dtype=np.uint8)
            ], axis=-1),
            np.array([0, 100, 200], dtype=np.uint8)
        ),
    ],
)
def test_calc_mean_color_patch(img: np.ndarray, expected_mean) -> None:
    mean_color = calc_mean_color_patch(img)
    np.testing.assert_array_equal(mean_color, expected_mean)
