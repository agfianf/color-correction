import numpy as np
import pytest
import cv2

from color_correction_asdfghjkl.utils.image_patch import (
    create_patch_tiled_image,
    visualize_patch_comparison,
)


@pytest.fixture
def sample_patches_bgr() -> list[tuple[int, int, int]]:
    return np.random.randint(0, 255, size=(24, 3))


@pytest.fixture
def sample_outer_patches() -> list[np.ndarray]:
    return np.random.randint(0, 255, size=(24, 3))


@pytest.fixture
def sample_inner_patches() -> list[np.ndarray]:
    return np.random.randint(0, 255, size=(24, 3))


def test_create_patch_tiled_image(sample_patches_bgr) -> None:  # noqa: ANN001
    patch_size = (50, 50, 1)
    image = create_patch_tiled_image(sample_patches_bgr, patch_size)
    assert image.shape == (4 * patch_size[0], 6 * patch_size[1], 3 * patch_size[2])
    assert image.dtype == np.uint8


def test_create_patch_tiled_image_custom_size(sample_patches_bgr) -> None:  # noqa: ANN001
    patch_size = (30, 30, 1)
    image = create_patch_tiled_image(sample_patches_bgr, patch_size)
    assert image.shape == (4 * patch_size[0], 6 * patch_size[1], 3 * patch_size[2])


def test_visualize_patch_comparison(sample_outer_patches, sample_inner_patches):
    patch_size = (100, 100, 1)
    image = visualize_patch_comparison(
        sample_outer_patches,
        sample_inner_patches,
        patch_size,
    )
    assert image.shape == (4 * patch_size[0], 6 * patch_size[1], 3 * patch_size[2])
    assert image.dtype == np.uint8


def test_visualize_inner_patch_center(sample_outer_patches, sample_inner_patches):
    patch_size = (100, 100, 1)
    h, w, _ = patch_size
    h_half = h // 2
    w_half = w // 2
    y1 = h_half - (h // 4)
    y2 = h_half + (h // 4)
    x1 = w_half - (w // 4)
    x2 = w_half + (w // 4)

    image = visualize_patch_comparison(
        sample_outer_patches,
        sample_inner_patches,
        patch_size,
    )
    inner_patch = sample_inner_patches[0]
    inner_patch = np.tile(inner_patch, (y2 - y1, x2 - x1, 1))
    assert np.array_equal(image[y1:y2, x1:x2], inner_patch)
