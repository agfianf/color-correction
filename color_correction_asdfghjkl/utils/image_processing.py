import numpy as np


def crop_region_with_margin(
    image: np.ndarray,
    coordinates: tuple[int, int, int, int],
    margin_ratio: float = 0.2,
) -> np.ndarray:
    """Crop a region from image with additional margin from given coordinates.

    Parameters
    ----------
    image : np.ndarray
        Input image array of shape (H, W, C) or (H, W).
    coordinates : np.ndarray
        Bounding box coordinates [x1, y1, x2, y2].
    margin_ratio : float, optional
        Ratio of margin to add relative to region size, by default 0.2.

    Returns
    -------
    np.ndarray
        Cropped image region with margins.
    """
    y1, y2 = coordinates[1], coordinates[3]
    x1, x2 = coordinates[0], coordinates[2]

    height = y2 - y1
    margin_y = height * margin_ratio
    width = x2 - x1
    margin_x = width * margin_ratio

    crop_y1 = int(y1 + margin_y)
    crop_y2 = int(y2 - margin_y)
    crop_x1 = int(x1 + margin_x)
    crop_x2 = int(x2 - margin_x)

    return image[crop_y1:crop_y2, crop_x1:crop_x2]


def calc_mean_color_patch(img: np.ndarray) -> np.ndarray:
    """Calculate mean RGB/BGR values across spatial dimensions.

    Parameters
    ----------
    img : np.ndarray
        Input image array of shape (H, W, C).

    Returns
    -------
    np.ndarray
        Array of mean RGB values, shape (C,), dtype uint8.
    """
    return np.mean(img, axis=(0, 1)).astype(np.uint8)
