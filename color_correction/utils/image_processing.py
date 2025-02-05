import base64
import io

import colour as cl
import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

ImageType = NDArray[np.uint8]


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


def calc_color_diff(
    image1: ImageType,
    image2: ImageType,
) -> dict[str, float]:
    """Calculate color difference metrics between two images.

    Parameters
    ----------
    image1, image2 : NDArray
        Images to compare in BGR format.

    Returns
    -------
    dict[str, float]
        Dictionary of color difference
        keys: min, max, mean, std
    """
    rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    lab1 = cl.XYZ_to_Lab(cl.sRGB_to_XYZ(rgb1 / 255))
    lab2 = cl.XYZ_to_Lab(cl.sRGB_to_XYZ(rgb2 / 255))

    delta_e = cl.difference.delta_E(lab1, lab2, method="CIE 2000")

    return {
        "min": round(float(np.min(delta_e)), 4),
        "max": round(float(np.max(delta_e)), 4),
        "mean": round(float(np.mean(delta_e)), 4),
        "std": round(float(np.std(delta_e)), 4),
    }


def numpy_array_to_base64(
    arr: np.ndarray,
    convert_bgr_to_rgb: bool = True,
) -> str:
    """Convert numpy array (image) to base64 string"""
    if arr is None:
        return ""

    if convert_bgr_to_rgb:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"
