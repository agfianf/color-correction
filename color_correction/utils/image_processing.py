import base64
import io

import colour as cl
import cv2
import numpy as np
from PIL import Image

from color_correction.schemas.custom_types import ImageBGR


def crop_region_with_margin(
    image: np.ndarray,
    coordinates: tuple[int, int, int, int],
    margin_ratio: float = 0.2,
) -> np.ndarray:
    """
    Crop a sub-region from an image with an additional margin.

    Parameters
    ----------
    image : np.ndarray
        The input image (H, W, C) or (H, W).
    coordinates : tuple[int, int, int, int]
        The bounding box defined as (x1, y1, x2, y2).
    margin_ratio : float, optional
        Ratio to determine the extra margin; default is 0.2.

    Returns
    -------
    np.ndarray
        The cropped image region including the margin.
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
    """
    Compute the mean color of an image patch across spatial dimensions.

    Parameters
    ----------
    img : np.ndarray
        The input image patch with shape (H, W, C).

    Returns
    -------
    np.ndarray
        Array containing the mean color for each channel (dtype uint8).
    """
    return np.mean(img, axis=(0, 1)).astype(np.uint8)


def calc_color_diff(
    image1: ImageBGR,
    image2: ImageBGR,
) -> dict[str, float]:
    """
    Calculate color difference metrics between two images using CIE 2000.

    Parameters
    ----------
    image1 : ImageBGR
        First input image in BGR format.
    image2 : ImageBGR
        Second input image in BGR format.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'min', 'max', 'mean', and 'std' for the color difference.
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
    """
    Convert a numpy image array into a base64-encoded PNG string.

    Parameters
    ----------
    arr : np.ndarray
        Input image array.
    convert_bgr_to_rgb : bool, optional
        Whether to convert BGR to RGB before encoding; default is True.

    Returns
    -------
    str
        Base64-encoded image string prefixed with the appropriate data URI.
    """
    if arr is None:
        return ""

    if convert_bgr_to_rgb:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"
