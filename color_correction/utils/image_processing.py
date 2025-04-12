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


def crop_segment_straighten(image: np.ndarray, segment: list[tuple[int, int]]) -> np.ndarray:
    """
    Straighten and crop a quadrilateral region from an image.

    This function takes a quadrilateral defined by four corner points and transforms
    it into a rectangular image by applying a perspective transformation.

    Parameters
    ----------
    image : np.ndarray
        The input image as a numpy array with shape (H, W, C) or (H, W).
    segment : list[tuple[int, int]]
        List of four (x, y) coordinates defining the quadrilateral region to be
        straightened and cropped. Points should be in clockwise or counter-clockwise
        order starting from any corner.

    Returns
    -------
    np.ndarray
        The straightened and cropped rectangular image.

    Raises
    ------
    ValueError
        If the segment doesn't contain exactly 4 points.

    Notes
    -----
    The function uses perspective transformation to rectify the quadrilateral
    region into a rectangular image.
    """
    # Validate the segment has exactly 4 points
    if len(segment) != 4:
        raise ValueError(f"Invalid segment: Expected 4 points, got {len(segment)} points.")

    # Convert segment points to numpy array for transformation
    quad_points = np.array(segment, dtype="float32")

    # Calculate dimensions of the output rectangle
    # Width is the distance between first two points
    rect_width = int(np.linalg.norm(quad_points[0] - quad_points[1]))
    # Height is the distance between second and third points
    rect_height = int(np.linalg.norm(quad_points[1] - quad_points[2]))

    # Define the destination points for the rectangular output
    rect_points = np.array(
        [
            [0, 0],  # Top-left
            [rect_width - 1, 0],  # Top-right
            [rect_width - 1, rect_height - 1],  # Bottom-right
            [0, rect_height - 1],  # Bottom-left
        ],
        dtype="float32",
    )

    # Compute the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(quad_points, rect_points)

    # Apply the perspective transformation to get the straightened image
    straightened_image = cv2.warpPerspective(image, transform_matrix, (rect_width, rect_height))

    return straightened_image


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
