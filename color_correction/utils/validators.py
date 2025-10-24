"""Input validation utilities for color correction operations."""

import numpy as np
from numpy.typing import NDArray

from color_correction.exceptions import InvalidBoxError, InvalidImageError


def validate_bgr_image(
    image: NDArray[np.uint8] | None,
    param_name: str = "image",
    min_height: int = 1,
    min_width: int = 1,
) -> None:
    """Validate that input is a valid BGR image.

    Parameters
    ----------
    image : NDArray[np.uint8] | None
        Image to validate
    param_name : str, optional
        Parameter name for error messages
    min_height : int, optional
        Minimum required height
    min_width : int, optional
        Minimum required width

    Raises
    ------
    InvalidImageError
        If image is invalid or doesn't meet requirements
    """
    if image is None:
        raise InvalidImageError(f"{param_name} cannot be None")

    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"{param_name} must be numpy array, got {type(image).__name__}")

    if image.ndim != 3:
        raise InvalidImageError(f"{param_name} must have 3 dimensions (H, W, C), got {image.ndim}")

    if image.shape[2] != 3:
        raise InvalidImageError(f"{param_name} must have 3 channels (BGR), got {image.shape[2]}")

    height, width = image.shape[:2]
    if height < min_height or width < min_width:
        raise InvalidImageError(
            f"{param_name} too small: {height}x{width}, minimum required: {min_height}x{min_width}",
        )


def validate_patches(
    patches: list | None,
    expected_count: int = 24,
    param_name: str = "patches",
    allow_none: bool = False,
) -> None:
    """Validate color patches list.

    Parameters
    ----------
    patches : list | None
        List of patches to validate
    expected_count : int, optional
        Expected number of patches (default 24)
    param_name : str, optional
        Parameter name for error messages
    allow_none : bool, optional
        Whether to allow None values in the list

    Raises
    ------
    ValueError
        If patches are invalid
    """
    if patches is None:
        raise ValueError(f"{param_name} cannot be None")

    if not isinstance(patches, list):
        raise ValueError(f"{param_name} must be a list, got {type(patches).__name__}")

    if len(patches) != expected_count:
        raise ValueError(f"{param_name} must have {expected_count} items, got {len(patches)}")

    if not allow_none:
        none_count = sum(1 for p in patches if p is None)
        if none_count > 0:
            raise ValueError(f"{param_name} contains {none_count} None values")


def validate_box(
    box: tuple[int, int, int, int],
) -> None:
    """Validate bounding box coordinates.

    Parameters
    ----------
    box : tuple[int, int, int, int]
        Bounding box in (x1, y1, x2, y2) format

    Raises
    ------
    InvalidBoxError
        If box coordinates are invalid
    """
    if not isinstance(box, tuple | list):
        raise InvalidBoxError(box, "must be tuple or list")

    if len(box) != 4:
        raise InvalidBoxError(box, f"must have 4 elements, got {len(box)}")

    x1, y1, x2, y2 = box

    if not all(isinstance(coord, int | float | np.integer | np.floating) for coord in box):
        raise InvalidBoxError(box, "coordinates must be numeric")

    if x1 >= x2:
        raise InvalidBoxError(box, f"x1 ({x1}) must be less than x2 ({x2})")

    if y1 >= y2:
        raise InvalidBoxError(box, f"y1 ({y1}) must be less than y2 ({y2})")

    if x1 < 0 or y1 < 0:
        raise InvalidBoxError(box, "coordinates cannot be negative")


def validate_confidence_threshold(
    threshold: float,
    param_name: str = "confidence_threshold",
) -> None:
    """Validate confidence threshold value.

    Parameters
    ----------
    threshold : float
        Confidence threshold to validate
    param_name : str, optional
        Parameter name for error messages

    Raises
    ------
    ValueError
        If threshold is not in valid range [0, 1]
    """
    if not isinstance(threshold, int | float):
        raise ValueError(f"{param_name} must be numeric, got {type(threshold).__name__}")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{param_name} must be between 0 and 1, got {threshold}")


def validate_iou_threshold(
    threshold: float,
    param_name: str = "iou_threshold",
) -> None:
    """Validate IoU threshold value.

    Parameters
    ----------
    threshold : float
        IoU threshold to validate
    param_name : str, optional
        Parameter name for error messages

    Raises
    ------
    ValueError
        If threshold is not in valid range [0, 1]
    """
    if not isinstance(threshold, int | float):
        raise ValueError(f"{param_name} must be numeric, got {type(threshold).__name__}")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{param_name} must be between 0 and 1, got {threshold}")
