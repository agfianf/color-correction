# Phase 1 Implementation Plan: Foundation

**Timeline:** Weeks 1-2 (10 working days)
**Estimated Effort:** 20-25 hours
**Goal:** Establish foundation for code quality improvements

---

## Overview

Phase 1 focuses on foundational improvements that will make the codebase more maintainable and set the stage for Phase 2 (Testing). These are relatively low-risk changes with high impact.

### Phase 1 Deliverables

1. ✅ Custom exception hierarchy (`color_correction/exceptions.py`)
2. ✅ Input validation utilities (`color_correction/utils/validators.py`)
3. ✅ Grid configuration constants (`color_correction/constant/grid_config.py`)
4. ✅ Updated docstrings (remove emojis, improve consistency)
5. ✅ Fixed type definition inconsistencies
6. ✅ Updated all affected modules to use new exceptions and constants

---

## Task Breakdown

### Task 1: Create Custom Exception Hierarchy
**Priority:** 🔴 HIGH
**Estimated Time:** 2 hours
**Dependencies:** None

#### Subtasks

**1.1. Create exceptions.py file**
- **File:** `color_correction/exceptions.py`
- **Lines of code:** ~80-100
- **Status:** ⬜ Not Started

**Code to implement:**

```python
"""Custom exceptions for color correction operations.

This module provides a hierarchy of exceptions for better error handling
and debugging throughout the color correction package.
"""


class ColorCorrectionError(Exception):
    """Base exception for all color correction errors."""
    pass


class DetectionError(ColorCorrectionError):
    """Raised when card detection fails."""
    pass


class NoCardDetectedError(DetectionError):
    """Raised when no color checker card is found in the image."""

    def __init__(self, message: str = "No color checker card detected in image"):
        self.message = message
        super().__init__(self.message)


class InsufficientPatchesError(DetectionError):
    """Raised when detected patches are incomplete.

    Attributes
    ----------
    detected_count : int
        Number of patches actually detected
    required_count : int
        Number of patches required (default 24)
    """

    def __init__(self, detected_count: int, required_count: int = 24):
        self.detected_count = detected_count
        self.required_count = required_count
        self.message = (
            f"Detected {detected_count} patches, but {required_count} are required. "
            f"Please ensure the color checker card is fully visible and well-lit."
        )
        super().__init__(self.message)


class LowConfidenceDetectionError(DetectionError):
    """Raised when detection confidence is below acceptable threshold.

    Attributes
    ----------
    max_confidence : float
        Highest confidence score among detections
    threshold : float
        Required confidence threshold
    """

    def __init__(self, max_confidence: float, threshold: float):
        self.max_confidence = max_confidence
        self.threshold = threshold
        self.message = (
            f"Detection confidence {max_confidence:.2f} is below threshold {threshold:.2f}. "
            f"Try improving lighting or image quality."
        )
        super().__init__(self.message)


class CorrectionError(ColorCorrectionError):
    """Raised when color correction fails."""
    pass


class ModelNotFittedError(CorrectionError):
    """Raised when attempting to predict with unfitted model.

    This error occurs when calling predict() or compute_correction()
    before calling fit() to train the correction model.
    """

    def __init__(self, message: str = "Model must be fitted before prediction"):
        self.message = f"{message}. Call fit() first."
        super().__init__(self.message)


class PatchesNotSetError(CorrectionError):
    """Raised when required patches are not set before an operation.

    Attributes
    ----------
    patch_type : str
        Type of patches not set ('reference' or 'input')
    """

    def __init__(self, patch_type: str):
        self.patch_type = patch_type
        self.message = (
            f"{patch_type.capitalize()} patches must be set before this operation. "
            f"Call set_{patch_type}_patches() first."
        )
        super().__init__(self.message)


class InvalidImageError(ColorCorrectionError):
    """Raised when input image is invalid or has incorrect format.

    Attributes
    ----------
    reason : str
        Specific reason why the image is invalid
    """

    def __init__(self, reason: str):
        self.reason = reason
        self.message = f"Invalid image: {reason}"
        super().__init__(self.message)


class ModelLoadError(ColorCorrectionError):
    """Raised when model loading fails.

    Attributes
    ----------
    model_path : str
        Path to the model that failed to load
    """

    def __init__(self, model_path: str, reason: str = ""):
        self.model_path = model_path
        if reason:
            self.message = f"Failed to load model from '{model_path}': {reason}"
        else:
            self.message = f"Failed to load model from '{model_path}'"
        super().__init__(self.message)


class UnsupportedModelError(ColorCorrectionError):
    """Raised when an unsupported model type is requested.

    Attributes
    ----------
    model_name : str
        Name of the unsupported model
    supported_models : list[str]
        List of supported model names
    """

    def __init__(self, model_name: str, supported_models: list[str]):
        self.model_name = model_name
        self.supported_models = supported_models
        self.message = (
            f"Unsupported model: '{model_name}'. "
            f"Supported models are: {', '.join(supported_models)}"
        )
        super().__init__(self.message)


class GeometryError(ColorCorrectionError):
    """Raised when geometric operations fail."""
    pass


class InvalidBoxError(GeometryError):
    """Raised when bounding box coordinates are invalid.

    Attributes
    ----------
    box : tuple
        The invalid box coordinates
    """

    def __init__(self, box: tuple, reason: str = ""):
        self.box = box
        if reason:
            self.message = f"Invalid box {box}: {reason}"
        else:
            self.message = f"Invalid box coordinates: {box}"
        super().__init__(self.message)
```

**1.2. Add __init__.py export**
- **File:** `color_correction/__init__.py`
- **Action:** Add exceptions to package exports

```python
# Add to __init__.py
from color_correction.exceptions import (
    ColorCorrectionError,
    DetectionError,
    NoCardDetectedError,
    InsufficientPatchesError,
    LowConfidenceDetectionError,
    CorrectionError,
    ModelNotFittedError,
    PatchesNotSetError,
    InvalidImageError,
    ModelLoadError,
    UnsupportedModelError,
    GeometryError,
    InvalidBoxError,
)

__all__ = [
    # ... existing exports
    # Exceptions
    "ColorCorrectionError",
    "DetectionError",
    "NoCardDetectedError",
    "InsufficientPatchesError",
    "LowConfidenceDetectionError",
    "CorrectionError",
    "ModelNotFittedError",
    "PatchesNotSetError",
    "InvalidImageError",
    "ModelLoadError",
    "UnsupportedModelError",
    "GeometryError",
    "InvalidBoxError",
]
```

**1.3. Write unit tests**
- **File:** `tests/test_exceptions.py`
- **Action:** Test that exceptions can be raised and have correct attributes

```python
"""Tests for custom exceptions."""

import pytest
from color_correction.exceptions import (
    InsufficientPatchesError,
    LowConfidenceDetectionError,
    ModelNotFittedError,
    PatchesNotSetError,
    InvalidImageError,
    UnsupportedModelError,
)


def test_insufficient_patches_error():
    """Test InsufficientPatchesError attributes and message."""
    error = InsufficientPatchesError(detected_count=18, required_count=24)
    assert error.detected_count == 18
    assert error.required_count == 24
    assert "18" in str(error)
    assert "24" in str(error)


def test_low_confidence_detection_error():
    """Test LowConfidenceDetectionError attributes and message."""
    error = LowConfidenceDetectionError(max_confidence=0.15, threshold=0.25)
    assert error.max_confidence == 0.15
    assert error.threshold == 0.25
    assert "0.15" in str(error)
    assert "0.25" in str(error)


def test_patches_not_set_error():
    """Test PatchesNotSetError for different patch types."""
    error = PatchesNotSetError("reference")
    assert error.patch_type == "reference"
    assert "Reference" in str(error)
    assert "set_reference_patches" in str(error)


def test_unsupported_model_error():
    """Test UnsupportedModelError with supported models list."""
    error = UnsupportedModelError("invalid_model", ["yolov8", "mcc"])
    assert error.model_name == "invalid_model"
    assert error.supported_models == ["yolov8", "mcc"]
    assert "yolov8" in str(error)
    assert "mcc" in str(error)
```

**Acceptance Criteria:**
- ✅ File `color_correction/exceptions.py` created
- ✅ All exceptions inherit from `ColorCorrectionError`
- ✅ Each exception has clear docstring
- ✅ Exceptions exported from package `__init__.py`
- ✅ Tests pass: `pytest tests/test_exceptions.py -v`

---

### Task 2: Create Input Validation Utilities
**Priority:** 🔴 HIGH
**Estimated Time:** 2 hours
**Dependencies:** Task 1 (exceptions)

#### Subtasks

**2.1. Create validators.py file**
- **File:** `color_correction/utils/validators.py`
- **Lines of code:** ~120-150

```python
"""Input validation utilities for color correction operations."""

import numpy as np
from numpy.typing import NDArray

from color_correction.exceptions import InvalidImageError, InvalidBoxError
from color_correction.schemas.custom_types import ImageBGR


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
        raise InvalidImageError(
            f"{param_name} must be numpy array, got {type(image).__name__}"
        )

    if image.ndim != 3:
        raise InvalidImageError(
            f"{param_name} must have 3 dimensions (H, W, C), got {image.ndim}"
        )

    if image.shape[2] != 3:
        raise InvalidImageError(
            f"{param_name} must have 3 channels (BGR), got {image.shape[2]}"
        )

    height, width = image.shape[:2]
    if height < min_height or width < min_width:
        raise InvalidImageError(
            f"{param_name} too small: {height}x{width}, "
            f"minimum required: {min_height}x{min_width}"
        )

    if image.size == 0:
        raise InvalidImageError(f"{param_name} is empty")


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
        raise ValueError(
            f"{param_name} must have {expected_count} items, got {len(patches)}"
        )

    if not allow_none:
        none_count = sum(1 for p in patches if p is None)
        if none_count > 0:
            raise ValueError(f"{param_name} contains {none_count} None values")


def validate_box(
    box: tuple[int, int, int, int],
    param_name: str = "box",
) -> None:
    """Validate bounding box coordinates.

    Parameters
    ----------
    box : tuple[int, int, int, int]
        Bounding box in (x1, y1, x2, y2) format
    param_name : str, optional
        Parameter name for error messages

    Raises
    ------
    InvalidBoxError
        If box coordinates are invalid
    """
    if not isinstance(box, (tuple, list)):
        raise InvalidBoxError(box, "must be tuple or list")

    if len(box) != 4:
        raise InvalidBoxError(box, f"must have 4 elements, got {len(box)}")

    x1, y1, x2, y2 = box

    if not all(isinstance(coord, (int, float)) for coord in box):
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
    if not isinstance(threshold, (int, float)):
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
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"{param_name} must be numeric, got {type(threshold).__name__}")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{param_name} must be between 0 and 1, got {threshold}")
```

**2.2. Write unit tests**
- **File:** `tests/utils/test_validators.py`

```python
"""Tests for validation utilities."""

import numpy as np
import pytest

from color_correction.exceptions import InvalidImageError, InvalidBoxError
from color_correction.utils.validators import (
    validate_bgr_image,
    validate_patches,
    validate_box,
    validate_confidence_threshold,
)


class TestValidateBGRImage:
    """Tests for validate_bgr_image function."""

    def test_valid_image(self):
        """Test that valid BGR image passes validation."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        validate_bgr_image(image)  # Should not raise

    def test_none_image(self):
        """Test that None image raises error."""
        with pytest.raises(InvalidImageError, match="cannot be None"):
            validate_bgr_image(None)

    def test_wrong_dimensions(self):
        """Test that wrong number of dimensions raises error."""
        image = np.zeros((100, 100), dtype=np.uint8)  # 2D instead of 3D
        with pytest.raises(InvalidImageError, match="must have 3 dimensions"):
            validate_bgr_image(image)

    def test_wrong_channels(self):
        """Test that wrong number of channels raises error."""
        image = np.zeros((100, 100, 4), dtype=np.uint8)  # 4 channels
        with pytest.raises(InvalidImageError, match="must have 3 channels"):
            validate_bgr_image(image)

    def test_too_small(self):
        """Test that image smaller than minimum raises error."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="too small"):
            validate_bgr_image(image, min_height=50, min_width=50)


class TestValidateBox:
    """Tests for validate_box function."""

    def test_valid_box(self):
        """Test that valid box passes validation."""
        validate_box((10, 20, 100, 200))  # Should not raise

    def test_inverted_coordinates(self):
        """Test that inverted coordinates raise error."""
        with pytest.raises(InvalidBoxError, match="x1.*must be less than x2"):
            validate_box((100, 20, 10, 200))

    def test_negative_coordinates(self):
        """Test that negative coordinates raise error."""
        with pytest.raises(InvalidBoxError, match="cannot be negative"):
            validate_box((-10, 20, 100, 200))

    def test_wrong_length(self):
        """Test that wrong number of elements raises error."""
        with pytest.raises(InvalidBoxError, match="must have 4 elements"):
            validate_box((10, 20, 100))


class TestValidateConfidenceThreshold:
    """Tests for validate_confidence_threshold function."""

    def test_valid_threshold(self):
        """Test that valid threshold passes validation."""
        validate_confidence_threshold(0.5)  # Should not raise

    def test_out_of_range_high(self):
        """Test that threshold > 1 raises error."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_confidence_threshold(1.5)

    def test_out_of_range_low(self):
        """Test that threshold < 0 raises error."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_confidence_threshold(-0.1)
```

**Acceptance Criteria:**
- ✅ File `color_correction/utils/validators.py` created
- ✅ All validation functions have clear docstrings
- ✅ Tests pass: `pytest tests/utils/test_validators.py -v`
- ✅ Coverage for validators.py > 90%

---

### Task 3: Extract Magic Numbers to Constants
**Priority:** 🟡 MEDIUM
**Estimated Time:** 3 hours
**Dependencies:** None

#### Subtasks

**3.1. Create grid_config.py**
- **File:** `color_correction/constant/grid_config.py`

```python
"""Configuration constants for color checker card grid layout.

This module defines the standard layout for the X-Rite ColorChecker Classic
24-patch card, which has a 6x4 grid (6 columns, 4 rows = 24 patches total).
"""

from typing import Final

# ============================================================================
# Grid Dimensions
# ============================================================================

GRID_ROWS: Final[int] = 4
"""Number of rows in the color checker grid."""

GRID_COLS: Final[int] = 6
"""Number of columns in the color checker grid."""

TOTAL_PATCHES: Final[int] = GRID_ROWS * GRID_COLS  # 24
"""Total number of patches in the color checker card."""


# ============================================================================
# Grid Position Indices
# ============================================================================

ROW_END_INDICES: Final[frozenset[int]] = frozenset([5, 11, 17, 23])
"""Indices of patches at the end of each row (rightmost column)."""

ROW_START_INDICES: Final[frozenset[int]] = frozenset([0, 6, 12, 18])
"""Indices of patches at the start of each row (leftmost column)."""

COL_END_INDICES: Final[frozenset[int]] = frozenset(range(18, 24))
"""Indices of patches in the last row (bottom row)."""

COL_START_INDICES: Final[frozenset[int]] = frozenset(range(0, 6))
"""Indices of patches in the first row (top row)."""


# ============================================================================
# Neighbor Offsets
# ============================================================================

NEIGHBOR_RIGHT_OFFSET: Final[int] = 1
"""Index offset to get right neighbor patch."""

NEIGHBOR_LEFT_OFFSET: Final[int] = -1
"""Index offset to get left neighbor patch."""

NEIGHBOR_BOTTOM_OFFSET: Final[int] = GRID_COLS  # 6
"""Index offset to get bottom neighbor patch (next row)."""

NEIGHBOR_TOP_OFFSET: Final[int] = -GRID_COLS  # -6
"""Index offset to get top neighbor patch (previous row)."""


# ============================================================================
# Visualization Defaults
# ============================================================================

DEFAULT_GRID_FIGSIZE_WIDTH: Final[int] = 15
"""Default figure width for grid visualizations."""

DEFAULT_GRID_FIGSIZE_HEIGHT_PER_ROW: Final[int] = 4
"""Default figure height per row for grid visualizations."""


# ============================================================================
# Detection Defaults
# ============================================================================

MIN_PATCHES_REQUIRED: Final[int] = 24
"""Minimum number of patches required for valid detection."""

DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.25
"""Default confidence threshold for card detection."""

DEFAULT_IOU_THRESHOLD: Final[float] = 0.7
"""Default Intersection over Union threshold for NMS."""


# ============================================================================
# Helper Functions
# ============================================================================

def is_row_end(index: int) -> bool:
    """Check if patch index is at row end (rightmost column).

    Parameters
    ----------
    index : int
        Patch index (0-23)

    Returns
    -------
    bool
        True if patch is at row end
    """
    return index in ROW_END_INDICES


def is_row_start(index: int) -> bool:
    """Check if patch index is at row start (leftmost column).

    Parameters
    ----------
    index : int
        Patch index (0-23)

    Returns
    -------
    bool
        True if patch is at row start
    """
    return index in ROW_START_INDICES


def get_row_number(index: int) -> int:
    """Get row number (0-3) for a given patch index.

    Parameters
    ----------
    index : int
        Patch index (0-23)

    Returns
    -------
    int
        Row number (0 for top row, 3 for bottom row)
    """
    return index // GRID_COLS


def get_col_number(index: int) -> int:
    """Get column number (0-5) for a given patch index.

    Parameters
    ----------
    index : int
        Patch index (0-23)

    Returns
    -------
    int
        Column number (0 for leftmost, 5 for rightmost)
    """
    return index % GRID_COLS
```

**3.2. Update geometry_processing.py**
- **File:** `color_correction/utils/geometry_processing.py`
- **Changes:** Replace hardcoded values with constants

```python
# Add imports at top
from color_correction.constant.grid_config import (
    GRID_ROWS,
    GRID_COLS,
    TOTAL_PATCHES,
    ROW_END_INDICES,
    ROW_START_INDICES,
    NEIGHBOR_RIGHT_OFFSET,
    NEIGHBOR_LEFT_OFFSET,
    NEIGHBOR_BOTTOM_OFFSET,
    NEIGHBOR_TOP_OFFSET,
)

# Update line 103-104
def generate_expected_patches(card_box: box_tuple) -> list[box_tuple]:
    """Generate a grid of expected patch coordinates within a card box."""
    card_x1, card_y1, card_x2, card_y2 = card_box
    card_width = card_x2 - card_x1
    card_height = card_y2 - card_y1

    # Use constants instead of hardcoded 6 and 4
    patch_width = card_width / GRID_COLS
    patch_height = card_height / GRID_ROWS

    expected_patches = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1 = int(card_x1 + col * patch_width)
            y1 = int(card_y1 + row * patch_height)
            x2 = int(x1 + patch_width)
            y2 = int(y1 + patch_height)
            expected_patches.append((x1, y1, x2, y2))

    return expected_patches


# Update line 186-200
def calculate_patch_statistics(ls_ordered_patch: list[box_tuple]) -> tuple:
    """Calculate mean differences in positions and sizes for patches."""
    ls_dx = []
    ls_dy = []
    ls_w_grid = []
    ls_h_grid = []
    for idx, patch in enumerate(ls_ordered_patch):
        if patch is None:
            continue

        ls_w_grid.append(patch[2] - patch[0])
        ls_h_grid.append(patch[3] - patch[1])

        # Use constants instead of hardcoded values
        if idx not in ROW_END_INDICES or idx == 0:
            x1 = patch[0]
            next_x1 = ls_ordered_patch[idx + 1]
            if next_x1 is not None:
                dx = next_x1[0] - x1
                ls_dx.append(dx)

        # Use constant instead of hardcoded 6
        syarat = idx + GRID_COLS
        if syarat < len(ls_ordered_patch):
            y1 = patch[1]
            next_y1 = ls_ordered_patch[idx + GRID_COLS]
            if next_y1 is not None:
                dy = next_y1[1] - y1
                ls_dy.append(dy)

    mean_dx = np.mean(ls_dx)
    mean_dy = np.mean(ls_dy)
    mean_w = np.mean(ls_w_grid)
    mean_h = np.mean(ls_h_grid)

    return mean_dx, mean_dy, mean_w, mean_h


# Update suggest_missing_patch_coordinates
# Replace hardcoded neighbor calculations with constants
def _find_neighbor_patches(
    idx: int,
    ls_ordered_patch: list[box_tuple],
) -> dict[str, box_tuple | None]:
    """Find neighboring patches for a given index."""
    neighbors = {
        'right': None,
        'left': None,
        'top': None,
        'bottom': None,
    }

    # Right neighbor
    id_right = idx + NEIGHBOR_RIGHT_OFFSET
    if id_right not in ROW_START_INDICES and id_right < TOTAL_PATCHES:
        neighbors['right'] = ls_ordered_patch[id_right]

    # Left neighbor
    id_left = idx + NEIGHBOR_LEFT_OFFSET
    if id_left not in ROW_END_INDICES and id_left >= 0:
        neighbors['left'] = ls_ordered_patch[id_left]

    # Top neighbor
    id_top = idx + NEIGHBOR_TOP_OFFSET
    if id_top >= 0:
        neighbors['top'] = ls_ordered_patch[id_top]

    # Bottom neighbor
    id_bottom = idx + NEIGHBOR_BOTTOM_OFFSET
    if id_bottom < TOTAL_PATCHES:
        neighbors['bottom'] = ls_ordered_patch[id_bottom]

    return neighbors
```

**3.3. Update other files using magic numbers**
- **File:** `color_correction/services/color_correction.py` (line 218)
- **File:** `color_correction/processor/detection.py`
- **File:** `color_correction/services/correction_analyzer.py`

**Acceptance Criteria:**
- ✅ File `color_correction/constant/grid_config.py` created
- ✅ All hardcoded 6, 4, 24 values replaced with constants
- ✅ All hardcoded indices [5, 11, 17, 23] replaced with `ROW_END_INDICES`
- ✅ All tests still pass after changes
- ✅ No ruff warnings introduced

---

### Task 4: Fix Type Definition Inconsistencies
**Priority:** 🔴 HIGH
**Estimated Time:** 1 hour
**Dependencies:** None

#### Subtasks

**4.1. Update methods.py**
- **File:** `color_correction/constant/methods.py`
- **Issue:** `LiteralModelDetection` only includes "yolov8", missing "mcc"

```python
"""Model type literal definitions."""

from typing import Literal

# Update to include both supported detection models
LiteralModelDetection = Literal["yolov8", "mcc"]

LiteralModelCorrection = Literal[
    "least_squares",
    "polynomial",
    "linear_reg",
    "affine_reg",
]
```

**4.2. Update custom_types.py**
- **File:** `color_correction/schemas/custom_types.py`
- **Action:** Remove duplicate type definitions, import from methods.py

```python
"""Custom type definitions for color correction."""

from typing import Any
import numpy as np
from numpy.typing import NDArray

# Import from methods.py instead of redefining
from color_correction.constant.methods import (
    LiteralModelDetection,
    LiteralModelCorrection,
)

# Image type aliases
ImageBGR = NDArray[np.uint8]
ImageRGB = NDArray[np.uint8]
ImageGray = NDArray[np.uint8]

# Color patch type
ColorPatchType = NDArray[np.uint8]

# Training model type
TrainedCorrection = Any

# Export all
__all__ = [
    "ImageBGR",
    "ImageRGB",
    "ImageGray",
    "ColorPatchType",
    "TrainedCorrection",
    "LiteralModelDetection",
    "LiteralModelCorrection",
]
```

**4.3. Verify imports across codebase**
- Run tests to ensure no import errors
- Check that type hints are consistent

**Acceptance Criteria:**
- ✅ `LiteralModelDetection` includes both "yolov8" and "mcc"
- ✅ No duplicate type definitions
- ✅ All imports work correctly
- ✅ Type hints are consistent across codebase
- ✅ Ruff passes with no errors

---

### Task 5: Remove Emojis from Docstrings
**Priority:** 🟢 LOW
**Estimated Time:** 1 hour
**Dependencies:** None

#### Subtasks

**5.1. Update color_correction.py docstrings**
- **File:** `color_correction/services/color_correction.py`
- **Lines to update:** 332-365

**Current docstring (line 332-365):**
```python
def set_input_patches(self, image: np.ndarray, debug: bool = False) -> None:
    """
    This function processes an input image to extract color patches and generates
    corresponding grid and debug visualizations 🔍


    Parameters
    ----------
    image : np.ndarray
        Input image to extract color patches from 📸
    debug : bool, optional
        If True, generates additional debug visualization, by default False 🐛
```

**Updated docstring:**
```python
def set_input_patches(self, image: np.ndarray, debug: bool = False) -> None:
    """Extract color patches from input image.

    This function processes an input image to detect color checker patches and
    generates corresponding grid and debug visualizations.

    Parameters
    ----------
    image : np.ndarray
        Input image to extract color patches from.
    debug : bool, optional
        If True, generates additional debug visualization. Default is False.

    Returns
    -------
    tuple
        Contains three elements:

        - input_patches : list[ColorPatchType]
            Extracted color patches from the image
        - input_grid_image : ImageBGR
            Visualization of the detected grid
        - input_debug_image : ImageBGR | None
            Debug visualization (if debug=True)

    Notes
    -----
    The function will set the following instance attributes:

    - self.input_patches
    - self.input_grid_image
    - self.input_debug_image

    These attributes are reset to None before processing.
    """
```

**5.2. Review all docstrings for consistency**
- Search for any other emojis in docstrings
- Ensure consistent NumPy-style formatting
- Fix any inconsistent parameter descriptions

**Acceptance Criteria:**
- ✅ No emojis in any docstrings
- ✅ All docstrings follow NumPy style consistently
- ✅ Documentation builds successfully: `mkdocs build`

---

### Task 6: Apply Custom Exceptions Throughout Codebase
**Priority:** 🔴 HIGH
**Estimated Time:** 4 hours
**Dependencies:** Task 1, Task 2

#### Subtasks

**6.1. Update ColorCorrection service**
- **File:** `color_correction/services/color_correction.py`

**Changes:**

```python
# Add imports at top
from color_correction.exceptions import (
    PatchesNotSetError,
    ModelNotFittedError,
    UnsupportedModelError,
    InvalidImageError,
)
from color_correction.utils.validators import validate_bgr_image

# Update __init__ method - add validation
def __init__(
    self,
    detection_model: LiteralModelDetection = "mcc",
    detection_conf_th: float = 0.25,
    correction_model: LiteralModelCorrection = "least_squares",
    reference_image: ImageBGR | None = None,
    use_gpu: bool = False,
    **kwargs: dict,
) -> None:
    # Validate reference_image if provided
    if reference_image is not None:
        validate_bgr_image(reference_image, param_name="reference_image")

    # ... rest of __init__

# Update _create_detector method (line 108-139)
def _create_detector(
    self,
    model_name: str,
    conf_th: float = 0.25,
    use_gpu: bool = False,
) -> YOLOv8CardDetector | MCCardDetector:
    """Create a card detector instance."""
    supported_models = ["yolov8", "mcc"]
    if model_name not in supported_models:
        raise UnsupportedModelError(model_name, supported_models)

    if model_name == "mcc":
        return MCCardDetector(use_gpu=use_gpu, conf_th=conf_th)
    return YOLOv8CardDetector(use_gpu=use_gpu, conf_th=conf_th)

# Update set_input_patches (line 332)
def set_input_patches(self, image: np.ndarray, debug: bool = False) -> None:
    """Extract color patches from input image."""
    # Add validation
    validate_bgr_image(image, param_name="image")

    # Reset attributes
    self.input_patches = None
    self.input_grid_image = None
    self.input_debug_image = None

    # Extract patches
    (
        self.input_patches,
        self.input_grid_image,
        self.input_debug_image,
    ) = self._extract_color_patches(image=image, debug=debug)

    return self.input_patches, self.input_grid_image, self.input_debug_image

# Update fit method (line 378)
def fit(self) -> TrainedCorrection:
    """Fit the color correction model."""
    if self.reference_patches is None:
        raise PatchesNotSetError("reference")

    if self.input_patches is None:
        raise PatchesNotSetError("input")

    # ... rest of fit

# Update predict method (line 428)
def predict(
    self,
    input_image: ImageBGR,
    debug: bool = False,
    debug_output_dir: str = "output-debug",
) -> ImageBGR:
    """Apply color correction to input image."""
    # Add validation
    validate_bgr_image(input_image, param_name="input_image")

    if self.trained_model is None:
        raise ModelNotFittedError()

    # ... rest of predict
```

**6.2. Update YOLOv8CardDetector**
- **File:** `color_correction/core/card_detection/det_yv8_onnx.py`

```python
# Add imports
from color_correction.exceptions import ModelLoadError, InvalidImageError
from color_correction.utils.validators import validate_bgr_image

# Update __initialize_model (line 108)
def __initialize_model(self, path: str) -> None:
    """Initialize ONNX model."""
    try:
        self.session = onnxruntime.InferenceSession(
            path,
            providers=onnxruntime.get_available_providers(),
        )
    except Exception as e:
        raise ModelLoadError(path, str(e)) from e

    # Get model info
    self.__get_input_details()
    self.__get_output_details()

# Update detect method (line 80)
def detect(self, image: np.ndarray) -> DetectionResult:
    """Detect objects in the given image."""
    # Add validation
    validate_bgr_image(image, param_name="image", min_height=32, min_width=32)

    input_tensor = self.__prepare_input(image.copy())
    outputs = self.__inference(input_tensor)
    boxes, scores, class_ids = self.__process_output(outputs)

    det_res = DetectionResult(
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
    )

    return det_res
```

**6.3. Update geometry_processing.py**
- **File:** `color_correction/utils/geometry_processing.py`

```python
# Add imports
from color_correction.exceptions import InvalidBoxError
from color_correction.utils.validators import validate_box

# Add validation to box-related functions
def box_to_xyxy(box: shapely.geometry.box) -> tuple[int, int, int, int]:
    """Convert a Shapely box to (x1, y1, x2, y2) format."""
    minx, miny, maxx, maxy = box.bounds
    result = (int(minx), int(miny), int(maxx), int(maxy))
    validate_box(result)  # Add validation
    return result
```

**Acceptance Criteria:**
- ✅ All generic `ValueError` and `RuntimeError` replaced with custom exceptions
- ✅ All public methods have input validation
- ✅ Helpful error messages with actionable guidance
- ✅ All existing tests still pass
- ✅ New validation catches invalid inputs

---

### Task 7: Update Tests for New Validation
**Priority:** 🔴 HIGH
**Estimated Time:** 3 hours
**Dependencies:** Task 6

#### Subtasks

**7.1. Add tests for validation errors**
- **File:** `tests/services/test_color_correction_validation.py` (new file)

```python
"""Tests for ColorCorrection input validation."""

import numpy as np
import pytest

from color_correction.services.color_correction import ColorCorrection
from color_correction.exceptions import (
    InvalidImageError,
    PatchesNotSetError,
    ModelNotFittedError,
    UnsupportedModelError,
)


class TestColorCorrectionValidation:
    """Test input validation in ColorCorrection."""

    def test_invalid_detection_model(self):
        """Test that invalid detection model raises error."""
        with pytest.raises(UnsupportedModelError, match="invalid_model"):
            ColorCorrection(detection_model="invalid_model")

    def test_set_input_patches_none_image(self):
        """Test that None image raises error."""
        cc = ColorCorrection()
        with pytest.raises(InvalidImageError, match="cannot be None"):
            cc.set_input_patches(image=None)

    def test_set_input_patches_wrong_dimensions(self):
        """Test that wrong image dimensions raise error."""
        cc = ColorCorrection()
        image = np.zeros((100, 100), dtype=np.uint8)  # 2D instead of 3D
        with pytest.raises(InvalidImageError, match="must have 3 dimensions"):
            cc.set_input_patches(image=image)

    def test_fit_without_reference_patches(self):
        """Test that fit without reference patches raises error."""
        cc = ColorCorrection(reference_image=None)
        cc.reference_patches = None  # Explicitly set to None
        with pytest.raises(PatchesNotSetError, match="Reference"):
            cc.fit()

    def test_fit_without_input_patches(self):
        """Test that fit without input patches raises error."""
        cc = ColorCorrection()
        with pytest.raises(PatchesNotSetError, match="Input"):
            cc.fit()

    def test_predict_without_fit(self):
        """Test that predict without fit raises error."""
        cc = ColorCorrection()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ModelNotFittedError, match="must be fitted"):
            cc.predict(image)

    def test_predict_invalid_image(self):
        """Test that predict with invalid image raises error."""
        cc = ColorCorrection()
        # Simulate fitted model
        cc.trained_model = "dummy"
        with pytest.raises(InvalidImageError, match="cannot be None"):
            cc.predict(None)
```

**7.2. Update existing tests**
- Review and update tests that expect generic exceptions
- Ensure tests use new custom exceptions
- Add parametrized tests for edge cases

**Acceptance Criteria:**
- ✅ All tests pass: `pytest -v`
- ✅ Test coverage for new validation code
- ✅ Edge cases covered (None, wrong types, invalid values)

---

### Task 8: Documentation Updates
**Priority:** 🟢 LOW
**Estimated Time:** 2 hours
**Dependencies:** All previous tasks

#### Subtasks

**8.1. Update docstrings with new exceptions**
- Update all method docstrings to document new exceptions in "Raises" section
- Example:

```python
def set_input_patches(self, image: np.ndarray, debug: bool = False) -> None:
    """Extract color patches from input image.

    Parameters
    ----------
    image : np.ndarray
        Input image to extract color patches from.
    debug : bool, optional
        If True, generates additional debug visualization. Default is False.

    Returns
    -------
    tuple
        Contains input_patches, input_grid_image, input_debug_image

    Raises
    ------
    InvalidImageError
        If image is None, has wrong dimensions, or wrong number of channels.
    InsufficientPatchesError
        If fewer than 24 patches are detected in the image.

    Notes
    -----
    This method resets input_patches, input_grid_image, and input_debug_image
    before processing.
    """
```

**8.2. Create exceptions documentation**
- **File:** `docs/reference/exceptions.md`

```markdown
# Exceptions Reference

This page documents all custom exceptions in the color-correction package.

## Exception Hierarchy

```
ColorCorrectionError
├── DetectionError
│   ├── NoCardDetectedError
│   ├── InsufficientPatchesError
│   └── LowConfidenceDetectionError
├── CorrectionError
│   ├── ModelNotFittedError
│   └── PatchesNotSetError
├── InvalidImageError
├── ModelLoadError
├── UnsupportedModelError
└── GeometryError
    └── InvalidBoxError
```

## Base Exceptions

::: color_correction.exceptions.ColorCorrectionError

## Detection Exceptions

::: color_correction.exceptions.DetectionError
::: color_correction.exceptions.NoCardDetectedError
::: color_correction.exceptions.InsufficientPatchesError

... (continue for all exceptions)
```

**8.3. Update mkdocs.yml**
- Add exceptions page to navigation

```yaml
nav:
  - Home: index.md
  - Tutorial:
      - Getting Started: tutorial/getting_started.md
      - ...
  - Reference:
      - Exceptions: reference/exceptions.md  # Add this
      - API: reference/
```

**8.4. Update README**
- Add section about error handling
- Show example of catching exceptions

**Acceptance Criteria:**
- ✅ All method docstrings updated with "Raises" sections
- ✅ Exceptions documentation page created
- ✅ Documentation builds successfully: `mkdocs build`
- ✅ README updated with error handling examples

---

## Phase 1 Checklist

### Pre-Implementation
- [ ] Create feature branch: `git checkout -b feature/phase1-foundation`
- [ ] Review all task descriptions
- [ ] Ensure development environment is set up
- [ ] Run initial tests to establish baseline: `pytest -v`

### Implementation (in order)
- [ ] **Task 1:** Create custom exception hierarchy (2 hours)
  - [ ] 1.1. Create `exceptions.py`
  - [ ] 1.2. Update `__init__.py` exports
  - [ ] 1.3. Write unit tests
  - [ ] Verify: `pytest tests/test_exceptions.py -v`

- [ ] **Task 2:** Create validation utilities (2 hours)
  - [ ] 2.1. Create `validators.py`
  - [ ] 2.2. Write unit tests
  - [ ] Verify: `pytest tests/utils/test_validators.py -v`

- [ ] **Task 4:** Fix type definitions (1 hour)
  - [ ] 4.1. Update `methods.py`
  - [ ] 4.2. Update `custom_types.py`
  - [ ] 4.3. Verify imports
  - [ ] Verify: `ruff check`

- [ ] **Task 3:** Extract magic numbers (3 hours)
  - [ ] 3.1. Create `grid_config.py`
  - [ ] 3.2. Update `geometry_processing.py`
  - [ ] 3.3. Update other files
  - [ ] Verify: All tests still pass

- [ ] **Task 5:** Remove emojis (1 hour)
  - [ ] 5.1. Update `color_correction.py` docstrings
  - [ ] 5.2. Review all other docstrings
  - [ ] Verify: `mkdocs build`

- [ ] **Task 6:** Apply exceptions throughout (4 hours)
  - [ ] 6.1. Update `ColorCorrection` service
  - [ ] 6.2. Update `YOLOv8CardDetector`
  - [ ] 6.3. Update `geometry_processing.py`
  - [ ] Verify: Existing tests still pass

- [ ] **Task 7:** Update tests (3 hours)
  - [ ] 7.1. Add validation tests
  - [ ] 7.2. Update existing tests
  - [ ] Verify: `pytest -v --cov`

- [ ] **Task 8:** Documentation (2 hours)
  - [ ] 8.1. Update docstrings with Raises sections
  - [ ] 8.2. Create exceptions documentation page
  - [ ] 8.3. Update mkdocs.yml
  - [ ] 8.4. Update README
  - [ ] Verify: `mkdocs build && mkdocs serve`

### Post-Implementation
- [ ] Run full test suite: `pytest -v --cov`
- [ ] Run linter: `ruff check color_correction/`
- [ ] Run formatter: `ruff format color_correction/`
- [ ] Build documentation: `mkdocs build`
- [ ] Review all changes
- [ ] Commit changes with descriptive message
- [ ] Push branch
- [ ] Create pull request

---

## Success Metrics

At the end of Phase 1, you should have:

| Metric | Target | How to Verify |
|--------|--------|---------------|
| New files created | 5+ | `git status` |
| Test coverage | Maintained or improved | `pytest --cov` |
| Ruff checks | 0 errors | `ruff check` |
| Documentation builds | Success | `mkdocs build` |
| All tests passing | 100% | `pytest -v` |
| Custom exceptions | 10+ | Count in `exceptions.py` |
| Magic numbers removed | 95%+ | Code review |
| Emojis in docstrings | 0 | `grep -r "📸\\|🔍\\|🐛" color_correction/` |

---

## Estimated Timeline

| Week | Days | Tasks | Hours |
|------|------|-------|-------|
| Week 1 | Mon-Wed | Tasks 1, 2, 4 | 5 hours |
| Week 1 | Thu-Fri | Task 3 | 3 hours |
| Week 2 | Mon-Tue | Tasks 5, 6 | 5 hours |
| Week 2 | Wed-Thu | Task 7 | 3 hours |
| Week 2 | Fri | Task 8, Testing, PR | 4 hours |
| **Total** | **10 days** | **8 tasks** | **20 hours** |

---

## Common Issues & Solutions

### Issue: Import circular dependencies
**Solution:** Ensure validators.py doesn't import from modules that import validators

### Issue: Tests failing after exception changes
**Solution:** Update test assertions to expect new exception types

### Issue: Ruff errors after changes
**Solution:** Run `ruff format` and `ruff check --fix`

### Issue: Documentation not building
**Solution:** Check for syntax errors in docstrings, ensure all references exist

---

## Next Steps

After completing Phase 1:

1. **Review & Merge:** Create PR and request code review
2. **Plan Phase 2:** Begin planning test coverage expansion
3. **Document Learnings:** Note any issues for future phases
4. **Celebrate:** You've laid a solid foundation!

Phase 2 (Testing) will build on this foundation by:
- Using custom exceptions in new tests
- Validating that error messages are helpful
- Testing edge cases with validators
- Achieving 70%+ test coverage
