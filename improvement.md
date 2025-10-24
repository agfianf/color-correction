# Code Quality Improvement Plan

## Executive Summary

This document outlines a comprehensive improvement plan for the `color-correction` project based on Clean Code principles and best practices. The codebase demonstrates strong fundamentals with modern Python practices, type safety, and good architecture. However, there are opportunities to enhance maintainability, testability, and code quality further.

**Current State:** Pre-release (v0.0.1-rc5) with solid foundation
**Code Quality Score:** 7.5/10
**Test Coverage:** 35% (minimum enforced)
**Architecture:** Layered with clear separation of concerns

---

## Priority Matrix

| Priority | Category | Impact | Effort |
|----------|----------|--------|--------|
| 🔴 HIGH | Test Coverage | High | Medium |
| 🔴 HIGH | Error Handling | High | Low |
| 🟡 MEDIUM | Code Complexity | Medium | Medium |
| 🟡 MEDIUM | Magic Numbers | Medium | Low |
| 🟢 LOW | Documentation | Low | Low |
| 🟢 LOW | Code Duplication | Low | Medium |

---

## 1. Testing Improvements (Priority: 🔴 HIGH)

### Current State
- **Coverage:** 35% minimum (CI-enforced)
- **Test Files:** 9 files, ~458 lines
- **Coverage Gaps:**
  - Service layer completely untested (ColorCorrection, ColorCorrectionAnalyzer)
  - Core detection models minimally tested
  - Report generation untested
  - Geometry processing utilities untested

### Recommendations

#### 1.1 Increase Test Coverage to 70%+

**Action Items:**
1. **Service Layer Tests** (Priority: 🔴 CRITICAL)
   - Create `tests/services/test_color_correction.py`
   - Create `tests/services/test_correction_analyzer.py`
   - Test complete workflows end-to-end
   - Mock detector and correction models for unit tests

   ```python
   # Example test structure
   def test_color_correction_workflow():
       """Test complete color correction pipeline"""
       # Given: A ColorCorrection instance with test image
       # When: set_input_patches -> fit -> predict
       # Then: Output image has correct shape and values
   ```

2. **Detection Model Tests** (Priority: 🔴 HIGH)
   - `tests/core/card_detection/test_mcc_detector.py`
   - `tests/core/card_detection/test_detector_factory.py`
   - Test edge cases: no detection, multiple cards, low confidence

3. **Correction Model Tests** (Priority: 🔴 HIGH)
   - `tests/core/correction/test_polynomial.py`
   - `tests/core/correction/test_affine_reg.py`
   - `tests/core/correction/test_linear_reg.py`
   - `tests/core/correction/test_factory.py`
   - Test with various input shapes and edge cases

4. **Geometry Processing Tests** (Priority: 🟡 MEDIUM)
   - `tests/utils/test_geometry_processing.py`
   - Test `suggest_missing_patch_coordinates()` - complex logic
   - Test `extract_intersecting_patches()` - boundary conditions
   - Test IoU calculations

5. **Integration Tests** (Priority: 🟡 MEDIUM)
   - Create `tests/integration/` directory
   - Test detector + correction pipelines
   - Test with real sample images (checked into repo)

**Metrics:**
- Target coverage: **70%** (up from 35%)
- Timeline: 2-3 weeks
- Files to create: ~8 new test files

#### 1.2 Add Property-Based Testing

Use `hypothesis` for complex geometric operations:

```python
from hypothesis import given, strategies as st

@given(
    boxes=st.lists(
        st.tuples(
            st.integers(0, 1000),  # x1
            st.integers(0, 1000),  # y1
            st.integers(0, 1000),  # x2
            st.integers(0, 1000),  # y2
        ),
        min_size=1,
        max_size=24
    )
)
def test_generate_expected_patches_invariants(boxes):
    """Test that patch generation maintains invariants"""
    # Test properties that should always hold
```

#### 1.3 Add Performance Benchmarks

Create `tests/benchmarks/` for performance regression detection:

```python
import pytest

@pytest.mark.benchmark
def test_detection_performance(benchmark):
    """Ensure detection completes within acceptable time"""
    result = benchmark(detector.detect, test_image)
    assert result is not None
```

---

## 2. Error Handling & Validation (Priority: 🔴 HIGH)

### Current State
- Basic `RuntimeError` for missing state
- `ValueError` for invalid model names
- Limited input validation
- No custom exception hierarchy

### Recommendations

#### 2.1 Create Custom Exception Hierarchy

**Location:** `color_correction/exceptions.py`

```python
"""Custom exceptions for color correction operations."""

class ColorCorrectionError(Exception):
    """Base exception for all color correction errors."""
    pass

class DetectionError(ColorCorrectionError):
    """Raised when card detection fails."""
    pass

class NoCardDetectedError(DetectionError):
    """Raised when no color checker card is found."""
    pass

class InsufficientPatchesError(DetectionError):
    """Raised when detected patches are incomplete (< 24)."""
    def __init__(self, detected_count: int, required_count: int = 24):
        self.detected_count = detected_count
        self.required_count = required_count
        super().__init__(
            f"Detected {detected_count} patches, but {required_count} are required"
        )

class CorrectionError(ColorCorrectionError):
    """Raised when color correction fails."""
    pass

class ModelNotFittedError(CorrectionError):
    """Raised when attempting to predict with unfitted model."""
    pass

class InvalidImageError(ColorCorrectionError):
    """Raised when input image is invalid."""
    pass

class ModelLoadError(ColorCorrectionError):
    """Raised when model loading fails."""
    pass
```

**Impact:**
- Better error messages for users
- Easier to catch specific errors
- Improved debugging experience

#### 2.2 Add Input Validation

**File:** `color_correction/services/color_correction.py`

Add validation decorators or helper functions:

```python
def _validate_image(image: np.ndarray, param_name: str = "image") -> None:
    """Validate that input is a valid BGR image."""
    if image is None:
        raise InvalidImageError(f"{param_name} cannot be None")

    if not isinstance(image, np.ndarray):
        raise InvalidImageError(
            f"{param_name} must be numpy array, got {type(image)}"
        )

    if image.ndim != 3:
        raise InvalidImageError(
            f"{param_name} must have 3 dimensions (H, W, C), got {image.ndim}"
        )

    if image.shape[2] != 3:
        raise InvalidImageError(
            f"{param_name} must have 3 channels (BGR), got {image.shape[2]}"
        )

    if image.size == 0:
        raise InvalidImageError(f"{param_name} is empty")
```

**Apply to methods:**

```python
def set_input_patches(self, image: np.ndarray, debug: bool = False) -> None:
    """Set input patches with validation."""
    self._validate_image(image, "image")  # Add validation

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

    # Validate extraction succeeded
    if self.input_patches is None or len(self.input_patches) < 24:
        raise InsufficientPatchesError(
            detected_count=len(self.input_patches) if self.input_patches else 0
        )

    return self.input_patches, self.input_grid_image, self.input_debug_image
```

#### 2.3 Replace Generic Exceptions

**Current code (line 409-413 in color_correction.py):**
```python
if self.reference_patches is None:
    raise RuntimeError("Reference patches must be set before fitting model")

if self.input_patches is None:
    raise RuntimeError("Input patches must be set before fitting model")
```

**Improved:**
```python
if self.reference_patches is None:
    raise ModelNotFittedError(
        "Reference patches must be set before fitting. "
        "Call set_reference_patches() first."
    )

if self.input_patches is None:
    raise ModelNotFittedError(
        "Input patches must be set before fitting. "
        "Call set_input_patches() first."
    )
```

**Files to modify:**
- `color_correction/services/color_correction.py` (lines 409-413, 455-456)
- `color_correction/core/card_detection/det_yv8_onnx.py`
- `color_correction/core/card_detection/mcc_det.py`

---

## 3. Code Complexity Reduction (Priority: 🟡 MEDIUM)

### Current State
- Some long functions (200+ lines in services)
- High cyclomatic complexity in `suggest_missing_patch_coordinates()` (noqa: C901)
- Deeply nested conditionals

### Recommendations

#### 3.1 Refactor Complex Functions

**File:** `color_correction/utils/geometry_processing.py:210`

**Current:** `suggest_missing_patch_coordinates()` - 104 lines, C901 complexity warning

**Refactor into smaller functions:**

```python
def _find_neighbor_patches(
    idx: int,
    ls_ordered_patch: list[box_tuple],
) -> dict[str, box_tuple | None]:
    """Find neighboring patches for a given index.

    Returns
    -------
    dict with keys: 'right', 'left', 'top', 'bottom'
    """
    neighbors = {
        'right': None,
        'left': None,
        'top': None,
        'bottom': None,
    }

    # Right neighbor (not at row boundary)
    id_right = idx + 1
    if id_right not in [0, 6, 12, 18] and id_right <= 23:
        neighbors['right'] = ls_ordered_patch[id_right]

    # Left neighbor
    id_left = idx - 1
    if id_left not in [5, 11, 17, 23] and id_left >= 0:
        neighbors['left'] = ls_ordered_patch[id_left]

    # Top neighbor
    id_top = idx - 6
    if id_top >= 0:
        neighbors['top'] = ls_ordered_patch[id_top]

    # Bottom neighbor
    id_bottom = idx + 6
    if id_bottom <= 23:
        neighbors['bottom'] = ls_ordered_patch[id_bottom]

    return neighbors


def _suggest_from_neighbor(
    neighbor: box_tuple,
    direction: str,
    mean_dx: float,
    mean_dy: float,
    mean_w: float,
    mean_h: float,
) -> box_tuple:
    """Suggest patch coordinates based on a neighboring patch."""
    if direction == 'right':
        x1 = neighbor[0] - mean_dx
        y1 = neighbor[1]
    elif direction == 'left':
        x1 = neighbor[0] + mean_dx
        y1 = neighbor[1]
    elif direction == 'top':
        x1 = neighbor[0]
        y1 = neighbor[1] + mean_dy
    elif direction == 'bottom':
        x1 = neighbor[0]
        y1 = neighbor[1] - mean_dy
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return (
        int(x1),
        int(y1),
        int(x1 + mean_w),
        int(y1 + mean_h),
    )


def suggest_missing_patch_coordinates(
    ls_ordered_patch: list[box_tuple],
) -> dict[int, box_tuple]:
    """Suggest coordinates for missing patches based on neighbors.

    Simplified by extracting helper functions.
    """
    d_suggest = {}
    mean_dx, mean_dy, mean_w, mean_h = calculate_patch_statistics(ls_ordered_patch)

    for idx, patch in enumerate(ls_ordered_patch):
        if patch is not None:
            continue

        # Find all neighbors
        neighbors = _find_neighbor_patches(idx, ls_ordered_patch)

        # Try to suggest from first available neighbor (priority order)
        suggested_patch = None
        for direction in ['right', 'left', 'top', 'bottom']:
            if neighbors[direction] is not None:
                suggested_patch = _suggest_from_neighbor(
                    neighbor=neighbors[direction],
                    direction=direction,
                    mean_dx=mean_dx,
                    mean_dy=mean_dy,
                    mean_w=mean_w,
                    mean_h=mean_h,
                )
                break

        d_suggest[idx] = suggested_patch

    return d_suggest
```

**Benefits:**
- Reduced complexity from C901 to acceptable levels
- Each function has single responsibility
- Easier to test individual components
- Better readability

#### 3.2 Extract Long Methods in ColorCorrection Service

**File:** `color_correction/services/color_correction.py:171-221`

**Method:** `_save_debug_output()` - Could be simplified

**Refactor:**

```python
def _prepare_debug_image_collection(
    self,
    input_image: ImageBGR,
    corrected_image: ImageBGR,
) -> list[tuple[str, ImageBGR | None]]:
    """Prepare collection of images for debug visualization."""
    before_comparison = visualize_patch_comparison(
        ls_mean_in=self.input_patches,
        ls_mean_ref=self.reference_patches,
    )
    after_comparison = visualize_patch_comparison(
        ls_mean_in=self.corrected_patches,
        ls_mean_ref=self.reference_patches,
    )

    return [
        ("Input Image", input_image),
        ("Corrected Image", corrected_image),
        ("Debug Preprocess", self.input_debug_image),
        ("Reference vs Input", before_comparison),
        ("Reference vs Corrected", after_comparison),
        ("[Free Space]", None),
        ("Patch Input", self.input_grid_image),
        ("Patch Corrected", self.corrected_grid_image),
        ("Patch Reference", self.reference_grid_image),
    ]


def _save_debug_output(
    self,
    input_image: ImageBGR,
    corrected_image: ImageBGR,
    output_directory: str,
) -> None:
    """Save debug visualizations to disk."""
    run_dir = self._create_debug_directory(output_directory)
    image_collection = self._prepare_debug_image_collection(
        input_image, corrected_image
    )

    save_path = os.path.join(run_dir, "debug.jpg")
    create_image_grid_visualization(
        images=image_collection,
        grid_size=((len(image_collection) // 3) + 1, 3),
        figsize=(15, ((len(image_collection) // 3) + 1) * 4),
        save_path=save_path,
    )
    print(f"Debug output saved to: {save_path}")
```

---

## 4. Magic Numbers & Constants (Priority: 🟡 MEDIUM)

### Current State
- Hardcoded values scattered throughout code
- Some in constants files, some inline
- Patch grid dimensions (6×4) hardcoded in multiple places

### Recommendations

#### 4.1 Extract Magic Numbers to Constants

**Create:** `color_correction/constant/grid_config.py`

```python
"""Configuration constants for color checker card grid."""

# Color Checker Classic 24-patch grid dimensions
GRID_ROWS = 4
GRID_COLS = 6
TOTAL_PATCHES = GRID_ROWS * GRID_COLS  # 24

# Grid layout constants
ROW_END_INDICES = frozenset([5, 11, 17, 23])
ROW_START_INDICES = frozenset([0, 6, 12, 18])

# Visualization defaults
DEFAULT_GRID_FIGSIZE_HEIGHT = 4  # per row
DEFAULT_GRID_FIGSIZE_WIDTH = 15

# Detection defaults
MIN_PATCHES_REQUIRED = 24
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.7
```

**Usage in geometry_processing.py:**

```python
from color_correction.constant.grid_config import (
    GRID_ROWS,
    GRID_COLS,
    ROW_END_INDICES,
    ROW_START_INDICES,
    TOTAL_PATCHES,
)

def generate_expected_patches(card_box: box_tuple) -> list[box_tuple]:
    """Generate expected patch grid."""
    card_x1, card_y1, card_x2, card_y2 = card_box
    card_width = card_x2 - card_x1
    card_height = card_y2 - card_y1

    patch_width = card_width / GRID_COLS  # Instead of hardcoded 6
    patch_height = card_height / GRID_ROWS  # Instead of hardcoded 4

    expected_patches = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1 = int(card_x1 + col * patch_width)
            y1 = int(card_y1 + row * patch_height)
            x2 = int(x1 + patch_width)
            y2 = int(y1 + patch_height)
            expected_patches.append((x1, y1, x2, y2))

    return expected_patches
```

**Files to update:**
- `color_correction/utils/geometry_processing.py` (lines 103-104, 186-187, 248-251)
- `color_correction/services/color_correction.py` (line 218)
- `color_correction/processor/detection.py`

#### 4.2 Configuration for Model Defaults

**File:** `color_correction/constant/yolov8_det.py`

Add defaults:

```python
# Model configuration
DEFAULT_CONF_THRESHOLD = 0.15
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_INPUT_SIZE = (640, 640)

# Model paths
DEFAULT_MODEL_NAME = "yv8-det.onnx"
MODEL_CACHE_DIR = ".model"
```

---

## 5. Documentation Improvements (Priority: 🟢 LOW)

### Current State
- Good docstrings with NumPy style
- Type hints throughout
- MkDocs documentation
- Some inconsistencies (emojis in docstrings)

### Recommendations

#### 5.1 Remove Emojis from Docstrings

**File:** `color_correction/services/color_correction.py:332-365`

**Current:**
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

**Improved:**
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
```

**Rationale:**
- Consistency with rest of codebase
- Professional appearance
- Better compatibility with documentation generators

#### 5.2 Add Architecture Documentation

**Create:** `docs/architecture.md`

Include:
- System architecture diagram
- Component interactions
- Data flow diagrams
- Extension points

#### 5.3 Add Examples to Docstrings

For complex functions, add examples:

```python
def suggest_missing_patch_coordinates(
    ls_ordered_patch: list[box_tuple],
) -> dict[int, box_tuple]:
    """Suggest coordinates for missing patches based on neighbors.

    Parameters
    ----------
    ls_ordered_patch : list[box_tuple]
        List of 24 patch coordinates, with None for missing patches.

    Returns
    -------
    dict[int, box_tuple]
        Keys are indices of missing patches, values are suggested coordinates.

    Examples
    --------
    >>> patches = [None] * 24
    >>> patches[0] = (10, 10, 50, 40)
    >>> patches[1] = (60, 10, 100, 40)
    >>> suggestions = suggest_missing_patch_coordinates(patches)
    >>> len(suggestions)
    22
    """
```

---

## 6. Code Duplication (Priority: 🟢 LOW)

### Current State
- Some repeated patterns in test files
- Duplicate validation logic
- Similar image processing code

### Recommendations

#### 6.1 Create Test Fixtures

**File:** `tests/conftest.py`

Expand with common fixtures:

```python
import pytest
import numpy as np
import cv2

@pytest.fixture
def sample_color_checker_image():
    """Provide a sample color checker image for testing."""
    # Load from test assets
    return cv2.imread("tests/assets/sample_color_checker.jpg")

@pytest.fixture
def mock_detection_result():
    """Provide a mock DetectionResult."""
    return DetectionResult(
        boxes=[[10, 10, 50, 50]],
        scores=[0.95],
        class_ids=[0],
    )

@pytest.fixture
def reference_patches():
    """Provide standard reference patches."""
    from color_correction.constant.color_checker import reference_color_d50_bgr
    return reference_color_d50_bgr

@pytest.fixture
def color_corrector(sample_color_checker_image):
    """Provide a configured ColorCorrection instance."""
    from color_correction.services.color_correction import ColorCorrection
    cc = ColorCorrection(
        detection_model="yolov8",
        correction_model="least_squares",
        use_gpu=False,
    )
    cc.set_input_patches(sample_color_checker_image)
    return cc
```

#### 6.2 Extract Common Validation

**Create:** `color_correction/utils/validators.py`

```python
"""Common validation utilities."""

import numpy as np
from color_correction.exceptions import InvalidImageError

def validate_bgr_image(
    image: np.ndarray,
    param_name: str = "image",
    min_height: int = 1,
    min_width: int = 1,
) -> None:
    """Validate BGR image format."""
    if image is None:
        raise InvalidImageError(f"{param_name} cannot be None")

    if not isinstance(image, np.ndarray):
        raise InvalidImageError(
            f"{param_name} must be numpy array, got {type(image)}"
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(
            f"{param_name} must be (H, W, 3) BGR image, got shape {image.shape}"
        )

    height, width = image.shape[:2]
    if height < min_height or width < min_width:
        raise InvalidImageError(
            f"{param_name} too small: {height}x{width}, "
            f"minimum: {min_height}x{min_width}"
        )

def validate_patches(
    patches: list,
    expected_count: int = 24,
    param_name: str = "patches",
) -> None:
    """Validate color patches."""
    if patches is None:
        raise ValueError(f"{param_name} cannot be None")

    if len(patches) != expected_count:
        raise ValueError(
            f"{param_name} must have {expected_count} items, "
            f"got {len(patches)}"
        )
```

---

## 7. Type Safety Enhancements (Priority: 🟢 LOW)

### Current State
- Good type hints throughout
- `py.typed` marker present
- Custom type aliases defined

### Recommendations

#### 7.1 Add Runtime Type Checking (Optional)

For critical functions, consider `beartype` or `typeguard`:

```python
from beartype import beartype

@beartype
def compute_correction(
    self,
    input_image: ImageBGR,
) -> ImageBGR:
    """Type-checked at runtime."""
    ...
```

#### 7.2 Align Type Definitions

**File:** `color_correction/constant/methods.py` and `color_correction/schemas/custom_types.py`

**Current inconsistency:**
- `methods.py` defines `LiteralModelDetection = Literal["yolov8"]` (line missing "mcc")
- `custom_types.py` includes both: `Literal["yolov8", "mcc"]`

**Fix:**

Update `methods.py`:

```python
from typing import Literal

LiteralModelDetection = Literal["yolov8", "mcc"]
LiteralModelCorrection = Literal[
    "least_squares",
    "polynomial",
    "linear_reg",
    "affine_reg",
]
```

Remove duplicates from `custom_types.py` and import from `methods.py`.

---

## 8. Performance Optimizations (Priority: 🟢 LOW)

### Recommendations

#### 8.1 Profile Critical Paths

Add profiling decorators for performance monitoring:

```python
import functools
import time
from typing import Callable

def profile(func: Callable) -> Callable:
    """Profile function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper
```

#### 8.2 Cache Model Loading

The ONNX session could be cached to avoid reloading:

```python
from functools import lru_cache

@lru_cache(maxsize=4)
def load_onnx_model(path: str, use_gpu: bool):
    """Cache ONNX model loading."""
    return onnxruntime.InferenceSession(
        path,
        providers=onnxruntime.get_available_providers(),
    )
```

#### 8.3 Vectorize Operations

In `geometry_processing.py`, some loops could be vectorized:

```python
# Instead of loop
ls_w_grid = []
for patch in ls_ordered_patch:
    if patch is not None:
        ls_w_grid.append(patch[2] - patch[0])

# Use numpy
valid_patches = [p for p in ls_ordered_patch if p is not None]
patches_array = np.array(valid_patches)
widths = patches_array[:, 2] - patches_array[:, 0]
```

---

## 9. Logging & Observability (Priority: 🟢 LOW)

### Current State
- Print statements for debugging
- No structured logging
- README mentions TODO for logging

### Recommendations

#### 9.1 Replace Print with Logging

**File:** `color_correction/services/color_correction.py`

**Replace:**
```python
print("Extracting color patches from reference image", image.shape)
print(f"Debug output saved to: {save_path}")
```

**With:**
```python
import logging

logger = logging.getLogger(__name__)

logger.info("Extracting color patches from reference image, shape=%s", image.shape)
logger.debug("Debug output saved to: %s", save_path)
```

#### 9.2 Add Structured Logging Configuration

**Create:** `color_correction/utils/logging_config.py`

```python
"""Logging configuration for color correction."""

import logging
import sys

def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the package."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
```

---

## 10. CI/CD Enhancements (Priority: 🟢 LOW)

### Recommendations

#### 10.1 Add Code Quality Checks

**.github/workflows/quality.yml:**

```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --dev

      - name: Check complexity
        run: |
          uv run ruff check --select C901 --statistics

      - name: Check typing
        run: |
          uv run mypy color_correction/

      - name: Security check
        run: |
          uv run bandit -r color_correction/
```

#### 10.2 Add Dependency Scanning

Use Dependabot or similar:

**.github/dependabot.yml:**

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- ✅ Create custom exception hierarchy
- ✅ Add input validation to all public methods
- ✅ Extract magic numbers to constants
- ✅ Remove emojis from docstrings
- ✅ Fix type definition inconsistencies

**Deliverables:**
- `color_correction/exceptions.py`
- Updated `color_correction/constant/grid_config.py`
- Updated docstrings in `color_correction/services/color_correction.py`

### Phase 2: Testing (Weeks 3-5)
- ✅ Write service layer tests (70% of effort)
- ✅ Write detection model tests
- ✅ Write correction model tests
- ✅ Write geometry processing tests
- ✅ Add integration tests
- ✅ Update CI coverage threshold to 70%

**Deliverables:**
- 8+ new test files
- Coverage report showing 70%+

### Phase 3: Refactoring (Weeks 6-7)
- ✅ Refactor `suggest_missing_patch_coordinates()`
- ✅ Extract helper methods in `ColorCorrection`
- ✅ Create validation utilities
- ✅ Add test fixtures

**Deliverables:**
- Reduced cyclomatic complexity
- Better code organization

### Phase 4: Polish (Week 8)
- ✅ Replace print with logging
- ✅ Add performance profiling
- ✅ Update documentation
- ✅ Add CI/CD enhancements

**Deliverables:**
- Structured logging
- Performance benchmarks
- Enhanced CI workflows

---

## Metrics & Success Criteria

### Code Quality Metrics

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Test Coverage | 35% | 70%+ | 5 weeks |
| Cyclomatic Complexity | C901 warnings | < 10 per function | 2 weeks |
| Type Coverage | ~90% | 95%+ | 1 week |
| Documentation Coverage | Good | Excellent | 2 weeks |
| Lint Warnings | 0 | 0 | Maintain |

### Success Criteria

1. **Test Coverage:** Minimum 70% with all critical paths tested
2. **No Complexity Warnings:** All C901 warnings resolved
3. **Custom Exceptions:** All error cases use specific exceptions
4. **No Magic Numbers:** All hardcoded values extracted to constants
5. **CI Passing:** All quality checks green
6. **Documentation:** Architecture docs added

---

## Tools & Dependencies

### Development Tools
- **pytest-cov:** Coverage reporting (✅ already installed)
- **hypothesis:** Property-based testing (add)
- **pytest-benchmark:** Performance testing (add)
- **mypy:** Static type checking (add)
- **bandit:** Security scanning (add)

### Installation

```bash
# Add to pyproject.toml [dependency-groups.dev]
dev = [
    "pytest-cov==6.0.0",
    "pytest==8.3.5",
    "ruff==0.11.2",
    "pre-commit==4.2.0",
    "hypothesis==6.98.0",  # Add
    "pytest-benchmark==4.0.0",  # Add
    "mypy==1.8.0",  # Add
    "bandit==1.7.6",  # Add
]
```

---

## References

### Clean Code Principles Applied

1. **Single Responsibility Principle (SRP)**
   - Each function does one thing
   - Refactor long methods into smaller helpers

2. **Don't Repeat Yourself (DRY)**
   - Extract common validation logic
   - Create reusable test fixtures

3. **Meaningful Names**
   - Use descriptive constants instead of magic numbers
   - Clear function and variable names

4. **Error Handling**
   - Use specific exceptions
   - Provide helpful error messages

5. **Testing**
   - High coverage of critical paths
   - Test edge cases and error conditions

6. **Comments & Documentation**
   - Code should be self-documenting
   - Use docstrings for API documentation
   - Remove unnecessary comments

### Resources

- **Clean Code** by Robert C. Martin
- **Python Best Practices:** PEP 8, PEP 257
- **Testing:** pytest documentation
- **Type Hints:** PEP 484, mypy documentation

---

## Conclusion

This improvement plan provides a comprehensive roadmap to enhance the `color-correction` codebase from its current solid foundation (7.5/10) to production-ready excellence (9+/10). The priority-based approach ensures high-impact changes are implemented first, with clear metrics and success criteria.

**Estimated Timeline:** 8 weeks
**Estimated Effort:** ~80-120 hours
**Impact:** Significant improvement in maintainability, testability, and code quality

The modular nature of the improvements allows for incremental implementation - each phase delivers value independently.
