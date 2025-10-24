# Exception Reference

This page documents all custom exceptions in the color-correction package for better error handling and debugging.

## Exception Hierarchy

All exceptions in the package inherit from `ColorCorrectionError`, allowing you to catch all package-specific errors with a single exception handler.

```
ColorCorrectionError (base)
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

---

## Base Exception

### ColorCorrectionError

**Base exception for all color correction errors.**

All custom exceptions in this package inherit from this base class. You can use this to catch any error raised by the color-correction library.

**Example:**
```python
from color_correction import ColorCorrection
from color_correction.exceptions import ColorCorrectionError

try:
    cc = ColorCorrection(detection_model="invalid")
except ColorCorrectionError as e:
    print(f"Color correction error: {e}")
```

---

## Detection Exceptions

Exceptions related to color checker card detection.

### DetectionError

**Base class for detection-related errors.**

Parent class for all detection-specific exceptions.

### NoCardDetectedError

**Raised when no color checker card is found in the image.**

This error occurs when the detection algorithm cannot locate a color checker card in the provided image.

**Attributes:**
- `message` (str): Error message

**Example:**
```python
from color_correction.exceptions import NoCardDetectedError

# In detection code
if len(detections) == 0:
    raise NoCardDetectedError("No color checker card detected in image")
```

### InsufficientPatchesError

**Raised when detected patches are incomplete (fewer than 24).**

The ColorChecker Classic card has 24 patches in a 6×4 grid. This error is raised when fewer patches are detected.

**Attributes:**
- `detected_count` (int): Number of patches actually detected
- `required_count` (int): Number of patches required (default: 24)
- `message` (str): Formatted error message

**Example:**
```python
from color_correction.exceptions import InsufficientPatchesError

try:
    cc.set_input_patches(image)
except InsufficientPatchesError as e:
    print(f"Only found {e.detected_count} patches, need {e.required_count}")
```

### LowConfidenceDetectionError

**Raised when detection confidence is below acceptable threshold.**

This error indicates that the detector found something, but the confidence score is too low to be reliable.

**Attributes:**
- `max_confidence` (float): Highest confidence score among detections
- `threshold` (float): Required confidence threshold
- `message` (str): Helpful error message with suggestions

**Example:**
```python
from color_correction.exceptions import LowConfidenceDetectionError

try:
    detector.detect(low_quality_image)
except LowConfidenceDetectionError as e:
    print(f"Confidence {e.max_confidence:.2f} below threshold {e.threshold:.2f}")
    print("Try improving lighting or image quality")
```

---

## Correction Exceptions

Exceptions related to color correction operations.

### CorrectionError

**Base class for correction-related errors.**

Parent class for all correction-specific exceptions.

### ModelNotFittedError

**Raised when attempting to predict with an unfitted model.**

This error occurs when calling `predict()` or `compute_correction()` before calling `fit()` to train the correction model.

**Attributes:**
- `message` (str): Error message

**Example:**
```python
from color_correction import ColorCorrection
from color_correction.exceptions import ModelNotFittedError

cc = ColorCorrection()

try:
    corrected = cc.predict(image)
except ModelNotFittedError as e:
    print(f"Error: {e}")
    # Solution: call fit() first
    cc.set_input_patches(image)
    cc.fit()
    corrected = cc.predict(image)
```

### PatchesNotSetError

**Raised when required patches are not set before an operation.**

This error helps ensure that the workflow is followed correctly: patches must be set before fitting the model.

**Attributes:**
- `patch_type` (str): Type of patches not set ('reference' or 'input')
- `message` (str): Error message with method name to call

**Example:**
```python
from color_correction import ColorCorrection
from color_correction.exceptions import PatchesNotSetError

cc = ColorCorrection()

try:
    cc.fit()
except PatchesNotSetError as e:
    if e.patch_type == "input":
        print("Need to call set_input_patches() first")
        cc.set_input_patches(image)
    elif e.patch_type == "reference":
        print("Need to call set_reference_patches() first")
        cc.set_reference_patches(ref_image)
```

---

## Image & Input Exceptions

Exceptions related to image validation and input checking.

### InvalidImageError

**Raised when input image is invalid or has incorrect format.**

This error provides detailed information about why an image is invalid (wrong dimensions, wrong type, etc.).

**Attributes:**
- `reason` (str): Specific reason why the image is invalid
- `message` (str): Full error message

**Common reasons:**
- Image is None
- Not a numpy array
- Wrong number of dimensions (not 3D)
- Wrong number of channels (not 3 for BGR)
- Image too small

**Example:**
```python
from color_correction import ColorCorrection
from color_correction.exceptions import InvalidImageError
import numpy as np

cc = ColorCorrection()

# Example 1: Wrong dimensions
try:
    grayscale_image = np.zeros((100, 100), dtype=np.uint8)
    cc.set_input_patches(grayscale_image)
except InvalidImageError as e:
    print(f"Invalid image: {e.reason}")
    # Output: "image must have 3 dimensions (H, W, C), got 2"

# Example 2: Wrong type
try:
    cc.set_input_patches("not an array")
except InvalidImageError as e:
    print(f"Invalid image: {e.reason}")
    # Output: "image must be numpy array, got str"
```

---

## Model Exceptions

Exceptions related to model loading and selection.

### ModelLoadError

**Raised when model loading fails.**

This error occurs when a model file cannot be loaded from disk or downloaded.

**Attributes:**
- `model_path` (str): Path to the model that failed to load
- `message` (str): Error message including the path and reason

**Example:**
```python
from color_correction.exceptions import ModelLoadError

try:
    detector = YOLOv8CardDetector(path="/invalid/path/model.onnx")
except ModelLoadError as e:
    print(f"Failed to load model from: {e.model_path}")
```

### UnsupportedModelError

**Raised when an unsupported model type is requested.**

This error helps users discover which models are actually supported.

**Attributes:**
- `model_name` (str): Name of the unsupported model
- `supported_models` (list[str]): List of supported model names
- `message` (str): Error message listing supported models

**Example:**
```python
from color_correction import ColorCorrection
from color_correction.exceptions import UnsupportedModelError

try:
    cc = ColorCorrection(detection_model="resnet")
except UnsupportedModelError as e:
    print(f"Model '{e.model_name}' is not supported")
    print(f"Supported models: {', '.join(e.supported_models)}")
    # Output: Supported models: yolov8, mcc
```

---

## Geometry Exceptions

Exceptions related to geometric operations.

### GeometryError

**Base class for geometry-related errors.**

Parent class for geometric operation exceptions.

### InvalidBoxError

**Raised when bounding box coordinates are invalid.**

This error helps catch issues with bounding box format or values.

**Attributes:**
- `box` (tuple): The invalid box coordinates
- `message` (str): Error message explaining the issue

**Common issues:**
- Not a tuple or list
- Wrong number of elements (not 4)
- Coordinates not numeric
- x1 >= x2 or y1 >= y2
- Negative coordinates

**Example:**
```python
from color_correction.utils.validators import validate_box
from color_correction.exceptions import InvalidBoxError

# Example 1: Inverted coordinates
try:
    validate_box((100, 50, 10, 200))  # x1 > x2
except InvalidBoxError as e:
    print(f"Invalid box {e.box}: {e.message}")
    # Output: "Invalid box (100, 50, 10, 200): x1 (100) must be less than x2 (10)"

# Example 2: Negative coordinates
try:
    validate_box((-10, 20, 100, 200))
except InvalidBoxError as e:
    print(e.message)
    # Output: "Invalid box (-10, 20, 100, 200): coordinates cannot be negative"
```

---

## Best Practices

### Catching Specific Exceptions

Always catch the most specific exception possible:

```python
from color_correction import ColorCorrection
from color_correction.exceptions import (
    UnsupportedModelError,
    PatchesNotSetError,
    ModelNotFittedError,
    InvalidImageError,
)

try:
    cc = ColorCorrection(detection_model="custom_model")
    cc.set_input_patches(image)
    cc.fit()
    result = cc.predict(image)

except UnsupportedModelError as e:
    print(f"Invalid model choice: {e}")
    # Handle by using a supported model

except InvalidImageError as e:
    print(f"Image validation failed: {e.reason}")
    # Handle by fixing image format

except PatchesNotSetError as e:
    print(f"Missing {e.patch_type} patches")
    # Handle by setting required patches

except ModelNotFittedError:
    print("Model not fitted yet")
    # Handle by calling fit()
```

### Catching All Package Exceptions

To catch any error from the color-correction package:

```python
from color_correction.exceptions import ColorCorrectionError

try:
    # Your color correction code
    pass
except ColorCorrectionError as e:
    print(f"Color correction error: {e}")
    # Handle or log the error
```

### Logging Exceptions

Use exception attributes for better logging:

```python
import logging
from color_correction.exceptions import InsufficientPatchesError

logger = logging.getLogger(__name__)

try:
    cc.set_input_patches(image)
except InsufficientPatchesError as e:
    logger.error(
        "Patch detection failed",
        extra={
            "detected": e.detected_count,
            "required": e.required_count,
        }
    )
```

---

## See Also

- [API Reference](index.md) - Complete API documentation
- [Tutorial](../tutorial/getting_started.md) - Getting started guide
- [Validators](validators.md) - Input validation utilities
