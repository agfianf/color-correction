# Getting Started with Color Correction

![Color Checker Classic](https://www.xrite.com/-/media/global-product-images/c/colorchecker-classic/colorchecker-classic_01.png)
///caption
Color Checker Classic (24 patches) by X-Rite
///

## Introduction

In computer vision projects, maintaining consistent color representation is crucial for reliable results. Variations in color can affect tasks from object detection to image segmentation. Instead of treating color correction as an afterthought, it serves as a fundamental step for enhancing image analysis.

Images captured under different lighting or camera settings can exhibit significant color differences. Correcting these variations simplifies analysis and improves the overall performance of vision models.

The [color-correction](https://agfianf.github.io/color-correction/) package offers a robust solution for calibrating colors using a 24-patch Color Checker Classic card. It streamlines the calibration process, ensuring that colors across diverse lighting conditions align accurately. This facilitates more reliable image analysis and enhances the overall performance of computer vision projects.

## Installation

### Requirements

- Python 3.11 or higher
- pip or uv package manager

### Basic Installation

```bash
pip install color-correction
```

### Installation with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer:

```bash
pip install uv
uv pip install color-correction
```

### Installation for MCC Detector

If you plan to use the OpenCV MCC (Macbeth ColorChecker) detector:

```bash
pip install "color-correction[mccdet]"
```

### Verify Installation

```python
import color_correction
print(color_correction.__version__)
```

## Quick Start

### Basic Color Correction Workflow

Here's a minimal example to get started:

```python
import cv2
from color_correction import ColorCorrection

# Load your image
image = cv2.imread("path/to/your/image.jpg")

# Initialize color correction
cc = ColorCorrection(
    detection_model="yolov8",  # or "mcc"
    correction_model="polynomial",
    degree=3,
)

# Extract patches and fit the model
cc.set_input_patches(image=image, debug=True)
cc.fit()

# Apply correction
corrected_image = cc.predict(input_image=image, debug=True)

# Save the result
cv2.imwrite("corrected_image.jpg", corrected_image)

# Evaluate the correction
metrics = cc.calc_color_diff_patches()
print(f"Mean color difference reduced from {metrics['initial']['mean']:.2f} to {metrics['corrected']['mean']:.2f}")
```

### Understanding the Output

The `calc_color_diff_patches()` method returns a dictionary with three keys:

- **initial**: Color difference before correction
- **corrected**: Color difference after correction
- **delta**: Improvement (initial - corrected)

Each contains:
- `min`: Minimum Delta E across all patches
- `max`: Maximum Delta E across all patches
- `mean`: Average Delta E
- `std`: Standard deviation of Delta E

A lower value indicates colors closer to the reference. Typical improvements reduce mean Delta E from 8-10 to 1-3.

## Step-by-Step Tutorial

### Step 1: Understanding Detection Models

The package offers two detection methods to locate the Color Checker card in your image:

#### YOLOv8 (Default, Recommended)

- **Pros**: Fast, accurate, works in various conditions
- **Cons**: Requires ONNX model file (auto-downloaded)
- **Use when**: You need robust detection across different scenes

```python
cc = ColorCorrection(
    detection_model="yolov8",
    detection_conf_th=0.25,  # Confidence threshold (0.0 - 1.0)
)
```

#### MCC Detector (OpenCV-based)

- **Pros**: No deep learning, deterministic
- **Cons**: Requires controlled environment, may fail with occlusions
- **Use when**: You have consistent lighting and card placement

```python
cc = ColorCorrection(detection_model="mcc")
```

### Step 2: Choosing a Correction Model

#### Least Squares (Fastest)

Best for: Quick corrections, real-time applications

```python
cc = ColorCorrection(correction_model="least_squares")
```

#### Polynomial (Most Popular)

Best for: Non-linear color shifts, general use

```python
cc = ColorCorrection(
    correction_model="polynomial",
    degree=3,  # Higher degree = more flexible, but may overfit
)
```

Recommended degrees:
- `degree=2`: Subtle corrections
- `degree=3`: General purpose (default)
- `degree=4`: Strong color casts

#### Linear Regression

Best for: Simple color shifts, baseline comparisons

```python
cc = ColorCorrection(correction_model="linear_reg")
```

#### Affine Registration

Best for: Geometric color transformations

```python
cc = ColorCorrection(correction_model="affine_reg")
```

### Step 3: Setting Reference Patches

By default, the package uses CIE Standard Illuminant D50 reference values. You can also use your own reference image:

#### Using Default D50 Reference

```python
cc.set_reference_patches(image=None)  # Uses built-in D50 values
```

#### Using Custom Reference Image

If you have an image captured under ideal conditions:

```python
reference_image = cv2.imread("ideal_lighting.jpg")
cc.set_reference_patches(image=reference_image, debug=True)
```

### Step 4: Debug Visualization

Enable debug mode to see detection and patch extraction:

```python
cc.set_input_patches(image=image, debug=True)
```

This generates:
- `input_debug_image`: Shows detected card and patches
- `input_grid_image`: 4×6 grid of extracted patch colors

Access these for inspection:

```python
cv2.imshow("Debug", cc.input_debug_image)
cv2.imshow("Patches", cc.input_grid_image)
cv2.waitKey(0)
```

### Step 5: Complete Workflow Example

```python
import cv2
from color_correction import ColorCorrection

def correct_image(input_path: str, output_path: str):
    """Apply color correction to an image."""

    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")

    # Initialize corrector
    cc = ColorCorrection(
        detection_model="yolov8",
        detection_conf_th=0.25,
        correction_model="polynomial",
        degree=3,
        use_gpu=False,  # Set to True if you have GPU
    )

    # Set reference (using default D50)
    cc.set_reference_patches(image=None)

    # Extract input patches
    try:
        cc.set_input_patches(image=image, debug=True)
    except Exception as e:
        print(f"Failed to extract patches: {e}")
        print("Make sure the Color Checker card is visible in the image")
        return None

    # Fit the correction model
    cc.fit()

    # Apply correction with debug output
    corrected = cc.predict(
        input_image=image,
        debug=True,
        debug_output_dir="debug_output"
    )

    # Evaluate
    metrics = cc.calc_color_diff_patches()

    print("\n=== Color Correction Results ===")
    print(f"Initial mean Delta E: {metrics['initial']['mean']:.2f}")
    print(f"Corrected mean Delta E: {metrics['corrected']['mean']:.2f}")
    print(f"Improvement: {metrics['delta']['mean']:.2f}")
    print(f"Initial range: [{metrics['initial']['min']:.2f}, {metrics['initial']['max']:.2f}]")
    print(f"Corrected range: [{metrics['corrected']['min']:.2f}, {metrics['corrected']['max']:.2f}]")

    # Save result
    cv2.imwrite(output_path, corrected)
    print(f"\nCorrected image saved to: {output_path}")

    return corrected

# Usage
if __name__ == "__main__":
    correct_image("input.jpg", "output.jpg")
```

## GPU Acceleration

For faster processing with ONNX Runtime GPU:

### Installation

```bash
# Install ONNX Runtime with GPU support
pip install onnxruntime-gpu
```

### Enable GPU

```python
cc = ColorCorrection(
    detection_model="yolov8",
    use_gpu=True,  # Enable GPU acceleration
)
```

### Check Available Providers

```python
import onnxruntime
print(onnxruntime.get_available_providers())
# Expected: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Segmentation-Based Detection

If you already have the Color Checker card detected by another system (e.g., SAM, YOLO segmentation), you can use those coordinates directly:

```python
from color_correction.processor.detection import extract_patches_from_segmentation

# Quadrilateral vertices: [top-left, top-right, bottom-right, bottom-left]
vertices = [
    [100, 50],   # top-left (x, y)
    [400, 50],   # top-right
    [400, 250],  # bottom-right
    [100, 250],  # bottom-left
]

patches, grid_image, debug_image = extract_patches_from_segmentation(
    image=image,
    vertices=vertices,
    debug=True
)
```

## Common Patterns

### Batch Processing

Process multiple images with the same correction:

```python
import glob
import cv2
from color_correction import ColorCorrection

# Initialize once
cc = ColorCorrection(correction_model="polynomial", degree=3)

# Get reference from first image
reference_image = cv2.imread("reference.jpg")
cc.set_reference_patches(image=reference_image)

# Get correction from calibration image
calibration_image = cv2.imread("calibration.jpg")
cc.set_input_patches(image=calibration_image)
cc.fit()

# Apply to all images in directory
for img_path in glob.glob("images/*.jpg"):
    img = cv2.imread(img_path)
    corrected = cc.predict(input_image=img, debug=False)

    output_path = img_path.replace("images/", "corrected/")
    cv2.imwrite(output_path, corrected)
    print(f"Processed: {img_path}")
```

### Integration with Image Processing Pipeline

```python
import cv2
from color_correction import ColorCorrection

def preprocessing_pipeline(image_path: str):
    """Complete image preprocessing with color correction."""

    # Load image
    img = cv2.imread(image_path)

    # 1. Color correction
    cc = ColorCorrection()
    cc.set_input_patches(image=img)
    cc.fit()
    img = cc.predict(input_image=img, debug=False)

    # 2. Additional processing
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)  # Contrast & brightness

    return img
```

### Conditional Correction

Only apply correction if initial color difference is above a threshold:

```python
cc = ColorCorrection()
cc.set_input_patches(image=image)
cc.fit()

# Check if correction is needed
metrics = cc.calc_color_diff_patches()
initial_error = metrics['initial']['mean']

if initial_error > 5.0:  # Threshold
    print(f"Applying correction (error: {initial_error:.2f})")
    corrected = cc.predict(input_image=image)
else:
    print(f"Image already well-calibrated (error: {initial_error:.2f})")
    corrected = image
```

## Next Steps

Now that you understand the basics, explore:

- [Color Correction Examples](color_correction.md) - Detailed correction examples
- [Detection Methods](detect_card.md) - Deep dive into detection
- [Correction Analyzer](correction_analyzer.md) - Compare multiple methods
- [Architecture](../architecture.md) - Understand the package design
- [API Reference](../reference/services/color_correction.md) - Complete API documentation

## Troubleshooting

See the [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.
