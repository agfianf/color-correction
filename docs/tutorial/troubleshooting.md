# Troubleshooting Guide

This guide helps you diagnose and fix common issues when using the color-correction package.

## Installation Issues

### Python Version Error

**Error:**
```
ERROR: Package 'color-correction' requires a different Python: 3.10.0 not in '>=3.11'
```

**Solution:**

The package requires Python 3.11 or higher. Upgrade your Python installation:

```bash
# Check current version
python --version

# Using pyenv (recommended)
pyenv install 3.11
pyenv global 3.11

# Or download from python.org
# https://www.python.org/downloads/
```

### Missing OpenCV MCC Module

**Error:**
```
AttributeError: module 'cv2' has no attribute 'mcc'
```

**Solution:**

Install the contrib version of OpenCV:

```bash
# Uninstall regular opencv
pip uninstall opencv-python

# Install opencv-contrib-python
pip install opencv-contrib-python>=4.11.0.86

# Or use the package extra
pip install "color-correction[mccdet]"
```

### ONNX Runtime Not Found

**Error:**
```
ModuleNotFoundError: No module named 'onnxruntime'
```

**Solution:**

ONNX Runtime should be installed automatically, but if not:

```bash
# For CPU
pip install onnxruntime

# For GPU
pip install onnxruntime-gpu
```

## Detection Issues

### No Color Checker Card Detected

**Error:**
```
RuntimeError: No color checker card detected in the image
```

**Common Causes & Solutions:**

#### 1. Card Not Visible or Occluded

- Ensure the entire 24-patch card is visible
- Remove any obstructions
- Check that lighting is adequate
- Verify the card is in focus

#### 2. Low Detection Confidence

Try lowering the confidence threshold:

```python
cc = ColorCorrection(
    detection_model="yolov8",
    detection_conf_th=0.15,  # Lower from default 0.25
)
```

#### 3. Card Too Small in Image

Crop the image to make the card larger:

```python
import cv2

# Crop to region of interest
image = cv2.imread("image.jpg")
roi = image[100:500, 200:600]  # Adjust coordinates

cc.set_input_patches(image=roi, debug=True)
```

#### 4. Try Alternative Detection Method

Switch from YOLOv8 to MCC or vice versa:

```python
# If YOLOv8 fails, try MCC
cc = ColorCorrection(detection_model="mcc")

# Or if MCC fails, try YOLOv8
cc = ColorCorrection(detection_model="yolov8")
```

#### 5. Use Manual Segmentation

If automatic detection fails, provide coordinates manually:

```python
from color_correction.processor.detection import extract_patches_from_segmentation

# Define card corners
vertices = [
    [x1, y1],  # top-left
    [x2, y2],  # top-right
    [x3, y3],  # bottom-right
    [x4, y4],  # bottom-left
]

patches, grid, debug = extract_patches_from_segmentation(
    image=image,
    vertices=vertices,
    debug=True
)

# Use extracted patches directly
cc.input_patches = patches
cc.fit()
```

### Insufficient Patches Detected

**Error:**
```
InsufficientPatchesError: Detected 18 patches, but 24 are required
```

**Solutions:**

1. **Improve Image Quality:**
   - Increase resolution
   - Improve lighting
   - Reduce glare/reflections

2. **Adjust Detection Parameters:**
```python
# Lower IoU threshold
cc = ColorCorrection(
    detection_model="yolov8",
    detection_iou_th=0.5,  # Lower from default 0.7
)
```

3. **Enable Debug Mode:**
```python
cc.set_input_patches(image=image, debug=True)
cv2.imshow("Debug", cc.input_debug_image)
cv2.waitKey(0)
# Check which patches are missing
```

### Multiple Cards Detected

**Issue:** Multiple color checkers in the scene

**Solution:**

1. **Crop to Single Card:**
```python
# Crop image to contain only one card
roi = image[y1:y2, x1:x2]
```

2. **Increase Confidence Threshold:**
```python
cc = ColorCorrection(
    detection_model="yolov8",
    detection_conf_th=0.5,  # Higher threshold
)
```

3. **Post-filter Detections:**
```python
# Keep only the largest detection
from color_correction.core.card_detection.det_yv8_onnx import YOLOv8Det

detector = YOLOv8Det(conf_th=0.25)
result = detector.detect(image)

if len(result.boxes) > 1:
    # Find largest box
    areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in result.boxes]
    largest_idx = areas.index(max(areas))

    # Keep only largest
    result.boxes = [result.boxes[largest_idx]]
    result.scores = [result.scores[largest_idx]]
```

## Correction Issues

### Model Not Fitted Error

**Error:**
```
ModelNotFittedError: Model must be fitted before prediction. Call fit() first.
```

**Solution:**

Ensure you call methods in the correct order:

```python
# Correct workflow
cc = ColorCorrection()
cc.set_reference_patches(image=None)  # Step 1
cc.set_input_patches(image=image)     # Step 2
cc.fit()                               # Step 3 - REQUIRED
corrected = cc.predict(image)          # Step 4
```

### Patches Not Set Error

**Error:**
```
PatchesNotSetError: Input patches must be set before this operation
```

**Solution:**

Call `set_input_patches()` before `fit()`:

```python
cc = ColorCorrection()
cc.set_input_patches(image=image)  # Must be called first
cc.fit()
```

### Poor Correction Results

**Issue:** Corrected image looks worse or similar to original

**Diagnostic Steps:**

1. **Check Initial Color Difference:**
```python
cc.fit()
metrics = cc.calc_color_diff_patches()
print(f"Initial error: {metrics['initial']['mean']:.2f}")

if metrics['initial']['mean'] < 3.0:
    print("Image already well-calibrated, correction not needed")
```

2. **Try Different Correction Models:**
```python
# Try polynomial with different degrees
for degree in [2, 3, 4, 5]:
    cc = ColorCorrection(correction_model="polynomial", degree=degree)
    cc.set_input_patches(image=image)
    cc.fit()
    corrected = cc.predict(image)

    metrics = cc.calc_color_diff_patches()
    print(f"Degree {degree}: {metrics['corrected']['mean']:.2f}")
```

3. **Use ColorCorrectionAnalyzer:**
```python
from color_correction import ColorCorrectionAnalyzer

analyzer = ColorCorrectionAnalyzer(
    list_correction_methods=[
        ("least_squares", {}),
        ("linear_reg", {}),
        ("polynomial", {"degree": 2}),
        ("polynomial", {"degree": 3}),
        ("affine_reg", {}),
    ],
    list_detection_methods=[("yolov8", {"detection_conf_th": 0.25})],
)

analyzer.run(input_image=image, output_dir="analysis")
# Check HTML report for best method
```

4. **Verify Reference Patches:**
```python
# Use custom reference instead of D50
reference_img = cv2.imread("ideal_reference.jpg")
cc.set_reference_patches(image=reference_img, debug=True)

# Inspect reference patches
cv2.imshow("Reference", cc.reference_grid_image)
cv2.waitKey(0)
```

### Color Cast Not Removed

**Issue:** Strong color cast remains after correction

**Solutions:**

1. **Use Higher Polynomial Degree:**
```python
cc = ColorCorrection(
    correction_model="polynomial",
    degree=4,  # or 5 for stronger casts
)
```

2. **Verify Lighting Consistency:**
   - Ensure card has same lighting as scene
   - Avoid partial shadows on the card
   - Check for colored reflections

3. **Try Affine Registration:**
```python
cc = ColorCorrection(correction_model="affine_reg")
```

## Image Issues

### Invalid Image Error

**Error:**
```
InvalidImageError: Invalid image: image must have 3 dimensions (H, W, C), got 2
```

**Solution:**

Ensure image is BGR format (not grayscale):

```python
import cv2

# Read as color (default)
image = cv2.imread("image.jpg", cv2.IMREAD_COLOR)

# Convert grayscale to BGR if needed
if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
```

### Image Too Large

**Issue:** Slow processing with high-resolution images

**Solutions:**

1. **Resize Image:**
```python
def resize_to_max_dimension(image, max_dim=1920):
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image

image = cv2.imread("large_image.jpg")
image = resize_to_max_dimension(image)
```

2. **Extract ROI First:**
```python
# Detect on full image but correct on ROI
detection_img = cv2.resize(image, (1280, 720))
cc.set_input_patches(image=detection_img)
cc.fit()

# Apply to full resolution
corrected = cc.predict(input_image=image)
```

## Performance Issues

### Slow Detection

**Issue:** YOLOv8 detection takes too long

**Solutions:**

1. **Enable GPU:**
```python
cc = ColorCorrection(
    detection_model="yolov8",
    use_gpu=True,
)
```

2. **Reduce Image Size:**
```python
image = cv2.resize(image, (1280, 720))
```

3. **Switch to MCC:**
```python
# MCC can be faster for simple scenes
cc = ColorCorrection(detection_model="mcc")
```

### Memory Issues

**Error:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Process Smaller Images:**
```python
# Downsample before processing
scale = 0.5
h, w = image.shape[:2]
small = cv2.resize(image, (int(w*scale), int(h*scale)))
```

2. **Disable Debug Mode:**
```python
# Debug mode stores additional images
cc.set_input_patches(image=image, debug=False)
corrected = cc.predict(image, debug=False)
```

3. **Clear Unused Variables:**
```python
import gc

# After processing
del cc
gc.collect()
```

## GPU Issues

### GPU Not Detected

**Issue:** `use_gpu=True` but still using CPU

**Diagnostic:**

```python
import onnxruntime as ort

# Check available providers
print(ort.get_available_providers())

# Should include 'CUDAExecutionProvider' for NVIDIA GPU
# Or 'TensorrtExecutionProvider' for TensorRT
```

**Solutions:**

1. **Install GPU Version:**
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

2. **Check CUDA Installation:**
```bash
# Verify CUDA
nvidia-smi

# Check CUDA version
nvcc --version
```

3. **Install Compatible CUDA:**
   - ONNX Runtime GPU requires CUDA 11.x or 12.x
   - Check compatibility: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

### CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce Batch Size:**
```python
# Process images one at a time
for img_path in image_paths:
    img = cv2.imread(img_path)
    corrected = cc.predict(img)
    # Save and release
    cv2.imwrite(output_path, corrected)
    del img, corrected
```

2. **Use CPU:**
```python
cc = ColorCorrection(use_gpu=False)
```

## Common Workflow Errors

### Applying Correction Without Fitting

**Incorrect:**
```python
cc = ColorCorrection()
cc.set_input_patches(image1)
# Forgot cc.fit()
corrected = cc.predict(image2)  # ERROR!
```

**Correct:**
```python
cc = ColorCorrection()
cc.set_input_patches(image1)
cc.fit()  # Must fit before predict
corrected = cc.predict(image2)
```

### Reusing Fitted Model

**Issue:** Want to apply same correction to multiple images

**Correct Approach:**
```python
# Fit once
cc = ColorCorrection()
cc.set_input_patches(calibration_image)
cc.fit()

# Apply to multiple images
for img_path in image_paths:
    img = cv2.imread(img_path)
    corrected = cc.predict(img, debug=False)  # Reuse fitted model
    cv2.imwrite(output_path, corrected)
```

**Incorrect:**
```python
# Don't refit for each image
for img_path in image_paths:
    cc = ColorCorrection()
    cc.set_input_patches(cv2.imread(img_path))
    cc.fit()  # Wasteful! Fit once, predict many
    corrected = cc.predict(cv2.imread(img_path))
```

## Debug Strategies

### Enable Verbose Debug Output

```python
# Set debug=True for all operations
cc = ColorCorrection()
cc.set_reference_patches(image=None, debug=True)
cc.set_input_patches(image=image, debug=True)
cc.fit()
corrected = cc.predict(image, debug=True, debug_output_dir="debug_output")

# Inspect debug images
import os
for filename in os.listdir("debug_output"):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join("debug_output", filename))
        cv2.imshow(filename, img)
cv2.waitKey(0)
```

### Visualize Intermediate Results

```python
# Check detection
cc.set_input_patches(image=image, debug=True)
cv2.imwrite("step1_detection.jpg", cc.input_debug_image)
cv2.imwrite("step2_patches.jpg", cc.input_grid_image)

# Check correction
cc.fit()
corrected = cc.predict(image)

# Compare patches
from color_correction.utils.visualization_utils import visualize_patch_comparison
comparison = visualize_patch_comparison(
    ls_mean_in=cc.corrected_patches,
    ls_mean_ref=cc.reference_patches,
)
cv2.imwrite("step3_comparison.jpg", comparison)
```

### Analyze Metrics

```python
metrics = cc.calc_color_diff_patches()

print("=== Per-Patch Analysis ===")
print(f"Initial - Mean: {metrics['initial']['mean']:.2f}, Std: {metrics['initial']['std']:.2f}")
print(f"          Range: [{metrics['initial']['min']:.2f}, {metrics['initial']['max']:.2f}]")
print(f"Corrected - Mean: {metrics['corrected']['mean']:.2f}, Std: {metrics['corrected']['std']:.2f}")
print(f"            Range: [{metrics['corrected']['min']:.2f}, {metrics['corrected']['max']:.2f}]")
print(f"Improvement: {metrics['delta']['mean']:.2f} ± {metrics['delta']['std']:.2f}")

# Identify problematic patches
if metrics['corrected']['max'] > 5.0:
    print("Warning: Some patches still have high color difference")
    print("Consider trying a different correction model")
```

## Getting Help

### Before Asking for Help

Please gather this information:

1. **Python and package versions:**
```python
import sys
import cv2
import color_correction
import onnxruntime

print(f"Python: {sys.version}")
print(f"color-correction: {color_correction.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"ONNX Runtime: {onnxruntime.__version__}")
```

2. **Full error traceback**

3. **Minimal reproducible example**

4. **Image characteristics:**
   - Resolution
   - Lighting conditions
   - Card visibility

### Reporting Issues

File issues on GitHub: https://github.com/agfianf/color-correction/issues

Include:
- Description of the problem
- Expected vs actual behavior
- System information (above)
- Code to reproduce
- Sample images (if possible)

### Community Support

- GitHub Discussions: Ask questions and share tips
- GitHub Issues: Report bugs and request features
- Documentation: https://agfianf.github.io/color-correction/

## FAQ

### Q: Can I use this package without a Color Checker card?

**A:** No, the package requires a Color Checker Classic 24-patch card visible in your images. The card provides the reference colors needed for calibration.

### Q: What's the best correction model to use?

**A:** It depends on your use case:
- **polynomial (degree=3)**: Good default choice
- **least_squares**: Fastest, for real-time
- **affine_reg**: For geometric color shifts
- Use `ColorCorrectionAnalyzer` to compare methods for your specific images

### Q: Can I correct images without the card in them?

**A:** Yes! Calibrate with an image containing the card, then apply the fitted correction to other images taken under the same conditions:

```python
cc.set_input_patches(calibration_image_with_card)
cc.fit()

# Apply to images without card
corrected1 = cc.predict(image1_no_card)
corrected2 = cc.predict(image2_no_card)
```

### Q: How accurate is the color correction?

**A:** Typical results reduce mean Delta E from 8-10 to 1-3. Real-world accuracy depends on:
- Lighting consistency
- Card condition
- Detection quality
- Correction model choice

### Q: Does this work with other color checker cards?

**A:** Currently, the package is designed for the X-Rite Color Checker Classic (24 patches). Support for other cards (Passport, Mini, etc.) is not yet available.

### Q: Can I use this for video processing?

**A:** Yes, but process frame-by-frame:

```python
cap = cv2.VideoCapture("video.mp4")

# Calibrate on first frame
ret, frame = cap.read()
cc.set_input_patches(frame)
cc.fit()

# Process remaining frames
while ret:
    corrected = cc.predict(frame, debug=False)
    # Write to output video
    ret, frame = cap.read()
```

### Q: Is this package thread-safe?

**A:** No, don't share `ColorCorrection` instances across threads. Create separate instances per thread or use locks.

### Q: Can I save and load a fitted model?

**A:** Currently, there's no built-in save/load functionality. You can pickle the correction model:

```python
import pickle

# After fitting
with open("correction_model.pkl", "wb") as f:
    pickle.dump(cc.correction_model, f)

# Load later
with open("correction_model.pkl", "rb") as f:
    cc.correction_model = pickle.load(f)

# Use loaded model
corrected = cc.predict(image)
```

Note: This is experimental and may not work across package versions.
