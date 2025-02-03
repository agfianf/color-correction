
# 🎨 Color Correction

> **Note:** The "asdfghjkl" is just a placeholder due to some naming difficulties.

This package is designed to perform color correction on images using the Color Checker Classic 24 Patch card. It provides a robust solution for ensuring accurate color representation in your images.

## Installation

```bash
pip install color-correction-asdfghjkl
```
## Usage

```python
import os

# Step 1: Define the path to the input image
image_path = "asset/images/cc-19.png"

# Step 2: Load the input image
input_image = cv2.imread(image_path)

# Step 3: Initialize the color correction model with specified parameters
color_corrector = ColorCorrection(
    detection_model="yolov8",
    detection_conf_th=0.25,
    correction_model="least_squares",
    degree=2, # for polynomial correction model
    use_gpu=True,
)

# Step 4: Extract color patches from the input image
color_corrector.set_input_patches(image=input_image, debug=True)
color_corrector.fit()
corrected_image = color_corrector.predict(
    input_image=input_image,
    debug=True,
    debug_output_dir="zzz",
)

```
Sample output:
![Sample Output](assets/sample-output-usage.png)

## 📈 Benefits
- **Consistency**: Ensure uniform color correction across multiple images.
- **Accuracy**: Leverage the color correction matrix for precise color adjustments.
- **Flexibility**: Adaptable for various image sets with different color profiles.

![How it works](assets/color-correction-how-it-works.png)


<!-- write reference -->
## 📚 References
- [Color Checker Classic 24 Patch Card](https://www.xrite.com/categories/calibration-profiling/colorchecker-classic)
- [Color Correction Tool ML](https://github.com/collinswakholi/ML_ColorCorrection_tool/tree/Pip_package)
- [Colour Science Python](https://www.colour-science.org/colour-checker-detection/)
- [Fast and Robust Multiple ColorChecker Detection ()](https://github.com/pedrodiamel/colorchecker-detection)
- [Automatic color correction with OpenCV and Python (PyImageSearch)](https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/)
- [ONNX-YOLOv8-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection)
- [yolov8-triton](https://github.com/omarabid59/yolov8-triton/tree/main)
- [Streamlined Data Science Development: Organizing, Developing and Documenting Your Code](https://medium.com/henkel-data-and-analytics/streamlined-data-science-development-organizing-developing-and-documenting-your-code-bfd69e3ef4fb)
