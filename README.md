
<div align="center">
<!-- image logo -->
<img src="assets/logo-v1.png" alt="Color Correction Logo" width="100"/>

# Color Correction

<br>

[![version](https://badge.fury.io/py/color-correction.svg)](https://badge.fury.io/py/color-correction)
[![downloads](https://img.shields.io/pypi/dm/color-correction-asdfghjkl)](https://pypistats.org/packages/color-correction-asdfghjkl)
[![python-version](https://img.shields.io/pypi/pyversions/color-correction)](https://badge.fury.io/py/color-correction)


</div>

> _Package formerly published as [`color-correction-asdfghjkl`](https://pypi.org/project/color-correction-asdfghjkl/) on PyPI. The name has been simplified for better accessibility and professional recognition._

This package is designed to perform color correction on images using the Color Checker Classic 24 Patch card. It provides a robust solution for ensuring accurate color representation in your images.

## 📦 Installation

```bash
pip install color-correction
```

## 🏋️‍♀️ How it works
![How it works](assets/color-correction-how-it-works.png)


## ⚡ How to use

```python
from color_correction import ColorCorrection

# Step 1: Define the path to the input image
image_path = "asset/images/cc-19.png"

# Step 2: Load the input image
input_image = cv2.imread(image_path)

# Step 3: Initialize the color correction model with specified parameters
color_corrector = ColorCorrection(
    detection_model="yolov8",
    detection_conf_th=0.25,
    correction_model="polynomial", # "least_squares", "affine_reg", "linear_reg"
    degree=3,  # for polynomial correction model
    use_gpu=True,
)

# Step 4: Extract color patches from the input image
# you can set reference patches from another image (image has color checker card)
# or use the default D50
# color_corrector.set_reference_patches(image=None, debug=True)
color_corrector.set_input_patches(image=input_image, debug=True)
color_corrector.fit()
corrected_image = color_corrector.predict(
    input_image=input_image,
    debug=True,
    debug_output_dir="zzz",
)

# Step 5: Evaluate the color correction results
eval_result = color_corrector.calc_color_diff_patches()
print(eval_result)
```

<details>
<summary>Sample Evaluation Output</summary>

```json
{
    "initial": {
        "min": 2.254003059526461,
        "max": 13.461066402633447,
        "mean": 8.3072755187654,
        "std": 3.123962754767539,
    },
    "corrected": {
        "min": 0.30910031798755183,
        "max": 5.422311999126372,
        "mean": 1.4965478752947827,
        "std": 1.2915738724958112,
    },
    "delta": {
        "min": 1.9449027415389093,
        "max": 8.038754403507074,
        "mean": 6.810727643470616,
        "std": 1.8323888822717276,
    },
}
```
</details>

<details>
<summary>Sample Output Debug Image</summary>

![Sample Output](assets/sample-output-debug.jpg)

</details>

## 🔎 Reporting
```python
import cv2

from color_correction import ColorCorrectionAnalyzer

# input_image_path = "assets/cc-19.png"
input_image_path = "assets/cc-1.jpg"

report = ColorCorrectionAnalyzer(
    list_correction_methods=[
        ("least_squares", {}),
        ("linear_reg", {}),
        ("affine_reg", {}),
        ("polynomial", {"degree": 2}),
        ("polynomial", {"degree": 3}),
        # ("polynomial", {"degree": 4}),
        # ("polynomial", {"degree": 5}),
    ],
    list_detection_methods=[
        ("yolov8", {"detection_conf_th": 0.25}),
    ],
)
report.run(
    input_image=cv2.imread(input_image_path),
    reference_image=None,
    output_dir="report-output",
)
```
<details>
<summary>Sample Report Output</summary>

![Sample Benchmark Output](assets/sample-benchmark.png)
</details>

## 📈 Benefits
- **Consistency**: Ensure uniform color correction across multiple images.
- **Accuracy**: Leverage the color correction matrix for precise color adjustments.
- **Flexibility**: Adaptable for various image sets with different color profiles.


## 🤸 TODO
- [ ] Add Loggers
- [ ] Add detection MCC:CCheckerDetector from opencv
- [ ] Add Segmentation Color Checker using YOLOv11 ONNX
- [ ] Improve validation preprocessing (e.g., auto-match-orientation CC)
- [ ] Add more analysis and evaluation metrics (Still thinking...)

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
