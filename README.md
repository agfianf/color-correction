
# 🎨 Color Correction 

> **Note:** The "asdfghjkl" is just a placeholder due to some naming difficulties.

This package is designed to perform color correction on images using the Color Checker Classic 24 Patch card. It provides a robust solution for ensuring accurate color representation in your images.

## Installation

```bash
pip install color-correction-asdfghjkl
```
## Usage

```python
import cv2

from color_correction_asdfghjkl import ColorCorrection

cc = ColorCorrection(
    detection_model="yolov8",
    correction_model="least_squares",
    use_gpu=False,
)

input_image = cv2.imread("cc-19.png")
cc.fit(input_image=input_image)
corrected_image = cc.correct_image(input_image=input_image)
cv2.imwrite("corrected_image.png", corrected_image)
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
