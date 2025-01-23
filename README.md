
# 🎨 Color Correction Toolkit

> **Note:** The "asdfghjkl" is just a placeholder due to some naming difficulties.

This package is designed to perform color correction on images using the Color Checker Classic 24 Patch card. It provides a robust solution for ensuring accurate color representation in your images.

## 🛠️ How It Works

1. **Input Requirements**:
   - An image containing the Color Checker Classic 24 Patch card (referred to as the color checker).
   - A reference image of the color checker.

2. **Process**:
   - The package computes a color correction matrix based on the input image and reference image.
   - This matrix can then be used to correct the colors of the input image or other images with the same color profile.

   **Workflow**:
   ```
   Input Image + Reference Image -> Color Correction Matrix -> Corrected Image
   ```

3. **Advanced Usage**:
   - For multiple images, it’s recommended to preprocess the images using white balance before calculating the color correction matrix. This ensures that all images are in the same color profile, leading to a more accurate correction matrix.

   **Enhanced Workflow**:
   ```
   Input Images -> White Balance -> Compute (Input Image with Color Checker, Reference Image) -> Color Correction Matrix -> Corrected Images
   ```

## 📈 Benefits
- **Consistency**: Ensure uniform color correction across multiple images.
- **Accuracy**: Leverage the color correction matrix for precise color adjustments.
- **Flexibility**: Adaptable for various image sets with different color profiles.

<!-- write reference -->
## 📚 References
- [Color Checker Classic 24 Patch Card](https://www.xrite.com/categories/calibration-profiling/colorchecker-classic)
- [Color Correction Tool ML](https://github.com/collinswakholi/ML_ColorCorrection_tool/tree/Pip_package)
- [Colour Science Python](https://www.colour-science.org/colour-checker-detection/)
- [Fast and Robust Multiple ColorChecker Detection ()](https://github.com/pedrodiamel/colorchecker-detection)
- [Automatic color correction with OpenCV and Python (PyImageSearch)](https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/)
- [ONNX-YOLOv8-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection)
- [yolov8-triton](https://github.com/omarabid59/yolov8-triton/tree/main)
