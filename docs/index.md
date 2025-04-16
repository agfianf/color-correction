# Color Correction

<figure markdown="span">
  ![Image title](assets/logo-v1.png){ width="200" }
  <figcaption>color correction</figcaption>
</figure>

<p align="center">
    <a href="https://badge.fury.io/py/color-correction">
        <img src="https://badge.fury.io/py/color-correction.svg" alt="version" />
    </a>
    <a href="https://pypistats.org/packages/color-correction">
        <img src="https://img.shields.io/pypi/dm/color-correction" alt="downloads" />
    </a>
    <a href="https://badge.fury.io/py/color-correction">
        <img src="https://img.shields.io/pypi/pyversions/color-correction" alt="python version" />
    </a>
    <a href="https://colab.research.google.com/drive/146SXHHihMmGLzaTSwdBXncVr3SU_I-Dm?usp=sharing">
        <img src="https://img.shields.io/badge/Try_color--correction_using-Google_Colab-blue?logo=googlecolab" alt="Try color-correction using - Google Colab" />
    </a>
    <a href="https://agfianf.github.io/color-correction/">
        <img src="https://img.shields.io/badge/read-docs-blue?logo=materialformkdocs" alt="read - docs" />
    </a>

</p>

!!! warning "Disclaimer"

    _Package formerly published as [`color-correction-asdfghjkl`](https://pypi.org/project/color-correction-asdfghjkl/) on PyPI. The name has been simplified for better accessibility and professional recognition._

This package is designed to perform color correction on images using the Color Checker Classic 24 Patch card. It provides a robust solution for ensuring accurate color representation in your images.

## 📦 Installation

```bash
pip install color-correction
```

## 🏋️‍♀️ How it works
![How it works](assets/color-correction-how-it-works.png)


## 📈 Benefits
- **Consistency**: Ensure uniform color correction across multiple images.
- **Accuracy**: Leverage the color correction matrix for precise color adjustments.
- **Flexibility**: Adaptable for various image sets with different color profiles.


## 🤸 TODO

- [ ] Add Loggers
- [x] Add detection MCC:CCheckerDetector from opencv
- [ ] Add Segmentation Color Checker using YOLOv11 ONNX
- [ ] Improve validation preprocessing (e.g., auto-match-orientation CC)
- [ ] Add more analysis and evaluation metrics _(still thinking...)_

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
