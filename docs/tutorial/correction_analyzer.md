# Correction Analyzer

## Introduction

This package provides a class for analyzing color correction by combining various correction methods with object detection models. The analyzer offers multiple correction methods and customizable parameters that you can experiment with. For object detection, it currently supports `YOLOv8` (ONNX), with plans to expand to more models in the future. For correction methods, it supports `least squares`, `linear regression`, `affine regression`, and `polynomial regression`.

The analyzer automatically runs all defined methods and generates a comprehensive report comparing visual results and `ΔE CIE 2000` values. You don't need to write complex code - simply define your desired methods and parameters, and let the analyzer do the work for you. This makes it easy to find the best color correction approach for your specific needs.


## Usage

=== "Code"

    ??? tip "If you don't have image to test"

        You can download the sample image from the following link:
        ```bash
        curl -L -o input_image.jpg "https://drive.google.com/uc?export=download&id=1syOqw9kC0tt01p7yEobU4MeLfh336DZA"
        ```

    ```python
    # your_script.py
    import cv2

    from color_correction import ColorCorrectionAnalyzer

    input_image_path = "your_path_image"

    report = ColorCorrectionAnalyzer(
        list_correction_methods=[
            ("least_squares", {}),
            ("linear_reg", {}),
            ("affine_reg", {}),
            ("polynomial", {"degree": 2}),
            ("polynomial", {"degree": 3}),
            ("polynomial", {"degree": 4}),
            ("polynomial", {"degree": 5}),
        ],
        list_detection_methods=[
            ("yolov8", {"detection_conf_th": 0.25}),
        ],
        use_gpu=False,
    )

    df_report = report.run(
        input_image=cv2.imread(input_image_path),
        reference_image=None,
        output_dir="report-output", # (1)
    )

    df_report.head()
    ```

    1. 💬 The output directory where the report files will be saved.

=== "Project Structure"

    ```bash
    ├── 📄 your_script.py
    └── 📂 report-output # (1)
        ├── report.html
        ├── report.pkl
        └── report_no_image.csv
    ```

    1. 💬 The output directory contains the following files:
        - `report.html`: The HTML report file.
        - `report.pkl`: The pickle file containing the report data.
        - `report_no_image.csv`: The CSV file containing the report data without images.

=== "Report HTML Output"

    ![Report HTML Output](../../assets/sample-output-html-v2.png){ loading=lazy }
    /// caption
    Comparison of Color Correction Methods: Visual Results and ΔE CIE 2000 Values (Click to enlarge)
    ///


## Reference

- [`class` ColorCorrectionAnalyzer](../reference/services/correction_analyzer.md)
