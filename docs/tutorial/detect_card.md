# Card Patches Detection

## Introduction
This package has built-in support for detecting card patches in an image. This feature is useful for detecting color checker cards in an image. The detection is based on the YOLOv8 model. You can use the [`YOLOv8CardDetector`](../reference/core/card_detection/yv8_onnx.md) class to detect card patches in an image.

| :simple-onnx: | :simple-python: | :simple-opencv: | :simple-pydantic: |

## Usage

=== "Code"

    ??? tip "If you don't have image to test"

        You can download the sample image from the following link:
        ```bash
        curl -L -o input_image.jpg "https://drive.google.com/uc?export=download&id=1syOqw9kC0tt01p7yEobU4MeLfh336DZA"
        ```


    ```python
    import cv2

    from color_correction import YOLOv8CardDetector
    from color_correction.schemas.det_yv8 import DetectionResult

    image = cv2.imread("your_image_path")
    detector = YOLOv8CardDetector(
        conf_th=0.15,
        iou_th=0.7,
        use_gpu=False, # (1)
    )

    result: DetectionResult = detector.detect(image=image)
    drawed_image = result.draw_detections(image=image)
    cv2.imwrite("drawed_detection.jpg", drawed_image)
    ```



    1. 💬 The model runs using onnx :simple-onnx:, which supports both CPU and GPU. The model will be **automatically downloaded** if not already present on your system.

=== "output: DetectionResult"

    ```json

    {
        "boxes": [
            (366, 426, 441, 500), // (1)
            ...,
            (248, 138, 805, 545),
        ],
        "scores": [
            0.9755859375,
            ...,
            0.8583984375,
        ],
        "class_ids": [
            0, // (2)
            ...,
            1, // (3)
        ],
    }
    ```

    1.  💬 (`x1`, `y1`, `x2`, `y2`): coordinates of the bounding box top-left and bottom-right corners
    2.  💬 for `patch` object
    3.  💬 for `card` object

=== "output: Drawed Detection"

    ![Detection result](../assets/drawed_detection.jpg){ loading=lazy }
    /// caption
    The model detects two types of objects: a complete `card` and individual color `patches` (24 patches per card). Click the image to view full resolution detection results.
    ///


## Reference

- [`class` YOLOv8CardDetector](../reference/core/card_detection/yv8_onnx.md)
- [`class` DetectionResult](../reference/schemas/yv8_onnx.md)
