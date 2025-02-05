import time

import cv2
import numpy as np
import onnxruntime

from color_correction.core.card_detection.base import BaseCardDetector
from color_correction.schemas.det_yv8 import DetectionResult
from color_correction.utils.downloader import downloader_model_yolov8
from color_correction.utils.yolo_utils import (
    multiclass_nms,
    xywh2xyxy,
)


class YOLOv8CardDetector(BaseCardDetector):
    """YOLOv8CardDetector is a class that implements card detection
    using the YOLOv8 model.

    Reference
    ---------
    https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/blob/main/yolov8/YOLOv8.py

    """

    def __init__(
        self,
        conf_th: float = 0.15,
        iou_th: float = 0.7,
        path: str | None = None,
        use_gpu: bool = False,
    ) -> None:
        self.conf_threshold = conf_th
        self.iou_threshold = iou_th
        self.use_gpu = use_gpu
        if path is None:
            path = downloader_model_yolov8(use_gpu)
        self.__initialize_model(path)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect objects in the given image using YOLOv8 model.

        Parameters
        ----------
        image : np.ndarray
            The input image BGR in which to detect objects.

        Returns
        -------
        DetectionResult
            A dataclass containing detected bounding boxes, confidence scores,
            and class IDs.
        """
        input_tensor = self.__prepare_input(image.copy())
        outputs = self.__inference(input_tensor)
        boxes, scores, class_ids = self.__process_output(outputs)

        det_res = DetectionResult(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
        )

        return det_res

    # Service functions
    def __initialize_model(self, path: str) -> None:
        self.session = onnxruntime.InferenceSession(
            path,
            providers=onnxruntime.get_available_providers(),
        )
        # Get model info
        self.__get_input_details()
        self.__get_output_details()

    def __prepare_input(self, original_image: np.ndarray) -> np.ndarray:
        [height, width, _] = original_image.shape

        # expected shape based on model input
        expected_width = self.input_width
        expected_height = self.input_height
        expected_length = min((expected_height, expected_width))

        length = max((height, width))
        # self.scale_to_expected = expected_length / length
        self.scale_to_ori = length / expected_length

        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        input_image = cv2.resize(image, (expected_width, expected_height))

        if self.use_gpu:
            input_image = (input_image / 255.0).astype(np.float16)
        else:
            input_image = (input_image / 255.0).astype(np.float32)
        # Channel first
        input_image = input_image.transpose(2, 0, 1)

        # Expand dimensions
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def __inference(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        start = time.perf_counter()  # noqa: F841
        outputs = self.session.run(
            self.output_names,
            {self.input_names[0]: input_tensor},
        )

        # print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def __process_output(
        self,
        output: list[np.ndarray],
    ) -> tuple[list[list[int]], list[float], list[int]]:
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.__extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(
            boxes,
            scores,
            class_ids,
            self.iou_threshold,
        )

        return (
            boxes[indices].astype(int).tolist(),
            scores[indices].tolist(),
            class_ids[indices].tolist(),
        )

    # Helper functions
    def __extract_boxes(self, predictions: np.ndarray) -> np.ndarray:
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.__rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def __rescale_boxes(self, boxes: np.ndarray) -> np.ndarray:
        # Rescale boxes to original image dimensions
        boxes *= self.scale_to_ori
        return boxes

    def __get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def __get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == "__main__":
    print("YOLOv8CardDetector")
    model_path = "color_correction/asset/.model/yv8-det.onnx"
    image_path = "color_correction/asset/images/cc-1.jpg"
    image_path = "color_correction/asset/images/Test 19.png"
    detector = YOLOv8CardDetector(conf_th=0.15, iou_th=0.7, use_gpu=True)

    input_image = cv2.imread(image_path)
    # input_image = cv2.resize(input_image, (640, 640))
    result = detector.detect(input_image)
    result.print_summary()
    image_drawed = result.draw_detections(input_image)
    cv2.imwrite("result.png", image_drawed)
