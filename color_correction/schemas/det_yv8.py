import numpy as np
from pydantic import BaseModel

from color_correction.utils.yolo_utils import draw_detections

BoundingBox = tuple[int, int, int, int]


class DetectionResult(BaseModel):
    boxes: list[BoundingBox]
    scores: list[float]
    class_ids: list[int]

    def draw_detections(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Draw detection boxes on image."""
        return draw_detections(image, self.boxes, self.scores, self.class_ids)
