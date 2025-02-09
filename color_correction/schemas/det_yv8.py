"""
Module for detection result schema using Pydantic.

Provides the DetectionResult model that contains detection boxes, scores, and class ids,
and a helper method to draw these detections on an image.
"""

import numpy as np
from pydantic import BaseModel

from color_correction.utils.yolo_utils import draw_detections

BoundingBox = tuple[int, int, int, int]


class DetectionResult(BaseModel):
    """
    Detection result model for YOLOv8 card and color patches detection.

    A data model that encapsulates YOLOv8 detection results for a standardized color
    card and its color patches. The model handles two distinct classes:
    patches (label 0) and card (label 1). In a typical detection scenario,
    the model captures one color calibration card and 24 color patches.


    Notes
    -----
    The detection typically yields 25 objects:

    - 1 calibration card (class_id: 1)
    - 24 color patches (class_id: 0)

    Attributes
    ----------
    boxes : list[tuple[int, int, int, int]]
        List of bounding boxes as (x1, y1, x2, y2).
        Representing the top-left and bottom-right corners of the detection.
        Class identifiers for each detected object where:

        - 0: represents color patches
        - 1: represents the calibration card

    scores : list[float]
        List of confidence scores for each detection.
    class_ids : list[int]
        List of class IDs corresponding to each detection.
    """

    boxes: list[BoundingBox]
    scores: list[float]
    class_ids: list[int]

    def draw_detections(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Draw detection boxes on the provided image.

        Parameters
        ----------
        image : numpy.ndarray
            The image on which the detection boxes will be drawn.

        Returns
        -------
        numpy.ndarray
            The image with the drawn detection boxes.
        """
        return draw_detections(image, self.boxes, self.scores, self.class_ids)
