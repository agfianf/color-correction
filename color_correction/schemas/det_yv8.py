"""
Module for detection result schema using Pydantic.

Provides the DetectionResult model that contains detection boxes, scores, and class ids,
and a helper method to draw these detections on an image.
"""

import numpy as np
from pydantic import BaseModel

from color_correction.utils.yolo_utils import draw_detections, draw_segmentation

BoundingBox = tuple[int, int, int, int]
SegmentPoint = list[tuple[int, int]]


class DetectionResult(BaseModel):
    """
    Detection result model for YOLOv8 or MCCDetector card and color patches detection.

    A data model that encapsulates YOLOv8 detection or MCCDetector results for a standardized color
    card and its color patches. The model handles two distinct classes:
    patches (label 0) and card (label 1). In a typical detection scenario,
    the model captures one card and 24 color patches inside/outside the card.

    Notes
    -----
    The detection typically yields 25 objects:

    - 1 color checker card 24 Patches (class_id: 1)
    - 24 color patches (class_id: 0)

    Attributes
    ----------
    boxes : list[tuple[int, int, int, int]] | None
        List of bounding boxes for detected objects.
        List of bounding boxes as (x1, y1, x2, y2).
        Representing the top-left and bottom-right corners of the detection.
        if not None, the boxes are not used. it means coming from Object Detection.

        Class identifiers for each detected object where:

        - 0: represents color patches
        - 1: represents the color checker card 24 Patches

    segment : list[SegmentPoint] | None
        List of segmentation points for detected objects.
        Each point is represented as a tuple of (x, y) coordinates.
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x1, y1]] for each patch.
        The last point is a repeat of the first to close the polygon.
        if not None, the segment is not used. it means coming from MCCDetector.

    scores : list[float]
        List of confidence scores for each detection.

    class_ids : list[int]
        List of class IDs corresponding to each detection.
    """

    boxes: list[BoundingBox] | None = None
    segment: list[SegmentPoint] | None = None
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
        if self.boxes is not None:
            return draw_detections(image, self.boxes, self.scores, self.class_ids)
        elif self.segment is not None:
            return draw_segmentation(image, self.segment, self.scores, self.class_ids)
