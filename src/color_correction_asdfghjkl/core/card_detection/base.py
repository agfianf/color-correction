from abc import ABC, abstractmethod

import numpy as np

from color_correction_asdfghjkl.schemas.yolov8_det import DetectionResult


class BaseCardDetector(ABC):
    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.15,
    ) -> DetectionResult: ...
