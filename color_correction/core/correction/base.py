from abc import ABC, abstractmethod

import numpy as np


class BaseComputeCorrection(ABC):
    @abstractmethod
    def fit(self, image: np.ndarray) -> np.ndarray: ...
