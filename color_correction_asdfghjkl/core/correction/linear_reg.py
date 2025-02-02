import time

import numpy as np
from sklearn.linear_model import LinearRegression

from color_correction_asdfghjkl.core.correction.base import BaseComputeCorrection
from color_correction_asdfghjkl.utils.correction import (
    postprocessing_compute,
    preprocessing_compute,
)


class LinearReg(BaseComputeCorrection):
    def __init__(self) -> None:
        self.model = None

    def fit(
        self,
        x_patches: np.ndarray,  # input patches
        y_patches: np.ndarray,  # reference patches
    ) -> np.ndarray:
        start_time = time.perf_counter()

        self.model = LinearRegression(fit_intercept=False).fit(x_patches, y_patches)

        exc_time = time.perf_counter() - start_time
        print(f"{self.__class__.__name__} Fit: {exc_time} seconds")
        return self.model

    def compute_correction(self, input_image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please call fit() method first.")

        org_input_shape = input_image.shape
        input_image = preprocessing_compute(input_image)
        image = self.model.predict(input_image)
        corrected_image = postprocessing_compute(org_input_shape, image)
        return corrected_image
