import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from color_correction.core.correction.base import BaseComputeCorrection
from color_correction.utils.correction import (
    postprocessing_compute,
    preprocessing_compute,
)


class Polynomial(BaseComputeCorrection):
    def __init__(self, **kwargs: dict) -> None:
        self.model = None
        self.degree = kwargs.get("degree", 2)

    def fit(
        self,
        x_patches: np.ndarray,  # input patches
        y_patches: np.ndarray,  # reference patches
        **kwargs: dict,
    ) -> np.ndarray:
        start_time = time.perf_counter()

        degree = kwargs.get("degree", self.degree)
        self.model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression(),
        ).fit(x_patches, y_patches)

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
