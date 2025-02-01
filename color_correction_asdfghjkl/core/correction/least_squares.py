import time

import numpy as np

from color_correction_asdfghjkl.core.correction.base import BaseComputeCorrection


class LeastSquaresRegression(BaseComputeCorrection):
    def __init__(self) -> None:
        self.model = None

    def fit(
        self,
        input_patches: np.ndarray,
        reference_patches: np.ndarray,
    ) -> np.ndarray:
        start_time = time.perf_counter()

        self.model = np.linalg.lstsq(
            a=input_patches,
            b=reference_patches,
            rcond=None,
        )[0]  # get only matrix of coefficients

        exc_time = time.perf_counter() - start_time
        print(f"Least Squares Regression: {exc_time} seconds")
        return self.model

    def compute_correction(self, input_image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please call fit() method first.")

        # Input adalah array (N,3) dari nilai warna patches
        if input_image.shape == (24, 3):  # Khusus untuk ColorChecker 24 patches
            image = input_image.astype(np.float32)
            # Aplikasikan koreksi warna
            image = np.dot(image, self.model)
            # Clip dan convert ke uint8
            corrected_image = np.clip(image, 0, 255).astype(np.uint8)
            return corrected_image

        # Untuk kasus gambar normal (h,w,c)
        # h, w, c = input_image.shape
        # image = input_image.reshape(-1, 3).astype(np.float32)
        # image = np.dot(image, self.model)
        # corrected_image = np.clip(image, 0, 255).astype(np.uint8).reshape(h, w, c)
        # return corrected_image
        # Reshape
        h, w, c = input_image.shape
        image = input_image.reshape(-1, 3).astype(np.float32)

        image = np.dot(input_image, self.model)

        # Clip dan convert kembali ke uint8
        corrected_image = np.clip(image, 0, 255).astype(np.uint8).reshape(h, w, c)
        return corrected_image
