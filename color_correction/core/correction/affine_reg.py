import time

import numpy as np
from sklearn.linear_model import LinearRegression

from color_correction.core.correction.base import BaseComputeCorrection
from color_correction.utils.correction import (
    postprocessing_compute,
    preprocessing_compute,
)


class AffineRegression(BaseComputeCorrection):
    """
    Apply an affine (linear) regression for color correction.

    This class uses a linear regression model without an intercept
    (by adding a bias column) to compute a correction transformation
    between input and reference patches.
    """

    def __init__(self) -> None:
        self.model = None

    def fit(
        self,
        x_patches: np.ndarray,  # input patches
        y_patches: np.ndarray,  # reference patches
    ) -> np.ndarray:
        """
        Fit a linear regression model using input and reference patches.

        Parameters
        ----------
        x_patches : np.ndarray
            Array of input patches with shape (n_samples, n_features).
        y_patches : np.ndarray
            Array of reference patches with shape (n_samples, n_targets).

        Returns
        -------
        np.ndarray
            The fitted linear regression model.
        """
        start_time = time.perf_counter()
        x_patches = np.array(x_patches)
        print("x_patches.shape", x_patches.shape)
        x_patches = np.hstack([x_patches, np.ones((x_patches.shape[0], 1))])
        self.model = LinearRegression(fit_intercept=False).fit(x_patches, y_patches)

        exc_time = time.perf_counter() - start_time
        print(f"{self.__class__.__name__} Fit: {exc_time} seconds")
        return self.model

    def compute_correction(self, input_image: np.ndarray) -> np.ndarray:
        """
        Compute the color-corrected image using the fitted regression model.

        Parameters
        ----------
        input_image : np.ndarray
            The input image to be corrected.

        Returns
        -------
        np.ndarray
            The color-corrected output image.

        Raises
        ------
        ValueError
            If the model has not been fitted before calling this method.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please call fit() method first.")

        org_input_shape = input_image.shape
        input_image = preprocessing_compute(input_image)
        input_image = np.hstack([input_image, np.ones((input_image.shape[0], 1))])
        image = self.model.predict(input_image)
        corrected_image = postprocessing_compute(org_input_shape, image)
        return corrected_image
