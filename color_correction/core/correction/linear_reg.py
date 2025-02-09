import time

import numpy as np
from sklearn.linear_model import LinearRegression as LinearRegressionSklearn

from color_correction.core.correction.base import BaseComputeCorrection
from color_correction.utils.correction import (
    postprocessing_compute,
    preprocessing_compute,
)


class LinearRegression(BaseComputeCorrection):
    """
    LinearRegression correction using scikit-learn's LinearRegression.

    This class fits a linear model to input and reference patches and computes
    the corresponding color correction on images.

    Attributes
    ----------
    model : sklearn.linear_model.LinearRegression, optional
        The fitted linear regression model. Defaults to None until fitted.
    """

    def __init__(self) -> None:
        """
        Initialize the LinearRegression correction.

        Notes
        -----
        The model is set to None until the fit method is called.
        """
        self.model = None

    def fit(
        self,
        x_patches: np.ndarray,
        y_patches: np.ndarray,
    ) -> np.ndarray:
        """
        Fit the linear regression model on input and reference patches.

        Parameters
        ----------
        x_patches : np.ndarray
            Array of input patches for fitting.
        y_patches : np.ndarray
            Array of reference patches corresponding to the input.

        Returns
        -------
        np.ndarray
            The fitted model coefficients.

        Notes
        -----
        The method uses scikit-learn's LinearRegression without intercept.
        """
        start_time = time.perf_counter()

        self.model = LinearRegressionSklearn(fit_intercept=False).fit(
            x_patches,
            y_patches,
        )

        exc_time = time.perf_counter() - start_time
        print(f"{self.__class__.__name__} Fit: {exc_time} seconds")
        return self.model

    def compute_correction(self, input_image: np.ndarray) -> np.ndarray:
        """
        Compute the color correction for the given input image.

        Parameters
        ----------
        input_image : np.ndarray
            The input image array to be corrected.

        Returns
        -------
        np.ndarray
            The corrected image array.

        Raises
        ------
        ValueError
            If the model has not been fitted prior to calling this method.

        Notes
        -----
        The method preprocesses the input image, applies the linear regression model,
        and postprocesses the output to maintain the original image shape.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please call fit() method first.")

        org_input_shape = input_image.shape
        input_image = preprocessing_compute(input_image)
        image = self.model.predict(input_image)
        corrected_image = postprocessing_compute(org_input_shape, image)
        return corrected_image
