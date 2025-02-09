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
    """
    Polynomial correction class using polynomial regression.

    Parameters
    ----------
    **kwargs : dict, optional
        Keyword arguments. Recognized keyword:
        degree : int, optional, default 2
            Degree of the polynomial.
    """

    def __init__(self, **kwargs: dict) -> None:
        """
        Initialize the Polynomial correction model.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for initialization.

        Other Parameters
        ----------------
        degree : int, optional
            Degree of the polynomial. Default is 2.
            The more complex the polynomial, the more flexible the model.
            But it may also lead to overfitting.
        """
        self.model = None
        self.degree = kwargs.get("degree", 2)

    def fit(
        self,
        x_patches: np.ndarray,  # input patches
        y_patches: np.ndarray,  # reference patches
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Fit the polynomial regression model.

        Parameters
        ----------
        x_patches : np.ndarray
            Input image patches.
        y_patches : np.ndarray
            Reference image patches.
        **kwargs : dict
            Additional keyword arguments. Recognized keyword:
            degree : int, optional
                Degree of the polynomial.

        Returns
        -------
        np.ndarray
            Fitted model pipeline.

        Notes
        -----
        The execution time for model fitting is printed.
        """
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
        """
        Compute and return the corrected image using the fitted model.

        Parameters
        ----------
        input_image : np.ndarray
            The input image to be corrected.

        Returns
        -------
        np.ndarray
            The corrected image.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please call fit() method first.")

        org_input_shape = input_image.shape
        input_image = preprocessing_compute(input_image)
        image = self.model.predict(input_image)
        corrected_image = postprocessing_compute(org_input_shape, image)
        return corrected_image
