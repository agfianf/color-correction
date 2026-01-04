from typing import Any

from color_correction.core.correction.affine_reg import AffineRegression
from color_correction.core.correction.least_squares import (
    LeastSquaresRegression,
)
from color_correction.core.correction.linear_reg import LinearRegression
from color_correction.core.correction.polynomial import Polynomial
from color_correction.schemas.custom_types import LiteralModelCorrection

# Type alias for correction models
CorrectionModel = LeastSquaresRegression | Polynomial | LinearRegression | AffineRegression


class CorrectionModelFactory:
    """Factory class for creating color correction models."""

    @staticmethod
    def create(
        model_name: LiteralModelCorrection,
        **kwargs: Any,  # noqa: ANN401
    ) -> CorrectionModel:
        """
        Create a correction model instance based on the model name.

        Parameters
        ----------
        model_name : LiteralModelCorrection
            Name of the correction model to create.
        **kwargs : Any
            Additional parameters passed to the model constructor.

        Returns
        -------
        CorrectionModel
            An instance of the requested correction model.

        Raises
        ------
        KeyError
            If model_name is not a valid correction model.
        """
        model_registry: dict[str, CorrectionModel] = {
            "least_squares": LeastSquaresRegression(),
            "polynomial": Polynomial(**kwargs),
            "linear_reg": LinearRegression(),
            "affine_reg": AffineRegression(),
        }
        model = model_registry.get(model_name)
        if model is None:
            valid_models = list(model_registry.keys())
            raise KeyError(f"Unknown model '{model_name}'. Valid options: {valid_models}")
        return model
