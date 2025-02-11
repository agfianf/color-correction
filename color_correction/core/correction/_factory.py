from color_correction.core.correction.affine_reg import AffineRegression
from color_correction.core.correction.least_squares import (
    LeastSquaresRegression,
)
from color_correction.core.correction.linear_reg import LinearRegression
from color_correction.core.correction.polynomial import Polynomial


class CorrectionModelFactory:
    @staticmethod
    def create(
        model_name: str,
        **kwargs: dict,
    ) -> LeastSquaresRegression | Polynomial | LinearRegression | AffineRegression:
        model_registry = {
            "least_squares": LeastSquaresRegression(),
            "polynomial": Polynomial(**kwargs),
            "linear_reg": LinearRegression(),
            "affine_reg": AffineRegression(),
        }
        return model_registry.get(model_name)
