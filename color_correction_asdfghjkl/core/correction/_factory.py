from color_correction_asdfghjkl.core.correction.affine_reg import AffineReg
from color_correction_asdfghjkl.core.correction.least_squares import (
    LeastSquaresRegression,
)
from color_correction_asdfghjkl.core.correction.linear_reg import LinearReg
from color_correction_asdfghjkl.core.correction.polynomial import Polynomial


class CorrectionModelFactory:
    @staticmethod
    def create(model_name: str, **kwargs: dict) -> ...:
        model_registry = {
            "least_squares": LeastSquaresRegression(),
            "polynomial": Polynomial(**kwargs),
            "linear_reg": LinearReg(),
            "affine_reg": AffineReg(),
        }
        return model_registry.get(model_name)
