__version__ = "0.0.1b2"

# fmt: off
from .constant.color_checker import reference_color_d50_bgr as REFERENCE_COLOR_D50_BGR  # noqa: N812, I001
from .constant.color_checker import reference_color_d50_rgb as REFERENCE_COLOR_D50_RGB  # noqa: N812, I001
from .core.card_detection.det_yv8_onnx import YOLOv8CardDetector
from .schemas.det_yv8 import DetectionResult as YOLOv8DetectionResult
from .services.color_correction import ColorCorrection
from .services.report import CorrectionReport
# fmt: on

__all__ = [
    "__version__",
    "REFERENCE_COLOR_D50_BGR",
    "REFERENCE_COLOR_D50_RGB",
    "ColorCorrection",
    "CorrectionReport",
    "YOLOv8CardDetector",
    "YOLOv8DetectionResult",
]
