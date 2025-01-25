__version__ = "0.0.1-alpha"


from color_correction_asdfghjkl.constant.color_checker import (
    reference_color_d50 as REFERENCE_COLOR_D50,  # noqa: N812
)
from color_correction_asdfghjkl.core.card_detection.yolov8_det_onnx import (
    YOLOv8CardDetector,
)
from color_correction_asdfghjkl.schemas.yolov8_det import (
    DetectionResult as YOLOv8DetectionResult,
)
from color_correction_asdfghjkl.services.color_correction import ColorCorrection
from color_correction_asdfghjkl.utils.image_processing import (
    calculate_mean_rgb,
    crop_region_with_margin,
    generate_image_patches,
)

__all__ = [
    "__version__",
    "REFERENCE_COLOR_D50",
    "ColorCorrection",
    "YOLOv8CardDetector",
    "YOLOv8DetectionResult",
]
