"""Custom exceptions for color correction operations.

This module provides a hierarchy of exceptions for better error handling
and debugging throughout the color correction package.
"""


class ColorCorrectionError(Exception):
    """Base exception for all color correction errors."""

    pass


class DetectionError(ColorCorrectionError):
    """Raised when card detection fails."""

    pass


class NoCardDetectedError(DetectionError):
    """Raised when no color checker card is found in the image."""

    def __init__(self, message: str = "No color checker card detected in image") -> None:
        self.message = message
        super().__init__(self.message)


class InsufficientPatchesError(DetectionError):
    """Raised when detected patches are incomplete.

    Attributes
    ----------
    detected_count : int
        Number of patches actually detected
    required_count : int
        Number of patches required (default 24)
    """

    def __init__(self, detected_count: int, required_count: int = 24) -> None:
        self.detected_count = detected_count
        self.required_count = required_count
        self.message = (
            f"Detected {detected_count} patches, but {required_count} are required. "
            f"Please ensure the color checker card is fully visible and well-lit."
        )
        super().__init__(self.message)


class LowConfidenceDetectionError(DetectionError):
    """Raised when detection confidence is below acceptable threshold.

    Attributes
    ----------
    max_confidence : float
        Highest confidence score among detections
    threshold : float
        Required confidence threshold
    """

    def __init__(self, max_confidence: float, threshold: float) -> None:
        self.max_confidence = max_confidence
        self.threshold = threshold
        self.message = (
            f"Detection confidence {max_confidence:.2f} is below threshold {threshold:.2f}. "
            f"Try improving lighting or image quality."
        )
        super().__init__(self.message)


class CorrectionError(ColorCorrectionError):
    """Raised when color correction fails."""

    pass


class ModelNotFittedError(CorrectionError):
    """Raised when attempting to predict with unfitted model.

    This error occurs when calling predict() or compute_correction()
    before calling fit() to train the correction model.
    """

    def __init__(self, message: str = "Model must be fitted before prediction") -> None:
        self.message = f"{message}. Call fit() first."
        super().__init__(self.message)


class PatchesNotSetError(CorrectionError):
    """Raised when required patches are not set before an operation.

    Attributes
    ----------
    patch_type : str
        Type of patches not set ('reference' or 'input')
    """

    def __init__(self, patch_type: str) -> None:
        self.patch_type = patch_type
        self.message = (
            f"{patch_type.capitalize()} patches must be set before this operation. "
            f"Call set_{patch_type}_patches() first."
        )
        super().__init__(self.message)


class InvalidImageError(ColorCorrectionError):
    """Raised when input image is invalid or has incorrect format.

    Attributes
    ----------
    reason : str
        Specific reason why the image is invalid
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.message = f"Invalid image: {reason}"
        super().__init__(self.message)


class ModelLoadError(ColorCorrectionError):
    """Raised when model loading fails.

    Attributes
    ----------
    model_path : str
        Path to the model that failed to load
    """

    def __init__(self, model_path: str, reason: str = "") -> None:
        self.model_path = model_path
        if reason:
            self.message = f"Failed to load model from '{model_path}': {reason}"
        else:
            self.message = f"Failed to load model from '{model_path}'"
        super().__init__(self.message)


class UnsupportedModelError(ColorCorrectionError):
    """Raised when an unsupported model type is requested.

    Attributes
    ----------
    model_name : str
        Name of the unsupported model
    supported_models : list[str]
        List of supported model names
    """

    def __init__(self, model_name: str, supported_models: list[str]) -> None:
        self.model_name = model_name
        self.supported_models = supported_models
        self.message = (
            f"Unsupported model: '{model_name}'. "
            f"Supported models are: {', '.join(supported_models)}"
        )
        super().__init__(self.message)


class GeometryError(ColorCorrectionError):
    """Raised when geometric operations fail."""

    pass


class InvalidBoxError(GeometryError):
    """Raised when bounding box coordinates are invalid.

    Attributes
    ----------
    box : tuple
        The invalid box coordinates
    """

    def __init__(self, box: tuple, reason: str = "") -> None:
        self.box = box
        if reason:
            self.message = f"Invalid box {box}: {reason}"
        else:
            self.message = f"Invalid box coordinates: {box}"
        super().__init__(self.message)
