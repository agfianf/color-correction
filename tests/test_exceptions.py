"""Tests for custom exceptions."""

import pytest

from color_correction.exceptions import (
    ColorCorrectionError,
    DetectionError,
    InsufficientPatchesError,
    InvalidBoxError,
    InvalidImageError,
    LowConfidenceDetectionError,
    ModelNotFittedError,
    NoCardDetectedError,
    PatchesNotSetError,
    UnsupportedModelError,
)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_base_exception(self):
        """Test that ColorCorrectionError inherits from Exception."""
        assert issubclass(ColorCorrectionError, Exception)

    def test_detection_error_hierarchy(self):
        """Test that DetectionError inherits from ColorCorrectionError."""
        assert issubclass(DetectionError, ColorCorrectionError)

    def test_specific_detection_errors(self):
        """Test that specific detection errors inherit from DetectionError."""
        assert issubclass(NoCardDetectedError, DetectionError)
        assert issubclass(InsufficientPatchesError, DetectionError)
        assert issubclass(LowConfidenceDetectionError, DetectionError)


class TestInsufficientPatchesError:
    """Test InsufficientPatchesError."""

    def test_attributes_and_message(self):
        """Test that error has correct attributes and message."""
        error = InsufficientPatchesError(detected_count=18, required_count=24)
        assert error.detected_count == 18
        assert error.required_count == 24
        assert "18" in str(error)
        assert "24" in str(error)
        assert "required" in str(error).lower()

    def test_default_required_count(self):
        """Test that required_count defaults to 24."""
        error = InsufficientPatchesError(detected_count=10)
        assert error.required_count == 24
        assert "24" in str(error)

    def test_can_be_raised(self):
        """Test that error can be raised and caught."""
        with pytest.raises(InsufficientPatchesError) as exc_info:
            raise InsufficientPatchesError(detected_count=5)
        assert exc_info.value.detected_count == 5


class TestLowConfidenceDetectionError:
    """Test LowConfidenceDetectionError."""

    def test_attributes_and_message(self):
        """Test that error has correct attributes and message."""
        error = LowConfidenceDetectionError(max_confidence=0.15, threshold=0.25)
        assert error.max_confidence == 0.15
        assert error.threshold == 0.25
        assert "0.15" in str(error)
        assert "0.25" in str(error)

    def test_helpful_message(self):
        """Test that error message is helpful."""
        error = LowConfidenceDetectionError(max_confidence=0.1, threshold=0.5)
        message = str(error)
        assert "confidence" in message.lower()
        assert "lighting" in message.lower() or "quality" in message.lower()


class TestModelNotFittedError:
    """Test ModelNotFittedError."""

    def test_default_message(self):
        """Test that error has helpful default message."""
        error = ModelNotFittedError()
        message = str(error)
        assert "fit" in message.lower()
        assert "must be fitted" in message.lower()

    def test_custom_message(self):
        """Test that custom message can be provided."""
        error = ModelNotFittedError("Custom message")
        assert "Custom message" in str(error)
        assert "fit() first" in str(error)


class TestPatchesNotSetError:
    """Test PatchesNotSetError."""

    def test_reference_patches(self):
        """Test error for reference patches."""
        error = PatchesNotSetError("reference")
        assert error.patch_type == "reference"
        assert "Reference" in str(error)
        assert "set_reference_patches" in str(error)

    def test_input_patches(self):
        """Test error for input patches."""
        error = PatchesNotSetError("input")
        assert error.patch_type == "input"
        assert "Input" in str(error)
        assert "set_input_patches" in str(error)


class TestInvalidImageError:
    """Test InvalidImageError."""

    def test_reason_in_message(self):
        """Test that reason is included in message."""
        error = InvalidImageError("image cannot be None")
        assert error.reason == "image cannot be None"
        assert "Invalid image" in str(error)
        assert "None" in str(error)

    def test_various_reasons(self):
        """Test with different reasons."""
        reasons = [
            "must have 3 dimensions",
            "must have 3 channels",
            "too small",
        ]
        for reason in reasons:
            error = InvalidImageError(reason)
            assert reason in str(error)


class TestUnsupportedModelError:
    """Test UnsupportedModelError."""

    def test_attributes_and_message(self):
        """Test that error has correct attributes and message."""
        error = UnsupportedModelError("invalid_model", ["yolov8", "mcc"])
        assert error.model_name == "invalid_model"
        assert error.supported_models == ["yolov8", "mcc"]
        assert "invalid_model" in str(error)
        assert "yolov8" in str(error)
        assert "mcc" in str(error)

    def test_single_supported_model(self):
        """Test with single supported model."""
        error = UnsupportedModelError("bad", ["good"])
        assert "good" in str(error)


class TestInvalidBoxError:
    """Test InvalidBoxError."""

    def test_box_coordinates_in_message(self):
        """Test that box coordinates are in message."""
        box = (10, 20, 100, 200)
        error = InvalidBoxError(box)
        assert error.box == box
        message = str(error)
        assert "Invalid box" in message

    def test_with_reason(self):
        """Test error with specific reason."""
        box = (100, 20, 10, 200)
        error = InvalidBoxError(box, "x1 must be less than x2")
        assert "x1 must be less than x2" in str(error)

    def test_without_reason(self):
        """Test error without specific reason."""
        box = (-10, 20, 100, 200)
        error = InvalidBoxError(box)
        assert "Invalid box coordinates" in str(error)


class TestNoCardDetectedError:
    """Test NoCardDetectedError."""

    def test_default_message(self):
        """Test default error message."""
        error = NoCardDetectedError()
        assert "No color checker card detected" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = NoCardDetectedError("Card not found in image")
        assert "Card not found in image" in str(error)
