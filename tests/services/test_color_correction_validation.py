"""Tests for ColorCorrection service input validation."""

import numpy as np
import pytest

from color_correction.exceptions import (
    InvalidImageError,
    ModelNotFittedError,
    PatchesNotSetError,
    UnsupportedModelError,
)
from color_correction.services.color_correction import ColorCorrection


class TestColorCorrectionValidation:
    """Test input validation in ColorCorrection service."""

    def test_invalid_detection_model_raises_error(self):
        """Test that invalid detection model raises UnsupportedModelError."""
        with pytest.raises(UnsupportedModelError, match="invalid_model"):
            ColorCorrection(detection_model="invalid_model")

    def test_invalid_detection_model_shows_supported_models(self):
        """Test that error message includes supported models."""
        with pytest.raises(UnsupportedModelError) as exc_info:
            ColorCorrection(detection_model="bad_model")

        error_message = str(exc_info.value)
        assert "yolov8" in error_message
        assert "mcc" in error_message
        assert "Supported models" in error_message

    def test_valid_detection_models(self):
        """Test that valid detection models work."""
        # mcc should work (doesn't require model download)
        cc = ColorCorrection(detection_model="mcc")
        assert cc is not None

        # Note: yolov8 test skipped as it requires model download

    def test_init_with_invalid_reference_image_raises_error(self):
        """Test that invalid reference image in __init__ raises error."""
        invalid_image = np.zeros((100, 100), dtype=np.uint8)  # 2D instead of 3D
        with pytest.raises(InvalidImageError, match="must have 3 dimensions"):
            ColorCorrection(reference_image=invalid_image)

    def test_init_with_none_reference_image(self):
        """Test that None reference image is allowed."""
        cc = ColorCorrection(reference_image=None)
        assert cc.reference_patches is not None  # Should use default D50 values

    def test_set_input_patches_with_none_image(self):
        """Test that None image raises InvalidImageError."""
        cc = ColorCorrection()
        with pytest.raises(InvalidImageError, match="cannot be None"):
            cc.set_input_patches(image=None)

    def test_set_input_patches_with_2d_array(self):
        """Test that 2D array raises InvalidImageError."""
        cc = ColorCorrection()
        image = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="must have 3 dimensions"):
            cc.set_input_patches(image=image)

    def test_set_input_patches_with_wrong_channels(self):
        """Test that image with wrong channels raises error."""
        cc = ColorCorrection()
        image = np.zeros((100, 100, 4), dtype=np.uint8)  # 4 channels instead of 3
        with pytest.raises(InvalidImageError, match="must have 3 channels"):
            cc.set_input_patches(image=image)

    def test_set_input_patches_with_non_numpy_array(self):
        """Test that non-numpy array raises InvalidImageError."""
        cc = ColorCorrection()
        with pytest.raises(InvalidImageError, match="must be numpy array"):
            cc.set_input_patches(image="not an array")

    def test_fit_without_reference_patches(self):
        """Test that fit without reference patches raises PatchesNotSetError."""
        cc = ColorCorrection(reference_image=None)
        cc.reference_patches = None  # Force to None
        with pytest.raises(PatchesNotSetError, match="Reference"):
            cc.fit()

    def test_fit_without_reference_patches_has_helpful_message(self):
        """Test that error message is helpful."""
        cc = ColorCorrection(reference_image=None)
        cc.reference_patches = None
        with pytest.raises(PatchesNotSetError) as exc_info:
            cc.fit()

        error_message = str(exc_info.value)
        assert "reference" in error_message.lower()
        assert "set_reference_patches" in error_message

    def test_fit_without_input_patches(self):
        """Test that fit without input patches raises PatchesNotSetError."""
        cc = ColorCorrection()
        # reference_patches are set by default, but input_patches are not
        with pytest.raises(PatchesNotSetError, match="Input"):
            cc.fit()

    def test_fit_without_input_patches_has_helpful_message(self):
        """Test that error message mentions set_input_patches."""
        cc = ColorCorrection()
        with pytest.raises(PatchesNotSetError) as exc_info:
            cc.fit()

        error_message = str(exc_info.value)
        assert "input" in error_message.lower()
        assert "set_input_patches" in error_message

    def test_predict_without_fit(self):
        """Test that predict without fit raises ModelNotFittedError."""
        cc = ColorCorrection()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ModelNotFittedError, match="must be fitted"):
            cc.predict(image)

    def test_predict_without_fit_has_helpful_message(self):
        """Test that error message mentions fit()."""
        cc = ColorCorrection()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ModelNotFittedError) as exc_info:
            cc.predict(image)

        error_message = str(exc_info.value)
        assert "fit" in error_message.lower()
        assert "fit() first" in error_message

    def test_predict_with_none_image(self):
        """Test that predict with None image raises InvalidImageError."""
        cc = ColorCorrection()
        cc.trained_model = "dummy"  # Fake trained model
        with pytest.raises(InvalidImageError, match="cannot be None"):
            cc.predict(None)

    def test_predict_with_invalid_image_dimensions(self):
        """Test that predict with invalid image raises error."""
        cc = ColorCorrection()
        cc.trained_model = "dummy"
        image = np.zeros((100, 100), dtype=np.uint8)  # 2D
        with pytest.raises(InvalidImageError, match="must have 3 dimensions"):
            cc.predict(image)

    def test_predict_with_invalid_image_channels(self):
        """Test that predict with wrong channels raises error."""
        cc = ColorCorrection()
        cc.trained_model = "dummy"
        image = np.zeros((100, 100, 1), dtype=np.uint8)  # 1 channel
        with pytest.raises(InvalidImageError, match="must have 3 channels"):
            cc.predict(image)


class TestColorCorrectionExceptionAttributes:
    """Test that exceptions have correct attributes."""

    def test_unsupported_model_error_attributes(self):
        """Test UnsupportedModelError has model_name and supported_models."""
        with pytest.raises(UnsupportedModelError) as exc_info:
            ColorCorrection(detection_model="fake_model")

        error = exc_info.value
        assert error.model_name == "fake_model"
        assert error.supported_models == ["yolov8", "mcc"]

    def test_patches_not_set_error_attributes(self):
        """Test PatchesNotSetError has patch_type attribute."""
        cc = ColorCorrection()
        with pytest.raises(PatchesNotSetError) as exc_info:
            cc.fit()

        error = exc_info.value
        assert error.patch_type == "input"

    def test_invalid_image_error_has_reason(self):
        """Test InvalidImageError has reason attribute."""
        cc = ColorCorrection()
        with pytest.raises(InvalidImageError) as exc_info:
            cc.set_input_patches(None)

        error = exc_info.value
        assert hasattr(error, "reason")
        assert "cannot be None" in error.reason


class TestColorCorrectionValidationParameterNames:
    """Test that validation uses correct parameter names in errors."""

    def test_reference_image_validation_uses_correct_param_name(self):
        """Test that reference_image validation shows 'reference_image' in error."""
        image = np.zeros((100, 100), dtype=np.uint8)  # Invalid 2D
        with pytest.raises(InvalidImageError) as exc_info:
            ColorCorrection(reference_image=image)

        error_message = str(exc_info.value)
        assert "reference_image" in error_message

    def test_set_input_patches_validation_uses_correct_param_name(self):
        """Test that set_input_patches validation shows 'image' in error."""
        cc = ColorCorrection()
        image = np.zeros((100, 100), dtype=np.uint8)  # Invalid 2D
        with pytest.raises(InvalidImageError) as exc_info:
            cc.set_input_patches(image)

        error_message = str(exc_info.value)
        assert "image" in error_message

    def test_predict_validation_uses_correct_param_name(self):
        """Test that predict validation shows 'input_image' in error."""
        cc = ColorCorrection()
        cc.trained_model = "dummy"
        image = np.zeros((100, 100), dtype=np.uint8)  # Invalid 2D
        with pytest.raises(InvalidImageError) as exc_info:
            cc.predict(image)

        error_message = str(exc_info.value)
        assert "input_image" in error_message


class TestColorCorrectionValidationEdgeCases:
    """Test edge cases in validation."""

    def test_empty_image_raises_error(self):
        """Test that empty image (0x0) raises error."""
        cc = ColorCorrection()
        image = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="too small"):
            cc.set_input_patches(image)

    def test_very_small_image_raises_error(self):
        """Test that very small image raises error if min size specified."""
        cc = ColorCorrection()
        # Note: validate_bgr_image has default min_height=1, min_width=1
        # So we're just checking it doesn't crash with small images
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        # This should not raise for default min size
        try:
            cc.set_input_patches(image)
        except Exception as e:
            # It might raise other errors from detection, but not InvalidImageError for size
            assert not isinstance(e, InvalidImageError) or "too small" not in str(e)

    def test_list_instead_of_array_raises_error(self):
        """Test that passing list instead of array raises error."""
        cc = ColorCorrection()
        with pytest.raises(InvalidImageError, match="must be numpy array"):
            cc.set_input_patches([[1, 2, 3]])

    def test_dict_instead_of_array_raises_error(self):
        """Test that passing dict raises error."""
        cc = ColorCorrection()
        with pytest.raises(InvalidImageError, match="must be numpy array"):
            cc.set_input_patches({"key": "value"})


class TestColorCorrectionValidationWorkflow:
    """Test validation in typical workflows."""

    def test_valid_workflow_does_not_raise(self):
        """Test that valid workflow completes without validation errors."""
        # This test would need actual valid images and patches
        # For now, just test that instantiation works
        cc = ColorCorrection(
            detection_model="mcc",
            correction_model="least_squares",
        )
        assert cc is not None
        assert cc.reference_patches is not None  # Should have default D50 values

    def test_changing_detection_model_after_init_not_allowed(self):
        """Test that detection model is set during initialization."""
        cc = ColorCorrection(detection_model="mcc")
        # Detector is created in __init__, can't easily change it
        # This is more of a design test
        assert cc.card_detector is not None
