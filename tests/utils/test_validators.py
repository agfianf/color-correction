"""Tests for validation utilities."""

import numpy as np
import pytest

from color_correction.exceptions import InvalidBoxError, InvalidImageError
from color_correction.utils.validators import (
    validate_bgr_image,
    validate_box,
    validate_confidence_threshold,
    validate_iou_threshold,
    validate_patches,
)


class TestValidateBGRImage:
    """Tests for validate_bgr_image function."""

    def test_valid_image(self):
        """Test that valid BGR image passes validation."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        validate_bgr_image(image)  # Should not raise

    def test_valid_image_large(self):
        """Test that large valid image passes validation."""
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        validate_bgr_image(image, min_height=100, min_width=100)

    def test_none_image(self):
        """Test that None image raises error."""
        with pytest.raises(InvalidImageError, match="cannot be None"):
            validate_bgr_image(None)

    def test_none_image_custom_param_name(self):
        """Test error message uses custom parameter name."""
        with pytest.raises(InvalidImageError, match="input_image cannot be None"):
            validate_bgr_image(None, param_name="input_image")

    def test_not_numpy_array(self):
        """Test that non-numpy array raises error."""
        with pytest.raises(InvalidImageError, match="must be numpy array"):
            validate_bgr_image([1, 2, 3])

    def test_wrong_dimensions_2d(self):
        """Test that 2D array raises error."""
        image = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="must have 3 dimensions"):
            validate_bgr_image(image)

    def test_wrong_dimensions_4d(self):
        """Test that 4D array raises error."""
        image = np.zeros((10, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="must have 3 dimensions"):
            validate_bgr_image(image)

    def test_wrong_channels_1(self):
        """Test that 1 channel raises error."""
        image = np.zeros((100, 100, 1), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="must have 3 channels"):
            validate_bgr_image(image)

    def test_wrong_channels_4(self):
        """Test that 4 channels raises error."""
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="must have 3 channels"):
            validate_bgr_image(image)

    def test_too_small_height(self):
        """Test that image smaller than minimum height raises error."""
        image = np.zeros((10, 100, 3), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="too small"):
            validate_bgr_image(image, min_height=50, min_width=50)

    def test_too_small_width(self):
        """Test that image smaller than minimum width raises error."""
        image = np.zeros((100, 10, 3), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="too small"):
            validate_bgr_image(image, min_height=50, min_width=50)

    def test_empty_image(self):
        """Test that empty image raises error."""
        image = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(InvalidImageError, match="too small"):
            validate_bgr_image(image)


class TestValidatePatches:
    """Tests for validate_patches function."""

    def test_valid_patches(self):
        """Test that valid patches list passes validation."""
        patches = [np.array([1, 2, 3]) for _ in range(24)]
        validate_patches(patches)  # Should not raise

    def test_none_patches(self):
        """Test that None patches raises error."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_patches(None)

    def test_not_list(self):
        """Test that non-list raises error."""
        with pytest.raises(ValueError, match="must be a list"):
            validate_patches("not a list")

    def test_wrong_count(self):
        """Test that wrong number of patches raises error."""
        patches = [1, 2, 3]
        with pytest.raises(ValueError, match="must have 24 items"):
            validate_patches(patches)

    def test_custom_expected_count(self):
        """Test validation with custom expected count."""
        patches = [1, 2, 3]
        validate_patches(patches, expected_count=3)  # Should not raise

    def test_contains_none_not_allowed(self):
        """Test that None values in list raise error when not allowed."""
        patches = [1, 2, None, 4] + [0] * 20
        with pytest.raises(ValueError, match="contains 1 None values"):
            validate_patches(patches, allow_none=False)

    def test_contains_none_allowed(self):
        """Test that None values pass when allowed."""
        patches = [1, None, 3] + [0] * 21
        validate_patches(patches, allow_none=True)  # Should not raise

    def test_multiple_none_values(self):
        """Test error message with multiple None values."""
        patches = [None] * 24
        with pytest.raises(ValueError, match="contains 24 None values"):
            validate_patches(patches, allow_none=False)


class TestValidateBox:
    """Tests for validate_box function."""

    def test_valid_box(self):
        """Test that valid box passes validation."""
        validate_box((10, 20, 100, 200))  # Should not raise

    def test_valid_box_float(self):
        """Test that box with float coordinates passes."""
        validate_box((10.5, 20.5, 100.5, 200.5))

    def test_valid_box_list(self):
        """Test that box as list passes validation."""
        validate_box([10, 20, 100, 200])

    def test_not_tuple_or_list(self):
        """Test that non-tuple/list raises error."""
        with pytest.raises(InvalidBoxError, match="must be tuple or list"):
            validate_box("not a box")

    def test_wrong_length_short(self):
        """Test that box with < 4 elements raises error."""
        with pytest.raises(InvalidBoxError, match="must have 4 elements"):
            validate_box((10, 20, 100))

    def test_wrong_length_long(self):
        """Test that box with > 4 elements raises error."""
        with pytest.raises(InvalidBoxError, match="must have 4 elements"):
            validate_box((10, 20, 100, 200, 300))

    def test_non_numeric_coordinates(self):
        """Test that non-numeric coordinates raise error."""
        with pytest.raises(InvalidBoxError, match="coordinates must be numeric"):
            validate_box((10, "20", 100, 200))

    def test_inverted_x_coordinates(self):
        """Test that x1 >= x2 raises error."""
        with pytest.raises(InvalidBoxError, match="x1.*must be less than x2"):
            validate_box((100, 20, 10, 200))

    def test_equal_x_coordinates(self):
        """Test that x1 == x2 raises error."""
        with pytest.raises(InvalidBoxError, match="x1.*must be less than x2"):
            validate_box((100, 20, 100, 200))

    def test_inverted_y_coordinates(self):
        """Test that y1 >= y2 raises error."""
        with pytest.raises(InvalidBoxError, match="y1.*must be less than y2"):
            validate_box((10, 200, 100, 20))

    def test_equal_y_coordinates(self):
        """Test that y1 == y2 raises error."""
        with pytest.raises(InvalidBoxError, match="y1.*must be less than y2"):
            validate_box((10, 100, 100, 100))

    def test_negative_x1(self):
        """Test that negative x1 raises error."""
        with pytest.raises(InvalidBoxError, match="cannot be negative"):
            validate_box((-10, 20, 100, 200))

    def test_negative_y1(self):
        """Test that negative y1 raises error."""
        with pytest.raises(InvalidBoxError, match="cannot be negative"):
            validate_box((10, -20, 100, 200))

    def test_numpy_integers(self):
        """Test that numpy integer types work."""
        box = (np.int32(10), np.int32(20), np.int32(100), np.int32(200))
        validate_box(box)  # Should not raise


class TestValidateConfidenceThreshold:
    """Tests for validate_confidence_threshold function."""

    def test_valid_threshold_zero(self):
        """Test that 0.0 is valid."""
        validate_confidence_threshold(0.0)

    def test_valid_threshold_one(self):
        """Test that 1.0 is valid."""
        validate_confidence_threshold(1.0)

    def test_valid_threshold_middle(self):
        """Test that 0.5 is valid."""
        validate_confidence_threshold(0.5)

    def test_non_numeric(self):
        """Test that non-numeric raises error."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_confidence_threshold("0.5")

    def test_too_low(self):
        """Test that value < 0 raises error."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_confidence_threshold(-0.1)

    def test_too_high(self):
        """Test that value > 1 raises error."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_confidence_threshold(1.5)

    def test_custom_param_name(self):
        """Test that custom parameter name appears in error."""
        with pytest.raises(ValueError, match="my_threshold must be between"):
            validate_confidence_threshold(2.0, param_name="my_threshold")


class TestValidateIoUThreshold:
    """Tests for validate_iou_threshold function."""

    def test_valid_threshold(self):
        """Test that valid threshold passes."""
        validate_iou_threshold(0.7)

    def test_edge_case_zero(self):
        """Test that 0.0 is valid."""
        validate_iou_threshold(0.0)

    def test_edge_case_one(self):
        """Test that 1.0 is valid."""
        validate_iou_threshold(1.0)

    def test_non_numeric(self):
        """Test that non-numeric raises error."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_iou_threshold([0.7])

    def test_out_of_range_low(self):
        """Test that value < 0 raises error."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_iou_threshold(-0.5)

    def test_out_of_range_high(self):
        """Test that value > 1 raises error."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_iou_threshold(1.1)
