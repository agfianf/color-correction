import numpy as np
import pytest

from color_correction.utils.yolo_utils import (
    compute_iou,
    multiclass_nms,
    nms,
    xywh2xyxy,
)


# ------------------- Test for nms ---------------------------
@pytest.mark.parametrize(
    "boxes,scores,iou_threshold,expected_len",
    [
        # 1. Happy path - 2 overlapping boxes and 1 separate box
        (
            np.array(
                [[0, 0, 10, 10], [9, 9, 30, 30], [1, 1, 11, 11]],
                dtype=np.float32,
            ),  # noqa: E501
            np.array([0.9, 0.8, 0.7]),
            0.5,
            2,
        ),
        # 2. Edge case - single box
        (
            np.array([[0, 0, 10, 10]], dtype=np.float32),
            np.array([0.9]),
            0.5,
            1,
        ),
        # 3. Edge case - no overlapping boxes
        (
            np.array(
                [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
                dtype=np.float32,
            ),  # noqa: E501
            np.array([0.9, 0.8, 0.7]),
            0.5,
            3,
        ),
        # 4. Edge case - all boxes overlap significantly
        (
            np.array(
                [[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 10, 10]],
                dtype=np.float32,
            ),
            np.array([0.9, 0.8, 0.7]),
            0.5,
            1,
        ),
        # 5. Edge case - very low IoU threshold
        (
            np.array([[0, 0, 10, 10], [2, 2, 12, 12]], dtype=np.float32),
            np.array([0.9, 0.8]),
            0.1,
            1,
        ),
        # 6. Edge case - very high IoU threshold
        (
            np.array([[0, 0, 10, 10], [2, 2, 12, 12]], dtype=np.float32),
            np.array([0.9, 0.8]),
            0.9,
            2,
        ),
    ],
)
def test_nms_parametrized(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    expected_len: int,
) -> None:
    keep_indices = nms(boxes, scores, iou_threshold)
    assert len(keep_indices) == expected_len
    assert 0 in keep_indices  # Highest scoring box should always be kept
    assert all(
        scores[i] >= scores[j]
        for i, j in zip(keep_indices[:-1], keep_indices[1:], strict=False)
    )  # Scores should be descending


def test_nms_empty() -> None:
    empty_boxes = np.array([], dtype=np.float32).reshape(0, 4)
    empty_scores = np.array([], dtype=np.float32)
    keep_indices = nms(boxes=empty_boxes, scores=empty_scores, iou_threshold=0.5)
    assert len(keep_indices) == 0


def test_nms_same_score() -> None:
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.9])
    keep_indices = nms(boxes=boxes, scores=scores, iou_threshold=0.5)
    assert len(keep_indices) == 2


# ------------------- Test for compute_iou -------------------
@pytest.mark.parametrize(
    "box,boxes,expected_ious",
    [
        # No overlap case
        (
            np.array([0, 0, 10, 10]),
            np.array([[20, 20, 30, 30], [40, 40, 50, 50]]),
            np.array([0.0, 0.0]),
        ),
        # Partial overlap case
        (
            np.array([0, 0, 20, 20]),
            np.array([[10, 10, 30, 30]]),
            np.array([0.14285714]),
        ),
        # Complete overlap (identical boxes)
        (
            np.array([0, 0, 10, 10]),
            np.array([[0, 0, 10, 10]]),
            np.array([1.0]),
        ),
        # Multiple boxes with varying overlap
        (
            np.array([0, 0, 10, 10]),
            np.array([[5, 5, 15, 15], [0, 0, 5, 5], [20, 20, 30, 30]]),
            np.array([0.14285714, 0.25, 0.0]),
        ),
    ],
)
def test_compute_iou(
    box: np.ndarray,
    boxes: np.ndarray,
    expected_ious: np.ndarray,
) -> None:
    ious = compute_iou(box, boxes)
    np.testing.assert_array_almost_equal(ious, expected_ious, decimal=7)


def test_compute_iou_empty_boxes() -> None:
    box = np.array([0, 0, 10, 10])
    empty_boxes = np.array([], dtype=np.float32).reshape(0, 4)
    ious = compute_iou(box, empty_boxes)
    assert len(ious) == 0


# ------------------- Test for multiclass_nms -------------------
@pytest.mark.parametrize(
    "boxes,scores,class_ids,iou_threshold,expected_len",
    [
        # 1. Happy path - multiple classes with some overlapping boxes
        (
            np.array(
                [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]],
                dtype=np.float32,
            ),
            np.array([0.9, 0.8, 0.7]),
            np.array([0, 0, 1]),
            0.5,
            2,
        ),
        # 2. Edge case - single class
        (
            np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32),
            np.array([0.9, 0.8]),
            np.array([0, 0]),
            0.5,
            1,
        ),
        # 3. Edge case - no overlapping boxes, different classes
        (
            np.array(
                [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
                dtype=np.float32,
            ),
            np.array([0.9, 0.8, 0.7]),
            np.array([0, 1, 2]),
            0.5,
            3,
        ),
        # 4. Edge case - multiple overlapping boxes with same class
        (
            np.array(
                [[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 10, 10]],
                dtype=np.float32,
            ),
            np.array([0.9, 0.8, 0.7]),
            np.array([0, 0, 0]),
            0.5,
            1,
        ),
        # 5. Edge case - multiple overlapping boxes with different classes
        (
            np.array(
                [[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]],
                dtype=np.float32,
            ),
            np.array([0.9, 0.8, 0.7]),
            np.array([0, 1, 2]),
            0.5,
            3,
        ),
    ],
)
def test_multiclass_nms_parametrized(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    expected_len: int,
) -> None:
    keep_indices = multiclass_nms(boxes, scores, class_ids, iou_threshold)
    assert len(keep_indices) == expected_len

    # Check that boxes with highest scores per class are kept
    unique_classes = np.unique(class_ids)
    for class_id in unique_classes:
        class_mask = class_ids == class_id
        class_scores = scores[class_mask]
        if len(class_scores) > 0:
            max_score_idx = np.where(class_mask)[0][np.argmax(class_scores)]
            assert max_score_idx in keep_indices


def test_multiclass_nms_empty() -> None:
    empty_boxes = np.array([], dtype=np.float32).reshape(0, 4)
    empty_scores = np.array([], dtype=np.float32)
    empty_class_ids = np.array([], dtype=np.int32)
    keep_indices = multiclass_nms(
        boxes=empty_boxes,
        scores=empty_scores,
        class_ids=empty_class_ids,
        iou_threshold=0.5,
    )
    assert len(keep_indices) == 0


def test_multiclass_nms_same_score_diff_class() -> None:
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.9])
    class_ids = np.array([0, 1])
    keep_indices = multiclass_nms(
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        iou_threshold=0.5,
    )
    assert len(keep_indices) == 2


# ------------------- Test for xywh2xyxy -------------------
@pytest.mark.parametrize(
    "input_boxes,expected_boxes",
    [
        # Single box test case
        (
            np.array([[10, 10, 20, 20]], dtype=np.float32),
            np.array([[0, 0, 20, 20]], dtype=np.float32),
        ),
        # Multiple boxes test case
        (
            np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float32),
            np.array([[0, 0, 20, 20], [10, 10, 50, 50]], dtype=np.float32),
        ),
        # Test with decimal values
        (
            np.array([[5.5, 5.5, 3, 3]], dtype=np.float32),
            np.array([[4, 4, 7, 7]], dtype=np.float32),
        ),
        # Test with zero width/height
        (
            np.array([[10, 10, 0, 0]], dtype=np.float32),
            np.array([[10, 10, 10, 10]], dtype=np.float32),
        ),
    ],
)
def test_xywh2xyxy_parametrized(
    input_boxes: np.ndarray,
    expected_boxes: np.ndarray,
) -> None:
    output_boxes = xywh2xyxy(input_boxes)
    np.testing.assert_array_almost_equal(output_boxes, expected_boxes)


def test_xywh2xyxy_empty() -> None:
    empty_boxes = np.array([], dtype=np.float32).reshape(0, 4)
    output = xywh2xyxy(empty_boxes)
    assert output.shape == (0, 4)
