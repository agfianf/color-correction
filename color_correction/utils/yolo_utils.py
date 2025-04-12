"""YOLOv8 utility functions for object detection tasks.

This module provides utility functions for processing YOLOv8 detections, including:
- Non-maximum suppression (NMS)
- Bounding box conversions
- Drawing functions for visualization
"""

import cv2
import numpy as np

from color_correction.constant.yolov8_det import class_names

# Constants
RANDOM_SEED = 3
DEFAULT_COLOR = (0, 0, 255)
DEFAULT_THICKNESS = 2
DEFAULT_MASK_ALPHA = 0.3

# Initialize colors for visualization
rng = np.random.default_rng(RANDOM_SEED)
COLORS = rng.uniform(0, 255, size=(len(class_names), 3))


# Detection Processing Functions
def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

    Parameters
    ----------
    boxes : np.ndarray
        Array of bounding boxes with shape (N, 4) in (x1, y1, x2, y2) format.
    scores : np.ndarray
        Confidence scores for each bounding box.
    iou_threshold : float
        Threshold for Intersection over Union (IoU) to filter boxes.

    Returns
    -------
    list[int]
        List of indices for boxes to keep.
    """
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []

    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # if just only one box, then break
        if sorted_indices.size == 1:
            break

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold,
        keep_indices = np.where(ious < iou_threshold)[0]

        # update sorted_indices
        sorted_indices = sorted_indices[1:][keep_indices]

    return keep_boxes


def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """
    Perform Non-Maximum Suppression (NMS) on boxes across multiple classes.

    Parameters
    ----------
    boxes : np.ndarray
        Array of bounding boxes in (x1, y1, x2, y2) format.
    scores : np.ndarray
        Confidence scores corresponding to each box.
    class_ids : np.ndarray
        Class identifier for each bounding box.
    iou_threshold : float
        IoU threshold to determine overlapping boxes.

    Returns
    -------
    list[int]
        List of indices for boxes to keep.
    """
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])
    return keep_boxes


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) between a box and an array of boxes.

    Parameters
    ----------
    box : np.ndarray
        Single bounding box in (x1, y1, x2, y2) format.
    boxes : np.ndarray
        Array of boxes to compare against.

    Returns
    -------
    np.ndarray
        Array containing the IoU of the input box with each box in 'boxes'.
    """
    box = box.astype(np.float32)
    boxes = boxes.astype(np.float32)

    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area
    return iou


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2) format.

    Parameters
    ----------
    x : np.ndarray
        Array of bounding boxes in (x, y, w, h) format.

    Returns
    -------
    np.ndarray
        Array of bounding boxes in (x1, y1, x2, y2) format.
    """
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# Visualization Functions
def draw_segmentation(
    image: np.ndarray,
    segments: list[tuple[int, int]],
    scores: list[float],
    class_ids: list[int],
    mask_alpha: float = DEFAULT_MASK_ALPHA,  # noqa: ARG001
) -> np.ndarray:
    """
    Draw segmentation polygons on an image.

    Parameters
    ----------
    image : np.ndarray
        The original image.
    segments : list[list[int]]
        List of segmentation points in (x, y) format.
    scores : list[float]
        Confidence scores for every box.
    class_ids : list[int]
        Class IDs corresponding to each detection.
    mask_alpha : float, optional
        Transparency of the drawn mask, default is DEFAULT_MASK_ALPHA.

    Returns
    -------
    np.ndarray
        The image with the polygon drawn on it.
    """
    for class_id, segment, score in zip(class_ids, segments, scores, strict=False):
        color = COLORS[class_id]
        cv2.polylines(image, [np.int32(segment)], True, color, 2)

        draw_text(
            image,
            f"{class_names[class_id]} {int(score * 100)}%",
            segment[0],
            color,
            font_size=0.001,
            text_thickness=2,
        )
    return image


def draw_detections(
    image: np.ndarray,
    boxes: list[list[int]],
    scores: list[float],
    class_ids: list[int],
    mask_alpha: float = DEFAULT_MASK_ALPHA,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, and semi-transparent masks on an image.

    Parameters
    ----------
    image : np.ndarray
        The input image on which detections will be drawn.
    boxes : list[list[int]]
        List of bounding boxes represented as (x1, y1, x2, y2).
    scores : list[float]
        Confidence scores for every box.
    class_ids : list[int]
        Class IDs corresponding to each detection.
    mask_alpha : float, optional
        Transparency of the drawn mask, default is DEFAULT_MASK_ALPHA.

    Returns
    -------
    np.ndarray
        The annotated image with drawn detections.
    """
    det_img = image.copy()
    img_height, img_width = image.shape[:2]

    # Calculate font and text properties based on image size
    font_size = min([img_height, img_width]) * 0.0005
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw masks first (background layer)
    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw boxes and labels (foreground layer)
    for class_id, box, score in zip(class_ids, boxes, scores, strict=False):
        color = COLORS[class_id]
        draw_box(det_img, box, color)

        # Create and draw label
        label = f"{class_names[class_id]} {int(score * 100)}%"
        draw_text(det_img, label, box, color, font_size, text_thickness)

    return det_img


def draw_box(
    image: np.ndarray,
    box: list[int],
    color: tuple[int, int, int] = DEFAULT_COLOR,
    thickness: int = DEFAULT_THICKNESS,
) -> np.ndarray:
    """
    Draw a single bounding box on an image.

    Parameters
    ----------
    image : np.ndarray
        The original image.
    box : list[int]
        Bounding box coordinates in (x1, y1, x2, y2) format.
    color : tuple[int, int, int], optional
        Color of the box in RGB format; default is DEFAULT_COLOR.
    thickness : int, optional
        Line thickness; default is DEFAULT_THICKNESS.

    Returns
    -------
    np.ndarray
        The image with the box drawn on it.
    """
    x1, y1, x2, y2 = box
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(
    image: np.ndarray,
    text: str,
    box: list[int],
    color: tuple[int, int, int] = DEFAULT_COLOR,
    font_size: float = 0.001,
    text_thickness: int = 2,
) -> np.ndarray:
    """
    Draw text with a background rectangle near a bounding box.

    Parameters
    ----------
    image : np.ndarray
        The image on which text is drawn.
    text : str
        Text string to be displayed.
    box : list[int]
        Bounding box coordinates (x1, y1, x2, y2) where the text will be placed.
    color : tuple[int, int, int], optional
        Background color for the text; default is DEFAULT_COLOR.
    font_size : float, optional
        Font scaling factor; default is 0.001.
    text_thickness : int, optional
        Thickness of the text stroke; default is 2.

    Returns
    -------
    np.ndarray
        Image with the annotated text.
    """
    x1, y1, x2, y2 = box
    (tw, th), _ = cv2.getTextSize(
        text=text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_size,
        thickness=text_thickness,
    )
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

    return cv2.putText(
        image,
        text,
        (x1, y1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_masks(
    image: np.ndarray,
    boxes: list[list[int]],
    classes: list[int],
    mask_alpha: float = DEFAULT_MASK_ALPHA,
) -> np.ndarray:
    """
    Overlay semi-transparent masks on the image for detected objects.

    Parameters
    ----------
    image : np.ndarray
        The original image.
    boxes : list[list[int]]
        List of bounding boxes in (x1, y1, x2, y2) format.
    classes : list[int]
        Class IDs corresponding to each bounding box.
    mask_alpha : float, optional
        Alpha value for mask transparency; default is DEFAULT_MASK_ALPHA.

    Returns
    -------
    np.ndarray
        The image with masks applied.
    """
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes, strict=False):
        color = COLORS[class_id]

        x1, y1, x2, y2 = box

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)
