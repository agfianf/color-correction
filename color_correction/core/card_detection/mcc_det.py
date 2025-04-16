import cv2
import cv2.mcc
import numpy as np

from color_correction.core.card_detection.base import BaseCardDetector
from color_correction.schemas.det_yv8 import DetectionResult
from color_correction.utils.image_processing import crop_segment_straighten


class MCCardDetector(BaseCardDetector):
    """MCCardDetector implements card detection using OpenCV's ColorChecker detector (cv2.mcc).

    This class detects ColorChecker cards and their color patches in an image using OpenCV's
    built-in MCC ([Macbeth ColorChecker](https://docs.opencv.org/4.11.0/dd/d19/group__mcc.html)) detection module.
    It provides methods to detect cards, extract patch coordinates, and visualize detections.

    Parameters
    ----------
    use_gpu : bool, optional
        Whether to use GPU for detection (not used in OpenCV MCC), by default False.
    conf_th : float, optional
        Confidence threshold for detection filtering, by default 0.15.
    """

    def __init__(self, use_gpu: bool = False, conf_th: float = 0.15) -> None:
        """
        Initialize the MCCardDetector.

        Parameters
        ----------
        use_gpu : bool, optional
            Whether to use GPU for detection (not used in OpenCV MCC), by default False.
        conf_th : float, optional
            Confidence threshold for detection filtering, by default 0.15.
        """
        self.use_gpu = use_gpu
        self.conf_th = conf_th

    def detect(self, image: np.ndarray, conf: float | None = None) -> DetectionResult:
        """
        Detect ColorChecker cards and their color patches in the input image.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR) in which to detect ColorChecker cards.
        conf : float, optional
            Confidence threshold for detection filtering, by default 0.15.

        Returns
        -------
        DetectionResult
            A dataclass containing detected segments (card and patch polygons),
            confidence scores, and class IDs. Boxes are not used in this case (None).
        """

        if conf is None:
            conf = self.conf_th

        ls_segments = []
        ls_class_ids = []
        ls_scores = []

        image_copy = image.copy()
        detector = cv2.mcc.CCheckerDetector.create()
        chart_type = cv2.mcc.MCC24
        number_color_checker_card = 1
        success = detector.process(image, chartType=chart_type, nc=number_color_checker_card, useNet=True)

        if not success:
            return DetectionResult(boxes=None, segment=None, scores=[], class_ids=[])

        # Get the detected ColorChecker(s)
        checkers = detector.getListColorChecker()

        # Loop through detected checkers and draw them
        for _, checker in enumerate(checkers):
            # we check the cost of the checker, if that is too high,
            # we skip it because it is not a good detection
            cost_detection = checker.getCost()
            if cost_detection > 1.0 - conf:
                continue
            # get bbox color checker card
            res_box = checker.getBox()
            res_box = [(int(point[0]), int(point[1])) for point in res_box]

            ls_segments.append(res_box)
            ls_class_ids.append(1)  # color checker card label
            ls_scores.append(1.0)  # confidence score

            # get bbox patch
            charts_rgb = checker.getColorCharts()
            ls_chart_coord = self._extract_chart_coordinates(image_copy, charts_rgb)

            for group_patch in ls_chart_coord:
                ls_segments.append(group_patch)
                ls_class_ids.append(0)  # color patch label
                ls_scores.append(1.0)

        detector.clear()

        return DetectionResult(
            boxes=None,
            segment=ls_segments,
            scores=ls_scores,
            class_ids=ls_class_ids,
        )

    def _extract_chart_coordinates(self, image_copy: np.ndarray, charts_rgb: np.ndarray) -> list[list[tuple[int, int]]]:
        """
        Extracts the coordinates of color patches from the detected ColorChecker.

        Parameters
        ----------
        image_copy : np.ndarray
            Copy of the input image for visualization (patch points drawn).
        charts_rgb : np.ndarray
            Array of shape (N, 2) containing the (x, y) coordinates of patch corners.

        Returns
        -------
        list of list of tuple of int
            List of patches, each patch is a list of 4 (x, y) tuples.
        """
        random_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        ls_chart_coord = []
        ls_group_patch = []
        # Input Format:
        # shape of chartsRGB: (96, 2)
        # [
        #   [x1, y1],
        #   [x2, y2],
        #   [x3, y3],
        #   [x4, y4],
        #   ...
        #   [xn, yn]
        # ]
        # Target format:
        # [
        #   [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], # patch 1 (4 points)
        #   ...
        #   [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], # patch n (4 points)
        # ]
        for _, chart in enumerate(charts_rgb):
            ls_group_patch.append((int(chart[0]), int(chart[1])))
            cv2.circle(image_copy, (int(chart[0]), int(chart[1])), 3, random_color, -1)

            if len(ls_group_patch) == 4:
                random_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                # ls_group_patch.append(ls_group_patch[0])  # close the patch with the first point
                ls_chart_coord.append(ls_group_patch)
                ls_group_patch = []
        return ls_chart_coord

    def _draw_box_color_path(self, image_copy: np.ndarray, i: int, group_patch: list[tuple[int, int]]) -> None:
        """
        Draws a polygon and label for a color patch on the image.

        Parameters
        ----------
        image_copy : np.ndarray
            Image on which to draw the patch.
        i : int
            Index or label for the patch.
        group_patch : list of tuple of int
            List of 4 (x, y) tuples representing the patch corners.
        """
        random_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        # draw patch
        cv2.polylines(image_copy, [np.int32(group_patch)], True, random_color, 2)
        # draw text
        cv2.putText(
            image_copy,
            str(i),
            (int(group_patch[0][0]), int(group_patch[0][1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            random_color,
            2,
        )


if __name__ == "__main__":
    image = cv2.imread("./assets/reference-image-2.jpg")
    image = cv2.imread("./assets/cc-1.jpg")
    # image = cv2.imread("./assets/Freeze.jpg")

    detector = MCCardDetector(
        use_gpu=False,
        conf_th=0.15,
    )
    result = detector.detect(image)

    straightened_image = crop_segment_straighten(image, result.segment[0])
    cv2.imwrite("straightened_segment_0.jpg", straightened_image)

    # Straighten and crop the third bounding box
    straightened_image = crop_segment_straighten(image, result.segment[2])
    cv2.imwrite("straightened_segment_2.jpg", straightened_image)

    print("Detection result:", result)
