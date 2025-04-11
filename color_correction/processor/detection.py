import cv2
import numpy as np

from color_correction.schemas.custom_types import BoundingBox, ColorPatchType, ImageBGR
from color_correction.schemas.det_yv8 import DetectionResult
from color_correction.utils.geometry_processing import (
    extract_intersecting_patches,
    generate_expected_patches,
    suggest_missing_patch_coordinates,
)
from color_correction.utils.image_processing import (
    calc_mean_color_patch,
    crop_region_with_margin,
)


class DetectionProcessor:
    """
    A class for processing detection results and transforming them using
    available utility methods. This class takes the output from an object
    detection model (e.g., YOLOv8) and provides functionality to separate,
    process, and visualize detected color checker cards and their patches.
    """

    @staticmethod
    def get_each_class_box(
        prediction: DetectionResult,
    ) -> tuple[list[BoundingBox], list[BoundingBox]]:
        """
        Separates detection boxes into card boxes and patch boxes.

        Parameters
        ----------
        prediction : DetectionResult
            The detection output that includes bounding boxes and class IDs.

        Returns
        -------
        tuple[list[BoundingBox], list[BoundingBox]]
            A tuple containing:

            - a list of card boxes (class id 1)
            - a list of patch boxes (class id 0).
        """
        ls_cards = [
            box
            for box, class_id in zip(
                prediction.boxes,
                prediction.class_ids,
                strict=False,
            )
            if class_id == 1
        ]
        ls_patches = [
            box
            for box, class_id in zip(
                prediction.boxes,
                prediction.class_ids,
                strict=False,
            )
            if class_id == 0
        ]
        return ls_cards, ls_patches

    @staticmethod
    def print_summary(prediction: DetectionResult) -> None:
        """
        Prints a summary of the detected cards and patches.

        Parameters
        ----------
        prediction : DetectionResult
            The detection result to summarize.
        """
        ls_cards, ls_patches = DetectionProcessor.get_each_class_box(prediction)
        print(f"Number of cards detected: {len(ls_cards)}")
        print(f"Number of patches detected: {len(ls_patches)}")

    @staticmethod
    def process_patches(
        input_image: ImageBGR,
        ordered_patches: list[tuple[BoundingBox, tuple[int, int]] | None],
    ) -> tuple[list[ColorPatchType], ImageBGR]:
        """
        Processes each detected patch by extracting its region from the image,
        computing its mean BGR color, and building a visualization grid.

        Parameters
        ----------
        input_image : ImageBGR
            The original image containing detected patches.
        ordered_patches : list[tuple[BoundingBox, tuple[int, int]] | None]
            The list of ordered patch coordinates paired with their center
            points, possibly with missing entries as None.

        Returns
        -------
        tuple[list[ColorPatchType], ImageBGR]
            a tuple containing:

            - list of mean BGR color values for each patch.
            - an image visualizing these patches in a grid layout.
        """
        patch_size = (50, 50, 1)
        ls_bgr_mean_patch = []
        ls_horizontal_patch = []
        ls_vertical_patch = []

        for idx, coord_patch in enumerate(ordered_patches, start=1):
            if coord_patch is None:
                continue

            bbox_patch, _ = coord_patch

            # Extract and process each patch
            cropped_patch = crop_region_with_margin(
                image=input_image,
                coordinates=bbox_patch,
                margin_ratio=0.2,
            )
            bgr_mean_patch = calc_mean_color_patch(cropped_patch)
            ls_bgr_mean_patch.append(bgr_mean_patch)

            # Build visualization
            patch_viz = np.tile(bgr_mean_patch, patch_size)
            ls_horizontal_patch.append(patch_viz)
            if idx % 6 == 0:
                ls_vertical_patch.append(np.hstack(ls_horizontal_patch))
                ls_horizontal_patch = []

        patches_image = np.vstack(ls_vertical_patch)
        return ls_bgr_mean_patch, patches_image

    @staticmethod
    def extract_color_patches(
        input_image: ImageBGR,
        prediction: DetectionResult,
        draw_processed_image: bool = False,
    ) -> tuple[list[ColorPatchType], ImageBGR, ImageBGR | None]:
        """
        Extracts and processes color patches from detected color checker cards,
        transforming the detection results with available methods.

        The method first separates detected cards and patches, generates an expected
        patch grid, and then matches the detected patches with this grid. If patches
        are missing, it attempts to auto-fill them with suggested coordinates.
        Finally, it computes the mean color for each patch and builds a visualization.

        Parameters
        ----------
        input_image : ImageBGR
            The original image containing the color checker card.
        prediction : DetectionResult
            The detection result output from the model.
        draw_processed_image : bool, optional
            If True, returns an additional image with visualized detections.
            Otherwise, only patch processing is performed. Defaults to False.

        Returns
        -------
        tuple[list[ColorPatchType], ImageBGR, ImageBGR | None]
            a tuple containing:

            - a list of mean BGR color values for each patch.
            - an image visualizing these patches in a grid layout.
            - an optional image with visualized detection results.

        Raises
        ------
        ValueError
            If no cards or patches are detected.
        """
        ls_cards, ls_patches = DetectionProcessor.get_each_class_box(prediction)

        if not ls_cards:
            raise ValueError("No cards detected")
        if not ls_patches:
            raise ValueError("No patches detected")

        # Generate expected patch grid
        card_box = ls_cards[0]
        ls_grid_card = generate_expected_patches(card_box)

        # Match detected patches with grid
        ls_ordered_patch_bbox = extract_intersecting_patches(
            ls_patches=ls_patches,
            ls_grid_card=ls_grid_card,
        )

        # Handle missing patches
        d_suggest = None
        if None in ls_ordered_patch_bbox:
            print("Auto filling missing patches...")
            ls_ordered_bbox_only = [patch[0] if patch is not None else None for patch in ls_ordered_patch_bbox]
            d_suggest = suggest_missing_patch_coordinates(ls_ordered_bbox_only)
            for idx, patch in d_suggest.items():
                cxpatch = (patch[0] + patch[2]) // 2
                cypatch = (patch[1] + patch[3]) // 2
                ls_ordered_patch_bbox[idx] = (patch, (cxpatch, cypatch))

        # Process patches and create visualizations
        ls_bgr_mean_patch, grid_patch_img = DetectionProcessor.process_patches(
            input_image=input_image,
            ordered_patches=ls_ordered_patch_bbox,
        )

        detection_viz = None
        if draw_processed_image:
            detection_viz = DetectionProcessor.draw_preprocess(
                image=input_image,
                expected_boxes=ls_grid_card,
                prediction=prediction,
                ls_ordered_patch_bbox=ls_ordered_patch_bbox,
                suggested_patches=d_suggest,
            )

        return ls_bgr_mean_patch, grid_patch_img, detection_viz

    @staticmethod
    def draw_preprocess(
        image: ImageBGR,
        expected_boxes: list[BoundingBox],
        prediction: DetectionResult,
        ls_ordered_patch_bbox: list[BoundingBox | None],
        suggested_patches: dict[int, BoundingBox] | None = None,
    ) -> ImageBGR:
        """
        Draws visualizations on the input image to compare the detected patches with
        the expected positions. It overlays expected boxes, connects detected patches to
        their corresponding expected boxes, and highlights individual patch detections.

        Parameters
        ----------
        image : ImageBGR
            The original image to draw on.
        expected_boxes : list[BoundingBox]
            List of expected bounding boxes for patches.
        prediction : DetectionResult
            The detection results containing predicted bounding boxes.
        ls_ordered_patch_bbox : list[BoundingBox | None]
            The list of ordered patch bounding boxes.
        suggested_patches : dict[int, BoundingBox], optional
            Dictionary of suggested patch coordinates for missing patches.

        Returns
        -------
        ImageBGR
            The image with drawn detection and patch visualizations.
        """
        color_green = (0, 255, 0)
        color_cyan = (255, 255, 10)
        color_violet = (255, 0, 255)
        color_red = (0, 0, 255)
        color_blue = (255, 0, 0)

        result_image = image.copy()

        # Draw all expected boxes
        for idx_b, box in enumerate(expected_boxes):
            cv2.rectangle(
                img=result_image,
                pt1=(box[0], box[1]),
                pt2=(box[2], box[3]),
                color=color_green,
                thickness=2,
            )

            # Draw connection lines between expected and detected patch boxes
            patch = ls_ordered_patch_bbox[idx_b]
            if patch is None:
                continue
            cx, cy = patch[1]
            crefx, crefy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            cv2.line(
                img=result_image,
                pt1=(cx, cy),
                pt2=(crefx, crefy),
                color=color_blue,
                thickness=1,
            )

        # Draw all predicted boxes
        for pbox, pids, pscore in zip(
            prediction.boxes,
            prediction.class_ids,
            prediction.scores,
            strict=False,
        ):
            if pids == 1:
                continue
            cv2.rectangle(
                img=result_image,
                pt1=(pbox[0], pbox[1]),
                pt2=(pbox[2], pbox[3]),
                color=color_cyan,
                thickness=2,
            )
            cv2.putText(
                img=result_image,
                text=f"{pids} {pscore:.2f}",
                org=(pbox[0] + 3, pbox[1] + 12),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=color_red,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        # Draw suggested patches if provided
        if suggested_patches:
            for box in suggested_patches.values():
                cv2.rectangle(
                    img=result_image,
                    pt1=(box[0], box[1]),
                    pt2=(box[2], box[3]),
                    color=color_violet,
                    thickness=2,
                )

        return result_image
