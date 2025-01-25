import numpy as np
from pydantic import BaseModel

from color_correction_asdfghjkl.utils.geometry_processing import (
    extract_intersecting_patches,
    generate_expected_patches,
    suggest_missing_patch_coordinates,
)
from color_correction_asdfghjkl.utils.image_processing import (
    calculate_mean_rgb,
    crop_region_with_margin,
)
from color_correction_asdfghjkl.utils.yolo_utils import draw_detections

box_tuple = tuple[int, int, int, int]


class DetectionResult(BaseModel):
    boxes: list[box_tuple]
    scores: list[float]
    class_ids: list[int]

    def get_each_class_box(self) -> tuple[list[box_tuple], list[box_tuple]]:
        """
        Return
        ------
        Tuple[list[box_tuple], list[box_tuple]]
            A tuple of two lists, where the first list contains the bounding boxes
            of the cards and the second list contains the bounding boxes of the patches.
        """
        ls_cards = []
        ls_patches = []
        for box, class_id in zip(self.boxes, self.class_ids, strict=False):
            if class_id == 0:
                ls_patches.append(box)
            if class_id == 1:
                ls_cards.append(box)
        return ls_cards, ls_patches

    def print_summary(self) -> None:
        ls_cards, ls_patches = self.get_each_class_box()
        print(f"Number of cards detected: {len(ls_cards)}")
        print(f"Number of patches detected: {len(ls_patches)}")

    def draw_detections(self, image: np.ndarray, mask_alpha: float = 0.2) -> np.ndarray:
        return draw_detections(
            image=image,
            boxes=self.boxes,
            scores=self.scores,
            class_ids=self.class_ids,
            mask_alpha=mask_alpha,
        )

    def get_list_patches(self, input_image: np.ndarray) -> list[np.ndarray]:
        ls_cards, ls_patches = self.get_each_class_box()

        if len(ls_cards) == 0:
            raise ValueError("No cards detected")

        if len(ls_patches) == 0:
            raise ValueError("No patches detected")

        # Extract card coordinates
        card_box = ls_cards[0]
        ls_grid_card = generate_expected_patches(card_box)

        # get ls grid card
        ls_ordered_patch_bbox = extract_intersecting_patches(
            ls_patches=ls_patches,
            ls_grid_card=ls_grid_card,
        )

        if None in ls_ordered_patch_bbox:
            # Auto fill missing patches ----------------
            print("Auto fill missing patch...", ls_ordered_patch_bbox)
            d_suggest = suggest_missing_patch_coordinates(ls_ordered_patch_bbox)
            for idx, patch in d_suggest.items():
                ls_ordered_patch_bbox[idx] = patch
            print(f"result len = {len(ls_ordered_patch_bbox)}")

        ls_rgb_mean_patch = []
        for coord_patch in ls_ordered_patch_bbox:
            cropped_patch = crop_region_with_margin(
                image=input_image,
                coordinates=coord_patch,
                margin_ratio=0.2,
            )
            rgb_mean_patch = calculate_mean_rgb(cropped_patch)
            ls_rgb_mean_patch.append(rgb_mean_patch)

        return ls_rgb_mean_patch
