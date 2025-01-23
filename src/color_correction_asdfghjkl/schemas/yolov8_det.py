import numpy as np
from pydantic import BaseModel

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
