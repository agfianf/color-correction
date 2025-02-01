import numpy as np
import shapely

box_tuple = tuple[int, int, int, int]


def get_max_iou_shapely(
    ref_box: shapely.geometry.box,
    target_boxes: list[shapely.geometry.box],
) -> tuple[float, int, shapely.geometry.box]:
    max_iou = 0
    max_idx = -1

    # Compare with each target box
    for idx, target_box in enumerate(target_boxes):
        # Calculate intersection and union
        intersection_area = ref_box.intersection(target_box).area
        union_area = ref_box.union(target_box).area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0

        # Update maximum IoU if current is larger
        if iou > max_iou:
            max_iou = iou
            max_idx = idx

    return max_iou, max_idx, target_boxes[max_idx]


def box_to_xyxy(box: shapely.geometry.box) -> tuple[int, int, int, int]:
    """Convert shapely box to xyxy format"""
    minx, miny, maxx, maxy = box.bounds
    return int(minx), int(miny), int(maxx), int(maxy)


def box_centroid_xy(box: shapely.geometry.box) -> tuple[int, int]:
    return int(box.centroid.x), int(box.centroid.y)


def generate_expected_patches(card_box: box_tuple) -> list[box_tuple]:
    card_x1, card_y1, card_x2, card_y2 = card_box
    card_width = card_x2 - card_x1
    card_height = card_y2 - card_y1

    # get expected grid of cards
    patch_width = card_width / 6
    patch_height = card_height / 4

    expected_patches = []
    for row in range(4):
        for col in range(6):
            x1 = int(card_x1 + col * patch_width)
            y1 = int(card_y1 + row * patch_height)
            x2 = int(x1 + patch_width)
            y2 = int(y1 + patch_height)
            expected_patches.append((x1, y1, x2, y2))

    return expected_patches


def extract_intersecting_patches(
    ls_patches: list[box_tuple],
    ls_grid_card: list[box_tuple],
) -> list[tuple[box_tuple, tuple[int, int]]]:
    ls_ordered_patch = []
    for _, grid_card in enumerate(ls_grid_card, start=1):
        # get intesect patch
        gx1, gy1, gx2, gy2 = grid_card
        grid_box = shapely.box(*grid_card)
        ls_intersect = [
            shapely.box(*xyxy)
            for xyxy in ls_patches
            if grid_box.intersects(shapely.box(*xyxy))
        ]
        len_intersect = len(ls_intersect)
        if len_intersect > 0:
            max_iou, max_id, intersect_box = get_max_iou_shapely(
                ref_box=grid_box,
                target_boxes=ls_intersect,
            )
            # intersect_box = ls_intersect[max_id]
            val = box_to_xyxy(intersect_box)
            xy = box_centroid_xy(intersect_box)
            ls_ordered_patch.append((val, xy))
        else:
            ls_ordered_patch.append(None)
    return ls_ordered_patch


def calculate_patch_statistics(ls_ordered_patch: list[box_tuple]) -> tuple:
    ls_dx = []
    ls_dy = []
    ls_w_grid = []
    ls_h_grid = []
    for idx, patch in enumerate(ls_ordered_patch):
        if patch is None:
            continue

        ls_w_grid.append(patch[2] - patch[0])
        ls_h_grid.append(patch[3] - patch[1])

        if idx not in [5, 11, 17, 23] or idx == 0:
            x1 = patch[0]
            next_x1 = ls_ordered_patch[idx + 1]
            if next_x1 is not None:
                dx = next_x1[0] - x1
                ls_dx.append(dx)

        syarat = idx + 6
        if syarat < len(ls_ordered_patch):
            y1 = patch[1]
            next_y1 = ls_ordered_patch[idx + 6]
            if next_y1 is not None:
                dy = next_y1[1] - y1
                ls_dy.append(dy)

    mean_dx = np.mean(ls_dx)
    mean_dy = np.mean(ls_dy)
    mean_w = np.mean(ls_w_grid)
    mean_h = np.mean(ls_h_grid)

    print(ls_dx, mean_dx)
    return mean_dx, mean_dy, mean_w, mean_h


def suggest_missing_patch_coordinates(  # noqa: C901
    ls_ordered_patch: list[box_tuple],
) -> dict[int, box_tuple]:
    d_suggest = {}

    print("ls_ordered_patch", ls_ordered_patch)
    mean_dx, mean_dy, mean_w, mean_h = calculate_patch_statistics(
        ls_ordered_patch=ls_ordered_patch,
    )

    for idx, patch in enumerate(ls_ordered_patch):
        if patch is not None:
            continue

        # looking for nearest neghbor
        neigh_right = None
        neigh_left = None
        neigh_top = None
        neigh_bottom = None

        id_neigh_right = idx + 1
        id_neigh_left = idx - 1
        id_neigh_top = idx - 6
        id_neigh_bottom = idx + 6

        if id_neigh_right not in [0, 6, 12, 18] and id_neigh_right <= 23:
            neigh_right = ls_ordered_patch[id_neigh_right]

        if id_neigh_left not in [5, 11, 17, 23] and id_neigh_left >= 0:
            neigh_left = ls_ordered_patch[id_neigh_left]

        if id_neigh_top >= 0:
            neigh_top = ls_ordered_patch[id_neigh_top]

        if id_neigh_bottom <= 23:
            neigh_bottom = ls_ordered_patch[id_neigh_bottom]

        suggested_patch = None

        # print(f"neigh_right: {neigh_right}")
        # print(f"neigh_left: {neigh_left}")
        # print(f"neigh_top: {neigh_top}")
        # print(f"neigh_bottom: {neigh_bottom}")

        if neigh_right is not None:
            # Dari kanan, geser ke kiri dengan mean_dx
            x1 = neigh_right[0] - mean_dx
            y1 = neigh_right[1]
            suggested_patch = (
                int(x1),
                int(y1),
                int(x1 + mean_w),
                int(y1 + mean_h),
            )

        elif neigh_left is not None:
            # Dari kiri, geser ke kanan dengan mean_dx
            x1 = neigh_left[0] + int(mean_dx)
            y1 = neigh_left[1]
            suggested_patch = (
                int(x1),
                int(y1),
                int(x1 + mean_w),
                int(y1 + mean_h),
            )

        elif neigh_top is not None:
            # Dari atas, geser ke bawah dengan mean_dy
            x1 = neigh_top[0]
            y1 = neigh_top[1] + mean_dy
            suggested_patch = (
                int(x1),
                int(y1),
                int(x1 + mean_w),
                int(y1 + mean_h),
            )

        elif neigh_bottom is not None:
            # Dari bawah, geser ke atas dengan mean_dy
            x1 = neigh_bottom[0]
            y1 = neigh_bottom[1] - mean_dy
            suggested_patch = (
                int(x1),
                int(y1),
                int(x1 + mean_w),
                int(y1 + mean_h),
            )

        d_suggest[idx] = suggested_patch

    return d_suggest
