import cv2
import numpy as np


def create_patch_tiled_image(
    ls_patches: list[tuple[int, int, int, int]],
    patch_size: tuple[int, int, int] = (50, 50, 1),
) -> np.ndarray:
    """Generate a color patch image from a list of RGB values.

    This function creates a color patch image by tiling RGB values into patches and
    arranging them in a 4x6 grid pattern. Each patch is repeated according to the
    specified patch size.

    Parameters
    ----------
    ls_patches : list of tuple
        List containing 24 RGB color tuples, where each tuple has three integers
        representing (R, G, B) values.
    patch_size : tuple of int, optional
        Size of each individual patch in pixels, by default (50, 50, 1).
        Format is (height, width, channels).

    Returns
    -------
    numpy.ndarray
        Generated image as a numpy array with shape determined by patch_size and
        arrangement (4 rows x 6 columns). Array type is uint8.

    Notes
    -----
    This function is specifically designed to work with 24 color patches arranged
    in a 4x6 grid pattern.

    Examples
    --------
    >>> patches = [(255, 0, 0), (0, 255, 0), ...] # 24 RGB tuples
    >>> patch_size = (50, 50, 1)
    >>> image = generate_image_patches(patches, patch_size)
    """

    ls_stack_h = []
    ls_stack_v = []

    for _idx, patch in enumerate(ls_patches, start=1):
        patch_img = np.tile(patch, patch_size)
        ls_stack_h.append(patch_img)
        if _idx % 6 == 0:
            row = np.hstack(ls_stack_h)
            ls_stack_v.append(row)
            ls_stack_h = []
    image = np.vstack(ls_stack_v).astype(np.uint8)
    return image


def visualize_patch_comparison(
    ls_mean_ref: list[np.ndarray],
    ls_mean_in: list[np.ndarray],
    patch_size: tuple[int, int, int] = (100, 100, 1),
) -> np.ndarray:
    """
    Compare two sets of image patches by inserting a resized inner patch into
    the center of an outer patch. This visualization grid helps in comparing
    the reference and input images in a structured manner.

    Parameters
    ----------
    ls_mean_ref : list of np.ndarray
        List of outer image patches. Each patch is repeated to form the full
        grid background.
    ls_mean_in : list of np.ndarray
        List of inner image patches meant to be resized and placed into the
        center of the outer patches.
    patch_size : tuple of int, optional
        A tuple specifying the size of the patch in the format (height, width, channels)
        by default (100, 100, 1).

    Returns
    -------
    np.ndarray
        The final composited image with each outer patch modified with the
        corresponding resized inner patch, arranged in a grid format.
    """
    ls_stack_h = []
    ls_stack_v = []

    h = patch_size[0]
    w = patch_size[1]
    h_2 = h // 2
    w_2 = w // 2
    y1 = h_2 - (h // 4) - 1
    y2 = h_2 + (h // 4)
    x1 = w_2 - (w // 4) - 1
    x2 = w_2 + (w // 4)

    for _idx, (patch_ref, patch_in) in enumerate(
        zip(ls_mean_ref, ls_mean_in, strict=False),
        start=1,
    ):
        img_patch_ref = np.tile(patch_ref, patch_size)
        img_patch_in = np.tile(patch_in, (h_2, w_2, 1))
        img_patch_in = cv2.resize(img_patch_in, (y2 - y1, x2 - x1))
        img_patch_ref[y1:y2, x1:x2] = img_patch_in
        ls_stack_h.append(img_patch_ref)

        if _idx % 6 == 0:
            row = np.hstack(ls_stack_h)
            ls_stack_v.append(row)
            ls_stack_h = []
    image = np.vstack(ls_stack_v).astype(np.uint8)
    return image
