import cv2
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def crop_region_with_margin(
    image: np.ndarray,
    coordinates: tuple[int, int, int, int],
    margin_ratio: float = 0.2,
) -> np.ndarray:
    """Crop a region from image with additional margin from given coordinates.

    Parameters
    ----------
    image : np.ndarray
        Input image array of shape (H, W, C) or (H, W).
    coordinates : np.ndarray
        Bounding box coordinates [x1, y1, x2, y2].
    margin_ratio : float, optional
        Ratio of margin to add relative to region size, by default 0.2.

    Returns
    -------
    np.ndarray
        Cropped image region with margins.
    """
    y1, y2 = coordinates[1], coordinates[3]
    x1, x2 = coordinates[0], coordinates[2]

    height = y2 - y1
    margin_y = height * margin_ratio
    width = x2 - x1
    margin_x = width * margin_ratio

    crop_y1 = int(y1 + margin_y)
    crop_y2 = int(y2 - margin_y)
    crop_x1 = int(x1 + margin_x)
    crop_x2 = int(x2 - margin_x)

    return image[crop_y1:crop_y2, crop_x1:crop_x2]


def calc_mean_color_patch(img: np.ndarray) -> np.ndarray:
    """Calculate mean RGB/BGR values across spatial dimensions.

    Parameters
    ----------
    img : np.ndarray
        Input image array of shape (H, W, C).

    Returns
    -------
    np.ndarray
        Array of mean RGB values, shape (C,), dtype uint8.
    """
    return np.mean(img, axis=(0, 1)).astype(np.uint8)


def generate_image_patches(
    ls_patches: list[tuple[int, int, int, int]],
    patch_size: tuple[int, int, int] = (50, 50, 1),
) -> np.ndarray:
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


def display_image_grid(
    images: list[tuple[str, np.ndarray | matplotlib.figure.Figure]],
    grid_size: tuple[int, int] = (2, 3),
    figsize: tuple[int, int] = (15, 10),
    save_path: str | None = None,
    dpi: int = 300,
) -> matplotlib.figure.Figure:
    """
    Display images in a grid layout with titles

    Parameters:
    -----------
    images : List[Tuple[str, Union[np.ndarray, matplotlib.figure.Figure]]]
        List of tuples containing (title, image)
    grid_size : Tuple[int, int]
        Grid layout in (rows, columns) format
    figsize : Tuple[int, int]
        Size of the entire figure in inches
    save_path : Optional[str]
        If provided, save the figure to this path
    dpi : int
        DPI for saved figure

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the grid
    """

    rows, cols = grid_size
    fig = plt.figure(figsize=figsize)

    for idx, (title, img) in enumerate(images):
        if idx >= rows * cols:
            print(
                f"Warning: Only showing first {rows * cols} images due to "
                "grid size limitation",
            )
            break

        ax = fig.add_subplot(rows, cols, idx + 1)

        # Handle different image types
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:  # Grayscale
                ax.imshow(img, cmap="gray")
            else:  # RGB/RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
        elif isinstance(img, matplotlib.figure.Figure):
            # Convert matplotlib figure to image array
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            ax.imshow(img_array)

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.close()  # Close the figure to free memory
    return fig


def compare_viz_two_patches(
    ls_mean_ref: list[np.ndarray],
    ls_mean_in: list[np.ndarray],
    patch_size: tuple[int, int, int] = (100, 100, 1),
) -> np.ndarray:
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
