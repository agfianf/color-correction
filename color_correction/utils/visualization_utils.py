import cv2
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def create_image_grid_visualization(
    images: list[tuple[str, np.ndarray | matplotlib.figure.Figure | None]],
    grid_size: tuple[int, int] = (2, 3),
    figsize: tuple[int, int] = (15, 10),
    save_path: str | None = None,
    dpi: int = 300,
) -> matplotlib.figure.Figure:
    """
    Create a grid visualization of images with optional saving.

    Parameters
    ----------
    images : list of tuple
        List where each tuple contains (title, image) and image can be a numpy array,
        a matplotlib Figure, or None.
    grid_size : tuple[int, int], optional
        Tuple of (rows, columns) defining the grid layout; default is (2, 3).
    figsize : tuple[int, int], optional
        Size of the matplotlib figure in inches; default is (15, 10).
    save_path : str | None, optional
        File path to save the figure; if None, the figure is not saved.
    dpi : int, optional
        Dots per inch for the saved image; default is 300.

    Returns
    -------
    matplotlib.figure.Figure
        The created matplotlib figure containing the image grid.
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
