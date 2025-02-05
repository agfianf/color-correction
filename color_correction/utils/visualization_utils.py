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
    Display images in a grid layout with titles

    Parameters:
    -----------
    images : List[Tuple[str, Union[np.ndarray, matplotlib.figure.Figure, None]]]
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
