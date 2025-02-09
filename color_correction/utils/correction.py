import numpy as np


def preprocessing_compute(input_image: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image for computation by reshaping and converting datatype.

    Parameters
    ----------
    input_image : np.ndarray
        Input image array that can be either a grid (24, 3) or a general image.

    Returns
    -------
    np.ndarray
        Processed image data as a float32 numpy array.
    """
    if input_image.shape == (24, 3):
        # to handle grid image patches only
        image = input_image.astype(np.float32)
    else:
        image = input_image.reshape(-1, 3).astype(np.float32)
    return image


def postprocessing_compute(
    original_shape: tuple,
    predict_image: np.ndarray,
) -> np.ndarray:
    """
    Convert predicted image data back into its original shape and type.

    Parameters
    ----------
    original_shape : tuple
        The original dimensions of the image. Should be 2 or (H, W, C).
    predict_image : np.ndarray
        The processed image data to be reshaped and clipped.

    Returns
    -------
    np.ndarray
        The final corrected image, reshaped to original dimensions and in uint8 format.
    """
    if len(original_shape) == 2:
        # to handle grid image patches only
        corrected_image = np.clip(predict_image, 0, 255).astype(np.uint8)
    else:
        h, w, c = original_shape
        corrected_image = (
            np.clip(predict_image, 0, 255).astype(np.uint8).reshape(h, w, c)
        )
    return corrected_image
