import numpy as np


def preprocessing_compute(input_image: np.ndarray) -> np.ndarray:
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
    if len(original_shape) == 2:
        # to handle grid image patches only
        corrected_image = np.clip(predict_image, 0, 255).astype(np.uint8)
    else:
        h, w, c = original_shape
        corrected_image = (
            np.clip(predict_image, 0, 255).astype(np.uint8).reshape(h, w, c)
        )
    return corrected_image
