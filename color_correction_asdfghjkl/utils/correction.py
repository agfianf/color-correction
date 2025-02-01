import numpy as np


def preprocessing_compute(input_image: np.ndarray) -> np.ndarray:
    # Input adalah array (N,3) dari nilai warna patches

    print("pre-processing", input_image.shape)
    if input_image.shape == (24, 3):  # Khusus untuk ColorChecker 24 patches
        image = input_image.astype(np.float32)
    else:
        image = input_image.reshape(-1, 3).astype(np.float32)
    return image


def postprocessing_compute(
    original_shape: tuple,
    predict_image: np.ndarray,
) -> np.ndarray:
    if len(original_shape) == 2:
        corrected_image = np.clip(predict_image, 0, 255).astype(np.uint8)
        print("post-processing", corrected_image.shape, original_shape)
    else:
        h, w, c = original_shape
        # Clip dan convert kembali ke uint8
        corrected_image = (
            np.clip(predict_image, 0, 255).astype(np.uint8).reshape(h, w, c)
        )
    return corrected_image
