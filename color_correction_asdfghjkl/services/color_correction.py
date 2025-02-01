from typing import Literal

import colour as cl
import cv2
import numpy as np
from numpy.typing import NDArray

from color_correction_asdfghjkl.constant.color_checker import reference_color_d50_bgr
from color_correction_asdfghjkl.core.card_detection.det_yv8_onnx import (
    YOLOv8CardDetector,
)
from color_correction_asdfghjkl.core.correction.least_squares import (
    LeastSquaresRegression,
)
from color_correction_asdfghjkl.processor.det_yv8 import DetectionProcessor
from color_correction_asdfghjkl.utils.image_processing import (
    compare_viz_two_patches,
    generate_image_patches,
)

ColorPatchType = NDArray[np.uint8]
ImageType = NDArray[np.uint8]


class ColorCorrection:
    """Color correction handler using color card detection and correction models.

    Parameters
    ----------
    detection_model : {'yolov8'}
        The model to use for color card detection.
    correction_model : {'least_squares'}
        The model to use for color correction.
    reference_color_card : str, optional
        Path to the reference color card image.
    use_gpu : bool, default=True
        Whether to use GPU for card detection.
    """

    def __init__(
        self,
        detection_model: Literal["yolov8"] = "yolov8",
        correction_model: Literal["least_squares"] = "least_squares",
        reference_color_card: str | None = None,
        use_gpu: bool = True,
    ) -> None:
        self.reference_color_card = reference_color_card or reference_color_d50_bgr
        self.correction_model = self._initialize_correction_model(correction_model)
        self.card_detector = self._initialize_detector(detection_model, use_gpu)
        self.correction_weights: np.ndarray | None = None

    def _initialize_correction_model(self, model_name: str) -> LeastSquaresRegression:
        if model_name == "least_squares":
            return LeastSquaresRegression()
        raise ValueError(f"Unsupported correction model: {model_name}")

    def _initialize_detector(
        self,
        model_name: str,
        use_gpu: bool,
    ) -> YOLOv8CardDetector:
        if model_name == "yolov8":
            return YOLOv8CardDetector(use_gpu=use_gpu, conf_th=0.25)
        raise ValueError(f"Unsupported detection model: {model_name}")

    def extract_color_patches(
        self,
        input_image: ImageType,
        debug: bool = False,
    ) -> tuple[list[ColorPatchType], ImageType, ImageType | None]:
        """Extract color patches from input image using card detection.

        Parameters
        ----------
        input_image : NDArray[np.uint8]
            Input image BGR from which to extract color patches.

        Returns
        -------
        Tuple[List[NDArray], NDArray, NDArray]
            - List of BGR mean values for each detected patch.
            Each element is an array of shape (3,) containing [B, G, R] values.
            - Visualization of detected patches.
            - Visualization of preprocessing card detection.
        """
        prediction = self.card_detector.detect(image=input_image)
        input_patches, patch_viz, debug_detection_viz = (
            DetectionProcessor.extract_color_patches(
                input_image=input_image,
                prediction=prediction,
                draw_processed_image=debug,
            )
        )
        return input_patches, patch_viz, debug_detection_viz

    def fit(
        self,
        input_patches: list[ColorPatchType],
        reference_patches: list[ColorPatchType] | None = None,
    ) -> tuple[NDArray, list[ColorPatchType], list[ColorPatchType]]:
        """Fit color correction model using input and reference images.

        Parameters
        ----------
        input_image : NDArray
            Image BGR to be corrected that contains color checker classic 24 patches.
        reference_image : NDArray, optional
            Image BGR to be reference that contains color checker classic 24 patches.

        Returns
        -------
        Tuple[NDArray, List[NDArray], List[NDArray]]
            Correction weights, input patches, and reference patches.
        """
        if reference_patches is None:
            reference_patches = self.reference_color_card

        self.correction_weights = self.correction_model.fit(
            input_patches=input_patches,
            reference_patches=reference_patches,
        )
        return self.correction_weights

    def correct_image(self, input_image: ImageType) -> ImageType:
        """Apply color correction to input image.

        Parameters
        ----------
        input_image : NDArray
            Image to be color corrected.

        Returns
        -------
        NDArray
            Color corrected image.
        """
        if self.correction_weights is None:
            raise RuntimeError("Model must be fitted before correction")

        return self.correction_model.compute_correction(
            input_image=input_image.copy(),
        )

    def calc_color_diff(
        self,
        image1: ImageType,
        image2: ImageType,
    ) -> tuple[float, float, float, float]:
        """Calculate color difference metrics between two images.

        Parameters
        ----------
        image1, image2 : NDArray
            Images to compare in BGR format.

        Returns
        -------
        Tuple[float, float, float, float]
            Minimum, maximum, mean, and standard deviation of delta E values.
        """
        rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        lab1 = cl.XYZ_to_Lab(cl.sRGB_to_XYZ(rgb1 / 255))
        lab2 = cl.XYZ_to_Lab(cl.sRGB_to_XYZ(rgb2 / 255))

        delta_e = cl.difference.delta_E(lab1, lab2, method="CIE 2000")

        return (
            float(np.min(delta_e)),
            float(np.max(delta_e)),
            float(np.mean(delta_e)),
            float(np.std(delta_e)),
        )


if __name__ == "__main__":
    import os

    image_path = "asset/images/cc-1.jpg"
    # image_path = "asset/images/cc-19.png"
    filename = os.path.basename(image_path)
    input_image = cv2.imread(image_path)

    cc = ColorCorrection(detection_model="yolov8", correction_model="least_squares")
    input_patches, input_grid_patches, drawed_debug_preprocess = (
        cc.extract_color_patches(
            input_image=input_image,
            debug=True,
        )
    )
    cc.fit(input_patches=input_patches)
    corrected_image = cc.correct_image(input_image=input_image)
    corrected_patches = cc.correct_image(input_image=input_grid_patches)

    # Display results with 4 grid: 1. Input patches, 2. Reference patches, 3. Corrected patches, 4. Corrected image
    input_vs_output = np.vstack([input_image, corrected_image])
    grid_patches_vs = np.vstack(
        [
            input_grid_patches,
            generate_image_patches(cc.reference_color_card),
            corrected_patches,
        ],
    )
    compare_viz = compare_viz_two_patches(
        ls_mean_in=input_patches,
        ls_mean_ref=cc.reference_color_card,
    )
    print(np.array(input_patches).shape, np.array(input_patches))
    output_patches = cc.correct_image(input_image=np.array(input_patches))

    cor_compare_viz = compare_viz_two_patches(
        ls_mean_in=output_patches,
        ls_mean_ref=cc.reference_color_card,
    )
    print(compare_viz.shape)
    cv2.imwrite("compare_input_vs_output.jpg", compare_viz)
    cv2.imwrite("cor_compare_input_vs_output.jpg", cor_compare_viz)
    cv2.imwrite(f"input_vs_output-{filename}.jpg", input_vs_output)
    cv2.imwrite(f"grid_patches_vs-{filename}.jpg", grid_patches_vs)
    cv2.imwrite(f"drawed_debug_preprocess-{filename}.jpg", drawed_debug_preprocess)
