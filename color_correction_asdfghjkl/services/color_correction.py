from typing import Literal

import colour as cl
import cv2
import numpy as np
from numpy.typing import NDArray

from color_correction_asdfghjkl.constant.color_checker import reference_color_d50_bgr
from color_correction_asdfghjkl.core.card_detection.det_yv8_onnx import (
    YOLOv8CardDetector,
)
from color_correction_asdfghjkl.core.correction import (
    AffineReg,
    LeastSquaresRegression,
    LinearReg,
    Polynomial,
)
from color_correction_asdfghjkl.processor.det_yv8 import DetectionProcessor
from color_correction_asdfghjkl.utils.image_processing import (
    compare_viz_two_patches,
    display_image_grid,
    generate_image_patches,
)

ColorPatchType = NDArray[np.uint8]
ImageType = NDArray[np.uint8]
LiteralModelCorrection = Literal[
    "least_squares",
    "polynomial",
    "linear_reg",
    "affine_reg",
]


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
        correction_model: LiteralModelCorrection = "least_squares",
        reference_color_card: str | None = None,
        use_gpu: bool = True,
        **kwargs: dict,
    ) -> None:
        self.reference_color_card = reference_color_card or reference_color_d50_bgr
        self.correction_model = self._initialize_correction_model(
            correction_model,
            **kwargs,
        )
        self.card_detector = self._initialize_detector(detection_model, use_gpu)
        self.trained_model = None

    @property
    def model_name(self) -> str:
        return self.correction_model.__class__.__name__

    @property
    def img_grid_patches_ref(self) -> np.ndarray:
        return generate_image_patches(self.reference_color_card)

    def _initialize_correction_model(
        self,
        model_name: str,
        **kwargs: dict,
    ) -> LeastSquaresRegression:
        d_model_selection = {
            "least_squares": LeastSquaresRegression(),
            "polynomial": Polynomial(**kwargs),
            "linear_reg": LinearReg(),
            "affine_reg": AffineReg(),
        }
        selected_model = d_model_selection.get(model_name)
        if selected_model:
            return selected_model
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

        self.trained_model = self.correction_model.fit(
            x_patches=input_patches,
            y_patches=reference_patches,
        )
        return self.trained_model

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
        if self.trained_model is None:
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

    # image_path = "asset/images/cc-1.jpg"
    image_path = "asset/images/cc-19.png"
    filename = os.path.basename(image_path)
    input_image = cv2.imread(image_path)

    cc = ColorCorrection(
        detection_model="yolov8",
        correction_model="polynomial",
        degree=4,
        # correction_model="least_squares",
    )
    input_patches, input_grid_patches_img, drawed_debug_preprocess = (
        cc.extract_color_patches(
            input_image=input_image,
            debug=True,
        )
    )
    cc.fit(input_patches=input_patches)
    corrected_image = cc.correct_image(input_image=input_image)
    corrected_patches = cc.correct_image(input_image=input_grid_patches_img)

    reff_grid_patches_img = cc.img_grid_patches_ref
    print(
        reff_grid_patches_img.shape,
        input_grid_patches_img.shape,
        corrected_patches.shape,
    )
    input_vs_output = np.vstack([input_image, corrected_image])
    grid_patches_vs = np.vstack(
        [
            input_grid_patches_img,
            reff_grid_patches_img,
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

    os.makedirs("zzz", exist_ok=True)
    ls_dir = os.listdir("zzz")
    run_rank = len(ls_dir) + 2
    folder = f"zzz/{run_rank}-{cc.model_name}"
    os.makedirs(folder, exist_ok=True)

    images_coll = [
        ("input_image", input_image),
        ("corrected_image", corrected_image),
        ("drawed_debug_preprocess", drawed_debug_preprocess),
        ("Reff (inside: input)", compare_viz),
        ("Reff (inside: corrected)", cor_compare_viz),
        ("None", None),
        ("patch_input", input_grid_patches_img),
        ("patch_corrected", corrected_patches),
        ("patch_reff", reff_grid_patches_img),
    ]

    display_image_grid(
        images=images_coll,
        grid_size=((len(images_coll) // 3) + 1, 3),
        figsize=(15, ((len(images_coll) // 3) + 1) * 4),
        save_path=f"{folder}/00-summary.jpg",
    )
