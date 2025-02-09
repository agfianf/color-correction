import os

import cv2
import numpy as np

from color_correction.constant.color_checker import reference_color_d50_bgr
from color_correction.core.card_detection.det_yv8_onnx import (
    YOLOv8CardDetector,
)
from color_correction.core.correction import CorrectionModelFactory
from color_correction.processor.det_yv8 import DetectionProcessor
from color_correction.schemas.custom_types import (
    ColorPatchType,
    ImageBGR,
    LiteralModelCorrection,
    LiteralModelDetection,
)
from color_correction.utils.image_patch import (
    create_patch_tiled_image,
    visualize_patch_comparison,
)
from color_correction.utils.image_processing import calc_color_diff
from color_correction.utils.visualization_utils import (
    create_image_grid_visualization,
)


class ColorCorrection:
    """Color correction handler using color `card_detection` and `correction_models`.
    This class handles the complete workflow of color correction, including:

    - Color card detection in images
    - Color patch extraction
    - Color correction model training
    - Image correction application
    - Evaluation of color correction patches

    Parameters
    ----------
    detection_model : LiteralModelDetection, optional
        The model to use for color card detection.
    detection_conf_th : float, optional
        Confidence threshold for card detection.
    correction_model : {'least_squares', 'polynomial', 'linear_reg', 'affine_reg'}
        The model to use for color correction.
    reference_image : NDArray[np.uint8] | None, optional
        Reference image containing color checker card.
        If None, uses standard D50 values.
    use_gpu : bool, default=True
        Whether to use GPU for card detection. False will use CPU.
    **kwargs : dict
        Additional parameters for the correction model.

    Other parameters
    ----------------
    degree : int, optional
        Degree of `polynomial` correction model. Default is 2.
        the more degree, the more complex the model. (e.g. 2, 3, 4, ...)

    Attributes
    ----------
    reference_patches : list[ColorPatchType] | None
        Extracted color patches from reference image.
    reference_grid_image : ImageBGR | None
        Visualization of reference color patches in grid format.
    reference_debug_image : ImageBGR | None
        Debug visualization of reference image preprocessing.
    """

    def __init__(
        self,
        detection_model: LiteralModelDetection = "yolov8",
        detection_conf_th: float = 0.25,
        correction_model: LiteralModelCorrection = "least_squares",
        reference_image: ImageBGR | None = None,
        use_gpu: bool = True,
        **kwargs: dict,
    ) -> None:
        # Initialize reference image attributes
        self.reference_patches = None
        self.reference_grid_image = None
        self.reference_debug_image = None

        # Initialize input image attributes
        self.input_patches = None
        self.input_grid_image = None
        self.input_debug_image = None

        # Initialize correction output attributes
        self.corrected_patches = None
        self.corrected_grid_image = None

        # Initialize model attributes
        self.trained_model = None
        self.correction_model = CorrectionModelFactory.create(
            model_name=correction_model,
            **kwargs,
        )
        self.card_detector = self._create_detector(
            model_name=detection_model,
            conf_th=detection_conf_th,
            use_gpu=use_gpu,
        )

        # Set reference patches
        self.set_reference_patches(image=reference_image)

    def _create_detector(
        self,
        model_name: str,
        conf_th: float = 0.25,
        use_gpu: bool = False,
    ) -> YOLOv8CardDetector:
        """Create a card detector instance.

        Parameters
        ----------
        model_name : str
            Name of the detector model to create.
        conf_th : float, optional
            Confidence threshold for card detection. Default is 0.25.
        use_gpu : bool, optional
            Whether to use GPU for detection. Default is False.

        Returns
        -------
        YOLOv8CardDetector
            Initialized detector instance.

        Raises
        ------
        ValueError
            If the model name is not supported.
        """
        if model_name != "yolov8":
            raise ValueError(f"Unsupported detection model: {model_name}")
        return YOLOv8CardDetector(use_gpu=use_gpu, conf_th=conf_th)

    def _extract_color_patches(
        self,
        image: ImageBGR,
        debug: bool = False,
    ) -> tuple[list[ColorPatchType], ImageBGR, ImageBGR | None]:
        """Extract color patches from an image using card detection.

        Parameters
        ----------
        image : ImageBGR
            Input image in BGR format.
        debug : bool, optional
            Whether to generate debug visualizations.

        Returns
        -------
        tuple[list[ColorPatchType], ImageType, ImageType | None]
            - List of BGR mean values for each detected patch
            - Grid visualization of detected patches
            - Debug visualization (if debug=True)
        """
        prediction = self.card_detector.detect(image=image)
        ls_bgr_mean_patch, grid_patch_img, debug_detection_viz = (
            DetectionProcessor.extract_color_patches(
                input_image=image,
                prediction=prediction,
                draw_processed_image=debug,
            )
        )
        return ls_bgr_mean_patch, grid_patch_img, debug_detection_viz

    def _save_debug_output(
        self,
        input_image: ImageBGR,
        corrected_image: ImageBGR,
        output_directory: str,
    ) -> None:
        """Save debug visualizations to disk.

        Parameters
        ----------
        input_image : ImageBGR
            The input image.
        corrected_image : ImageType
            The color-corrected image.
        output_directory : str
            Directory to save debug outputs.
        """
        before_comparison = visualize_patch_comparison(
            ls_mean_in=self.input_patches,
            ls_mean_ref=self.reference_patches,
        )
        after_comparison = visualize_patch_comparison(
            ls_mean_in=self.corrected_patches,
            ls_mean_ref=self.reference_patches,
        )

        # Create output directories
        run_dir = self._create_debug_directory(output_directory)

        # Prepare debug image grid
        image_collection = [
            ("Input Image", input_image),
            ("Corrected Image", corrected_image),
            ("Debug Preprocess", self.input_debug_image),
            ("Reference vs Input", before_comparison),
            ("Reference vs Corrected", after_comparison),
            ("[Free Space]", None),
            ("Patch Input", self.input_grid_image),
            ("Patch Corrected", self.corrected_grid_image),
            ("Patch Reference", self.reference_grid_image),
        ]

        # Save debug grid
        save_path = os.path.join(run_dir, "debug.jpg")
        create_image_grid_visualization(
            images=image_collection,
            grid_size=((len(image_collection) // 3) + 1, 3),
            figsize=(15, ((len(image_collection) // 3) + 1) * 4),
            save_path=save_path,
        )
        print(f"Debug output saved to: {save_path}")

    def _create_debug_directory(self, base_dir: str) -> str:
        """Create and return a unique debug output directory.

        Parameters
        ----------
        base_dir : str
            Base directory for debug outputs.

        Returns
        -------
        str
            Path to the created directory.
        """
        os.makedirs(base_dir, exist_ok=True)
        run_number = len(os.listdir(base_dir)) + 1
        run_dir = os.path.join(base_dir, f"{run_number}-{self.model_name}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    @property
    def model_name(self) -> str:
        """
        Return the name of the correction model.

        Returns
        -------
        str
            The name of the correction model class.
        """
        return self.correction_model.__class__.__name__

    @property
    def ref_patches(self) -> np.ndarray:
        """
        Return grid image of reference color patches.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            A tuple containing:

            - **self.reference_patches**: The array representing the reference
            color patches.
            - **self.reference_grid_image**: The array depicting the grid layout of
            the reference patches.
            - **self.reference_debug_image**: The array used for debugging the color
            correction process.
        """
        return (
            self.reference_patches,
            self.reference_grid_image,
            self.reference_debug_image,
        )

    def set_reference_patches(
        self,
        image: np.ndarray | None,
        debug: bool = False,
    ) -> None:
        """Set reference patches for color correction.

        This function sets up reference color patches either from a default set of D50 BGR values
        or by extracting patches from a provided reference image.

        Parameters
        ----------
        image : np.ndarray | None
            Input reference image from which to extract color patches.
            If None, uses default D50 BGR reference values.
        debug : bool, default=False
            If True, generates additional debug visualization.

        Returns
        -------
        None
            Sets the following instance attributes:

            - `self.reference_patches`: Color values of reference patches
            - `self.reference_grid_image`: Grid image of reference patches
            - `self.reference_debug_image`: Debug visualization (only if debug=True and image provided)

        """  # noqa: E501
        if image is None:
            self.reference_patches = reference_color_d50_bgr
            self.reference_grid_image = create_patch_tiled_image(self.reference_patches)
        else:
            (
                self.reference_patches,
                self.reference_grid_image,
                self.reference_debug_image,
            ) = self._extract_color_patches(image=image, debug=debug)

    def set_input_patches(self, image: np.ndarray, debug: bool = False) -> None:
        """
        This function processes an input image to extract color patches and generates
        corresponding grid and debug visualizations 🔍


        Parameters
        ----------
        image : np.ndarray
            Input image to extract color patches from 📸
        debug : bool, optional
            If True, generates additional debug visualization, by default False 🐛

        Returns
        -------
        tuple
            Contains three elements:

            - `self.input_patches` : np.ndarray
                Extracted color patches from the image
            - `self.input_grid_image` : np.ndarray
                Visualization of the detected grid
            - `self.input_debug_image` : np.ndarray
                Debug visualization (if debug=True)

        Notes
        -----
        The function will set class attributes:

        - `self.input_patches`
        - `self.input_grid_image`
        - `self.input_debug_image`

        The function first resets these attributes to None before processing 🔄
        """
        self.input_patches = None
        self.input_grid_image = None
        self.input_debug_image = None

        (
            self.input_patches,
            self.input_grid_image,
            self.input_debug_image,
        ) = self._extract_color_patches(image=image, debug=debug)
        return self.input_patches, self.input_grid_image, self.input_debug_image

    def fit(self) -> tuple[np.ndarray, list[ColorPatchType], list[ColorPatchType]]:
        """Fit color correction model using input and reference images.

        Parameters
        ----------
        input_image : np.ndarray
            Image `BGR` to be corrected that contains color checker classic 24 patches.
        reference_image : np.ndarray, optional
            Image `BGR` to be reference that contains color checker classic 24 patches.

        Returns
        -------
        tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]
            Correction weights, input patches, and reference patches.
        """
        if self.reference_patches is None:
            raise RuntimeError("Reference patches must be set before fitting model")

        if self.input_patches is None:
            raise RuntimeError("Input patches must be set before fitting model")

        self.trained_model = self.correction_model.fit(
            x_patches=self.input_patches,
            y_patches=self.reference_patches,
        )

        # Compute corrected patches
        self.corrected_patches = self.correction_model.compute_correction(
            input_image=np.array(self.input_patches),
        )
        self.corrected_grid_image = create_patch_tiled_image(self.corrected_patches)

        return self.trained_model

    def predict(
        self,
        input_image: ImageBGR,
        debug: bool = False,
        debug_output_dir: str = "output-debug",
    ) -> ImageBGR:
        """Apply color correction to input image.

        Parameters
        ----------
        input_image : ImageBGR
            Image to be color corrected.
        debug : bool, optional
            Whether to save debug visualizations.
        debug_output_dir : str, optional
            Directory to save debug outputs.

        Returns
        -------
        ImageBGR
            Color corrected image.

        Raises
        ------
        RuntimeError
            If model has not been fitted.
        """
        if self.trained_model is None:
            raise RuntimeError("Model must be fitted before correction")

        corrected_image = self.correction_model.compute_correction(
            input_image=input_image.copy(),
        )

        if debug:
            self._save_debug_output(
                input_image=input_image,
                corrected_image=corrected_image,
                output_directory=debug_output_dir,
            )

        return corrected_image

    def calc_color_diff_patches(self) -> dict:
        """
        Calculate color difference metrics for image patches using the dE CIE 2000 metric.

        This method computes the color differences between:

          - The initial (uncorrected) input patches and the reference patches.
          - The corrected patches and the reference patches.

        It then calculates the delta as the difference between the initial and corrected color differences
        (i.e., initial minus corrected) to assess the change in color discrepancy after correction.

        Notes
        -----
        This function processes patches only, not whole images. The calculations compare the color differences
        between patches before correction and patches after correction against the same reference patches.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - `initial`: dict containing the color difference metrics for the initial patches versus the reference.
            - `corrected`: dict containing the color difference metrics for the corrected patches versus the reference.
            - `delta`: dict with metrics representing the difference between the initial and corrected color differences.
                      Each metric is computed as:
                      ```python
                        metric_delta = metric_initial - metric_corrected,
                      ```
                      where metrics include `min`, `max`, `mean`, and `std`.

        """  # noqa: E501
        initial_color_diff = calc_color_diff(
            image1=self.input_grid_image,
            image2=self.reference_grid_image,
        )

        corrected_color_diff = calc_color_diff(
            image1=self.corrected_grid_image,
            image2=self.reference_grid_image,
        )

        delta_color_diff = {
            "min": initial_color_diff["min"] - corrected_color_diff["min"],
            "max": initial_color_diff["max"] - corrected_color_diff["max"],
            "mean": initial_color_diff["mean"] - corrected_color_diff["mean"],
            "std": initial_color_diff["std"] - corrected_color_diff["std"],
        }

        info = {
            "initial": initial_color_diff,
            "corrected": corrected_color_diff,
            "delta": delta_color_diff,
        }

        return info


if __name__ == "__main__":
    # Step 1: Define the path to the input image
    image_path = "asset/images/cc-19.png"
    image_path = "asset/images/cc-1.jpg"

    # Step 2: Load the input image
    input_image = cv2.imread(image_path)

    # Step 3: Initialize the color correction model with specified parameters
    color_corrector = ColorCorrection(
        detection_model="yolov8",
        detection_conf_th=0.25,
        correction_model="polynomial",
        # correction_model="least_squares",
        # correction_model="affine_reg",
        # correction_model="linear_reg",
        degree=3,  # for polynomial correction model
        use_gpu=True,
    )

    # Step 4: Extract color patches from the input image
    # you can set reference patches from another image (image has color checker card)
    # or use the default D50
    # color_corrector.set_reference_patches(image=None, debug=True)
    color_corrector.set_input_patches(image=input_image, debug=True)
    color_corrector.fit()
    corrected_image = color_corrector.predict(
        input_image=input_image,
        debug=True,
        debug_output_dir="zzz",
    )

    eval_result = color_corrector.calc_color_diff_patches()
    print(eval_result)
