import os

import numpy as np
import pandas as pd

from color_correction.constant.methods import (
    LiteralModelCorrection,
    LiteralModelDetection,
)
from color_correction.services.color_correction import ColorCorrection
from color_correction.utils.image_patch import (
    visualize_patch_comparison,
)
from color_correction.utils.image_processing import calc_color_diff
from color_correction.utils.report_generator import ReportGenerator


class ColorCorrectionAnalyzer:
    def __init__(
        self,
        list_correction_methods: list[tuple[LiteralModelCorrection, dict]],
        list_detection_methods: list[tuple[LiteralModelDetection, dict]],
        use_gpu: bool = True,
    ) -> None:
        self.list_correction_methods = list_correction_methods
        self.list_detection_methods = list_detection_methods
        self.use_gpu = use_gpu
        self.rg = ReportGenerator()

    def _run_single_exp(
        self,
        idx: int,
        input_image: np.ndarray,
        det_method: LiteralModelDetection,
        det_params: dict,
        cc_method: LiteralModelCorrection,
        cc_params: dict,
        reference_image: np.ndarray | None = None,
    ) -> dict:
        cc = ColorCorrection(
            correction_model=cc_method,
            detection_model=det_method,
            detection_conf_th=det_params.get("detection_conf_th", 0.25),
            use_gpu=self.use_gpu,
            **cc_params,
        )

        if reference_image is not None:
            cc.set_reference_image(reference_image)
        cc.set_input_patches(input_image, debug=True)
        cc.fit()
        corrected_image = cc.predict(input_image=input_image)
        eval_results = cc.calc_color_diff_patches()

        before_comparison = visualize_patch_comparison(
            ls_mean_in=cc.input_patches,
            ls_mean_ref=cc.reference_patches,
        )
        after_comparison = visualize_patch_comparison(
            ls_mean_in=cc.corrected_patches,
            ls_mean_ref=cc.reference_patches,
        )

        dE_image = calc_color_diff(  # noqa: N806
            image1=input_image,
            image2=corrected_image,
        )

        one_row = {
            "Index": idx,
            "Detection Method": det_method,
            "Detection Parameters": det_params,
            "Drawed Preprocessing Input": cc.input_debug_image,
            "Drawed Preprocessing Reference": cc.reference_debug_image,
            "Correction Method": cc_method,
            "Correction Parameters": cc_params,
            "Color Patches - Before": before_comparison,
            "Color Patches - After": after_comparison,
            "Input Image": input_image,
            "Corrected Image": corrected_image,
            "Patch ΔE (Before) - Min": eval_results["initial"]["min"],
            "Patch ΔE (Before) - Max": eval_results["initial"]["max"],
            "Patch ΔE (Before) - Mean": eval_results["initial"]["mean"],
            "Patch ΔE (Before) - Std": eval_results["initial"]["std"],
            "Patch ΔE (After) - Min": eval_results["corrected"]["min"],
            "Patch ΔE (After) - Max": eval_results["corrected"]["max"],
            "Patch ΔE (After) - Mean": eval_results["corrected"]["mean"],
            "Patch ΔE (After) - Std": eval_results["corrected"]["std"],
            "Image ΔE - Min": dE_image["min"],
            "Image ΔE - Max": dE_image["max"],
            "Image ΔE - Mean": dE_image["mean"],
            "Image ΔE - Std": dE_image["std"],
        }
        return one_row

    def run(
        self,
        input_image: np.ndarray,
        output_dir: str = "benchmark_debug",
        reference_image: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Fungsi ini menjalankan benchmark untuk model color correction.
        """
        ls_data = []
        idx = 1
        for det_method, det_params in self.list_detection_methods:
            for cc_method, cc_params in self.list_correction_methods:
                print(
                    f"Running benchmark for {cc_method} method with {cc_params}",
                )
                data = self._run_single_exp(
                    idx=idx,
                    input_image=input_image,
                    det_method=det_method,
                    det_params=det_params,
                    cc_method=cc_method,
                    cc_params=cc_params,
                    reference_image=reference_image,
                )
                idx += 1
                ls_data.append(data)
        df_results = pd.DataFrame(ls_data)

        # Generate HTML report path
        os.makedirs(output_dir, exist_ok=True)
        html_report_path = os.path.join(output_dir, "report.html")
        pickel_report_path = os.path.join(output_dir, "report.pkl")

        # Report Generator -----------------------------------------------------
        self.rg.generate_html_report(df=df_results, path_html=html_report_path)
        self.rg.save_dataframe(df=df_results, filepath=pickel_report_path)

        # Save CSV report, but without image data
        df_results.drop(
            columns=[
                "Drawed Preprocessing Input",
                "Drawed Preprocessing Reference",
                "Color Patches - Before",
                "Color Patches - After",
                "Corrected Image",
                "Input Image",
            ],
        ).to_csv(os.path.join(output_dir, "report_no_image.csv"), index=False)

        print("DataFrame shape:", df_results.shape)
        print("\nDataFrame columns:", df_results.columns.tolist())


if __name__ == "__main__":
    # Pastikan path image sesuai dengan lokasi image Anda
    input_image_path = "asset/images/cc-19.png"

    benchmark = ColorCorrectionAnalyzer(
        list_correction_methods=[
            ("least_squares", {}),
            ("linear_reg", {}),
            ("affine_reg", {}),
            ("polynomial", {"degree": 2}),
            ("polynomial", {"degree": 3}),
            ("polynomial", {"degree": 4}),
        ],
        list_detection_methods=[
            ("yolov8", {"detection_conf_th": 0.25}),
        ],
    )

    benchmark.run(
        input_image_path,
        reference_image=None,
        output_dir="benchmark_debug",
    )
