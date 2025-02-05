import os
import re
from typing import Final

import httpx

from color_correction.schemas.device import GPUType
from color_correction.utils.device_info import get_device_specs


def download_google_drive_file(file_id: str, output_file: str) -> None:
    """
    Download a file from Google Drive using its file ID.

    Parameters
    ----------
    file_id : str
        The unique ID of the file on Google Drive.
    output_file : str
        The name of the output file where the content will be saved.

    Returns
    -------
    None
        Downloads the file and saves it locally.
    """
    url: Final = f"https://drive.google.com/uc?export=download&id={file_id}"

    with httpx.Client(follow_redirects=True, timeout=60) as client:
        print(f"Start downloading file from: {url}")
        # First request to get confirmation token
        response = client.get(url)
        print(f"Response status code: {response.status_code}")
        if response.status_code == 303 or "download_warning" in response.text:
            # Handle large file confirmation
            print("Large file confirmation needed if file is not downloaded.")
            confirm_token = re.findall(r"confirm=([^&]+)", response.url.query)
            print(f"Confirm token: {confirm_token}")
            if confirm_token:
                url = f"{url}&confirm={confirm_token[0]}"
                print(f"New URL: {url}")

        # Stream download
        with client.stream("GET", url, timeout=60) as response:
            print(f"Downloading file to: {output_file}")
            response.raise_for_status()
            with open(output_file, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        print(f"File downloaded: {output_file}")


def downloader_model_yolov8(use_gpu: bool = False) -> str:
    specs = get_device_specs()
    model_folder = os.path.join(os.getcwd(), "tmp", "models")
    if use_gpu:
        if specs.is_apple_silicon:
            print("Apple Silicon device detected.")
            fileid = "19O02x_Co2ceBQHHNUib-aVp0cjgG9V-r"
            filename = "yv8-det-mps.onnx"
        elif specs.gpu_type == GPUType.NVIDIA:
            print("NVIDIA GPU detected.")
            fileid = "19O02x_Co2ceBQHHNUib-aVp0cjgG9V-r"
            filename = "yv8-det-gpu.onnx"
        else:
            raise ValueError(
                "GPU not detected or not supported. "
                "Please use CPU device. Device Info: ",
                specs,
            )
    else:
        print("CPU device detected.")
        fileid = "1d1p2HCltiJeVGi6NtDanJvqLbzXs8NLP"
        filename = "yv8-det-cpu.onnx"

    os.makedirs(model_folder, exist_ok=True)
    fullpath = os.path.join(model_folder, filename)
    if os.path.exists(fullpath):
        return fullpath
    print("Auto downloading YOLOv8 model...")
    download_google_drive_file(fileid, fullpath)
    return fullpath


if __name__ == "__main__":
    # Example usage
    # file_id = "19O02x_Co2ceBQHHNUib-aVp0cjgG9V-r"
    # output_file = "yv8-det-mps.onnx"
    filename = downloader_model_yolov8()
    print(f"Output file name: {filename}")
