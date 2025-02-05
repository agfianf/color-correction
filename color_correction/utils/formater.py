import json

import numpy as np

from color_correction.utils.image_processing import numpy_array_to_base64


def format_value(value: np.ndarray | dict | list | float | str) -> str:
    """Format different types of values for HTML display"""
    if isinstance(value, np.ndarray):  # Image arrays
        return f'<img src="{numpy_array_to_base64(value, convert_bgr_to_rgb=True)}"/>'
    elif isinstance(value, dict | list):  # Dictionaries or lists
        return json.dumps(value)
    elif isinstance(value, np.float64 | np.float32):  # Numpy float types
        return f"{float(value):.4f}"
    return str(value)
