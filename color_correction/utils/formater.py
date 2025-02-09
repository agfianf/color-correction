import json

import numpy as np

from color_correction.utils.image_processing import numpy_array_to_base64


def format_value(value: np.ndarray | dict | list | float | str) -> str:
    """
    Format different types of values for HTML display.

    Parameters
    ----------
    value : np.ndarray, dict, list, float, or str
        The input value that needs to be formatted. If the value is:

          - an `np.ndarray`, it is assumed to represent an image array and will be converted
            to a base64-encoded HTML image.
          - a `dict` or `list`, it will be converted to its JSON string representation.
          - an `np.float64` or `np.float32`, it will be formatted as a float with 4 decimal places.
          - any other type, it will be converted to a string using the str() function.

    Returns
    -------
    str
        A string representation of the input value formatted for HTML display.
    """  # noqa: E501
    if isinstance(value, np.ndarray):  # Image arrays
        return f'<img src="{numpy_array_to_base64(value, convert_bgr_to_rgb=True)}"/>'
    elif isinstance(value, dict | list):  # Dictionaries or lists
        return json.dumps(value)
    elif isinstance(value, np.float64 | np.float32):  # Numpy float types
        return f"{float(value):.4f}"
    return str(value)
