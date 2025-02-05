import json
import numpy as np
import pytest
from color_correction.utils.formater import format_value

@pytest.mark.parametrize("value,expected_check", [
    # For numpy array image: expecting an <img src="..."/> tag.
    (np.array([[1, 2, 3]], dtype=np.uint8), lambda out: out.startswith('<img src="')),
    # For dict: expecting a valid json string that decodes back to the same dict.
    ({"a": 1, "b": 2}, lambda out: json.loads(out) == {"a": 1, "b": 2}),
    # For list: expecting a valid json string that decodes back to the same list.
    ([1, 2, 3], lambda out: json.loads(out) == [1, 2, 3]),
    # For numpy float types: expecting formatted float with 4 decimals.
    (np.float64(3.14159265), lambda out: out == "3.1416"),
    (np.float32(2.71828), lambda out: out == "2.7183"),
    # For built-in float: the function doesn't format, returns str(value)
    (2.0, lambda out: out == "2.0"),
    # For plain string: should return same string.
    ("test string", lambda out: out == "test string"),
])
def test_format_value(value, expected_check):
    result = format_value(value)
    assert expected_check(result)
