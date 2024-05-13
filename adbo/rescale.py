"""Rescale the parameter inputs to fall in 0.0 to 1.0 range.

This is used to rescale the inputs to the GP, and then rescale the outputs back to the original range.

The parameters are defined in the config file, and the rescaling is done by subtracting the minimum value, and dividing by the range.

TODO: add a test for inverse.
"""

import numpy as np


def rescale(input: np.ndarray, config: dict) -> np.ndarray:
    """Rescale the numbers to fall in 0.0 to 1.0 range.

    Should write a test that the inverse works.

    Args:
        input (np.ndarray): Input array [N, F].
        config (dict): Config dictionary. Should have key "order".

    Returns:
        np.ndarray: Rescaled array [N, F].
    """
    # this will only deal with 1 dimensional arrays at the moment
    print("input", input, input.shape)
    # print("config", config)
    order = config["order"]
    ones = np.ones((input.shape[0]))
    diffs = np.array(
        [config[i]["max"] - config[i]["min"] for i in order]
    )  # .reshape(input.shape[0], 1)
    # print("diffs", diffs)
    mins = np.array([config[i]["min"] for i in order])  # .reshape(input.shape[0], 1)
    # print("mins", mins)
    print(diffs.shape, mins.shape, input.shape, ones.shape)
    # return (input - np.dot(ones, mins)) * np.dot(ones, 1 / diffs)
    output = []
    for i in range(input.shape[0]):
        print(input[i], mins[i], diffs[i])
        output.append((input[i] - mins[i]) / diffs[i])
    # print("Output", output)
    out = (input - mins) / diffs
    # a test that all outputs are between 0 and 1, otherwise raise an error
    assert np.all(out >= 0) and np.all(out <= 1)

    assert out.shape == input.shape
    return out


def rescale_inverse(input: np.ndarray, config: dict) -> np.ndarray:
    """Rescale back the numbers to fall in original range.

    Args:
        input (np.ndarray): Input array [N, F].
        config (dict): Config dictionary. Should have key "order".

    Returns:
        np.ndarray: rescaled array [N, F].
    """
    # print("input", input)
    # print("config", config)
    order = config["order"]
    ones = np.ones((input.shape[0]))  # , 1
    diffs = np.array([config[i]["max"] - config[i]["min"] for i in order])
    mins = np.array([config[i]["min"] for i in order])
    # assert diffs.shape[0] == mins.shape[0] and diffs.shape[0] == ones.shape[0]
    print(diffs.shape, mins.shape, input.shape, ones.shape)
    # return np.dot(input, np.dot(ones, diffs)) + np.dot(ones, mins)
    out = input * diffs + mins
    assert out.shape == input.shape
    return out


# let's design a simple round trip test
def _test_rescale() -> None:
    """Test rescale function."""
    config = {
        "order": ["a", "b"],
        "a": {"min": 0.0, "max": 10.0},
        "b": {"min": 5.0, "max": 15.0},
    }
    real_ex = np.array([[1.0, 10.0, 7], [5.0, 10.0, 13]]).T  # .swapaxes(0, 1)
    scaled_ex = np.array([[0.1, 0.5, 0.7], [0.0, 0.5, 0.8]]).T  # .swapaxes(0, 1)
    print("input", real_ex)
    output = rescale(real_ex, config)
    print("output", output, "scaled_ex", scaled_ex)
    assert np.allclose(output, scaled_ex, atol=1e-6)
    output = rescale_inverse(scaled_ex, config)
    print("output", output, "real_ex", real_ex)
    assert np.allclose(output, real_ex, atol=1e-6)


if __name__ == "__main__":
    # python -m adbo.rescale
    _test_rescale()
