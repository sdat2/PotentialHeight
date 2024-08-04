"""Rescale the parameter inputs to fall in 0.0 to 1.0 range.

This is used to rescale the inputs to the GP, and then rescale the outputs back to the original range.

The parameters are defined in the config file, and the rescaling is done by subtracting the minimum value, and dividing by the range.

Input/output numpy arrays are assumed to be of shape [N, F] where N is the number of points and F is the number of features.
"""

import numpy as np


def rescale(inputs: np.ndarray, config: dict, verbose: bool = False) -> np.ndarray:
    """Rescale the numbers to fall in 0.0 to 1.0 range.

    Args:
        inputs (np.ndarray): Input array [N, F].
        config (dict): Config dictionary. Should have key "order".
        verbose (bool): Print debug information. Default is False.

    Returns:
        np.ndarray: Rescaled array [N, F].
    """
    # this will only deal with 1 dimensional arrays at the moment
    print("inputs", inputs, inputs.shape)
    # print("config", config)
    order = config["order"]
    ones = np.ones((inputs.shape[0]))
    diffs = np.array(
        [config[i]["max"] - config[i]["min"] for i in order]
    )  # .reshape(inputs.shape[0], 1)
    # print("diffs", diffs)
    mins = np.array([config[i]["min"] for i in order])  # .reshape(inputs.shape[0], 1)
    # print("mins", mins)
    if verbose:
        print(diffs.shape, mins.shape, inputs.shape, ones.shape)
    # return (inputs - np.dot(ones, mins)) * np.dot(ones, 1 / diffs)
    # outputs = []
    if verbose:
        for i in range(inputs.shape[1]):
            print(inputs[:, i], mins[i], diffs[i])
            # outputs.append((inputs[:, i] - mins[i]) / diffs[i])
    # print("Output", outputs)
    outputs = (inputs - mins) / diffs
    # a test that all outputs are between 0 and 1, otherwise raise an error
    assert np.all(outputs >= 0) and np.all(outputs <= 1)

    assert outputs.shape == inputs.shape
    return outputs


def rescale_inverse(
    inputs: np.ndarray, config: dict, verbose: bool = False
) -> np.ndarray:
    """Rescale back the numbers to fall in original range.

    Args:
        inputs (np.ndarray): Input array [N, F].
        config (dict): Config dictionary. Should have key "order".
        verbose (bool): Print debug information. Default is False.

    Returns:
        np.ndarray: rescaled array size [N, F] where N is number of points and F is number of features.
    """
    # print("inputs", inputs)
    # print("config", config)
    order = config["order"]
    ones = np.ones((inputs.shape[0]))  # , 1
    diffs = np.array([config[i]["max"] - config[i]["min"] for i in order])
    mins = np.array([config[i]["min"] for i in order])
    # assert diffs.shape[0] == mins.shape[0] and diffs.shape[0] == ones.shape[0]
    if verbose:
        print(diffs.shape, mins.shape, inputs.shape, ones.shape)
    # return np.dot(inputs, np.dot(ones, diffs)) + np.dot(ones, mins)
    outputs = inputs * diffs + mins
    assert outputs.shape == inputs.shape
    return outputs


# let's design a simple round trip test
def test_rescale_test(verbose: bool = True) -> None:
    """Test rescale function can round trip/ is self inverse and accurate.

    Args:
        verbose (bool): Print debug information. Default is True.
    """

    def test_roundtrip(real_ex, scaled_ex, config):
        outputs = rescale(real_ex, config)
        if verbose:
            print("outputs", outputs, "\n scaled_ex", scaled_ex)
        assert np.allclose(outputs, scaled_ex, atol=1e-6)
        outputs = rescale_inverse(scaled_ex, config)
        if verbose:
            print("outputs", outputs, "\n real_ex", real_ex)
        assert np.allclose(outputs, real_ex, atol=1e-6)

    real_ex = np.array([[1.0, 10.0, 7], [5.0, 10.0, 13]]).T
    scaled_ex = np.array([[0.1, 1.0, 0.7], [0.0, 0.5, 0.8]]).T
    config = {
        "order": ("a", "b"),
        "a": {"min": 0.0, "max": 10.0},
        "b": {"min": 5.0, "max": 15.0},
    }
    test_roundtrip(real_ex, scaled_ex, config)
    test_roundtrip(np.array([[1.0], [11.0]]).T, np.array([[0.1], [0.6]]).T, config)
    test_roundtrip(
        np.array([[1.0]]).T,
        np.array([[0.1]]).T,
        {"order": ["a"], "a": {"min": 0.0, "max": 10.0}},
    )
    real_ex = np.random.rand(500, 2)
    assert np.allclose(
        rescale(rescale_inverse(real_ex, config), config), real_ex, atol=1e-6
    )
    print("Rescaling round trip test passed.")


if __name__ == "__main__":
    # python -m adbo.rescale
    test_rescale_test()
