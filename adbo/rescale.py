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
        input (np.ndarray): Input array.
        config (dict): Config dictionary. Should have key "order".

    Returns:
        np.ndarray: Rescaled array.
    """
    # this will only deal with 1 dimensional arrays at the moment
    # print("input", input)
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
    output = (input - mins) / diffs
    # a test that all outputs are between 0 and 1, otherwise raise an error
    assert np.all(output >= 0) and np.all(output <= 1)
    return output


def rescale_inverse(input: np.ndarray, config: dict) -> np.ndarray:
    """Rescale back the numbers to fall in original range.

    Args:
        input (np.ndarray): Input array.
        config (dict): Config dictionary. Should have key "order".

    Returns:
        np.ndarray: rescaled array.
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
    return input * diffs + mins
