"""GP experiment module; check the gains from different kernels, and\or different prior mean functions."""

import os
import numpy as np
import pandas as pd
import gpflow


def gather_data() -> None:
    """Gather data from existing experiments.

    Get all the LHS samples from the 3D experiments.

    Save it all as a large csv file
    x(displacement, bearing, angle), z (maximum storm height).
    """
    raise NotImplementedError("Gather data not implemented yet.")


def fit_gp(
    x: np.ndarray,
    y: np.ndarray,
    kernel: str,
    mean_function: str,
) -> gpflow.models.GPR:
    """Fit a GP model to the data.

    Args:
        x (np.ndarray): Input data.
        y (np.ndarray): Output data.
        kernel (str): Kernel to use.
        mean_function (str): Mean function to use.

    Returns:
        gpflow.models.GPR: Fitted GP model.
    """
    if kernel == "Matern52":
        k = gpflow.kernels.Matern52()
    elif kernel == "Matern12":
        k = gpflow.kernels.Matern12()
    elif kernel == "SE" or kernel == "SquaredExponential" or kernel == "RBF":
        k = gpflow.kernels.SquaredExponential()
    elif kernel == "RationalQuadratic":
        k = gpflow.kernels.RationalQuadratic()
    elif kernel == "Exponential":
        k = gpflow.kernels.Exponential()
    elif kernel == "Linear":
        k = gpflow.kernels.Linear()
    elif kernel == "Periodic":
        k = gpflow.kernels.Periodic()
    elif kernel == "Polynomial":
        k = gpflow.kernels.Polynomial()
    elif kernel == "Constant":
        k = gpflow.kernels.Constant()
    elif kernel == "White":
        k = gpflow.kernels.White()
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    if mean_function == "Constant":
        mean_function = gpflow.mean_functions.Constant()
    elif mean_function == "Linear":
        mean_function = gpflow.mean_functions.Linear()
    elif mean_function == "Polynomial":
        mean_function = gpflow.mean_functions.Polynomial()
    elif mean_function == "Zero":
        mean_function = gpflow.mean_functions.Zero()
    elif mean_function == "Exponential":
        mean_function = gpflow.mean_functions.Exponential()
    elif mean_function == "Periodic":
        mean_function = gpflow.mean_functions.Periodic()
    elif mean_function == "RationalQuadratic":
        mean_function = gpflow.mean_functions.RationalQuadratic()
    elif mean_function == "LinearPeriodic":
        mean_function = gpflow.mean_functions.LinearPeriodic()
    elif mean_function == "LinearExponential":
        mean_function = gpflow.mean_functions.LinearExponential()
    elif mean_function == "LinearRationalQuadratic":
        mean_function = gpflow.mean_functions.LinearRationalQuadratic()
    elif mean_function == "LinearWhite":
        mean_function = gpflow.mean_functions.LinearWhite()
    elif mean_function == "LinearConstant":
        mean_function = gpflow.mean_functions.LinearConstant()
    elif mean_function == "LinearPolynomial":
        mean_function = gpflow.mean_functions.LinearPolynomial()
    elif mean_function == "LinearGaussian":
        mean_function = gpflow.mean_functions.LinearGaussian()
    elif mean_function == "LinearGaussianPeriodic":
        mean_function = gpflow.mean_functions.LinearGaussianPeriodic()
    elif mean_function == "LinearGaussianRationalQuadratic":
        mean_function = gpflow.mean_functions.LinearGaussianRationalQuadratic()
    elif mean_function == "LinearGaussianWhite":
        mean_function = gpflow.mean_functions.LinearGaussianWhite()
    elif mean_function == "LinearGaussianConstant":
        mean_function = gpflow.mean_functions.LinearGaussianConstant()
    elif mean_function == "LinearGaussianPolynomial":
        mean_function = gpflow.mean_functions.LinearGaussianPolynomial()
    elif mean_function == "LinearGaussianExponential":
        mean_function = gpflow.mean_functions.LinearGaussianExponential()
    elif mean_function == "LinearGaussianPeriodicExponential":
        mean_function = gpflow.mean_functions.LinearGaussianPeriodicExponential()
    else:
        raise ValueError(f"Unknown mean function: {mean_function}")

    model = gpflow.models.GPR(data=(x, y), kernel=k, mean_function=mean_function)

    # model fit
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss,
        model.trainable_variables,
        method="L-BFGS-B",
        options={"maxiter": 100},
    )

    return model


if __name__ == "__main__":
    # gather_data()
    print(
        fit_gp(
            x=np.random.rand(100, 3),
            y=np.random.rand(100, 1),
            kernel="Matern52",
            mean_function="Constant",
        )
    )
