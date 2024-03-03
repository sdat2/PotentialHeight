"""Non bayesian optimization wrap examples.s"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sithom.time import timeit
from sithom.plot import plot_defaults
from sithom.io import write_json
from src.constants import NEW_ORLEANS
from tcpips.constants import FIGURE_PATH
from adforce import setup_new, read_results, run_wrapped

ROOT: str = "/work/n01/n01/sithom/adcirc-swan/"


@timeit
def run_angle_exp() -> None:
    exp_dir = os.path.join(ROOT, "angle_test")
    os.makedirs(exp_dir, exist_ok=True)
    for i, angle in enumerate(np.linspace(-90, 90, num=10)):
        tmp_dir = os.path.join(exp_dir, f"exp_{i:03}")
        print(tmp_dir, angle)
        setup_new(tmp_dir, angle)


@timeit
def read_angle_exp() -> None:
    results = []
    exp_dir = os.path.join(ROOT, "angle_test")
    for i, angle in enumerate(np.linspace(-90, 90, num=10)):
        tmp_dir = os.path.join(exp_dir, f"exp_{i:03}")
        print(tmp_dir, angle)
        res = read_results(tmp_dir)
        print(i, angle, res)
        results += [[i, angle, res]]
    results = np.array(results)

    plot_defaults()
    plt.plot(results[:, 1], results[:, 2])
    plt.xlabel("Angle [$^{\circ}$]")
    plt.ylabel("Height [m]")
    plt.savefig("angle_test.png")


@timeit
def run_angle_new() -> None:
    # https://github.com/sdat2/new-orleans/blob/main/src/models/emu6d.py
    exp_dir = os.path.join(ROOT, "angle")
    os.makedirs(exp_dir, exist_ok=True)
    res_l = []
    angles = np.linspace(-80, 80, num=100)
    for i, angle in enumerate(angles):
        tmp_dir = os.path.join(exp_dir, f"exp_{i:03}")
        print(tmp_dir, angle)
        res_l += [run_wrapped(out_path=tmp_dir, angle=angle)]
    print(angles, res_l)
    plot_defaults()
    plt.plot(angles, res_l)
    plt.xlabel("Angle [$^{\circ}$]")
    plt.ylabel("Height [m]")
    plt.savefig(
        os.path.join(
            FIGURE_PATH,
            "angle_test.png",
        )
    )


@timeit
def run_speed() -> None:
    # https://github.com/sdat2/new-orleans/blob/main/src/models/emu6d.py
    exp_dir = os.path.join(ROOT, "trans_speed")
    os.makedirs(exp_dir, exist_ok=True)
    res_l = []
    speeds = np.linspace(1, 25, num=50)
    for i, speed in enumerate(speeds):
        tmp_dir = os.path.join(exp_dir, f"exp_{i:03}")
        print(tmp_dir, speed)
        res_l += [run_wrapped(out_path=tmp_dir, trans_speed=speed)]
    print(speeds, res_l)
    plt.plot(speeds, res_l)
    plt.xlabel("Translation speed [m/s]")
    plt.ylabel("Height [m]")
    plt.savefig(
        os.path.join(
            FIGURE_PATH,
            "trans_speed_test.png",
        )
    )


@timeit
def run_disp() -> None:
    # https://github.com/sdat2/new-orleans/blob/main/src/models/emu6d.py
    exp_dir = os.path.join(ROOT, "displacment")
    os.makedirs(exp_dir, exist_ok=True)
    res_l = []
    tmp_dirs = []

    displacements = np.linspace(-2, 2, num=50)

    def save_output() -> None:
        nonlocal tmp_dirs, res_l, displacements
        output = {
            i: {
                "dir": tmp_dirs[i],
                "res": res_l[i],
                "displacement": displacements[i],
            }
            for i in range(len(tmp_dirs))
        }
        write_json(output, os.path.join(exp_dir, "displacement_test.json"))

    for i, displacement in enumerate(displacements):
        tmp_dir = os.path.join(exp_dir, f"exp_{i:03}")
        print(tmp_dir, displacement)
        tmp_dirs += [tmp_dir]
        res_l += [
            run_wrapped(out_path=tmp_dir, impact_lon=NEW_ORLEANS.lon + displacement)
        ]
        save_output()
    print(displacements, res_l)

    plot_defaults()

    plt.plot(displacements, res_l)
    plt.xlabel("Displacement, $c$ [$^{\circ}$]")
    plt.ylabel("Height [m]")
    plt.savefig(
        os.path.join(
            FIGURE_PATH,
            "displacement_test.png",
        )
    )


if __name__ == "__main__":
    # run_angle_exp()
    # read_angle_exp()
    # run_angle_new()
    # run_speed()
    run_disp()
