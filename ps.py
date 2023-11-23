from typing import Callable
import numpy as np
import matplotlib.pyplot as plt



def g(y):
    return 5 * y - np.exp(y)

def w22_func(a: float = 0.062, b: float = 0.031, c: float = 0.008) -> Callable:
    def f(y: float) -> float:
        return np.exp(a * y + b * np.log(y) * y + c)
    return f



def wang_diff(a: float = 0.062, b: float = 0.031, c: float = 0.008) -> Callable:
    def f(y: float) -> float:
        return y - np.exp(a * y + b * np.log(y) * y + c)
    return f


def bisection(f: Callable, left: float, right: float, tol: float) -> float:
    # https://en.wikipedia.org/wiki/Root-finding_algorithms#Bisection_method
    fleft = f(left)
    fright = f(right)
    if fleft * fright > 0:
        print("Error: f(left) and f(right) must have opposite signs.")
        return None

    while fleft * fright < 0 and right - left > tol:
        mid = (left + right) / 2
        fmid = f(mid)
        if fleft * fmid < 0:
            right = mid
            fright = fmid
        else:
            left = mid
            fleft = fmid
    return (left + right) / 2


def plot_w22_func():
    f = w22_func()
    y = np.linspace(0.1, 21, 1000)
    plt.plot(y, y, label="y=y")
    plt.plot(y, f(y), label="y=exp(a*y + b*log(y)*y + c)")
    lower = bisection(wang_diff(), 0.01, 3, 1e-5)
    if lower is not None:
        print(lower, wang_diff()(lower))
    upper = bisection(wang_diff(), 6, 21, 1e-5)
    if upper is not None:
        print(upper, wang_diff()(upper))
    plt.plot(lower, f(lower), "o", label="lower, y={:.3f}".format(lower))
    plt.plot(upper, f(upper), "o", label="upper, y={:.3f}".format(upper))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # python ps.py
    # 
    bi = bisection(g, 1, 10, 1e-5)
    print(bi, g(bi))
    bi = bisection(wang_diff(), 0.01, 3, 1e-5)
    if bi is not None:
        print(bi, wang_diff()(bi))
    
    bi = bisection(wang_diff(), 6, 21, 1e-5)
    if bi is not None:
        print(bi, wang_diff()(bi))
    plot_w22_func()
