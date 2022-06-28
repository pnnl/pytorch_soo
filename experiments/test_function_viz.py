from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from viz_soo import rastrigin, rosenbrock, sphere


def plot_both(
    func: Callable,
    minima: Tuple[float, float],
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    step: float = 0.25,
    levels: int = 20,
):
    fname = func.__name__.capitalize()

    X = np.arange(x_bounds[0], x_bounds[1], step)
    Y = np.arange(y_bounds[0], y_bounds[1], step)
    X, Y = np.meshgrid(X, Y)
    Z = func([X, Y], lib=np)

    fig = plt.figure(figsize=(16, 8))
    ax_contour = fig.add_subplot(1, 2, 1)
    ax_contour.contour(X, Y, Z, levels, cmap="jet")
    plt.xlim(x_bounds)
    plt.ylim(y_bounds)
    ax_contour.set_xlabel("x")
    ax_contour.set_ylabel("y")
    plt.plot(*minima, "gD")
    plt.title("Contour Plot")

    ax_surface = fig.add_subplot(1, 2, 2, projection="3d")
    ax_surface.plot_surface(X, Y, Z, cmap=plt.get_cmap("jet"))
    ax_surface.set_zlabel("z")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Surface Plot")
    plt.suptitle(f"{fname} function")
    plt.savefig("imgs/func_plots/" + fname + "_contour_and_surface.png", dpi=300)


def plot_surface(
    func: Callable,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    step: float = 0.25,
):
    X = np.arange(x_bounds[0], x_bounds[1], step)
    Y = np.arange(y_bounds[0], y_bounds[1], step)
    X, Y = np.meshgrid(X, Y)
    Z = func([X, Y], lib=np)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))
    surf = ax.plot_surface(X, Y, Z, cmap=plt.get_cmap("jet"))
    ax.set_zlabel("z")
    plt.xlabel("x")
    plt.ylabel("y")
    fname = func.__name__.capitalize()
    plt.savefig("imgs/func_plots/" + fname + "_surface.png", dpi=300)


def plot_contour(
    func: Callable,
    minima: Tuple[float, float],
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    step: float = 0.25,
    levels: int = 20,
):
    X = np.arange(x_bounds[0], x_bounds[1], step)
    Y = np.arange(y_bounds[0], y_bounds[1], step)
    X, Y = np.meshgrid(X, Y)
    Z = func([X, Y], lib=np)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, levels, cmap="jet")
    plt.xlim(x_bounds)
    plt.ylim(y_bounds)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(*minima, "gD")
    plt.tight_layout()
    fname = func.__name__.capitalize()
    plt.savefig("imgs/func_plots/" + fname + "_contour.png", dpi=300)


"""
plot_both(sphere, (0, 0), (-4.5, 4.5), (-4.5, 4.5), 0.01)
plot_both(rastrigin, (0, 0), (-4.5, 4.5), (-4.5, 4.5), 0.01)
plot_both(rosenbrock, (1, 1), (-2.0, 2.0), (-1.0, 3.0), 0.01, levels=90)
"""
plot_surface(sphere, (-4.5, 4.5), (-4.5, 4.5), 0.01)
plot_contour(sphere, (0, 0), (-4.5, 4.5), (-4.5, 4.5), 0.01)
plot_surface(rastrigin, (-4.5, 4.5), (-4.5, 4.5), 0.01)
plot_contour(rastrigin, (0, 0), (-4.5, 4.5), (-4.5, 4.5), 0.01)
plot_surface(rosenbrock, (-2.0, 2.0), (-2.0, 2.0), 0.01)
plot_contour(rosenbrock, (1, 1), (-2, 2), (-1, 3), 0.01, levels=90)

plt.show()
