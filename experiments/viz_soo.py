#!/usr/bin/env python3
"""
Taken and modified from here: https://github.com/jettify/pytorch-optimizer
Novik, M. (2020). torch-optimizer -- collection of optimization algorithms for PyTorch.
(Version 1.0.1) [Computer software]
"""
import math
from typing import Iterable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

import matplotlib.pyplot as plt
from pytorch_sso.optim_hfcr_newton import HFCR_Newton
from pytorch_sso.nonlinear_conjugate_gradient import (
    FletcherReeves,
    PolakRibiere,
    HestenesStiefel,
    DaiYuan,
)
from pytorch_sso.line_search_spec import LineSearchSpec
from pytorch_sso.quasi_newton import (
    DavidonFletcherPowell,
    DavidonFletcherPowellInverse,
    SymmetricRankOne,
    SymmetricRankOneInverse,
    Broyden,
    BrodyenInverse,
    BFGS,
)
from hyperopt import fmin, hp, tpe

plt.style.use("seaborn-white")


@dataclass(frozen=True)
class ExptSpec:
    """"""

    optimizer: torch.optim.Optimizer
    second_order: bool
    min_lr: float = 0.0
    max_lr: float = 1.0

    # Newton Iteration Specs
    min_newton: int = 0
    max_newton: int = 50
    min_newton_tol: float = 1e-6
    max_newton_tol: float = 0.1

    # Krylov/Conjugate Residual Specs
    min_krylov: int = 0
    max_krylov: int = 50
    min_krylov_tol: float = 1e-6
    max_krylov_tol: float = 0.1

    min_momentum: float = 0.0
    max_momentum: float = 1.0

    min_extrapolation_factor: float = 0.0
    max_extrapolation_factor: float = 1.0
    min_sufficient_decrease: float = 1.0e-4
    max_sufficient_decrease: float = 1.0e-4
    min_curvature_constant: float = 0.9
    max_curvature_constant: float = 0.9
    min_max_searches: int = 10
    max_max_searches: int = 10

    min_quasi_newton_memory: int = None
    max_quasi_newton_memory: int = None


# Add kwargs to the optimizers so that any unexpected kwarg is ignored
# As well as mapping between the "krylov" vs "cr" terms
class VizSGD(optim.SGD):
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        **kwargs,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
        _ = kwargs


class VizAdam(optim.Adam):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1.0e-08,
        weight_decay=0,
        amsgrad=False,
        **kwargs,
    ):
        super().__init__(params, lr=lr)
        _ = kwargs


class VizHFCR_Newton(HFCR_Newton):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_cr=10,
        max_newton=10,
        newton_tol=1.0e-3,
        cr_tol=1.0e-3,
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_cr=int(max_cr),
            max_newton=int(max_newton),
            newton_tol=newton_tol,
            cr_tol=cr_tol,
        )
        _ = kwargs


class VizSymmetricRankOne(SymmetricRankOne):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=10,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        quasi_newton_memory=None,
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            max_krylov=int(max_cr),
            newton_tol=newton_tol,
            krylov_tol=cr_tol,
            matrix_free_memory=quasi_newton_memory,
        )
        _ = kwargs


class VizSymmetricRankOneDual(SymmetricRankOneInverse):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=10,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        quasi_newton_memory=None,
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=newton_tol,
            matrix_free_memory=quasi_newton_memory,
        )
        _ = kwargs


class VizBFGS(BFGS):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=10,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        quasi_newton_memory=None,
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            max_krylov=int(max_cr),
            newton_tol=newton_tol,
            krylov_tol=cr_tol,
            matrix_free_memory=quasi_newton_memory,
        )
        _ = kwargs


class VizDFPInverse(DavidonFletcherPowellInverse):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=10,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        quasi_newton_memory=None,
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=newton_tol,
            matrix_free_memory=quasi_newton_memory,
        )
        _ = kwargs


class VizFletcherReeves(FletcherReeves):
    def __init__(self, params, lr=0.3333, max_newton=20, newton_tol=1.0e-3, **kwargs):
        max_newton = int(max_newton)
        super().__init__(
            params, lr=lr, max_newton=int(max_newton), abs_newton_tol=newton_tol
        )
        _ = kwargs


class VizPolakRebiere(PolakRibiere):
    def __init__(self, params, lr=0.3333, max_newton=20, newton_tol=1.0e-3, **kwargs):
        max_newton = int(max_newton)
        super().__init__(
            params, lr=lr, max_newton=int(max_newton), abs_newton_tol=newton_tol
        )
        _ = kwargs


class VizHestenesStiefel(HestenesStiefel):
    def __init__(self, params, lr=0.3333, max_newton=20, newton_tol=1.0e-3, **kwargs):
        max_newton = int(max_newton)
        super().__init__(
            params, lr=lr, max_newton=int(max_newton), abs_newton_tol=newton_tol
        )
        _ = kwargs


class VizDaiYuan(DaiYuan):
    def __init__(self, params, lr=0.3333, max_newton=20, newton_tol=1.0e-3, **kwargs):
        max_newton = int(max_newton)
        super().__init__(
            params, lr=lr, max_newton=int(max_newton), abs_newton_tol=newton_tol
        )
        _ = kwargs


# Line Search methods!
class VizHFCR_NewtonLineSearch(HFCR_Newton):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_cr=10,
        max_newton=10,
        newton_tol=1.0e-3,
        cr_tol=1.0e-3,
        extrapolation_factor=0.5,
        sufficient_decrease=1.0e-4,
        curvature_constant=0.9,
        max_searches=10,
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        line_search_spec = LineSearchSpec(
            max_searches=max_searches,
            sufficient_decrease=sufficient_decrease,
            curvature_constant=curvature_constant,
            extrapolation_factor=extrapolation_factor,
        )
        super().__init__(
            params,
            lr=lr,
            max_cr=int(max_cr),
            max_newton=int(max_newton),
            abs_newton_tol=newton_tol,
            cr_tol=cr_tol,
            # Not currently supported...
            # line_search=line_search_spec,
        )
        _ = kwargs


class VizSymmetricRankOneLineSearch(SymmetricRankOne):
    def __init__(
        self,
        params,
        lr=2.0,
        max_newton=1,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        extrapolation_factor=0.5,
        sufficient_decrease=1.0e-4,
        curvature_constant=None,
        max_searches=10,
        quasi_newton_memory=10,
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        line_search_spec = LineSearchSpec(
            max_searches=max_searches,
            sufficient_decrease=sufficient_decrease,
            curvature_constant=curvature_constant,
            extrapolation_factor=extrapolation_factor,
        )
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=newton_tol,
            max_krylov=max_cr,
            krylov_tol=cr_tol,
            line_search=line_search_spec,
            matrix_free_memory=quasi_newton_memory,
        )
        _ = kwargs


class VizSymmetricRankOneDualLineSearch(SymmetricRankOneInverse):
    def __init__(
        self,
        params,
        lr=2.0,
        max_newton=1,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        extrapolation_factor=0.5,
        sufficient_decrease=1.0e-4,
        curvature_constant=None,
        max_searches=10,
        quasi_newton_memory=10,
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        line_search_spec = LineSearchSpec(
            max_searches=max_searches,
            sufficient_decrease=sufficient_decrease,
            curvature_constant=curvature_constant,
            extrapolation_factor=extrapolation_factor,
        )
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=newton_tol,
            line_search=line_search_spec,
            matrix_free_memory=quasi_newton_memory,
        )
        _ = kwargs


class VizBFGSLineSearch(BFGS):
    def __init__(
        self,
        params,
        lr=2.0,
        max_newton=1,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        extrapolation_factor=0.5,
        sufficient_decrease=1.0e-4,
        curvature_constant=None,
        max_searches=10,
        quasi_newton_memory=10,
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        line_search_spec = LineSearchSpec(
            max_searches=max_searches,
            sufficient_decrease=sufficient_decrease,
            curvature_constant=curvature_constant,
            extrapolation_factor=extrapolation_factor,
        )
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=newton_tol,
            max_krylov=max_cr,
            krylov_tol=cr_tol,
            line_search=line_search_spec,
            matrix_free_memory=quasi_newton_memory,
        )
        _ = kwargs


class VizDFPLineSearch(DavidonFletcherPowell):
    def __init__(
        self,
        params,
        lr=2.0,
        max_newton=1,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        extrapolation_factor=0.5,
        sufficient_decrease=1.0e-4,
        curvature_constant=None,
        max_searches=10,
        quasi_newton_memory=10,
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        line_search_spec = LineSearchSpec(
            max_searches=max_searches,
            sufficient_decrease=sufficient_decrease,
            curvature_constant=curvature_constant,
            extrapolation_factor=extrapolation_factor,
        )
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=newton_tol,
            max_krylov=max_cr,
            krylov_tol=cr_tol,
            line_search=line_search_spec,
            matrix_free_memory=quasi_newton_memory,
        )
        _ = kwargs


class VizDFPInverseLineSearch(DavidonFletcherPowellInverse):
    def __init__(
        self,
        params,
        lr=2.0,
        max_newton=1,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        extrapolation_factor=0.5,
        sufficient_decrease=1.0e-4,
        curvature_constant=None,
        max_searches=10,
        quasi_newton_memory=10,
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        line_search_spec = LineSearchSpec(
            max_searches=max_searches,
            sufficient_decrease=sufficient_decrease,
            curvature_constant=curvature_constant,
            extrapolation_factor=extrapolation_factor,
        )
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=newton_tol,
            line_search=line_search_spec,
            matrix_free_memory=quasi_newton_memory,
        )
        _ = kwargs


class VizFletcherReevesLineSearch(FletcherReeves):
    def __init__(
        self,
        params,
        lr=2.0,
        max_newton=1,
        newton_tol=1.0e-3,
        extrapolation_factor=0.1,
        sufficient_decrease=1.0e-4,
        curvature_constant=None,
        max_searches=100,
        **kwargs,
    ):
        max_newton = int(max_newton)
        line_search_spec = LineSearchSpec(
            max_searches=max_searches,
            sufficient_decrease=sufficient_decrease,
            curvature_constant=curvature_constant,
            extrapolation_factor=extrapolation_factor,
        )
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            abs_newton_tol=newton_tol,
            line_search_spec=line_search_spec,
        )
        _ = kwargs


class VizPolakRebiereLineSearch(PolakRibiere):
    def __init__(
        self,
        params,
        lr=2.0,
        max_newton=1,
        newton_tol=1.0e-3,
        extrapolation_factor=0.1,
        sufficient_decrease=1.0e-4,
        curvature_constant=None,
        max_searches=100,
        **kwargs,
    ):
        max_newton = int(max_newton)
        line_search_spec = LineSearchSpec(
            max_searches=max_searches,
            sufficient_decrease=sufficient_decrease,
            curvature_constant=curvature_constant,
            extrapolation_factor=extrapolation_factor,
        )
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            abs_newton_tol=newton_tol,
            line_search_spec=line_search_spec,
        )
        _ = kwargs


class VizHestenesStiefelLineSearch(HestenesStiefel):
    def __init__(
        self,
        params,
        lr=2.0,
        max_newton=1,
        newton_tol=1.0e-3,
        extrapolation_factor=0.1,
        sufficient_decrease=1.0e-4,
        curvature_constant=None,
        max_searches=100,
        **kwargs,
    ):
        max_newton = int(max_newton)
        line_search_spec = LineSearchSpec(
            max_searches=max_searches,
            sufficient_decrease=sufficient_decrease,
            curvature_constant=curvature_constant,
            extrapolation_factor=extrapolation_factor,
        )
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            abs_newton_tol=newton_tol,
            line_search_spec=line_search_spec,
        )
        _ = kwargs


class VizDaiYuanLineSearch(DaiYuan):
    def __init__(
        self,
        params,
        lr=2.0,
        max_newton=1,
        newton_tol=1.0e-3,
        extrapolation_factor=0.1,
        sufficient_decrease=1.0e-4,
        curvature_constant=None,
        max_searches=100,
        **kwargs,
    ):
        max_newton = int(max_newton)
        line_search_spec = LineSearchSpec(
            max_searches=max_searches,
            sufficient_decrease=sufficient_decrease,
            curvature_constant=curvature_constant,
            extrapolation_factor=extrapolation_factor,
        )
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            abs_newton_tol=newton_tol,
            line_search_spec=line_search_spec,
        )
        _ = kwargs


def rosenbrock(tensor, lib=torch):
    _ = lib
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rastrigin(tensor, lib=torch):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    A = 10
    f = (
        A * 2
        + (x ** 2 - A * lib.cos(x * math.pi * 2))
        + (y ** 2 - A * lib.cos(y * math.pi * 2))
    )
    return f


def easom(tensor, lib=torch):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    f = -lib.cos(x) * lib.cos(y) * lib.exp(-((x - math.pi) ** 2) + (y - math.pi) ** 2)

    return f


def sphere(tensor, lib=torch):
    _ = lib
    x, y = tensor
    f = x ** 2 + y ** 2

    return f


def execute_steps(func, initial_state, optimizer_class, optimizer_config, num_iter=500):
    x = torch.Tensor(initial_state).requires_grad_(True)
    optimizer = optimizer_class([x], **optimizer_config)
    steps = []
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        f = func(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        optimizer.step()
        steps[:, i] = x.detach().numpy()
    return steps


def execute_steps_sso(
    func, initial_state, optimizer_class, optimizer_config, num_iter=500
):
    x = torch.Tensor(initial_state).requires_grad_(True)
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    try:
        optimizer = optimizer_class([x], **optimizer_config)
    except ValueError as err:
        print("Bad init")
        print(err)
        steps[:] = 1000 * np.ones_like(steps)
        return steps

    def closure():
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            f = func(x)
            f.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(x, 1.0)

        return f

    for i in range(1, num_iter + 1):
        f = func(x)
        # f.backward(create_graph=True, retain_graph=True)
        f.backward(retain_graph=True)
        optimizer.step(closure)
        steps[:, i] = x.detach().numpy()
        if torch.isnan(x).any():
            print(f"Broke on iteration {i}!")
            steps[:, i:] = np.empty_like(steps[:, i:]) * np.nan
            try:
                print(f"Last step: {steps[:, i-1]}")
            except:
                pass
            break

    return steps


def objective_rastrigin(params):
    lr = params["lr"]
    optimizer_class = params["optimizer_class"]
    second_order = params["second_order"]
    initial_state = (-2.0, 3.5)
    minimum = (0, 0)
    optimizer_config = dict(lr=lr)
    num_iter = 100
    if not second_order:
        steps = execute_steps(
            rastrigin, initial_state, optimizer_class, optimizer_config, num_iter
        )
    else:
        steps = execute_steps_sso(
            rastrigin, initial_state, optimizer_class, optimizer_config, num_iter
        )

    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def objective_rosenbrok(params):
    lr = params["lr"]
    optimizer_class = params["optimizer_class"]
    second_order = params["second_order"]
    minimum = (1.0, 1.0)
    initial_state = (-2.0, 2.0)
    optimizer_config = dict(lr=lr)
    num_iter = 100
    if not second_order:
        steps = execute_steps(
            rosenbrock, initial_state, optimizer_class, optimizer_config, num_iter
        )
    else:
        steps = execute_steps_sso(
            rosenbrock, initial_state, optimizer_class, optimizer_config, num_iter
        )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def objective_easom(params):
    lr = params["lr"]
    optimizer_class = params["optimizer_class"]
    second_order = params["second_order"]
    minimum = (math.pi, math.pi)
    initial_state = (-2.0, 2.0)
    optimizer_config = dict(lr=lr)
    num_iter = 100
    if not second_order:
        steps = execute_steps(
            easom, initial_state, optimizer_class, optimizer_config, num_iter
        )
    else:
        steps = execute_steps_sso(
            easom, initial_state, optimizer_class, optimizer_config, num_iter
        )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def objective_sphere(params):
    lr = params["lr"]
    optimizer_class = params["optimizer_class"]
    second_order = params["second_order"]
    minimum = (0.0, 0.0)
    initial_state = (-2.0, 2.0)
    optimizer_config = dict(lr=lr)
    num_iter = 100
    if not second_order:
        steps = execute_steps(
            sphere, initial_state, optimizer_class, optimizer_config, num_iter
        )
    else:
        steps = execute_steps_sso(
            sphere, initial_state, optimizer_class, optimizer_config, num_iter
        )
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def plot_rastrigin(grad_iter, optimizer_name: str, specs: Mapping):
    x = np.linspace(-4.5, 4.5, 250)
    y = np.linspace(-4.5, 4.5, 250)
    minimum = (0, 0)

    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y], lib=np)

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 20, cmap="jet")
    ax.plot(iter_x, iter_y, color="r", marker="x")
    opt_name = optimizer_name.replace("Viz", "")
    ax.set_title(f"Rastrigin func: {opt_name} with {len(iter_x)} iterations")
    plt.ylim((y.min(), y.max()))
    plt.xlim((x.min(), x.max()))
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("imgs/rastrigin_{}.png".format(optimizer_name))


def plot_sphere(grad_iter, optimizer_name, specs):
    x = np.linspace(-4.5, 4.5, 250)
    y = np.linspace(-4.5, 4.5, 250)
    minimum = (0, 0)

    X, Y = np.meshgrid(x, y)
    Z = sphere([X, Y])

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 20, cmap="jet")
    ax.plot(iter_x, iter_y, color="r", marker="x")
    opt_name = optimizer_name.replace("Viz", "")
    ax.set_title(f"Sphere func: {opt_name} with {len(iter_x)} iterations")
    plt.ylim((y.min(), y.max()))
    plt.xlim((x.min(), x.max()))
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("imgs/sphere_{}.png".format(optimizer_name))


def plot_rosenbrok(grad_iter, optimizer_name, specs):
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)

    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap="jet")
    ax.plot(iter_x, iter_y, color="r", marker="x")

    opt_name = optimizer_name.replace("Viz", "")
    ax.set_title(f"Rosenbrock func: {opt_name} with {len(iter_x)} iterations")
    plt.ylim((y.min(), y.max()))
    plt.xlim((x.min(), x.max()))
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("imgs/rosenbrock_{}.png".format(optimizer_name))


def plot_easom(grad_iter, optimizer_name, lr):
    x = np.linspace(-5, 5, 250)
    y = np.linspace(-5, 5, 250)
    minimum = (math.pi, math.pi)

    X, Y = np.meshgrid(x, y)
    Z = easom([X, Y], lib=np)

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap="jet")
    ax.plot(iter_x, iter_y, color="r", marker="x")

    ax.set_title(
        "Easom func: {} with {} "
        "iterations, lr={:.6}".format(optimizer_name, len(iter_x), lr)
    )
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("imgs/easom_{}.png".format(optimizer_name))


def execute_experiments(
    expt_specs: Iterable[ExptSpec],
    objective: Callable,
    func: Callable,
    plot_func: Callable,
    initial_state: tuple,
    seed=1,
):
    seed = seed
    for expt_spec in expt_specs:
        optimizer_class = expt_spec.optimizer
        second_order = expt_spec.second_order
        lr_low = expt_spec.min_lr
        lr_hi = expt_spec.max_lr

        min_newton = expt_spec.min_newton
        max_newton = expt_spec.max_newton
        min_newton_tol = expt_spec.min_newton_tol
        max_newton_tol = expt_spec.max_newton_tol

        min_krylov = expt_spec.min_newton
        max_krylov = expt_spec.max_newton
        min_krylov_tol = expt_spec.min_krylov_tol
        max_krylov_tol = expt_spec.max_krylov_tol

        min_momentum = expt_spec.min_momentum
        max_momentum = expt_spec.max_momentum

        # LineSearch params
        min_extrapolation_factor = expt_spec.min_extrapolation_factor
        max_extrapolation_factor = expt_spec.max_extrapolation_factor
        print(
            f"extrapolation range: ({min_extrapolation_factor}, {max_extrapolation_factor})"
        )
        min_sufficient_decrease = expt_spec.min_sufficient_decrease
        max_sufficient_decrease = expt_spec.max_sufficient_decrease
        min_curvature_constant = expt_spec.min_curvature_constant
        max_curvature_constant = expt_spec.max_curvature_constant
        min_max_searches = expt_spec.min_max_searches
        max_max_searches = expt_spec.max_max_searches

        min_quasi_newton_memory = expt_spec.min_quasi_newton_memory
        max_quasi_newton_memory = expt_spec.max_quasi_newton_memory

        print("=" * 80)
        print(optimizer_class)
        print(func)
        print("=" * 80)
        space = {
            "optimizer_class": hp.choice("optimizer_class", [optimizer_class]),
            "lr": hp.loguniform("lr", lr_low, lr_hi),
            "second_order": second_order,
        }

        if None in [min_newton, max_newton, min_newton_tol, max_newton_tol]:
            print("Not using Newton iterations")
            space["max_newton"] = None
            space["newton_tol"] = None
        else:
            space["max_newton"] = hp.uniformint("max_newton", min_newton, max_newton)
            space["newton_tol"] = hp.uniform(
                "newton_tol", min_newton_tol, max_newton_tol
            )
        if None in [min_krylov, max_krylov, min_krylov_tol, max_krylov_tol]:
            print("Not using Krylov/CR iterations")
            space["max_cr"] = None
            space["cr_tol"] = None
        else:
            space["max_cr"] = hp.uniformint("max_cr", min_krylov, max_krylov)
            space["cr_tol"] = hp.uniform("cr_tol", min_krylov_tol, max_krylov_tol)
        if None in [min_momentum, max_momentum]:
            print("Not using momentums")
            space["momentum"] = None
        else:
            space["momentum"] = hp.uniform("momentum", min_momentum, max_momentum)

        if None in [
            min_extrapolation_factor,
            max_extrapolation_factor,
            min_sufficient_decrease,
            max_sufficient_decrease,
            min_max_searches,
            max_max_searches,
        ]:
            print("Not using a line search")
            space["extrapolation_factor"] = None
            space["sufficient_decrease"] = None
            space["curvature_constant"] = None
            space["max_searches"] = None
        else:
            space["extrapolation_factor"] = hp.uniform(
                "extrapolation_factor",
                min_extrapolation_factor,
                max_extrapolation_factor,
            )
            space["sufficient_decrease"] = hp.uniform(
                "sufficient_decrease", min_sufficient_decrease, max_sufficient_decrease
            )
            if None not in (max_curvature_constant, min_curvature_constant):
                space["curvature_constant"] = hp.uniform(
                    "curvature_constant", min_curvature_constant, max_curvature_constant
                )
            else:
                space["curvature_constant"] = None

            space["max_searches"] = hp.uniformint(
                "max_searches", min_max_searches, max_max_searches
            )

        if None in (min_quasi_newton_memory, max_quasi_newton_memory):
            print("Not limiting memory if quasi-newton method")
            space["quasi_newton_memory"] = None
        else:
            space["quasi_newton_memory"] = hp.uniformint(
                "quasi_newton_memory", min_quasi_newton_memory, max_quasi_newton_memory
            )

        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            rstate=np.random.default_rng(seed),
        )

        for key in ("max_cr", "max_newton", "max_searches", "quasi_newton_memory"):
            try:
                best[key] = int(best[key])
            except KeyError:
                # no key present, wasn't used in the optimizer
                pass

        if not second_order:
            steps = execute_steps(
                func,
                initial_state,
                optimizer_class,
                # {'lr': best['lr']},
                best,
                num_iter=200,
            )
        else:
            steps = execute_steps_sso(
                func,
                initial_state,
                optimizer_class,
                # {'lr': best['lr']},
                best,
                num_iter=200,
            )

        plot_func(steps, optimizer_class.__name__, best)


def main():
    # fmt: off
    optimizers = [
        # baselines
        # ExptSpec(VizAdam,                False, -8, 0.5, None, None, None, None, None, None, None, None, None, None),
        # ExptSpec(VizSGD,                 False, -8., -1.0, None, None, None, None, None, None, None, None, 0.0, 1.0),
        # ExptSpec(VizHFCR_Newton,         True, -8, 1.0, 1, 50, 1e-6, 0.1, None, None, None, None, None, None),
        # ExptSpec(VizFletcherReeves,      True, -8, 1., 1, 50, 1e-6, 0.1, None, None, None, None, None, None ),
        # ExptSpec(VizHestenesStiefel,     True, -8, 1., 1, 50, 1e-6, 0.1, None, None, None, None, None, None ),
        # ExptSpec(VizPolakRebiere,        True, -8, 1., 1, 50, 1e-6, 0.1, None, None, None, None, None, None ),
        # ExptSpec(VizDaiYuan,             True, -8, 1., 1, 50, 1e-6, 0.1, None, None, None, None, None, None ),
        # ExptSpec(VizSymmetricRankOne,    True, -8., 1.0, 1, 50, 1e-6, 0.1, 1, 50, 1e-6, 1., None, None)
        # ExptSpec(VizFletcherReeves,       True, -8, 1., 1, 50, 1e-6, 0.1, None, None, None, None, None, None, None, None, None, None, None, None, None, None ),
        # ExptSpec(VizFletcherReevesLineSearch, True, -1.0, 0.0, 1, 50, 1e-6, 0.1, None, None, None, None, None, None, 1.1, 10.0, 1e-6, 0.99, 1e-6, 0.99, 1, 10),
        ExptSpec(optimizer=VizSymmetricRankOneLineSearch, second_order=True, min_lr=0.1, max_lr=2.0, min_newton=1, max_newton=2, min_newton_tol=1e-6, max_newton_tol=1e-2, min_krylov=1, max_krylov=50, min_krylov_tol=1e-6, max_krylov_tol=1e-2, min_momentum=None, max_momentum=None, min_extrapolation_factor=1e-2, max_extrapolation_factor=0.999, min_max_searches=1, max_max_searches=50, min_sufficient_decrease=1e-6, max_sufficient_decrease=1e-2, min_curvature_constant=None, max_curvature_constant=None, min_quasi_newton_memory=1, max_quasi_newton_memory=10),
        ExptSpec(optimizer=VizSymmetricRankOneDualLineSearch, second_order=True, min_lr=0.1, max_lr=2.0, min_newton=1, max_newton=2, min_newton_tol=1e-6, max_newton_tol=1e-2, min_krylov=1, max_krylov=50, min_krylov_tol=1e-6, max_krylov_tol=1e-2, min_momentum=None, max_momentum=None, min_extrapolation_factor=1e-2, max_extrapolation_factor=0.999, min_max_searches=1, max_max_searches=50, min_sufficient_decrease=1e-6, max_sufficient_decrease=1e-2, min_curvature_constant=None, max_curvature_constant=None, min_quasi_newton_memory=1, max_quasi_newton_memory=10),
        ExptSpec(optimizer=VizDFPLineSearch, second_order=True, min_lr=0.1, max_lr=2.0, min_newton=1, max_newton=2, min_newton_tol=1e-6, max_newton_tol=1e-2, min_krylov=1, max_krylov=50, min_krylov_tol=1e-6, max_krylov_tol=1e-2, min_momentum=None, max_momentum=None, min_extrapolation_factor=1e-2, max_extrapolation_factor=0.999, min_max_searches=1, max_max_searches=50, min_sufficient_decrease=1e-6, max_sufficient_decrease=1e-2, min_curvature_constant=None, max_curvature_constant=None, min_quasi_newton_memory=1, max_quasi_newton_memory=10),
        ExptSpec(optimizer=VizDFPInverseLineSearch, second_order=True, min_lr=0.1, max_lr=2.0, min_newton=1, max_newton=2, min_newton_tol=1e-6, max_newton_tol=1e-2, min_krylov=1, max_krylov=50, min_krylov_tol=1e-6, max_krylov_tol=1e-2, min_momentum=None, max_momentum=None, min_extrapolation_factor=1e-2, max_extrapolation_factor=0.999, min_max_searches=1, max_max_searches=50, min_sufficient_decrease=1e-6, max_sufficient_decrease=1e-2, min_curvature_constant=None, max_curvature_constant=None, min_quasi_newton_memory=1, max_quasi_newton_memory=10),
        ExptSpec(optimizer=VizBFGSLineSearch, second_order=True, min_lr=0.1, max_lr=2.0, min_newton=1, max_newton=2, min_newton_tol=1e-6, max_newton_tol=1e-2, min_krylov=1, max_krylov=50, min_krylov_tol=1e-6, max_krylov_tol=1e-2, min_momentum=None, max_momentum=None, min_extrapolation_factor=1e-2, max_extrapolation_factor=0.999, min_max_searches=1, max_max_searches=50, min_sufficient_decrease=1e-6, max_sufficient_decrease=1e-2, min_curvature_constant=None, max_curvature_constant=None, min_quasi_newton_memory=1, max_quasi_newton_memory=10),

        ExptSpec(optimizer=VizFletcherReevesLineSearch, second_order=True, min_lr=0.1, max_lr=2.0, min_newton=1, max_newton=2, min_newton_tol=1e-6, max_newton_tol=1e-2, min_krylov=None, max_krylov=None, min_krylov_tol=None, max_krylov_tol=None, min_momentum=None, max_momentum=None, min_extrapolation_factor=1e-2, max_extrapolation_factor=0.999, min_max_searches=1, max_max_searches=50, min_sufficient_decrease=1e-6, max_sufficient_decrease=1e-2, min_curvature_constant=None, max_curvature_constant=None, min_quasi_newton_memory=None, max_quasi_newton_memory=None),
        ExptSpec(optimizer=VizDaiYuanLineSearch, second_order=True, min_lr=0.1, max_lr=2.0, min_newton=1, max_newton=2, min_newton_tol=1e-6, max_newton_tol=1e-2, min_krylov=None, max_krylov=None, min_krylov_tol=None, max_krylov_tol=None, min_momentum=None, max_momentum=None, min_extrapolation_factor=1e-2, max_extrapolation_factor=0.999, min_max_searches=1, max_max_searches=50, min_sufficient_decrease=1e-6, max_sufficient_decrease=1e-2, min_curvature_constant=None, max_curvature_constant=None, min_quasi_newton_memory=None, max_quasi_newton_memory=None),
        ExptSpec(optimizer=VizHestenesStiefelLineSearch, second_order=True, min_lr=0.1, max_lr=2.0, min_newton=1, max_newton=2, min_newton_tol=1e-6, max_newton_tol=1e-2, min_krylov=None, max_krylov=None, min_krylov_tol=None, max_krylov_tol=None, min_momentum=None, max_momentum=None, min_extrapolation_factor=1e-2, max_extrapolation_factor=0.999, min_max_searches=1, max_max_searches=50, min_sufficient_decrease=1e-6, max_sufficient_decrease=1e-2, min_curvature_constant=None, max_curvature_constant=None, min_quasi_newton_memory=None, max_quasi_newton_memory=None),
        ExptSpec(optimizer=VizPolakRebiereLineSearch, second_order=True, min_lr=0.1, max_lr=2.0, min_newton=1, max_newton=2, min_newton_tol=1e-6, max_newton_tol=1e-2, min_krylov=None, max_krylov=None, min_krylov_tol=None, max_krylov_tol=None, min_momentum=None, max_momentum=None, min_extrapolation_factor=1e-2, max_extrapolation_factor=0.999, min_max_searches=1, max_max_searches=50, min_sufficient_decrease=1e-6, max_sufficient_decrease=1e-2, min_curvature_constant=None, max_curvature_constant=None, min_quasi_newton_memory=None, max_quasi_newton_memory=None),
    ]
    # fmt: on

    Path("./imgs/").mkdir(exist_ok=True)

    execute_experiments(
        optimizers,
        objective_sphere,
        sphere,
        plot_sphere,
        (-2.0, 2.0),
    )

    execute_experiments(
        optimizers,
        objective_rastrigin,
        rastrigin,
        plot_rastrigin,
        (-2.0, 3.5),
    )

    execute_experiments(
        optimizers,
        objective_rosenbrok,
        rosenbrock,
        plot_rosenbrok,
        (-2.0, 2.0),
    )

    """
    execute_experiments(
        optimizers,
        objective_easom,
        easom,
        plot_easom,
        (-2.0, 2.0),
    )
    """


if __name__ == "__main__":
    main()
