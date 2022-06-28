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
from pytorch_sso.matrix_free_operators import BFGSInverse
from pytorch_sso.optim_hfcr_newton import HFCR_Newton
from pytorch_sso.nonlinear_conjugate_gradient import (
    FletcherReeves,
    NonlinearConjugateGradient,
    PolakRibiere,
    HestenesStiefel,
    DaiYuan,
)
from pytorch_sso.line_search_spec import LineSearchSpec
from pytorch_sso.trust_region import TrustRegionSpec
from pytorch_sso.quasi_newton import (
    DavidonFletcherPowell,
    DavidonFletcherPowellInverse,
    SymmetricRankOne,
    SymmetricRankOneInverse,
    Broyden,
    BrodyenInverse,
    BFGS,
    BFGSInverse,
    SymmetricRankOneTrust,
    SymmetricRankOneDualTrust,
)

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

    min_extrapolation_factor: float = 2.0
    max_extrapolation_factor: float = 2.0
    min_sufficient_decrease: float = 1.0e-4
    max_sufficient_decrease: float = 1.0e-4
    min_curvature_constant: float = 0.9
    max_curvature_constant: float = 0.9
    min_max_searches: int = 10
    max_max_searches: int = 10


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
            abs_newton_tol=newton_tol,
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
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            max_krylov=int(max_cr),
            abs_newton_tol=newton_tol,
            krylov_tol=cr_tol,
            matrix_free_memory=2,
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
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=newton_tol,
            matrix_free_memory=2,
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
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            max_krylov=int(max_cr),
            abs_newton_tol=newton_tol,
            krylov_tol=cr_tol,
            matrix_free_memory=2,
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
        **kwargs,
    ):
        max_cr = int(max_cr)
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=newton_tol,
            matrix_free_memory=2,
        )
        _ = kwargs


class VizFletcherReeves(FletcherReeves):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=20,
        newton_tol=1.0e-3,
        viz_steps=True,
        **kwargs,
    ):
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            abs_newton_tol=newton_tol,
            viz_steps=viz_steps,
        )
        _ = kwargs


class VizPolakRebiere(PolakRibiere):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=20,
        newton_tol=1.0e-3,
        viz_steps=True,
        **kwargs,
    ):
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            abs_newton_tol=newton_tol,
            viz_steps=viz_steps,
        )
        _ = kwargs


class VizHestenesStiefel(HestenesStiefel):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=20,
        newton_tol=1.0e-3,
        viz_steps=True,
        **kwargs,
    ):
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            abs_newton_tol=newton_tol,
            viz_steps=viz_steps,
        )
        _ = kwargs


class VizDaiYuan(DaiYuan):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=20,
        newton_tol=1.0e-3,
        viz_steps=True,
        **kwargs,
    ):
        max_newton = int(max_newton)
        super().__init__(
            params,
            lr=lr,
            max_newton=int(max_newton),
            abs_newton_tol=newton_tol,
            viz_steps=viz_steps,
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
        extrapolation_factor=2.0,
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


class VizSymmetricRankOneTrust(SymmetricRankOneTrust):
    def __init__(
        self,
        params,
        newton_tol=1e-3,
        memory=5,
        initial_radius=1.0,
        max_radius=1000,
        nabla0=0.0,
        nabla1=0.25,
        nabla2=0.75,
        shrink_factor=0.25,
        growth_factor=2.0,
        **kwargs,
    ):
        trust_region_spec = TrustRegionSpec(
            initial_radius=initial_radius,
            max_radius=max_radius,
            nabla0=nabla0,
            nabla1=nabla1,
            nabla2=nabla2,
            shrink_factor=shrink_factor,
            growth_factor=growth_factor,
            trust_region_subproblem_iter=100,
        )
        super().__init__(
            params,
            max_newton=1,
            abs_newton_tol=newton_tol,
            trust_region=trust_region_spec,
            matrix_free_memory=memory,
        )
        _ = kwargs


class VizSymmetricRankOneDualTrust(SymmetricRankOneDualTrust):
    def __init__(
        self,
        params,
        newton_tol=1e-3,
        memory=5,
        initial_radius=1.0,
        max_radius=1000,
        nabla0=0.0,
        nabla1=0.25,
        nabla2=0.75,
        shrink_factor=0.25,
        growth_factor=2.0,
        **kwargs,
    ):
        trust_region_spec = TrustRegionSpec(
            initial_radius=initial_radius,
            max_radius=max_radius,
            nabla0=nabla0,
            nabla1=nabla1,
            nabla2=nabla2,
            shrink_factor=shrink_factor,
            growth_factor=growth_factor,
            trust_region_subproblem_iter=100,
        )
        super().__init__(
            params,
            max_newton=1,
            abs_newton_tol=newton_tol,
            trust_region=trust_region_spec,
            matrix_free_memory=memory,
        )
        _ = kwargs


class VizBFGSLineSearch(BFGS):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=10,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        extrapolation_factor=2.0,
        sufficient_decrease=1.0e-4,
        curvature_constant=0.9,
        max_searches=10,
        memory=10,
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
            matrix_free_memory=memory,
        )
        _ = kwargs


class VizBFGSInverseLineSearch(BFGSInverse):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=10,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        extrapolation_factor=2.0,
        sufficient_decrease=1.0e-4,
        curvature_constant=0.9,
        max_searches=10,
        memory=10,
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
            matrix_free_memory=memory,
        )
        _ = kwargs


class VizDFPLineSearch(DavidonFletcherPowell):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=10,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        extrapolation_factor=2.0,
        sufficient_decrease=1.0e-4,
        curvature_constant=0.9,
        max_searches=10,
        memory=10,
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
            matrix_free_memory=memory,
        )
        _ = kwargs


class VizDFPInverseLineSearch(DavidonFletcherPowellInverse):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=10,
        max_cr=10,
        newton_tol=1e-3,
        cr_tol=1e-3,
        extrapolation_factor=2.0,
        sufficient_decrease=1.0e-4,
        curvature_constant=0.9,
        max_searches=10,
        memory=10,
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
            matrix_free_memory=memory,
        )
        _ = kwargs


class VizFletcherReevesLineSearch(FletcherReeves):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=20,
        newton_tol=1.0e-3,
        extrapolation_factor=2.0,
        sufficient_decrease=1.0e-4,
        curvature_constant=0.9,
        max_searches=10,
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


class VizPolakRibiereLineSearch(PolakRibiere):
    def __init__(
        self,
        params,
        lr=0.3333,
        max_newton=20,
        newton_tol=1.0e-3,
        extrapolation_factor=2.0,
        sufficient_decrease=1.0e-4,
        curvature_constant=0.9,
        max_searches=10,
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
        lr=0.3333,
        max_newton=20,
        newton_tol=1.0e-3,
        extrapolation_factor=2.0,
        sufficient_decrease=1.0e-4,
        curvature_constant=0.9,
        max_searches=10,
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
        lr=0.3333,
        max_newton=20,
        newton_tol=1.0e-3,
        extrapolation_factor=2.0,
        sufficient_decrease=1.0e-4,
        curvature_constant=0.9,
        max_searches=10,
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
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rastrigin(tensor, lib=torch):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    A = 10
    f = (
        A * 2
        + (x**2 - A * lib.cos(x * math.pi * 2))
        + (y**2 - A * lib.cos(y * math.pi * 2))
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
    f = x**2 + y**2

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


def execute_steps_nlcg(
    func, initial_state, optimizer_class, optimizer_config, num_iter=500
):
    x = torch.Tensor(initial_state).requires_grad_(True)
    max_newton = optimizer_config["max_newton"]
    steps = np.zeros((2, (num_iter + 1) * max_newton))
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
        _, nlcg_steps = optimizer.step(closure)
        for j, j_step in zip(range(i, i + max_newton), nlcg_steps):
            # probably a clever slicing way to do this, oh well
            steps[:, j] = j_step.detach().numpy()
            if torch.isnan(j_step).any():
                print(f"Broke on iteration {i}, {j}!")
                steps[:, j:] = np.empty_like(steps[:, j:]) * np.nan
                try:
                    print(f"Last step: {steps[:, j-1]}")
                except:
                    pass
                break

    return steps


def plot_all(
    rastrigin_grad_iter: Iterable,
    rosenbrock_grad_iter: Iterable,
    sphere_grad_iter: Iterable,
    optimizer_name: str,
):

    # sphere
    x = np.linspace(-4.5, 4.5, 250)
    y = np.linspace(-4.5, 4.5, 250)
    minimum = (0, 0)

    X, Y = np.meshgrid(x, y)
    Z = sphere([X, Y])

    iter_x, iter_y = sphere_grad_iter[0, :], sphere_grad_iter[1, :]

    plt.figure(figsize=(8, 8))
    plt.contour(X, Y, Z, 20, cmap="jet")
    plt.plot(iter_x, iter_y, color="r", marker="x")
    plt.ylim((y.min(), y.max()))
    plt.xlim((x.min(), x.max()))
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("imgs/{}_sphere.png".format(optimizer_name))

    # rastrigin
    x = np.linspace(-4.5, 4.5, 250)
    y = np.linspace(-4.5, 4.5, 250)
    minimum = (0, 0)

    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y], lib=np)

    iter_x, iter_y = rastrigin_grad_iter[0, :], rastrigin_grad_iter[1, :]

    plt.figure(figsize=(8, 8))
    plt.contour(X, Y, Z, 20, cmap="jet")
    plt.plot(iter_x, iter_y, color="r", marker="x")
    plt.ylim((y.min(), y.max()))
    plt.xlim((x.min(), x.max()))
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("imgs/{}_rastrigin.png".format(optimizer_name))

    # rosenbrock
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)

    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])

    iter_x, iter_y = rosenbrock_grad_iter[0, :], rosenbrock_grad_iter[1, :]

    plt.figure(figsize=(8, 8))
    plt.contour(X, Y, Z, 90, cmap="jet")
    plt.plot(iter_x, iter_y, color="r", marker="x")
    plt.ylim((y.min(), y.max()))
    plt.xlim((x.min(), x.max()))
    plt.plot(*minimum, "gD")
    plt.plot(iter_x[-1], iter_y[-1], "rD")
    plt.savefig("imgs/{}_rosenbrock.png".format(optimizer_name))


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


def main():
    Path("./imgs/").mkdir(exist_ok=True)
    funcs = ("sphere", "rosenbrock", "rastrigin")
    # funcs = ("rosenbrock",)
    opt_param_tuples = []
    opt_param_tuples.append(
        (
            VizSGD,
            {
                "lr": 0.001,
                "optimizer_class": "SGD",
                "momentum": 0.9,
            },
            200,
        )
    )
    opt_param_tuples.append(
        (
            VizAdam,
            {
                "lr": 1.0,
                "optimizer_class": "ADAM",
                "momentum": 0.9,
            },
            200,
        )
    )
    opt_param_tuples.append(
        (
            VizFletcherReevesLineSearch,
            {
                "lr": 1.0,
                "max_newton": 5,
                "newton_tol": 0.001,
                "extrapolation_factor": 0.5,
                "sufficient_decrease": 1e-4,
                "curvature_constant": None,
                "max_searches": 100,
                "optimizer_class": "Fletcher-Reeves",
            },
            10,
        )
    )
    opt_param_tuples.append(
        (
            VizDaiYuanLineSearch,
            {
                "lr": 2.0,
                "max_newton": 5,
                "newton_tol": 0.001,
                "extrapolation_factor": 0.1,
                "sufficient_decrease": 1e-4,
                "curvature_constant": None,
                "max_searches": 10,
                "optimizer_class": "Dai-Yuan",
            },
            10,
        )
    )
    opt_param_tuples.append(
        (
            VizPolakRibiereLineSearch,
            {
                "lr": 2.0,
                "max_newton": 5,
                "newton_tol": 0.001,
                "extrapolation_factor": 0.1,
                "sufficient_decrease": 1e-4,
                "curvature_constant": None,
                "max_searches": 10,
                "optimizer_class": "Polak-Ribiere",
            },
            10,
        )
    )
    opt_param_tuples.append(
        (
            VizHestenesStiefelLineSearch,
            {
                "lr": 2.0,
                "max_newton": 5,
                "newton_tol": 0.001,
                "extrapolation_factor": 0.1,
                "sufficient_decrease": 1e-4,
                "curvature_constant": None,
                "max_searches": 10,
                "optimizer_class": "Hestenes-Stiefel",
            },
            10,
        )
    )
    opt_param_tuples.append(
        (
            VizDFPInverseLineSearch,
            {
                "lr": 2.0,
                "max_newton": 1,
                "newton_tol": 0.001,
                "extrapolation_factor": 0.1,
                "sufficient_decrease": 1e-4,
                "curvature_constant": None,
                "max_searches": 10,
                "optimizer_class": "DFP Inverse",
            },
            100,
        )
    )
    opt_param_tuples.append(
        (
            VizDFPLineSearch,
            {
                "lr": 2.0,
                "max_newton": 1,
                "newton_tol": 0.001,
                "max_cr": 10,
                "cr_tol": 1e-3,
                "extrapolation_factor": 0.1,
                "sufficient_decrease": 1e-4,
                "curvature_constant": None,
                "max_searches": 10,
                "optimizer_class": "DFP",
            },
            100,
        )
    )
    opt_param_tuples.append(
        (
            VizHFCR_Newton,
            {
                "lr": 1.0,
                "max_newton": 1,
                "newton_tol": 0.001,
                "max_cr": 10,
                "cr_tol": 1e-3,
                "optimizer_class": "HFCR-Newton",
            },
            12,
        )
    )
    opt_param_tuples.append(
        (
            VizBFGSInverseLineSearch,
            {
                "lr": 1.0,
                "max_newton": 1,
                "newton_tol": 0.0001,
                "max_cr": 10,
                "cr_tol": 1e-3,
                "extrapolation_factor": 0.1,
                "sufficient_decrease": 1e-4,
                "curvature_constant": 0.9,
                "max_searches": 10,
                "memory": 5,
                "optimizer_class": "BFGSInverse",
            },
            20,
        )
    )
    opt_param_tuples.append(
        (
            VizBFGSLineSearch,
            {
                "lr": 1.0,
                "max_newton": 1,
                "newton_tol": 0.0001,
                "max_cr": 3,
                "cr_tol": 1e-3,
                "extrapolation_factor": 0.1,
                "sufficient_decrease": 1e-4,
                "curvature_constant": 0.9,
                "max_searches": 10,
                "memory": 5,
                "optimizer_class": "BFGS",
            },
            40,
        )
    )
    opt_param_tuples.append(
        (
            VizSymmetricRankOneTrust,
            {
                "newton_tol": 1e-3,
                "memory": 25,
                "initial_radius": 5.5,
                "max_radius": 10000.0,
                "nabla0": 1e-4,
                "nabla1": 0.25,
                "nabla2": 0.75,
                "shrink_factor": 0.5,
                "growth_factor": 4.0,
                "optimizer_class": "SR1 Trust",
            },
            500,
        )
    )
    opt_param_tuples.append(
        (
            VizSymmetricRankOneDualTrust,
            {
                "newton_tol": 1e-5,
                "memory": 5,
                "initial_radius": 5.0,
                "max_radius": 1000,
                "nabla0": 1e-4,
                "nabla1": 0.25,
                "nabla2": 0.75,
                "shrink_factor": 0.25,
                "growth_factor": 2.0,
                "optimizer_class": "SR1D Trust",
            },
            200,
        )
    )
    for optimizer_class, best, num_iter in opt_param_tuples:

        func = rosenbrock
        initial_state = (-2.0, 2.0)
        print("Optimizer:", optimizer_class.__name__)
        print("Rosenbrock:")
        if isinstance(optimizer_class, NonlinearConjugateGradient):
            rosenbrock_steps = execute_steps_nlcg(
                func,
                initial_state,
                optimizer_class,
                best,
                num_iter=num_iter,
            )
        else:
            rosenbrock_steps = execute_steps_sso(
                func,
                initial_state,
                optimizer_class,
                best,
                num_iter=num_iter,
            )

        func = sphere
        initial_state = (-2.0, 2.0)
        print("Sphere:")
        if isinstance(optimizer_class, NonlinearConjugateGradient):
            sphere_steps = execute_steps_nlcg(
                func,
                initial_state,
                optimizer_class,
                best,
                num_iter=num_iter,
            )
        else:
            sphere_steps = execute_steps_sso(
                func,
                initial_state,
                optimizer_class,
                best,
                num_iter=num_iter,
            )

        func = rastrigin
        initial_state = (-2.0, 3.5)
        initial_state = (-3.5, 4.0)
        print("Rastrigin:")
        if isinstance(optimizer_class, NonlinearConjugateGradient):
            rastrigin_steps = execute_steps_nlcg(
                func,
                initial_state,
                optimizer_class,
                best,
                num_iter=num_iter,
            )
        else:
            rastrigin_steps = execute_steps_sso(
                func,
                initial_state,
                optimizer_class,
                best,
                num_iter=num_iter,
            )

        plot_all(
            rastrigin_steps, rosenbrock_steps, sphere_steps, optimizer_class.__name__
        )

    return


if __name__ == "__main__":
    main()
