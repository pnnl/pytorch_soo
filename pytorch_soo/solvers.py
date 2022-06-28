"""
A collection of linear system solvers for use in our optimization routines.
"""
import abc
import torch
from torch import Tensor
from pytorch_soo.matrix_free_operators import MatrixFreeOperator


class Solver(abc.ABC):
    """
    An iterative solver for problems of the form Ax=b where A may be a matrix-free
    operator.

    args:
        - max_iter: The maximum iterations the solver can attempt per call
        - tolerance: the tolerance for determining convergence
    """

    def __init__(self, max_iter: int, tolerance: float) -> None:
        self.max_iter = max_iter
        self.tol = tolerance

    def __call__(self, A: MatrixFreeOperator, x: Tensor, b: Tensor) -> Tensor:
        return self.solve(A, x, b)

    @abc.abstractmethod
    def solve(self, A: MatrixFreeOperator, x0: Tensor, b: Tensor) -> Tensor:
        """
        The base solve function.
        args:
            A: A matrix or matrix-free operator that supports matrix multiplicaton
            x0, b: the vectors to solve for.
        """


class ConjugateGradient(Solver):
    def solve(self, A: MatrixFreeOperator, x0: Tensor, b: Tensor) -> Tensor:
        rk = b - (A * x0)
        if torch.norm(rk).item() <= self.tol:
            return x0

        xk = x0.clone()
        pk = rk.clone()
        rk_inner = torch.dot(rk, rk)
        for _ in range(self.max_iter):
            Apk = A * pk
            alpha = torch.div(rk_inner, torch.dot(pk, Apk))
            xk = torch.add(xk, pk, alpha=alpha)
            rk = torch.sub(rk, Apk, alpha=alpha)

            rk_inner_new = torch.dot(rk, rk)
            if rk_inner_new.item() <= self.tol:
                return xk

            beta = torch.div(rk_inner_new, rk_inner)
            pk = torch.add(rk, pk, alpha=beta)
            rk_inner = rk_inner_new

        return xk


class ConjugateResidual(Solver):
    def solve(self, A: MatrixFreeOperator, x0: Tensor, b: Tensor) -> Tensor:
        rk = b - A * x0
        xk = x0
        if torch.norm(rk) <= self.tol:
            return x0
        pk = torch.clone(rk)
        Apk = A * pk
        for _ in range(self.max_iter):
            alpha = torch.div(torch.dot(rk, A * rk), torch.dot(Apk, Apk))
            xk1 = torch.add(xk, pk, alpha=alpha)
            rk1 = torch.sub(rk, Apk, alpha=alpha)
            if torch.norm(rk1) <= self.tol:
                xk = xk1
                break
            beta = torch.div(torch.dot(rk1, A * rk1), torch.dot(rk, A * rk))
            pk1 = torch.add(rk1, pk, alpha=beta)
            Apk = torch.add(A * rk1, Apk, alpha=beta)
            xk = xk1
            pk = pk1
            rk = rk1

        return xk
