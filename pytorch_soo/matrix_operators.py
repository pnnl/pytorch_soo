"""
Mostly for debugging, full matrix operators.
"""
import abc
from typing import Optional
from warnings import warn
import torch
from torch import Tensor


class MatrixOperator(abc.ABC):
    """
    Base class to create Hessians (matrix operators) for use in the
    """

    def __init__(self, B0: Tensor, n: Optional[int] = None) -> None:
        self.B0 = B0
        self.matrix = torch.clone(B0)
        self.n = n
        if self.n is not None:
            err = "Limited memory matrix operators not implemented, using full history!"
            warn(err, RuntimeWarning)

    def reset(self) -> None:
        self.matrix = torch.clone(self.B0)

    def multiply(self, x: Tensor) -> Tensor:
        return torch.matmul(self.matrix, x)

    def __mul__(self, x: Tensor) -> Tensor:
        return self.multiply(x)

    def __rmul__(self, x: Tensor) -> None:
        return torch.matmul(x, self.matrix)

    @abc.abstractmethod
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        """
        the quasi-newton update
        """


class Broyden(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        self.matrix += torch.outer(
            (yk - self.multiply(delta_x)) / torch.dot(delta_x, delta_x), delta_x
        )


class BroydenInverse(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        num = (delta_x - self.multiply(yk)) @ delta_x @ self.matrix
        den = delta_x @ self.matrix @ yk
        self.matrix += num / den


class SymmetricRankOne(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        Bk_delta_x = self.multiply(delta_x)
        vk = yk - Bk_delta_x
        num = torch.outer(vk, vk)
        den = torch.dot(vk, delta_x)
        if torch.abs(delta_x @ vk) < 1e-6 * torch.norm(delta_x) * torch.norm(vk):
            # Don't apply the update
            return

        self.matrix += num / den


class SymmetricRankOneInverse(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        tmp = delta_x - torch.matmul(self.matrix, yk)
        num = torch.matmul(tmp, tmp.T)
        den = torch.dot(tmp, yk)

        self.matrix += num / den


class DavidonFletcherPowell(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        dim = grad_f_k.shape[0]
        I = torch.eye(dim)
        num1 = torch.matmul(yk, delta_x.T)
        num2 = torch.matmul(delta_x, yk.T)
        den = torch.dot(yk, delta_x)
        left = I - num1 / den
        right = I - num2 / den
        product = torch.matmul(torch.matmul(left, self.matrix), right)
        summand = torch.matmul(yk, yk.T) / den

        self.matrix += product + summand


class DavidonFletcherPowellInverse(MatrixOperator):
    def update(
        self, grad_f_k: Tensor, grad_f_k_plus_one: Tensor, delta_x: Tensor
    ) -> None:
        yk = grad_f_k_plus_one - grad_f_k
        num1 = torch.matmul(delta_x, delta_x.T)
        den1 = torch.dot(delta_x, yk)
        num2 = torch.matmul(
            torch.matmul(torch.matmul(self.matrix, yk), yk.T), self.matrix
        )
        den2 = torch.matmul(torch.matmul(yk.T, self.matrix), yk)

        self.matrix += num1 / den1 - num2 / den2
