"""
A collection of Matrix-Free operators that act as Hessians in Quasi-Newton
methods.
"""
from typing import Callable, Optional
from collections import deque

import torch
from torch import Tensor

# A constant used to check if the denominator is too small for an update
_R = 1e-8


class NonCommutativeOperatorError(RuntimeError):
    """
    A custom exception for the Matrix Free operators to indicate that
    "left" multiplication (p^TA vs. Ap) is not defined.
    """


class MatrixFreeOperator:
    """
    A matrix free operator for direct problems
    args:
        - B0p: A callable that accepts a tensor and returns a tensor.
            Alternatively, update this later if required with the change_B0p method
        - n: The size of the memory. Setting it to None will make this a
            full-memory model; an integer retains the n most recent updates
    """

    def __init__(
        self, B0p: Callable[[Tensor], Tensor], n: Optional[int] = None
    ) -> None:
        self.B0p = B0p
        if n is None:
            self.memory = []
        else:
            self.memory = deque(maxlen=n)

    def change_B0p(self, B0p: Callable[[Tensor, Callable[[], float]], Tensor]) -> None:
        """
        A hack, basically...B0p relies on a closure, but that closure
        is handed in to the "step" method of the optimizer. Either we need
        to significantly redesign the interface...or we'll just use this.
        """
        self.B0p = B0p

    def reset(self) -> None:
        """
        Convenience function to reset the state of the operator
        """
        self.memory.clear()

    def update(self, grad_fk_, grad_fk_plus_one_, delta_x_) -> None:
        """
        Given relevant vectors, update the memory of this object
        """
        raise NotImplementedError("Update method must be provided in base class!")

    def multiply(self, p: Tensor) -> Tensor:
        """
        Implement the actual matrix vector multiplication
        """
        raise NotImplementedError("Multiply method must be provided in base class!")

    def __mul__(self, p: Tensor) -> Tensor:
        return self.multiply(p)

    def __rmul__(self, p: Tensor) -> None:
        raise NonCommutativeOperatorError(
            "Matrix free methods don't commute! In the case of something like q^T*A*p, try q^T(A*p)"
        )

    def construct_full_matrix(
        self,
        m: int,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        These are all matrix-vector approximations. We can "rehydrate" the underlying matrix by
        multiplying it times the standard basis vectors and "stacking" the results.

        Note that this is really for debugging/exploratory purposes, and very impractical
        or impossible for large systems (recall the O(n^2) memory requirements!).
        """
        H = torch.zeros((m, m), dtype=dtype, device=device)
        mat_device = self.memory[0][0].device
        for i in range(m):
            basis_i = torch.zeros((m,), dtype=dtype)
            basis_i[i] = 1.0
            basis_i = basis_i.to(mat_device)
            col = self.multiply(basis_i).to(device)
            H[:, i] = col

        return H

    def determinant(
        self,
        m: int,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
    ) -> float:
        """
        Construct the full matrix and calculate its determinant
        """
        mat = self.construct_full_matrix(m, dtype=dtype, device=device)
        det = torch.linalg.det(mat)

        return det


class InverseMatrixFreeOperator:
    """
    A matrix free operator for inverse problems
    """

    def __init__(self, n: Optional[int] = None) -> None:
        if n is None:
            self.memory = []
        else:
            self.memory = deque(maxlen=n)

    def reset(self) -> None:
        """
        Convenience function to reset the state of the operator
        """
        self.memory.clear()

    def update(self, grad_fk_, grad_fk_plus_one_, delta_x_) -> None:
        """
        Given relevant vectors, update the memory of this object
        """
        raise NotImplementedError("Update method must be provided in base class!")

    def multiply(self, p: Tensor) -> Tensor:
        """
        Implement the actual matrix vector multiplication
        """
        raise NotImplementedError("Multiply method must be provided in base class!")

    def __mul__(self, p: Tensor) -> Tensor:
        return self.multiply(p)

    def __rmul__(self, p: Tensor) -> None:
        raise NonCommutativeOperatorError("Matrix free methods don't commute!")


class SymmetricRankOne(MatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_f_k = grad_f_k_.detach().clone()
        grad_f_k_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        yk = grad_f_k_plus_one - grad_f_k
        Bk_delta_x = self.multiply(delta_x)
        vk = yk - Bk_delta_x
        den = torch.dot(vk, delta_x)
        if torch.abs(delta_x @ vk) < 1e-6 * torch.norm(delta_x) * torch.norm(vk):
            # Don't apply the update
            return

        self.memory.append((vk.clone(), den.clone()))

    def multiply(self, p: Tensor) -> Tensor:
        Bp = self.B0p(p.clone())
        for vk, den in self.memory:
            Bp += (vk * (vk @ p)) / den

        return Bp


class SymmetricRankOneInverse(InverseMatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        yk = grad_fk_plus_one - grad_fk
        vk = delta_x - self.multiply(yk)
        den = torch.dot(vk, yk)
        self.memory.append((vk, den))

    def multiply(self, p: Tensor) -> Tensor:
        """
        Assumes Hk0 = I
        """
        # TODO evaulate alternate Hk0 choices (Nocedal & Wright have a suggestion)
        Hkp = p.clone()
        for vk, den in self.memory:
            Hkp += vk * torch.dot(vk, p) / den

        return Hkp


class BFGS(MatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        y = grad_fk_plus_one - grad_fk
        B_delta_x = self.multiply(delta_x)
        den1 = torch.dot(y, delta_x)
        den2 = torch.dot(delta_x, B_delta_x)
        self.memory.append((y, B_delta_x, den1, den2))

    def multiply(self, p: Tensor) -> Tensor:
        Bp = self.B0p(p.clone())
        for y, B_delta_x, den1, den2 in self.memory:
            Bp += ((y * torch.dot(y, p)) / den1) + (
                (B_delta_x * torch.dot(B_delta_x, p)) / den2
            )

        return Bp


class BFGSInverse(InverseMatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        y = grad_fk_plus_one - grad_fk
        Hy = self.multiply(y)
        den = torch.dot(y, delta_x)
        self.memory.append((y, Hy, delta_x, den))

    def multiply(self, p: Tensor) -> Tensor:
        Hp = p.clone()
        for y, Hy, delta_x, den in self.memory:
            term1 = -(delta_x * torch.dot(y, Hp)) / den
            term2 = -(Hy * torch.dot(delta_x, p)) / den
            term3 = (delta_x * torch.dot(y, Hy) * torch.dot(delta_x, p)) / (den**2)
            term4 = (delta_x * torch.dot(delta_x, p)) / den

            Hp += term1 + term2 + term3 + term4

        return Hp


class DavidonFletcherPowell(MatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        y = grad_fk_plus_one - grad_fk
        B_delta_x = self.multiply(delta_x)
        self.memory.append((B_delta_x, delta_x, y))

    def multiply(self, p: Tensor) -> Tensor:
        Bp = self.B0p(p.clone())
        for B_delta_x, delta_x, y in self.memory:
            den = torch.dot(y, delta_x)
            yTp = torch.dot(y, p)
            B_deltax_yTp = torch.mul(B_delta_x, yTp)

            num2 = torch.mul(y, torch.dot(delta_x, Bp))
            num3 = torch.mul(y, yTp)
            num4 = torch.dot(y, torch.mul(delta_x, B_deltax_yTp))

            num1_sum = torch.add(torch.add(B_deltax_yTp, num2), num3)
            term1 = torch.div(num1_sum, den)
            term2 = torch.div(num4, torch.mul(den, den))
            Bp = torch.add(Bp, torch.add(term1, term2))

        return Bp


class DavidonFletcherPowellInverse(InverseMatrixFreeOperator):
    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        y = grad_fk_plus_one - grad_fk
        Hy = self.multiply(y)
        den1 = torch.dot(delta_x, y)
        den2 = torch.dot(y, Hy)
        self.memory.append((y, Hy, delta_x, den1, den2))

    def multiply(self, p: Tensor) -> Tensor:
        Hkp = p.clone()
        for y, Hy, delta_x, den1, den2 in self.memory:
            Hkp += (delta_x * torch.dot(delta_x, p) / den1) - (
                Hy * torch.dot(y, Hkp) / den2
            )

        return Hkp


class Broyden(MatrixFreeOperator):
    """
    https://en.wikipedia.org/wiki/Broyden%27s_method
    """

    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        yk = grad_fk_plus_one - grad_fk
        vk = yk - self.multiply(delta_x)
        left = vk / torch.dot(delta_x, delta_x)
        self.memory.append((left, delta_x))

    def multiply(self, p: Tensor) -> Tensor:
        Bkp = self.B0p(p.clone())
        for left_k, delta_xk in self.memory:
            Bkp += left_k * torch.dot(delta_xk, p)

        return Bkp


class BroydenInverse(InverseMatrixFreeOperator):
    """
    https://en.wikipedia.org/wiki/Broyden%27s_method
    """

    def update(
        self,
        grad_f_k_: Tensor,
        grad_f_k_plus_one_: Tensor,
        delta_x_: Tensor,
    ) -> None:
        grad_fk = grad_f_k_.detach().clone()
        grad_fk_plus_one = grad_f_k_plus_one_.detach().clone()
        delta_x = delta_x_.detach().clone()
        yk = grad_fk_plus_one - grad_fk
        vk = delta_x - self.multiply(yk)
        den = torch.dot(delta_x, self.multiply(yk))
        self.memory.append((vk, delta_x, den))

    def multiply(self, p: Tensor) -> Tensor:
        Hkp = p.clone()
        for vk, delta_x, den in self.memory:
            Hkp += vk * torch.dot(delta_x, Hkp) / den

        return Hkp
