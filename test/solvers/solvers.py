import unittest
import requests

import numpy as np
from scipy.io import mmread
import torch
from scipy.stats import ortho_group
from torch._C import dtype
from pytorch_sso import solvers
from scipy.linalg import solve

MATRIX_SIZE = 100
ITERATIONS = 5000
SOLVER_TOLERANCE = 1e-6
TEST_TOLERANCE = 1e-3
MAX_SINGULAR_VALUE = 100


class Matrix:
    def __init__(self, A: torch.Tensor):
        self.A = A

    def __mul__(self, p: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.A, p)

    def __str__(self) -> str:
        return self.A.__str__()


def is_spd(M: torch.Tensor) -> bool:
    symmetric = torch.equal(M, M.T)
    if not symmetric:
        return False
    eigenvalues = torch.real(torch.linalg.eig(M).eigenvalues)
    if torch.any(torch.lt(eigenvalues, 0.0)):
        return False

    return True


def generate_sym_pd_matrix(size: int):
    passes = False
    while not passes:
        A = generate_sym_matrix(size)
        A = A + (size * torch.eye(size))
        passes = is_spd(A)

    return A


def is_invertible(M: torch.Tensor, tol: float = 1e-3) -> bool:
    det = torch.det(M)
    return torch.abs(det) >= tol


def generate_sym_matrix(size: int):
    A = torch.rand((size, size))
    A = 0.5 * (A + A.T)

    return A


def generate_sym_inv_matrix(size: int) -> torch.Tensor:
    while True:
        m = torch.Tensor(ortho_group.rvs(dim=MATRIX_SIZE))
        V = torch.matmul(m, m.T)
        eigenvalues = (
            2 * MAX_SINGULAR_VALUE * np.random.random_sample((MATRIX_SIZE,))
            - MAX_SINGULAR_VALUE
        )
        # Replace any ~=0 values
        eigenvalues[np.abs(eigenvalues) <= 1e-3] = MAX_SINGULAR_VALUE
        eigenvalues = torch.diag(torch.Tensor(eigenvalues))

        A = torch.matmul(V, torch.matmul(eigenvalues, V.T))
        if is_invertible(A):
            break

    return A


class TestConjugateGradient(unittest.TestCase):
    def test_identity(self):
        I = torch.diag(torch.ones((MATRIX_SIZE,)))
        A = Matrix(I)
        x = 100.0 * torch.rand((MATRIX_SIZE,))
        x0 = 100.0 * torch.rand((MATRIX_SIZE,))
        b = A * x

        solver = solvers.ConjugateGradient(ITERATIONS, SOLVER_TOLERANCE)
        x_prime = solver(A, x0, b)
        error = torch.norm(x_prime - x).item()
        self.assertLessEqual(error, TEST_TOLERANCE)

    def test_random_matrix(self):
        A = Matrix(generate_sym_pd_matrix(MATRIX_SIZE))
        x = 100.0 * torch.rand((MATRIX_SIZE,))
        x0 = 100.0 * torch.rand((MATRIX_SIZE,))
        b = A * x

        solver = solvers.ConjugateGradient(ITERATIONS, SOLVER_TOLERANCE)
        x_prime = solver(A, x0, b)
        error = torch.norm(x_prime - x).item()
        self.assertLessEqual(error, TEST_TOLERANCE)

    def test_power_system(self):
        solver = solvers.ConjugateGradient(10 * ITERATIONS, SOLVER_TOLERANCE)

        fname = "1138_bus.mtx.gz"
        url = "https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/psadmit/1138_bus.mtx.gz"
        r = requests.get(url)
        with open(fname, "wb") as f:
            f.write(r.content)

        A = torch.Tensor(mmread(fname).todense()).to(torch.float64)
        data = A.dtype
        cond = torch.linalg.cond(A)
        A = Matrix(A)

        for _ in range(10):
            x = 100.0 * torch.rand((1138,), dtype=torch.float64)
            x0 = 100.0 * torch.rand((1138,), dtype=torch.float64)
            b = A * x

            x_prime = solver(A, x0, b)
            error = torch.norm(x_prime - x).item()
            self.assertLessEqual(error, TEST_TOLERANCE)

    def test_power_system_scipy(self):
        fname = "1138_bus.mtx.gz"
        url = "https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/psadmit/1138_bus.mtx.gz"
        r = requests.get(url)
        with open(fname, "wb") as f:
            f.write(r.content)

        A = np.array(mmread(fname).todense())

        x = (100.0 * torch.rand((1138,))).numpy()
        b = np.matmul(A, x)

        x_prime = solve(A, b)
        error = np.linalg.norm(x_prime - x)
        self.assertLessEqual(error, TEST_TOLERANCE)


class TestConjugateResidual(unittest.TestCase):
    def test_identity(self):
        I = torch.diag(torch.ones((MATRIX_SIZE,)))
        A = Matrix(I)
        x = 100.0 * torch.rand((MATRIX_SIZE,))
        x0 = 100.0 * torch.rand((MATRIX_SIZE,))
        b = A * x

        solver = solvers.ConjugateResidual(ITERATIONS, SOLVER_TOLERANCE)
        x_prime = solver(A, x0, b)
        error = torch.norm(x_prime - x).item()
        self.assertLessEqual(error, TEST_TOLERANCE)

    def test_random_sym_pd_matrix(self):
        A = Matrix(generate_sym_pd_matrix(MATRIX_SIZE))
        x = 100.0 * torch.rand((MATRIX_SIZE,))
        x0 = 100.0 * torch.rand((MATRIX_SIZE,))
        b = A * x

        solver = solvers.ConjugateResidual(ITERATIONS, SOLVER_TOLERANCE)
        x_prime = solver(A, x0, b)
        error = torch.norm(x_prime - x).item()
        self.assertLessEqual(error, TEST_TOLERANCE)

    def test_random_sym_inv_matrix(self):
        A = Matrix(generate_sym_inv_matrix(MATRIX_SIZE))
        x = 100.0 * torch.rand((MATRIX_SIZE,))
        x0 = 100.0 * torch.rand((MATRIX_SIZE,))
        b = A * x

        solver = solvers.ConjugateResidual(ITERATIONS, SOLVER_TOLERANCE)
        x_prime = solver(A, x0, b)
        error = torch.norm(x_prime - x).item()
        self.assertLessEqual(error, TEST_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
