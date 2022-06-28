import math
import torch
from torch import Tensor


class Rosenbrock:
    def __call__(self, tensor: Tensor) -> Tensor:
        return self.f(tensor)

    @staticmethod
    def f(tensor: Tensor) -> Tensor:
        x, y = tensor
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    @staticmethod
    def grad(tensor):
        x, y = tensor
        dx = 2 * (200 * x ** 3) - 200 * x * y + x - 1
        dy = 200 * (y - x ** 2)

        return torch.Tensor([dx, dy])

    @staticmethod
    def hessian(tensor):
        x, y = tensor
        h11 = -400 * (y - x ** 2) + 800 * x ** 2 + 2
        h12 = -400 * x
        h21 = h12
        h22 = 200
        hess = Tensor([[h11, h12], [h21, h22]])

        return hess


class Rastrigin:
    def __init___(self) -> None:
        self.A = 10

    def __call__(self, tensor: Tensor) -> Tensor:
        return self.f(tensor)

    def f(self, tensor: Tensor) -> Tensor:
        x, y = tensor
        self.A = 10
        f = (
            self.A * 2
            + (x ** 2 - self.A * torch.cos(x * math.pi * 2))
            + (y ** 2 - self.A * torch.cos(y * math.pi * 2))
        )
        return f

    def grad(self, tensor: Tensor) -> Tensor:
        x, y = tensor
        dx = 2 * (x + torch.pi * self.A * torch.sin(2 * torch.pi * x))
        dy = 2 * (y + torch.pi * self.A * torch.sin(2 * torch.pi * y))

        return Tensor([dx, dy])

    def hessian(self, tensor: Tensor) -> Tensor:
        x, y = tensor
        h11 = 4 * torch.pi ** 2 * torch.cos(2 * torch.pi * x) + 2
        h12 = 0
        h21 = 0
        h22 = 4 * torch.pi ** 2 * torch.cos(2 * torch.pi * y) + 2
        hess = Tensor([[h11, h12], [h21, h22]])

        return hess


class Sphere:
    def __call__(self, tensor: Tensor) -> Tensor:
        return self.f(tensor)

    @staticmethod
    def f(tensor: Tensor) -> Tensor:
        x, y = tensor
        f = x ** 2 + y ** 2

        return f

    @staticmethod
    def grad(tensor: Tensor) -> Tensor:
        x, y = tensor

        return Tensor([2 * x, 2 * y])

    @staticmethod
    def hessian(tensor: Tensor) -> Tensor:
        _ = tensor
        return Tensor([[2, 0], [0, 2]])
