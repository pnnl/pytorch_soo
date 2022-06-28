"""
pytorch_sso: A collection of Scalable Second-order optimizers for PyTorch
"""

try:
    import torch
except ModuleNotFoundError as module_error:

    class TorchNotFound(Exception):
        pass

    err_str = (
        "Unable to import Torch. Torch is not currently easy to track as a dependency; "
        "as such, you must install it yourself manually. Please go to https://pytorch.org/ "
        "and select the appropriate version for your platform."
    )
    raise TorchNotFound(err_str) from module_error

__version__ = "0.1.0"

from .mfcr_levenberg import LevenbergEveryStep
from .nonlinear_conjugate_gradient import (
    NonlinearConjugateGradient,
    FletcherReeves,
    Daniels,
    PolakRibiere,
    HestenesStiefel,
    DaiYuan,
)
from .optim_hfcr_newton import HFCR_Newton
from .quasi_newton import (
    SymmetricRankOne,
    SymmetricRankOneInverse,
    DavidonFletcherPowell,
    DavidonFletcherPowellInverse,
    Broyden,
    BrodyenInverse,
)

__all__ = (
    "quasi_newton",
    "DFPInverse",
    "LevenbergEveryStep",
    "NonlinearConjugateGradient",
    "FletcherReeves",
    "Daniels",
    "PolakRibiere",
    "HestenesStiefel",
    "DaiYuan",
    "HFCR_Newton",
    "SymmetricRankOne",
    "SymmetricRankOneDual",
    "SymmetricRankOneInverse",
    "DavidonFletcherPowell",
    "DavidonFletcherPowellInverse",
    "Broyden",
    "BrodyenInverse",
)
