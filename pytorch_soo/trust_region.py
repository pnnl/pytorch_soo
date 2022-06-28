"""
A Data Class for specifying Trust Region behaviors
"""
from dataclasses import dataclass
from typing import Optional

import torch


class BadTrustRegionSpec(Exception):
    """Occurs when something about the Trust Region Specification is invalid"""


@dataclass(frozen=True)
class TrustRegionSpec:
    """
    Params:
        - initial_radius: The initial radius of the trust region model
        - max_radius: The maximum allowed radius of the trust region model. A very large value is
            akin to an unlimited radius
        - nabla0: The minimum value the step-size pk must be; else, the step is rejected. Must be
            >= 0, but should be a fairly small value.
        - nabla1: The minimum value for the model to be considered "Good enough" and not prompt a
            reduction in trust region radius
        - nabla2: The minimum value for a model to be "better than good" and to prompt and increase
            in the trust region radius
        - shrink_factor: If the model is not good, this is the factor by which we will reduce the
            trust region radius. Must be >0.0 and <1.0.
        - growth_factor: If the model is very good, this is the factor by which we will increase the
            trust region radius. Must be > 1.0
    """

    initial_radius: float = 1.0
    max_radius: float = 1e4
    nabla0: float = 1e-4
    nabla1: float = 0.25
    nabla2: float = 0.75

    shrink_factor: float = 0.25
    growth_factor: float = 2.0

    trust_region_subproblem_solver: str = "cg"
    trust_region_subproblem_tol: Optional[float] = 1e-4
    trust_region_subproblem_iter: Optional[int] = None

    def __post_init__(self):
        try:
            if self.initial_radius <= 0.0:
                err_str = f"Initial radius ({self.initial_radius}) must be >0.0!"
                raise BadTrustRegionSpec(err_str)

            if self.max_radius <= 0.0 or self.max_radius < self.initial_radius:
                err_str = (
                    f"Maximum radius ({self.max_radius}) must be >0 and >= the initial radius "
                    f"({self.initial_radius})"
                )

            if self.nabla0 < 0.0:
                raise BadTrustRegionSpec(f"nabla0 ({self.nabla0}) must be >=0!")

            if not (0.0 <= self.nabla0 <= self.nabla1 <= self.nabla2):
                err_str = (
                    "Nabla's must be set s.t. 0.0 <= nabla0 <= nabla1 <= nabla2, are currently: "
                    f"{self.nabla0}, {self.nabla1}, {self.nabla2}"
                )
                raise BadTrustRegionSpec(err_str)
            if not (0.0 < self.shrink_factor < 1.0):
                err_str = f"Shrink factor ({self.shrink_factor}) must be >0 and <1!"
                raise BadTrustRegionSpec(err_str)
            if self.growth_factor <= 1.0:
                err_str = f"Growth factor ({self.growth_factor}) must be >1!"

            if self.trust_region_subproblem_solver == "cg" and (
                self.trust_region_subproblem_tol < 0.0
                or self.trust_region_subproblem_tol is None
            ):
                err_str = (
                    "If the Trust-Region Subproblem solver is Conjugated-Gradient Steihaug, the "
                    f"tolerance ({self.trust_region_subproblem_tol}) must be specified and >= 0.0!"
                )
                raise BadTrustRegionSpec(err_str)
        except TypeError as type_error:
            raise BadTrustRegionSpec("An invalid value was passed in!") from type_error


class QuadraticSubproblem:
    """
    Taken from Scipy. A Function object representing the quadratic subproblem, with
    neat lazy evaluation techniques for the coeff values
    Modified to work in our case
    """

    # TODO typehints
    def __init__(self, x, loss, grad, hess, max_iter) -> None:
        self._x = x
        self._f = None
        self._g = None
        self._h = hess
        self._g_mag = None

        self._fun = loss
        self._grad = grad
        self._max_iter = max_iter
        if self._max_iter <= 0:
            raise ValueError(
                "Maximum iterations in trust region subproblem must be >= 1"
            )

    def __call__(self, p):
        return self.fun + torch.dot(self.grad, p) + 0.5 * torch.dot(p, self.hess * p)

    @property
    def fun(self):
        if self._f is None:
            self._f = self._fun(self._x)
        return self._f

    @property
    def grad(self):
        if self._g is None:
            self._g = self._grad(self._x)
        return self._g

    @property
    def hess(self):
        # This won't be none, we're doing this differently
        return self._h

    @property
    def grad_mag(self):
        if self._g_mag is None:
            self._g_mag = torch.linalg.norm(self.grad)
        return self._g_mag

    def get_boundaries_intersections(self, z, d, trust_radius):
        """
        Solve ||z+t*d|| == trust_radius
        return both values of t, sorted from low to high
        """
        a = torch.dot(d, d)
        b = 2 * torch.dot(z, d)
        c = torch.dot(z, z) - trust_radius**2
        sqrt_discriminant = torch.sqrt(b * b - 4 * a * c)
        aux = b + torch.copysign(sqrt_discriminant, b)
        ta = -aux / (2 * a)
        tb = -2 * c / aux
        return sorted([ta, tb])

    def solve(self, trust_radius):
        raise NotImplementedError(
            "The solve method must be provided by an inheriting class!"
        )


class ConjugateGradientSteihaug(QuadraticSubproblem):
    """
    The Conjugate Gradient Steihaug method. Very similar to the normal CG method,
    but handles the case where the dBd <= 0 by instead assuming a linear model
    """

    def solve(self, trust_radius):
        # get the norm of jacobian and define the origin
        p_origin = torch.zeros_like(self.grad)

        # define a default tolerance
        tolerance = min(0.5, torch.sqrt(self.grad_mag)) * self.grad_mag

        # Stop the method if the search direction
        # is a direction of nonpositive curvature.
        if self.grad_mag < tolerance:
            hits_boundary = False
            return p_origin, hits_boundary

        # init the state for the first iteration
        z = p_origin
        r = self.grad
        d = -r

        # Search for the min of the approximation of the objective function.
        for _ in range(self._max_iter):
            # do an iteration
            Bd = self.hess * d
            dBd = torch.dot(d, Bd)
            if dBd <= 0:
                # Look at the two boundary points.
                # Find both values of t to get the boundary points such that
                # ||z + t d|| == trust_radius
                # and then choose the one with the predicted min value.
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                if self(pa) < self(pb):
                    p_boundary = pa
                else:
                    p_boundary = pb
                hits_boundary = True
                return p_boundary, hits_boundary
            r_squared = torch.dot(r, r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if torch.norm(z_next) >= trust_radius:
                # Find t >= 0 to get the boundary point such that
                # ||z + t d|| == trust_radius
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                p_boundary = z + tb * d
                hits_boundary = True
                return p_boundary, hits_boundary
            r_next = r + alpha * Bd
            r_next_squared = torch.dot(r_next, r_next)
            if torch.sqrt(r_next_squared) < tolerance:
                hits_boundary = False
                return z_next, hits_boundary
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d

            # update the state for the next iteration
            z = z_next
            r = r_next
            d = d_next
        if torch.norm(z) >= trust_radius:
            # Find t >= 0 to get the boundary point such that
            # ||z + t d|| == trust_radius
            ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
            p_boundary = z + tb * d
            hits_boundary = True
            return p_boundary, hits_boundary
        return (z_next, False)


class CauchyPoint(QuadraticSubproblem):
    def solve(self, trust_radius):
        _ = trust_radius
        raise NotImplementedError("Haven't implemented Cauchy-Point yet")


class Dogleg(QuadraticSubproblem):
    def solve(self, trust_radius):
        _ = trust_radius
        raise NotImplementedError("Haven't implemented Dogleg yet")
