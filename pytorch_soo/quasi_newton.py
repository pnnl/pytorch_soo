"""
A collection of Quasi-Newton methods that rely on matrix-free operators to approximate a
matrix-vector product rather than forming the Hessian explicitly.
"""
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import warnings
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

import pytorch_soo.matrix_free_operators as matrix_ops

from pytorch_soo.solvers import Solver, ConjugateGradient, ConjugateResidual
from pytorch_soo.line_search_spec import LineSearchSpec
from pytorch_soo.trust_region import CauchyPoint, ConjugateGradientSteihaug, Dogleg
from pytorch_soo.trust_region import BadTrustRegionSpec, TrustRegionSpec


class LineSearchWarning(UserWarning):
    """Raise when an error occurs with a Line Search"""


class B0p:
    """
    An expression for the initial Hessian B0 that uses a taylor series approximation
    to determine the Matrix-vector product:
        B0p := (1/mu)(F(x0+mu*p) - F(x0))
    """

    def __init__(
        self, params: Iterable, closure: Callable[[], float], mu: float = 1e-6
    ) -> None:
        self._params = params
        self.closure = closure
        self.mu = mu

    def _get_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel().zero_())
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _get_perturbed_grad(self, p: Tensor) -> Tensor:
        current_params = parameters_to_vector(self._params)
        vec = torch.add(current_params, torch.mul(self.mu, p))
        vector_to_parameters(vec, self._params)
        _ = self.closure()
        new_flat_grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        self._zero_grad()

        return new_flat_grad

    def _zero_grad(self):
        for p in self._params:
            if p.grad is not None:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()

    def __call__(self, p: Tensor) -> Tensor:
        Fx = self._get_flat_grad()
        Fx_mu_p = self._get_perturbed_grad(p)
        diff = torch.sub(Fx_mu_p, Fx)
        b0p = torch.div(diff, self.mu)

        return b0p


@dataclass
class _LineSearchReturn:
    """
    xk1: the new parameters
    fx1: the new gradient
    dxk: the change in parameters
    new_loss: the new loss
    """

    xk1: Tensor
    fx1: Tensor
    dxk: Tensor
    new_loss: float


class QuasiNewtonWarning(RuntimeWarning):
    """
    Something went wrong in the quasi-newton method that's recoverable
    """


class QuasiNewton(Optimizer):
    """
    A base class for the other quasi-newton methods to inherit from, providing
    common code (as they really only vary by their matrix update methods).
    By direct, we mean solving Bd=-F(x) for d
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1.0,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 1e-3,
        rel_newton_tol: float = 1e-5,
        krylov_tol: float = 1e-3,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
        mu: float = 1e-6,
        verbose=False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Learning Rate ({lr} must be > 0!")
        if max_newton < 1:
            raise ValueError(f"Max Newton ({max_newton} must be > 0!")
        if max_krylov < 1:
            raise ValueError(f"Max Krylov ({max_krylov} must be > 0!")
        if abs_newton_tol < 0.0:
            raise ValueError(
                f"Absolute Newton Tolerance ({abs_newton_tol} must be > 0!"
            )
        if rel_newton_tol < 0.0:
            raise ValueError(
                f"Relative Newton Tolerance ({rel_newton_tol} must be > 0!"
            )
        if krylov_tol < 0.0:
            raise ValueError(f"Krylov tolerance ({krylov_tol} must be > 0!")

        if mu <= 0.0:
            raise ValueError(f"Finite difference size mu ({mu}) for B0p must be >0.0!")

        if matrix_free_memory is not None and matrix_free_memory < 1:
            raise ValueError(
                f"Matrix-free memory size ({matrix_free_memory}) must be None (unlimited) or >0!"
            )

        if line_search is not None:
            if line_search.max_searches < 1:
                raise ValueError(
                    f"Line search max search ({line_search.max_searches}) must be >0!"
                )
            if (
                line_search.extrapolation_factor is not None
                and line_search.extrapolation_factor >= 1.0
            ):
                raise ValueError(
                    f"Extrapolation factor ({line_search.extrapolation_factor}) must be <= 1.0!"
                )
            if not 0.0 < line_search.sufficient_decrease < 1.0:
                raise ValueError(
                    (
                        f"Line search sufficient decrease ({line_search.sufficient_decrease}) must "
                        "be in (0.0, 1.0)!"
                    )
                )
            if (
                line_search.curvature_constant is not None
                and not 0.0 < line_search.curvature_constant < 1.0
            ):
                raise ValueError(
                    (
                        "Line search curvature constant specified ("
                        f"{line_search.curvature_constant}); must be <= 1.0!"
                    )
                )

        defaults = dict(
            lr=lr,
            max_krylov=max_krylov,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            line_search=line_search,
            mu=mu,
        )

        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(
                "The Quasi-Newton methods don't support per-parameter "
                "options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self.solver = Solver(max_krylov, krylov_tol)
        self.mf_op = matrix_ops.MatrixFreeOperator(lambda p: p, n=matrix_free_memory)
        self.verbose = verbose

    def step(self, closure: Callable[[], float]):
        group = self.param_groups[0]
        lr = group["lr"]
        max_newton = group["max_newton"]
        abs_newton_tol = group["abs_newton_tol"]
        rel_newton_tol = group["rel_newton_tol"]
        line_search = group["line_search"]

        b0p = lambda p: p
        try:
            self.mf_op.change_B0p(b0p)
        except AttributeError:
            pass

        def f(y):
            """
            Convenience method for finding the loss at y
            """
            with torch.no_grad():
                x = torch.clone(y)
                saved_x = parameters_to_vector(self._params)
                vector_to_parameters(x, self._params)
                loss = closure()
                vector_to_parameters(saved_x, self._params)

            return loss

        def F(y):
            """
            Convenience method for finding the gradient at y
            F(y) = df(y)/dx
            """
            nonlocal closure
            return self._get_changed_grad(y, closure)

        x0 = parameters_to_vector(self._params)
        fx = F(x0)
        original_gradient_norm = torch.norm(fx).item()

        x = x0.clone()
        d = fx.clone()

        new_loss = None
        fx1 = None

        for _ in range(max_newton):
            if (
                torch.norm(fx).item() <= abs_newton_tol
                or torch.norm(fx).item() <= original_gradient_norm * rel_newton_tol
            ):
                # converged
                break
            d = self.solver(self.mf_op, d, -fx)
            if not torch.isfinite(d).all():
                if self.verbose:
                    msg = (
                        "Solver produced invalid step, assumptions of matrix structure likely "
                        "violated. Resetting matrix free operator and taking gradient step."
                    )
                    print(msg)
                d = -fx.clone()
                self.mf_op.reset()

            if line_search is None:
                dx = torch.mul(lr, d)
                xk1 = torch.add(x, dx)
            elif line_search.curvature_constant is None:
                line_search_return = self._backtracking_line_search(
                    x, fx, d, lr, f, F, line_search
                )
                xk1 = line_search_return.xk1
                fx1 = line_search_return.fx1
                dx = line_search_return.dxk
                new_loss = line_search_return.new_loss
            else:
                line_search_return = self._wolfe_line_search(
                    x, fx, d, lr, f, F, line_search
                )
                xk1 = line_search_return.xk1
                fx1 = line_search_return.fx1
                dx = line_search_return.dxk
                new_loss = line_search_return.new_loss

            if not torch.isfinite(xk1).all():
                if self.verbose:
                    msg = (
                        "Something broke when stepping. Resetting the MF operator and skipping "
                        "this step. If this is occuring regularly, this may be an unstable "
                        "configuration."
                    )
                    print(msg)
                self.mf_op.reset()
                xk1 = x.clone()

            if fx1 is None:
                fx1 = F(xk1)
            self.mf_op.update(fx, fx1, dx)
            x = xk1
            fx = fx1
            fx1 = None

        vector_to_parameters(x, self._params)
        if new_loss is None:
            new_loss = f(x)

        return new_loss

    def _backtracking_line_search(
        self,
        x: Tensor,
        fx: Tensor,
        d: Tensor,
        lr: float,
        f: Callable[[Tensor], Tensor],
        F: Callable[[Tensor], Tensor],
        line_search: LineSearchSpec,
    ) -> _LineSearchReturn:
        max_searches = line_search.max_searches
        extrapolation_factor = line_search.extrapolation_factor
        sufficient_decrease = line_search.sufficient_decrease
        curvature_constant = line_search.curvature_constant
        x_orig = x.clone()
        orig_loss = f(x_orig)
        orig_gradient = fx
        orig_curvature = torch.dot(orig_gradient, d)
        fx1 = None
        new_loss = None
        for _ in range(max_searches):
            dx = d.mul(lr)
            x_new = x_orig.add(dx)
            new_loss = f(x_new)
            decreased = (
                orig_loss >= new_loss + sufficient_decrease * lr * orig_curvature
            )
            if decreased:
                if curvature_constant is None:
                    xk1 = x_new
                    break
                fx1 = F(x_new)
                new_curvature = torch.dot(fx1, d)
                curvature = -new_curvature <= -curvature_constant * orig_curvature
                if curvature:
                    xk1 = x_new
                    break

            lr *= extrapolation_factor
        else:
            warnings.warn(f"Maximum number of line searches ({max_searches}) reached!")
            xk1 = x_new

        return _LineSearchReturn(xk1=xk1, fx1=fx1, dxk=dx, new_loss=new_loss)

    def _wolfe_line_search(
        self,
        x: Tensor,
        fx: Tensor,
        pk: Tensor,
        lr: float,
        f: Callable[[Tensor], Tensor],
        F: Callable[[Tensor], Tensor],
        spec: LineSearchSpec,
    ):
        """Basically the same as the SciPy implementation"""

        grad = torch.empty_like(pk)

        def phi(alpha):
            return f(x + alpha * pk)

        def dphi(alpha):
            nonlocal grad
            grad = F(x + alpha * pk)
            return torch.dot(grad, pk)

        amax = lr
        c1 = spec.sufficient_decrease
        c2 = spec.curvature_constant
        phi0 = phi(0.0)
        dphi0 = dphi(0.0)
        alpha0 = 0.0
        alpha1 = 1.0
        phi_a1 = phi(alpha1)
        phi_a0 = phi0
        dphi_a0 = dphi0

        for i in range(spec.max_searches):
            if alpha1 == 0 or alpha0 == amax:
                alpha_star = None
                phi_star = phi0
                phi0 = None
                dphi_star = None
                if alpha1 == 0:
                    msg = (
                        "Rounding errors prevent the Wolfe line search from converging"
                    )
                else:
                    msg = (
                        "The line search algorithm could not find a solution <= the learning rate "
                        f"({lr})"
                    )
                warnings.warn(msg, LineSearchWarning)
                break

            not_first_iteration = i > 0
            if (phi_a1 > phi0 + c1 * alpha1 * dphi0) or (
                (phi_a1 >= phi_a0) and not_first_iteration
            ):
                alpha_star, phi_star, dphi = self._zoom(
                    alpha0,
                    alpha1,
                    phi_a0,
                    phi_a1,
                    dphi_a0,
                    phi,
                    dphi,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                )
                break

            dphi_a1 = dphi(alpha1)
            if abs(dphi_a1) <= -c2 * dphi0:
                alpha_star = alpha1
                phi_star = phi_a1
                dphi_star = dphi_a1
                break

            if dphi_a1 >= 0:
                alpha_star, phi_star, dphi_star = self._zoom(
                    alpha1,
                    alpha0,
                    phi_a1,
                    phi_a0,
                    dphi_a1,
                    phi,
                    dphi,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                )
                break

            alpha2 = 2 * alpha1
            alpha2 = min(alpha2, amax)
            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            phi_a1 = phi(alpha1)
            dphi_a0 = dphi_a1

        else:
            alpha_star = alpha1
            phi_star = phi_a1
            dphi_star = None
            warnings.warn(
                "The Wolfe Line Search algorithm did not converge", LineSearchWarning
            )

        if alpha_star is None:
            alpha_star = 0.0
        delta_x = alpha_star * pk
        xk1 = x + delta_x

        retval = _LineSearchReturn(xk1=xk1, fx1=grad, dxk=delta_x, new_loss=phi_star)
        return retval

    def _zoom(
        self,
        a_lo,
        a_hi,
        phi_lo,
        phi_hi,
        derphi_lo,
        phi,
        derphi,
        phi0,
        derphi0,
        c1,
        c2,
    ):
        """
        Zoom stage of approximate linesearch satisfying strong Wolfe conditions.
        Taken from SciPy's Optimization toolbox

        Notes
        -----
        Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
        'Numerical Optimization', 1999, pp. 61.

        """

        maxiter = 10
        i = 0
        delta1 = 0.2  # cubic interpolant check
        delta2 = 0.1  # quadratic interpolant check
        phi_rec = phi0
        a_rec = 0
        while True:
            dalpha = a_hi - a_lo
            if dalpha < 0:
                a, b = a_hi, a_lo
            else:
                a, b = a_lo, a_hi

            if i > 0:
                cchk = delta1 * dalpha
                a_j = self._cubicmin(
                    a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                )
            if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
                qchk = delta2 * dalpha
                a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                    a_j = a_lo + 0.5 * dalpha

            # Check new value of a_j
            phi_aj = phi(a_j)
            if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_j
                phi_hi = phi_aj

            else:
                derphi_aj = derphi(a_j)
                if abs(derphi_aj) <= -c2 * derphi0:
                    a_star = a_j
                    val_star = phi_aj
                    valprime_star = derphi_aj
                    break

                if derphi_aj * (a_hi - a_lo) >= 0:
                    phi_rec = phi_hi
                    a_rec = a_hi
                    a_hi = a_lo
                    phi_hi = phi_lo

                else:
                    phi_rec = phi_lo
                    a_rec = a_lo

                a_lo = a_j
                phi_lo = phi_aj
                derphi_lo = derphi_aj

            i += 1
            if i > maxiter:
                # Failed to find a conforming step size
                a_star = None
                val_star = None
                valprime_star = None
                break

        return a_star, val_star, valprime_star

    def _cubicmin(self, a, fa, fpa, b, fb, c, fc):
        """
        Finds the minimizer for a cubic polynomial that goes through the
        points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

        If no minimizer can be found, return None.

        """
        # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

        device = self._params[0].device
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = torch.empty((2, 2)).to(device)
            d1[0, 0] = dc**2
            d1[0, 1] = -(db**2)
            d1[1, 0] = -(dc**3)
            d1[1, 1] = db**3
            [A, B] = torch.matmul(
                d1,
                torch.tensor([fb - fa - C * db, fc - fa - C * dc]).flatten().to(device),
            )
            A = A / denom
            B = B / denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + torch.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
        if not torch.isfinite(xmin):
            return None
        return xmin

    def _quadmin(self, a, fa, fpa, b, fb):
        """
        Finds the minimizer for a quadratic polynomial that goes through
        the points (a,fa), (b,fb) with derivative at a of fpa.

        """
        # f(x) = B*(x-a)^2 + C*(x-a) + D
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
        if not torch.isfinite(xmin):
            return None
        return xmin

    def _get_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel().zero_())
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _get_changed_grad(self, vec: Tensor, closure: Callable[[], float]) -> Tensor:
        current_params = parameters_to_vector(self._params)
        vector_to_parameters(vec, self._params)
        _ = closure()
        new_flat_grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        self.zero_grad()

        return new_flat_grad


class QuasiNewtonTrust(Optimizer):
    """
    A base class for the other quasi-newton methods to inherit from, providing
    common code (as they really only vary by their matrix update methods).
    By direct, we mean solving Bd=-F(x) for d
    This appears to work for the SR1 Dual, even though the dual of the matrix is also the inverse.
    Neat.

    Uses a Trust Region method, instead of a fixed step size or line-search
    """

    def __init__(
        self,
        params: Iterable,
        trust_region: TrustRegionSpec,
        lr: float = 1.0,
        max_newton: int = 10,
        abs_newton_tol: float = 1e-3,
        rel_newton_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        mu: float = 1e-6,
    ):
        if lr <= 0.0:
            raise ValueError(f"Learning Rate ({lr} must be > 0!")
        if max_newton < 1:
            raise ValueError(f"Max Newton ({max_newton} must be > 0!")
        if abs_newton_tol < 0.0:
            raise ValueError(
                f"Absolute Newton Tolerance ({abs_newton_tol} must be > 0!"
            )
        if rel_newton_tol < 0.0:
            raise ValueError(
                f"Relative Newton Tolerance ({rel_newton_tol} must be > 0!"
            )
        if mu <= 0.0:
            raise ValueError(f"Finite difference size mu ({mu}) for B0p must be >0.0!")

        if matrix_free_memory is not None and matrix_free_memory < 1:
            raise ValueError(
                f"Matrix-free memory size ({matrix_free_memory}) must be None (unlimited) or >0!"
            )

        defaults = dict(
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            trust_region=trust_region,
            mu=mu,
        )

        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(
                "The Quasi-Newton methods don't support per-parameter "
                "options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self.mf_op = matrix_ops.MatrixFreeOperator(lambda p: p, n=matrix_free_memory)
        self.trust_region_spec = trust_region
        if self.trust_region_spec.trust_region_subproblem_solver == "cauchy":
            self._trust_region_subproblem = CauchyPoint
        elif self.trust_region_spec.trust_region_subproblem_solver == "dogleg":
            self._trust_region_subproblem = Dogleg
        elif self.trust_region_spec.trust_region_subproblem_solver == "cg":
            self._trust_region_subproblem = ConjugateGradientSteihaug
        else:
            raise BadTrustRegionSpec(
                "Invalid trust-region subproblem solver requested: "
                f"{self.trust_region_spec.trust_region_subproblem_solver}"
            )

        self._trust_region_radius = self.trust_region_spec.initial_radius

    def step(self, closure: Callable[[], float]):
        group = self.param_groups[0]
        max_newton = group["max_newton"]
        abs_newton_tol = group["abs_newton_tol"]
        rel_newton_tol = group["rel_newton_tol"]

        # b0p = B0p(self._params, closure)
        b0p = lambda p: p
        try:
            self.mf_op.change_B0p(b0p)
        except AttributeError:
            pass

        def f(y):
            """
            Convenience method for finding the loss at y
            """
            with torch.no_grad():
                x = torch.clone(y)
                saved_x = parameters_to_vector(self._params)
                vector_to_parameters(x, self._params)
                loss = closure()
                vector_to_parameters(saved_x, self._params)

            return loss

        def F(y):
            """
            Convenience method for finding the gradient at y
            F(y) = df(y)/dx
            """
            nonlocal closure
            return self._get_changed_grad(y, closure)

        x0 = parameters_to_vector(self._params)
        x = x0.clone()
        fx = F(x0)
        original_gradient_norm = torch.norm(fx).item()
        m = self._trust_region_subproblem(
            x, f, F, self.mf_op, self.trust_region_spec.trust_region_subproblem_iter
        )
        newton_iter = 0

        # We use while loops to allow for a repeated iteration in the event of a bad model
        while newton_iter < max_newton:
            newton_iter += 1
            if (
                torch.norm(fx).item() <= abs_newton_tol
                or torch.norm(fx).item() <= original_gradient_norm * rel_newton_tol
            ):
                # converged
                break
            p, hits_boundary = m.solve(self._trust_region_radius)
            predicted_value = m(p)
            x_proposed = x + p
            m_proposed = self._trust_region_subproblem(
                x_proposed,
                f,
                F,
                self.mf_op,
                self.trust_region_spec.trust_region_subproblem_iter,
            )
            actual_reduction = m.fun - m_proposed.fun
            predicted_reduction = m.fun - predicted_value
            if predicted_reduction < 0.0:
                warnings.warn("Predicted improvement was negative!")
                break
            rho = actual_reduction / predicted_reduction
            if rho < self.trust_region_spec.nabla1:
                self._trust_region_radius *= self.trust_region_spec.shrink_factor
            elif rho > 0.75 and hits_boundary:
                self._trust_region_radius = min(
                    self.trust_region_spec.growth_factor * self._trust_region_radius,
                    self.trust_region_spec.max_radius,
                )

            if rho > self.trust_region_spec.nabla0:
                # accept the step
                self.mf_op.update(
                    m.grad.clone(), m_proposed.grad.clone(), (x_proposed - x).clone()
                )
                x = x_proposed.clone()
                m = m_proposed

        vector_to_parameters(x, self._params)
        new_loss = m.fun

        return new_loss

    def _get_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel().zero_())
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _get_changed_grad(self, vec: Tensor, closure: Callable[[], float]) -> Tensor:
        current_params = parameters_to_vector(self._params)
        vector_to_parameters(vec, self._params)
        _ = closure()
        new_flat_grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        self.zero_grad()

        return new_flat_grad


class InverseQuasiNewton(Optimizer):
    """
    A base class for the inverse quasi-newton methods to inherit from, providing
    common code (as they really only vary by their matrix update methods).
    By inverse, we mean they rely on the direct calculation of the inverse matrix H=B^{-1}
    to find d = -HF(x)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1.0,
        max_newton: int = 10,
        abs_newton_tol: float = 1e-3,
        rel_newton_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
        trust_region: Optional[TrustRegionSpec] = None,
        verbose=False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Learning Rate ({lr} must be > 0!")
        if max_newton < 1:
            raise ValueError(f"Max Newton ({max_newton} must be > 0!")
        if abs_newton_tol < 0.0:
            raise ValueError(
                f"Absolute Newton Tolerance ({abs_newton_tol} must be > 0!"
            )
        if rel_newton_tol < 0.0:
            raise ValueError(
                f"Relative Newton Tolerance ({rel_newton_tol} must be > 0!"
            )
        if matrix_free_memory is not None and matrix_free_memory < 1:
            raise ValueError(
                f"Matrix-free memory size ({matrix_free_memory}) must be None (unlimited) or >0!"
            )

        # Line search and Trust Region validation
        if line_search is not None and trust_region is not None:
            raise ValueError(
                "Line search and trust region are mutually exclusive; "
                "please only provide one specification!"
            )
        if line_search is not None:
            if line_search.max_searches < 1:
                raise ValueError(
                    f"Line search max search ({line_search.max_searches}) must be >0!"
                )
            if (
                line_search.extrapolation_factor is not None
                and line_search.extrapolation_factor >= 1.0
            ):
                raise ValueError(
                    f"Extrapolation factor ({line_search.extrapolation_factor}) must be <= 1.0!"
                )
            if not 0.0 < line_search.sufficient_decrease < 1.0:
                raise ValueError(
                    (
                        f"Line search sufficient decrease ({line_search.sufficient_decrease}) must "
                        "be in (0.0, 1.0)!"
                    )
                )
            if (
                line_search.curvature_constant is not None
                and not 0.0 < line_search.curvature_constant < 1.0
            ):
                raise ValueError(
                    (
                        "Line search curvature constant specified ("
                        f"{line_search.curvature_constant}); must be <= 1.0!"
                    )
                )

        if trust_region is not None:
            raise NotImplementedError("Trust region methods not yet supported!")

        defaults = dict(
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            line_search=line_search,
            trust_region=trust_region,
        )

        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(
                "The Quasi-Newton methods don't support per-parameter "
                "options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self.mf_op = matrix_ops.InverseMatrixFreeOperator(n=matrix_free_memory)
        self.verbose = verbose

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        group = self.param_groups[0]
        lr = group["lr"]
        max_newton = group["max_newton"]
        abs_newton_tol = group["abs_newton_tol"]
        rel_newton_tol = group["rel_newton_tol"]
        line_search = group["line_search"]

        new_loss = None

        def f(y):
            """Convenience method to get the loss"""
            with torch.no_grad():
                x = torch.clone(y)
                saved_x = parameters_to_vector(self._params)
                vector_to_parameters(x, self._params)
                loss = closure()
                vector_to_parameters(saved_x, self._params)

            return loss

        def F(y):
            """Convenience method to get the gradient"""
            x = torch.clone(y)
            saved_x = parameters_to_vector(self._params)
            vector_to_parameters(x, self._params)
            with torch.set_grad_enabled(True):
                _ = closure()
            z = torch.clone(self._get_flat_grad())
            vector_to_parameters(saved_x, self._params)
            self.zero_grad()

            return z

        x0 = parameters_to_vector(self._params)
        fx = F(x0)
        original_gradient_norm = torch.norm(fx).item()

        x = x0.clone()
        d = fx.clone()

        new_loss = None
        fx1 = None

        for _ in range(max_newton):
            if (
                torch.norm(fx).item() <= abs_newton_tol
                or torch.norm(fx).item() <= original_gradient_norm * rel_newton_tol
            ):
                # converged
                break
            d = -(self.mf_op * fx)
            if not torch.isfinite(d).all():
                if self.verbose:
                    msg = (
                        "Matrix-Free operator produced an invalid step, assumptions of matrix "
                        "structure likely violated.  Resetting matrix free operator and taking "
                        "gradient step."
                    )
                    print(msg)
                    print(f"Currently at {len(self.mf_op.memory)} entries.")
                self.mf_op.reset()
                d = -fx.clone()

            if line_search is None:
                dx = torch.mul(lr, d)
                xk1 = torch.add(x, dx)
                fx1 = F(xk1)
            elif line_search.curvature_constant is None:
                line_search_return = self._backtracking_line_search(
                    x, fx, d, lr, f, F, line_search
                )
                xk1 = line_search_return.xk1
                fx1 = line_search_return.fx1
                dx = line_search_return.dxk
                new_loss = line_search_return.new_loss
            else:
                line_search_return = self._wolfe_line_search(
                    x, fx, d, lr, f, F, line_search
                )
                xk1 = line_search_return.xk1
                fx1 = line_search_return.fx1
                dx = line_search_return.dxk
                new_loss = line_search_return.new_loss

            if not torch.isfinite(xk1).all():
                if self.verbose:
                    msg = (
                        "Something broke when stepping. Resetting the MF operator and skipping this "
                        "step. If this is occuring regularly, this may be an unstable configuration."
                    )
                    print(msg)
                self.mf_op.reset()
                xk1 = x.clone()

            if fx1 is None:
                fx1 = F(xk1)
            self.mf_op.update(fx, fx1, dx)
            x = xk1
            fx = fx1

        vector_to_parameters(x, self._params)
        if new_loss is None:
            new_loss = f(x)

        return new_loss

    def _backtracking_line_search(
        self,
        x: Tensor,
        fx: Tensor,
        d: Tensor,
        lr: float,
        f: Callable[[Tensor], Tensor],
        F: Callable[[Tensor], Tensor],
        line_search: LineSearchSpec,
    ) -> _LineSearchReturn:
        max_searches = line_search.max_searches
        extrapolation_factor = line_search.extrapolation_factor
        sufficient_decrease = line_search.sufficient_decrease
        curvature_constant = line_search.curvature_constant

        x_orig = x.clone()
        orig_loss = f(x_orig)
        orig_gradient = fx
        orig_curvature = torch.dot(orig_gradient, d)
        fx1 = None
        new_loss = None

        for _ in range(max_searches):
            dx = d.mul(lr)
            x_new = x_orig.add(dx)
            new_loss = f(x_new)
            decreased = (
                new_loss <= orig_loss + sufficient_decrease * lr * orig_curvature
            )
            if decreased:
                if curvature_constant is None:
                    xk1 = x_new
                    break
                fx1 = F(x_new)
                new_curvature = torch.dot(fx1, d)
                curvature = -new_curvature <= -curvature_constant * orig_curvature
                if curvature:
                    xk1 = x_new
                    break

            lr *= extrapolation_factor
        else:
            warnings.warn(f"Maximum number of line searches ({max_searches}) reached!")
            xk1 = x_new

        return _LineSearchReturn(xk1=xk1, fx1=fx1, dxk=dx, new_loss=new_loss)

    def _wolfe_line_search(
        self,
        x: Tensor,
        fx: Tensor,
        pk: Tensor,
        lr: float,
        f: Callable[[Tensor], Tensor],
        F: Callable[[Tensor], Tensor],
        spec: LineSearchSpec,
    ):
        """Basically the same as the SciPy implementation"""

        grad = torch.empty_like(pk)

        def phi(alpha):
            return f(x + alpha * pk)

        def dphi(alpha):
            nonlocal grad
            grad = F(x + alpha * pk)
            return torch.dot(grad, pk)

        amax = lr
        c1 = spec.sufficient_decrease
        c2 = spec.curvature_constant
        phi0 = phi(0.0)
        dphi0 = dphi(0.0)
        alpha0 = 0.0
        alpha1 = 1.0
        phi_a1 = phi(alpha1)
        phi_a0 = phi0
        dphi_a0 = dphi0

        for i in range(spec.max_searches):
            if alpha1 == 0 or alpha0 == amax:
                alpha_star = None
                phi_star = phi0
                phi0 = None
                dphi_star = None
                if alpha1 == 0:
                    msg = (
                        "Rounding errors prevent the Wolfe line search from converging"
                    )
                else:
                    msg = (
                        "The line search algorithm could not find a solution <= the learning rate "
                        f"({lr})"
                    )
                warnings.warn(msg, LineSearchWarning)
                break

            not_first_iteration = i > 0
            if (phi_a1 > phi0 + c1 * alpha1 * dphi0) or (
                (phi_a1 >= phi_a0) and not_first_iteration
            ):
                alpha_star, phi_star, dphi = self._zoom(
                    alpha0,
                    alpha1,
                    phi_a0,
                    phi_a1,
                    dphi_a0,
                    phi,
                    dphi,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                )
                break

            dphi_a1 = dphi(alpha1)
            if abs(dphi_a1) <= -c2 * dphi0:
                alpha_star = alpha1
                phi_star = phi_a1
                dphi_star = dphi_a1
                break

            if dphi_a1 >= 0:
                alpha_star, phi_star, dphi_star = self._zoom(
                    alpha1,
                    alpha0,
                    phi_a1,
                    phi_a0,
                    dphi_a1,
                    phi,
                    dphi,
                    phi0,
                    dphi0,
                    c1,
                    c2,
                )
                break

            alpha2 = 2 * alpha1
            alpha2 = min(alpha2, amax)
            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            phi_a1 = phi(alpha1)
            dphi_a0 = dphi_a1

        else:
            alpha_star = alpha1
            phi_star = phi_a1
            dphi_star = None
            warnings.warn(
                "The Wolfe Line Search algorithm did not converge", LineSearchWarning
            )

        if alpha_star is None:
            alpha_star = 0.0
        delta_x = alpha_star * pk
        xk1 = x + delta_x

        retval = _LineSearchReturn(xk1=xk1, fx1=grad, dxk=delta_x, new_loss=phi_star)
        return retval

    def _zoom(
        self,
        a_lo,
        a_hi,
        phi_lo,
        phi_hi,
        derphi_lo,
        phi,
        derphi,
        phi0,
        derphi0,
        c1,
        c2,
    ):
        """
        Zoom stage of approximate linesearch satisfying strong Wolfe conditions.
        Taken from SciPy's Optimization toolbox

        Notes
        -----
        Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
        'Numerical Optimization', 1999, pp. 61.

        """

        maxiter = 10
        i = 0
        delta1 = 0.2  # cubic interpolant check
        delta2 = 0.1  # quadratic interpolant check
        phi_rec = phi0
        a_rec = 0
        while True:
            dalpha = a_hi - a_lo
            if dalpha < 0:
                a, b = a_hi, a_lo
            else:
                a, b = a_lo, a_hi

            if i > 0:
                cchk = delta1 * dalpha
                a_j = self._cubicmin(
                    a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                )
            if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
                qchk = delta2 * dalpha
                a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                    a_j = a_lo + 0.5 * dalpha

            # Check new value of a_j
            phi_aj = phi(a_j)
            if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_j
                phi_hi = phi_aj

            else:
                derphi_aj = derphi(a_j)
                if abs(derphi_aj) <= -c2 * derphi0:
                    a_star = a_j
                    val_star = phi_aj
                    valprime_star = derphi_aj
                    break

                if derphi_aj * (a_hi - a_lo) >= 0:
                    phi_rec = phi_hi
                    a_rec = a_hi
                    a_hi = a_lo
                    phi_hi = phi_lo

                else:
                    phi_rec = phi_lo
                    a_rec = a_lo

                a_lo = a_j
                phi_lo = phi_aj
                derphi_lo = derphi_aj

            i += 1
            if i > maxiter:
                # Failed to find a conforming step size
                a_star = None
                val_star = None
                valprime_star = None
                break

        return a_star, val_star, valprime_star

    def _cubicmin(self, a, fa, fpa, b, fb, c, fc):
        """
        Finds the minimizer for a cubic polynomial that goes through the
        points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

        If no minimizer can be found, return None.

        """
        # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

        device = self._params[0].device
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = torch.empty((2, 2)).to(device)
            d1[0, 0] = dc**2
            d1[0, 1] = -(db**2)
            d1[1, 0] = -(dc**3)
            d1[1, 1] = db**3
            [A, B] = torch.matmul(
                d1,
                torch.tensor([fb - fa - C * db, fc - fa - C * dc]).flatten().to(device),
            )
            A = A / denom
            B = B / denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + torch.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
        if not torch.isfinite(xmin):
            return None
        return xmin

    def _quadmin(self, a, fa, fpa, b, fb):
        """
        Finds the minimizer for a quadratic polynomial that goes through
        the points (a,fa), (b,fb) with derivative at a of fpa.

        """
        # f(x) = B*(x-a)^2 + C*(x-a) + D
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
        if not torch.isfinite(xmin):
            return None
        return xmin

    def _get_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel().zero_())
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)

        return torch.cat(views, 0)

    def _get_changed_grad(self, vec: Tensor, closure: Callable[[], float]) -> Tensor:
        current_params = parameters_to_vector(self._params)
        vector_to_parameters(vec, self._params)
        _ = closure()
        new_flat_grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        self.zero_grad()

        return new_flat_grad


class SymmetricRankOne(QuasiNewton):
    """
    https://en.wikipedia.org/wiki/Symmetric_rank-one
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = matrix_ops.SymmetricRankOne(lambda p: p, n=matrix_free_memory)
        self.solver = ConjugateResidual(max_krylov, krylov_tol)


class SymmetricRankOneInverse(InverseQuasiNewton):
    """
    https://en.wikipedia.org/wiki/Symmetric_rank-one
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
            trust_region=trust_region,
        )
        self.mf_op = matrix_ops.SymmetricRankOneInverse(matrix_free_memory)


class SymmetricRankOneTrust(QuasiNewtonTrust):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            trust_region=trust_region,
        )
        self.mf_op = matrix_ops.SymmetricRankOne(lambda p: p, n=matrix_free_memory)


class SymmetricRankOneDualTrust(QuasiNewtonTrust):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            trust_region=trust_region,
        )
        # SR1 dual is the inverse
        self.mf_op = matrix_ops.SymmetricRankOneInverse(matrix_free_memory)


class BFGS(QuasiNewton):
    """
    https://en.wikipedia.org/wiki/Davidon-Fletcher-Powell_formula
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 0.001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = matrix_ops.BFGS(lambda p: p, n=matrix_free_memory)
        self.solver = ConjugateGradient(max_krylov, krylov_tol)


class BFGSInverse(InverseQuasiNewton):
    """
    https://en.wikipedia.org/wiki/Davidon-Fletcher-Powell_formula
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = matrix_ops.BFGSInverse(matrix_free_memory)


class BFGSTrust(QuasiNewtonTrust):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            trust_region=trust_region,
        )
        self.mf_op = matrix_ops.BFGS(lambda p: p, n=matrix_free_memory)
        self.solver = ConjugateGradient(max_krylov, krylov_tol)


class DavidonFletcherPowell(QuasiNewton):
    """
    https://en.wikipedia.org/wiki/Davidon-Fletcher-Powell_formula
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 0.001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = matrix_ops.DavidonFletcherPowell(lambda p: p, n=matrix_free_memory)
        self.solver = ConjugateGradient(max_krylov, krylov_tol)


class DavidonFletcherPowellInverse(InverseQuasiNewton):
    """
    https://en.wikipedia.org/wiki/Davidon-Fletcher-Powell_formula
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = matrix_ops.DavidonFletcherPowellInverse(matrix_free_memory)


class DavidonFletcherPowellTrust(QuasiNewtonTrust):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            trust_region=trust_region,
        )
        self.mf_op = matrix_ops.DavidonFletcherPowell(lambda p: p, n=matrix_free_memory)
        self.solver = ConjugateGradient(max_krylov, krylov_tol)


class Broyden(QuasiNewton):
    """
    https://en.wikipedia.org/wiki/Brodyden%27s_method
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = matrix_ops.Broyden(lambda p: p, n=matrix_free_memory)
        self.solver = None
        err_msg = (
            "Need to implement a solver that handles non-symmetric, non-PD solver "
            "such as GMRES, Arnoldi, or GCR."
        )
        raise NotImplementedError(err_msg)


class BrodyenInverse(InverseQuasiNewton):
    """
    https://en.wikipedia.org/wiki/Broyden%27s_method
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        matrix_free_memory: Optional[int] = None,
        line_search: Optional[LineSearchSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            matrix_free_memory=matrix_free_memory,
            line_search=line_search,
        )
        self.mf_op = matrix_ops.BroydenInverse(matrix_free_memory)


class BroydenTrust(QuasiNewtonTrust):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1,
        max_newton: int = 10,
        max_krylov: int = 10,
        abs_newton_tol: float = 0.001,
        rel_newton_tol: float = 0.00001,
        krylov_tol: float = 1e-5,
        matrix_free_memory: Optional[int] = None,
        trust_region: Optional[TrustRegionSpec] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            max_newton=max_newton,
            max_krylov=max_krylov,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            krylov_tol=krylov_tol,
            matrix_free_memory=matrix_free_memory,
            trust_region=trust_region,
        )
        self.mf_op = matrix_ops.Broyden(lambda p: p, n=matrix_free_memory)
        self.solver = None
        err_msg = (
            "Need to implement a solver that handles non-symmetric, non-PD solver "
            "such as GMRES, Arnoldi, or GCR."
        )
        raise NotImplementedError(err_msg)
