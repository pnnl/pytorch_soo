"""
Implements several popular Nonlinear Conjugate Gradient Methods
https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
"""
from typing import Callable, Iterable, Optional
from dataclasses import dataclass
import warnings
import torch
from torch.functional import Tensor
from torch.optim import Optimizer
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from pytorch_soo.line_search_spec import LineSearchSpec


@dataclass
class _LineSearchReturn:
    xk1: Tensor
    fx1: Tensor
    dxk: Tensor
    new_loss: float


class NonlinearConjugateGradient(Optimizer):
    """
    The only difference between several NLGC methods is just the calculation of the value beta.
    This base class codifies this; inheriting classes must provide _beta_calc().

    args:
        - params: The model parameters
        - lr: The Learning rate. If a line-search is being used, this instead becomes the maximum
            step-size search for the backtracking line-search
        - max_newton: The maximum number of "newton" iterations (e.g. steps) that can occur
            per individual "step"
        - rel_newton_tol: the relative change in gradient for convergence at this step
        - abs_newton_tol: the absolute gradient for convergence
        - line_search_spec: A LineSearchSpec object that describes the backtracking line-search
            parameters
        - viz_steps: Used for plotting purposes, will record and return the values for each step.
            Not advisable for real problems, think Rosenbrock or similar
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.3333,
        max_newton: int = 20,
        rel_newton_tol: float = 1.0e-5,
        abs_newton_tol: float = 1.0e-8,
        line_search_spec: Optional[LineSearchSpec] = None,
        viz_steps=False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if max_newton < 1:
            raise ValueError(f"Invalid max_newton: {max_newton} - should be >= 1")
        if abs_newton_tol < 0.0:
            raise ValueError(
                f"Invalid abs_newton_tol: {abs_newton_tol} - should be >= 0.0"
            )
        if rel_newton_tol < 0.0:
            raise ValueError(
                f"Invalid abs_newton_tol: {rel_newton_tol} - should be >= 0.0"
            )
        if line_search_spec is not None:
            extrapolation_factor = line_search_spec.extrapolation_factor
            sufficient_decrease = line_search_spec.sufficient_decrease
            curvature_constant = line_search_spec.curvature_constant
            max_searches = line_search_spec.max_searches

            if extrapolation_factor >= 1.0:
                raise ValueError("Extrapolation factor is a multiplier, must be <1.0!")
            if not 0.0 < sufficient_decrease < 1.0:
                raise ValueError("Sufficient decrease must be strictly in (0, 1)!")
            if curvature_constant is not None:
                # Wolfe
                if not 0.0 < curvature_constant < 1.0:
                    raise ValueError("Curvature Constant must be strictly in (0, 1)!")
                if curvature_constant <= sufficient_decrease:
                    raise ValueError(
                        "Curvature Constant must be greater than sufficient decrease!"
                    )
            if max_searches <= 1:
                raise ValueError(
                    "If specifying a line search you must have at least one line search!"
                )

        defaults = dict(
            lr=lr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            line_search_spec=line_search_spec,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "The Nonlinear Conjugate Gradient algorithms don't support per-parameter options "
                "(parameter groups)"
            )
        self._params = self.param_groups[0]["params"]
        self.viz_steps = viz_steps

    def _get_flat_grad(self):
        views = []
        for param in self._params:
            if param.grad is None:
                view = param.data.new(param.data.numel()).zero_()
            elif param.grad.data.is_sparse:
                view = param.grad.data.to_dense().view(-1)
            else:
                view = param.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        _ = (gradient, last_gradient, last_conjugate_gradient)

        raise NotImplementedError("This is the base class!")

    @staticmethod
    def _convergence_check(
        grad: torch.Tensor,
        rel_newton_tol: float,
        abs_newton_tol: float,
        og_grad_norm: float,
    ):
        g = torch.norm(grad).item()
        return g < abs_newton_tol or g < rel_newton_tol * og_grad_norm

    def step(self, closure: Callable):
        """The optimizer step function."""
        group = self.param_groups[0]
        lr = group["lr"]
        max_newton = group["max_newton"]
        abs_newton_tol = group["abs_newton_tol"]
        rel_newton_tol = group["rel_newton_tol"]
        line_search_spec = group["line_search_spec"]
        if self.viz_steps:
            steps = []

        with torch.no_grad():
            orig_loss = None
            new_loss = None

            def f(y):
                """Wrapper to obtain the loss"""
                x = torch.clone(y)
                saved_x = parameters_to_vector(self._params)
                vector_to_parameters(x, self._params)
                loss = closure()
                vector_to_parameters(saved_x, self._params)

                return loss

            def F(y):
                """Wrapper to obtain the gradient"""
                nonlocal orig_loss
                x = torch.clone(y)

                saved_x = parameters_to_vector(self._params)
                vector_to_parameters(x, self._params)

                with torch.enable_grad():
                    orig_loss = closure()

                z = self._get_flat_grad().clone()
                vector_to_parameters(saved_x, self._params)
                self.zero_grad()

                return z

            x0 = parameters_to_vector(self._params).clone()
            gradient = -1.0 * F(x0)

            p = gradient.clone()
            rho = torch.dot(gradient, gradient)

            original_gradient_norm = torch.norm(gradient).item()
            for _ in range(max_newton):
                if self._convergence_check(
                    gradient, rel_newton_tol, abs_newton_tol, original_gradient_norm
                ):
                    # Converged
                    break
                if line_search_spec is None:
                    denom = torch.dot(p, p)
                    if denom.item() < 1.0e-6:
                        # Skip this update if the denominator is too small
                        break
                    alpha = torch.div(rho, denom)
                    x0 += lr * alpha * p
                else:
                    line_search_return = self._backtracking_line_search(
                        x0.clone(),
                        gradient.clone(),
                        p.clone(),
                        lr,
                        f,
                        F,
                        line_search_spec,
                    )
                    xk1 = line_search_return.xk1
                    fx1 = line_search_return.fx1
                    dx = line_search_return.dxk
                    new_loss = line_search_return.new_loss
                    x0 = xk1

                if self.viz_steps:
                    steps.append(x0)
                last_gradient = gradient.clone()
                gradient = -1.0 * F(x0).clone()

                beta = self._beta_calc(gradient, last_gradient, p)
                # direction reset
                beta = max(beta, 0.0)
                converged = self._convergence_check(
                    gradient, rel_newton_tol, abs_newton_tol, original_gradient_norm
                )
                if beta == 0.0 and converged:
                    # Converged
                    break
                rho = torch.dot(gradient, gradient)

                p = gradient + beta * p

            vector_to_parameters(x0, self._params)
            if new_loss is None:
                new_loss = f(x0)

            if self.viz_steps:
                return new_loss, steps

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
            new_loss = f(xk1)

        return _LineSearchReturn(xk1=xk1, fx1=fx1, dxk=dx, new_loss=new_loss)


class FletcherReeves(NonlinearConjugateGradient):
    """
    Fletcher, R.; Reeves; C, C.M. "Function Minimization by conjugate gradients" (1964)
    """

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        _ = last_conjugate_gradient
        num = torch.dot(gradient, gradient)
        den = torch.dot(last_gradient, last_gradient)
        beta = torch.div(num, den)

        return beta


class Daniels(NonlinearConjugateGradient):
    """
    Daniel, James W., "The Conjugate Gradient Method for Linear and Nonlinear Operator Equations"
    (1967)

    Not currently implemented, requires a Hessian-Vector product.
    """

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        # TODO This will require a bit of thought, and likely a change in function signature
        # the update is (gradient^T * Hessian * gradient) / (last_gradient^T * last_hessian * last_gradient)
        # Some way of storing "last numerator"?
        _ = (gradient, last_gradient, last_conjugate_gradient)
        raise NotImplementedError(
            "This method requires a hessian-vector product, which is not currently supported!"
            " Feel free to open a PR if you require it."
        )


class PolakRibiere(NonlinearConjugateGradient):
    """
    Polak, E.; Ribiere, G., "Note sur la convergence de methodes de directions conjuguees" (1969)
    """

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        _ = last_conjugate_gradient
        num = torch.dot(gradient, torch.sub(gradient, last_gradient))
        den = torch.dot(last_gradient, last_gradient)
        beta = torch.div(num, den)

        return beta


class HestenesStiefel(NonlinearConjugateGradient):
    """
    Hestenes, M.R.; Stiefel, E. "Methods of Conjugate Gradients for Solving Linear Systems" (1952)
    """

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        diff = torch.sub(gradient, last_gradient)
        num = torch.dot(gradient, diff)
        den = torch.dot(-last_conjugate_gradient, diff)
        beta = torch.div(num, den)

        return beta


class DaiYuan(NonlinearConjugateGradient):
    """
    Dai, Y.-H; Yuan, Y. "A nonlinear conjugate gradient method with strong global convergence
    property" (1999)
    """

    def _beta_calc(
        self, gradient: Tensor, last_gradient: Tensor, last_conjugate_gradient: Tensor
    ) -> Tensor:
        diff = torch.sub(gradient, last_gradient)
        num = torch.dot(gradient, gradient)
        den = torch.dot(-last_conjugate_gradient, diff)
        beta = torch.div(num, den)

        return beta
