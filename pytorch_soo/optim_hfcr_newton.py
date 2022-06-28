"""
An implementation of a Hessian-free, conjugate residual newton optimizer.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import warnings
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from pytorch_soo.line_search_spec import LineSearchSpec


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


class HFCR_Newton(Optimizer):
    """
    Implements the Inexact Newton algorithm with Hessian matrix free conjugate
    residual method.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
                            parameter groups
        lr (float, optional): learning rate (default=0.3333). If line_search_spec is not none,
            this will be the maximum linesearch step (consider setting =1.0)
        max_cr (int, optional): how many conjugate residual iterations to run (default = 10)
        max_newton (int, optional): how many newton iterations to run (default = 10)
        abs_newton_tol (float, optional): absolute tolerance for Newton iteration convergence (default=1.E-3)
        rel_newton_tol (float, optional): relative tolerance for Newton iteration convergence (default=1.E-3)
        cr_tol (float, optional): tolerance for conjugate residual iteration convergence (default=1.E-3)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.3333,
        max_cr: int = 10,
        max_newton: int = 10,
        abs_newton_tol: float = 1.0e-3,
        rel_newton_tol: float = 1.0e-5,
        cr_tol: float = 1.0e-3,
        line_search_spec: Optional[LineSearchSpec] = None,
    ):
        # ensure inputs are valid
        if lr < 0.0:
            raise ValueError(f"Invalid learnign rate: {lr} - should be >= 0.0")
        if max_cr < 1:
            raise ValueError(f"Invalid max_cr: {max_cr} - should be >= 1")
        if max_newton < 1:
            raise ValueError(f"Invalid max_newton: {max_newton} - should be >= 1")
        if abs_newton_tol <= 0.0:
            raise ValueError(
                f"Invalid Absolute Newton tolerance: {abs_newton_tol} - must be > 0.0!"
            )
        if rel_newton_tol <= 0.0:
            raise ValueError(
                f"Invalid Relative Newton tolerance: {rel_newton_tol} - must be > 0.0!"
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
            max_cr=max_cr,
            max_newton=max_newton,
            abs_newton_tol=abs_newton_tol,
            rel_newton_tol=rel_newton_tol,
            cr_tol=cr_tol,
            line_search_spec=line_search_spec,
        )
        super(HFCR_Newton, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "HFCR_Newton doesn't support per-parameter options "
                "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]

    def _Hessian_free_product(self, grad_x0, x0, d, closure):
        """
        Computes the Hessian vector product by using a finite-difference approximation
        Hd = (1 / eps) * delta_f(x + eps * d) - delta_f(x), where H is the Hessian,
        d is the vector which has the same dimension as x, delta_f is the first order
        derivative, eps is a small scalar.

        Arguments:
        grad_x0 (torch.tensor): flatten tensor, denotes the flat grad of x0
        x0 (torch.tensor): flatten tensor, denotes the flat parameters
        d(torch.tensor): flatten tensor.

        Return:
        Hv_free (torch.tensor): Flat tensor
        """
        # calculate eps
        a = torch.norm(d).item()
        eps = 1.0e-3 * torch.div(torch.norm(x0).item(), a)

        x_new = torch.add(x0, d, alpha=eps)

        grad_x_new = self._get_changed_grad(x_new, closure)

        Hv_free = torch.div(1.0, eps) * (grad_x_new - grad_x0)

        return Hv_free

    def _Hessian_free_cr(self, grad_x0, x0, dk, rk, max_iter, tol, closure):
        """
        Use conjugate residual for Hessian free.
        """
        A = lambda d: self._Hessian_free_product(grad_x0, x0, d, closure)
        return self._cr(A, dk, rk, max_iter, tol)

    def _cr(self, A, dk, rk, max_cr, tol):
        """
        The conjugate residual method method to solve ``Ax = b``, where A is
        required to be symmetric.

        Arguments:
            A (callable): An operator implementing the Hessian free
                Hessian vector product Ax.
            dk (torch.Tensor): An initial guess for x.
            rk (torch.Tensor): The vector b in ``Ax = b``.
            max_cr (int): maximum iterations.
            tol (float, optional): Termination tolerance for convergence.

        Return:
            dk (torch.Tensor): The approximation x in ``Ax = b``.
        """

        r = rk.clone()
        p = r.clone()
        w = A(p)
        q = w.clone()

        norm0 = torch.norm(r).item()
        rho_0 = torch.dot(q, r).item()

        iter = 0

        converged = False

        while iter < max_cr:
            iter += 1
            denom = torch.dot(w, w).item()
            if denom < 1.0e-8:
                break

            alpha = torch.div(rho_0, denom)

            dk.add_(p, alpha=alpha)
            r.sub_(w, alpha=alpha)

            res_i_norm = torch.norm(r).item()

            if torch.div(res_i_norm, norm0) < tol:
                break

            q = A(r)
            rho_1 = torch.dot(q, r).item()

            if abs(rho_1) < 1.0e-8:
                break

            beta = torch.div(rho_1, rho_0)
            rho_0 = rho_1
            p.mul_(beta).add_(r)
            w.mul_(beta).add_(q)

        return dk

    def _get_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _get_changed_grad(self, vec, closure):
        """Calculate the gradient of model parameters given the new parameters.
        Note that we are not really changing model parameters at this moment.

        Argument:
        vec (torch.tensor): a flatten tensor, the new model parameters.
        closure: used to re-evaluate model.

        Return:
        new_flat_grad (torch.tensor): a flatten tensor, the gradient of such
                                      new parameters.
        """
        current_params = parameters_to_vector(self._params)
        vector_to_parameters(vec, self._params)

        with torch.set_grad_enabled(True):
            orig_loss = closure()
            new_flat_grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        self.zero_grad()

        return new_flat_grad

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

    def step(self, closure: Callable):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr = group["lr"]
        max_cr = group["max_cr"]
        max_newton = group["max_newton"]
        abs_newton_tol = group["abs_newton_tol"]
        rel_newton_tol = group["rel_newton_tol"]
        cr_tol = group["cr_tol"]
        line_search_spec = group["line_search_spec"]

        # evaluate initial
        orig_loss = closure()

        # Initialize
        x0 = parameters_to_vector(self._params)
        rk = self._get_flat_grad().neg()

        res_norm_1 = torch.norm(rk).item()
        res_norm_0 = torch.norm(rk).item()
        res_ratio = float("inf")

        n_iter = 0

        def f(y):
            """Wrapper to obtain the loss"""
            x = torch.clone(y)
            saved_x = parameters_to_vector(self._params)
            vector_to_parameters(x, self._params)
            loss = closure()
            vector_to_parameters(saved_x, self._params)
            self.zero_grad()

            return loss

        def F(y):
            return self._get_changed_grad(y, closure)

        grad_x0 = None
        while (
            n_iter < max_newton
            and res_ratio > rel_newton_tol
            and res_norm_1 > abs_newton_tol
        ):
            n_iter += 1

            dk = torch.zeros_like(rk)

            if grad_x0 is None:
                grad_x0 = F(x0)

            dk = self._Hessian_free_cr(grad_x0, x0, dk, rk, max_cr, cr_tol, closure)
            if line_search_spec is None:
                # Update parameters
                x0.add_(dk, alpha=lr)
            else:
                line_search_ret = self._backtracking_line_search(
                    x0, grad_x0, dk, lr, f, F, line_search_spec
                )
                x0 = line_search_ret.xk1
                grad_x0 = line_search_ret.fx1
                orig_loss = line_search_ret.new_loss
                print("dx: ", torch.norm(line_search_ret.dxk))

            # update grad based on new parameters
            rk = self._get_changed_grad(x0, closure).neg()
            res_norm_1 = torch.norm(rk).item()
            res_ratio = torch.div(res_norm_1, res_norm_0)

        # set parameters
        vector_to_parameters(x0, self._params)

        return orig_loss
