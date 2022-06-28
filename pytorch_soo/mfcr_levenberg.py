"""
Implements several variants of the Levenberg algorithm.
    - LevenbergEveryStep adjusts the value of lambda with each step of the Newton iteration
    - LevenbergEndStep takes several steps and then adjusts lambda, retracting all if needed.
    - LevenbergBatch requires an additional call to adjust lambda after all batches have been
      used for that training epoch.
"""
import torch
from torch.optim import Optimizer
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

# TODO Eventually write this using inheritance...
class LevenbergEveryStep(Optimizer):
    """
    Implements the Levenberg method using a quasi-newton method and Hessian Free
    conjugate residual solver. Adjusts lambda per newton step.
        lr: The learning rate. Must be >0.0
        lambda0: The initial value of lambda, or the influence of gradient vs. Hessian. Must be >0.0
        max_lambda: The largest value lambda can scale to. Must be >0.0 and > min_lambda
        min_lambda: The smallest value lambda can scale to. Must be >0.0 and < max_lambda
        nu: The multiplier used to scale lambda as the optimizer steps improve or fail to improve
            the loss. Must be >1.0
        max_cr: The maximum number of conjugate residual iterations that can be taken when solving
            Ax=b. Must be >=1.
        max_newton: The maximum number of newton iterations that can be taken for this batch. Must
            be >=1
        cr_tol: The tolerance to consider the conjugate residual as converged, prompting early
            return. Must be >=0.0
        newton_tol: The tolerance to consider the newton iteration as converged, prompting early
            return. Must be >=0.0
        debug: Whether to track the values for lambda and loss as the algorithm progresses. If
            enabled, the step() method will return orig loss as well as these values in a tuple.
    """

    def __init__(
        self,
        params,
        lr=0.3333,
        lambda0=1.0,
        max_lambda=1000.0,
        min_lambda=1e-6,
        nu=2.0**0.5,
        max_cr=10,
        max_newton=1,
        newton_tol=1.0e-3,
        cr_tol=1.0e-3,
        debug=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if lambda0 <= 0.0:
            raise ValueError(f"Invalid lambda0: {lambda0} - should be > 0.0")
        if max_lambda <= 0.0:
            raise ValueError(f"Invalid max_lambda: {max_lambda} - should be > 0.0")
        if min_lambda <= 0.0:
            raise ValueError(f"Invalid min_lambda: {min_lambda} - should be > 0.0")
        if max_lambda <= min_lambda:
            raise ValueError(
                f"Invalid max and min lambda: {max_lambda}, {min_lambda} - max_lambda should be > min_lambda"
            )
        if nu <= 1.0:
            raise ValueError(f"Invalid nu: {nu} - should be > 1.0")
        if max_cr < 1:
            raise ValueError(f"Invalid max_cr: {max_cr} - should be >= 1")
        if cr_tol < 0.0:
            raise ValueError(f"Invalid cr_tol: {cr_tol} - should be >= 0.0")
        if max_newton < 1:
            raise ValueError(f"Invalid max_newton: {max_newton} - should be >= 1")
        if newton_tol < 0.0:
            raise ValueError(f"Invalid newton_tol: {newton_tol} - should be >= 0.0")

        defaults = dict(
            lr=lr,
            lambda_=lambda0,
            max_lambda=max_lambda,
            min_lambda=min_lambda,
            nu=nu,
            max_cr=max_cr,
            max_newton=max_newton,
            newton_tol=newton_tol,
            cr_tol=cr_tol,
            debug=debug,
        )

        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "HFCR_Newton doesn't support per-parameter options "
                "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self.lambda_ = lambda0

    def _Hessian_free_product(self, grad_x0, x0, d, lambda_, closure):
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
        eps = 1.0e-6 * torch.div(torch.norm(x0).item(), a)

        x_new = torch.add(x0, d, alpha=eps)

        grad_x_new = self._get_changed_grad(x_new, closure)

        Hv_free = torch.div(1.0, eps) * (grad_x_new - grad_x0) + torch.mul(lambda_, d)

        return Hv_free

    def _Hessian_free_cr(self, grad_x0, x0, dk, rk, lambda_, max_iter, tol, closure):
        """
        Use conjugate residual for Hessian free.
        """
        A = lambda d: self._Hessian_free_product(grad_x0, x0, d, lambda_, closure)
        return self._cr(A, dk, rk, max_iter, tol)

    @staticmethod
    def _cr(A, dk, rk, max_cr, tol):
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

        cr_iter = 0

        while cr_iter < max_cr:
            cr_iter += 1
            denom = torch.dot(w, w).item()
            if denom < 1.0e-16:
                break

            alpha = torch.div(rho_0, denom)

            dk.add_(p, alpha=alpha)
            r.sub_(w, alpha=alpha)

            res_i_norm = torch.norm(r).item()

            if torch.div(res_i_norm, norm0) < tol or cr_iter == (max_cr - 1):
                break

            q = A(r)
            rho_1 = torch.dot(q, r).item()

            if abs(rho_1) < 1.0e-16:
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
        """
        Calculate the gradient of model parameters given the new parameters.
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
        _ = closure()
        new_flat_grad = self._get_flat_grad()
        vector_to_parameters(current_params, self._params)
        self.zero_grad()

        return new_flat_grad

    def step(self, closure):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        assert len(self.param_groups) == 1

        group = self.param_groups[0]

        lr = group["lr"]
        min_lambda = group["min_lambda"]
        max_lambda = group["max_lambda"]
        nu = group["nu"]
        max_cr = group["max_cr"]
        max_newton = group["max_newton"]
        newton_tol = group["newton_tol"]
        cr_tol = group["cr_tol"]
        debug = group["debug"]

        # evaluate initial
        orig_loss = closure()

        # Completely original params, if multiple newton steps are taken
        x0 = parameters_to_vector(self._params)
        # "Current" params for the newton step
        x1 = x0.clone()

        rk = self._get_flat_grad().neg()

        res_norm_1 = torch.norm(rk).item()
        res_norm_0 = torch.norm(rk).item()

        n_iter = 0
        # Insurance so each epoch will at least try 1 iteration
        if self.lambda_ < min_lambda:
            self.lambda_ *= nu
        if self.lambda_ > max_lambda:
            self.lambda_ /= nu

        if debug:
            losses = []

        while (
            n_iter < max_newton
            and torch.div(res_norm_1, res_norm_0) > newton_tol
            and min_lambda < self.lambda_ < max_lambda
        ):
            if debug:
                iter_losses = []
            n_iter += 1

            dk = torch.zeros_like(rk)

            grad_xk = self._get_changed_grad(x1, closure)

            # Hessian free Conjugate Residual
            dk = self._Hessian_free_cr(
                grad_xk, x1, dk, rk, self.lambda_, max_cr, cr_tol, closure
            )

            # Update parameters
            x1.add_(dk, alpha=lr)

            vector_to_parameters(x1, self._params)
            new_loss = closure()
            vector_to_parameters(x0, self._params)
            if debug:
                iter_losses.append(new_loss.item())

            if new_loss < orig_loss:
                # Decrease the direct influence of the gradient
                self.lambda_ /= nu
                # update grad based on new parameters
                rk = self._get_changed_grad(x1, closure).neg()
                res_norm_1 = torch.norm(rk).item()
                x0 = x1.clone()
                if debug:
                    losses.append(iter_losses)

            else:
                # Retract the step
                n_iter -= 1
                x1 = x0.clone()
                # Increase the direct influence of the gradient
                self.lambda_ *= nu

        # set parameters
        vector_to_parameters(x0, self._params)
        if debug:
            return (
                orig_loss,
                lambdas,
                losses,
            )

        return orig_loss
