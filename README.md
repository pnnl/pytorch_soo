# PyTorch SOO: Second Order Optimizers in PyTorch
This repository is intended to enable the use of second-order (i.e. including curvature information)
optimizers in PyTorch. This can be for machine learning applications or for generic optimization
problems.

There is also a significant body of code used for running and analyzing experiments for
understanding the convergence properties of these optimizers as applied to machine learning.

Further documentation, examples, and overall clean-up of the API is forthcoming. Please feel free to
open issues and PR's as required on [GitHub](https://github.com/pnnl/pytorch_soo).

## Installation
This project relies solely upon [PyTorch](https://pytorch.org/). Install either the latest version:
```bash
python -m pip install torch torchvision torchaudio
```
or your preferred subvariant (older CUDA version or CPU only) using their instructions. Then, simply
run:
```bash
python -m pip install pytorch-soo
```

### Dependencies for experimental code
There is a significant body of code contained in the repository (not the package itself) that was
used for conducting studies on the behavior of these optimizers. These have not been
specifically enumerated but can be readily determined by inspection of the relevant scripts and Jupyter notebooks under the `experiments/` subdirectory.

## The Optimizers
Three primary classes of optimizers have been implemented herein:

- Hessian-free Conjugate Residual Krylov-Newton
- Nonlinear Conjugate Gradient
- Quasi-Newton

### Hessian-free Conjugate Residual Krylov-Newton (`HFCR_Newton`)
An adaptation of [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method), but uses a
[Krylov Subspace](https://en.wikipedia.org/wiki/Krylov_subspace) solver rather inverting a matrix
directly. Specifically, it utilizes the [Conjugate
Residual](https://en.wikipedia.org/wiki/Conjugate_residual_method) solver, which only requires that
the matrix in question need be hermitian, but not positive-definite (which is not guaranteed in
non-convex optimizaton).

Arguments:

- `params`: The model parameters
- `lr`: The learning rate/maximum stepsize. If no line search is used, it functions as a damping coefficient. If one is used, it is the starting/maximum value of the backtracking line search
- `max_cr`: The maximum number of conjugate residual iterations that can be performed. Note that an inexact solution may still be "good enough" and convergence of the solver isn't required.
- `max_newton`: The maximum number of newton steps that may be taken per optimizer step.
- `abs_newton_tol`: The maximum absolute value of the grad norm before the optimizer is considered converged
- `rel_newton_tol`: The relative absolute value of the grad norm before the optimizer is considered converged
- `cr_tol`: The absolute value of the residual in the CR solver before it is considered converged
- `line_search_spec`: (Optional) A dataclass defining the specifications for the line search algorithm.

### Nonlinear Conjugate Gradient (`NonlinearConjugateGradient`)
[Nonlinear Conjugate Gradient Methods](https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method)
are a generalization of
[Conjugate Gradient Methods](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
to nonlinear optimization problems. Four are implemented herein, varying only in the
calculation of the value β. These are:

- Fletcher-Reeves
- Polak-Ribiere
- Hestenes-Stiefel
- Dai-Yuan

All methods use an automatic reset of β to zero when it is negative. β is also reset between
minibatches (no attempt is made to retain conjugacy).  All methods support both a backtracking line
search that must satisfy the Armijo condition and optionally  the Wolfe conditions (not
recommended).

As a note, there is a class for the Daniel's method; however, this requires a Hessian-Gradient product that will require some change in the current implementation to support.

Arguments:

- `params`: The model parameters
- `lr`: The learning rate/maximum stepsize. If no line search is used, it functions as a damping coefficient. If one is used, it is the starting/maximum value of the backtracking line search
- `max_newton`: A misnomer, but the maximum number of steps that may be taken per optimizer step.
- `abs_newton_tol`: The maximum absolute value of the grad norm before the optimizer is considered converged
- `rel_newton_tol`: The relative absolute value of the grad norm before the optimizer is considered converged
- `line_search_spec`: (Optional) A dataclass defining the specifications for the line search algorithm.
- `viz_steps`: An extra argument used for the purposes of visualization; if true, the algorithm will track all conjugate gradient steps and return them as a tuple with the loss

### (Limited Memory) Matrix-Free Quasi-Newton Methods
An adaptation of [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method), but uses an
approximation of the Hessian -- hence,
[Quasi-Newton](https://en.wikipedia.org/wiki/Quasi-Newton_method).
Furthermore, no Hessian is explicitly formed, and instead the Hessian-vector product is
approximated. There are four possible basic versions:

| Implemented? | Line Search | Trust Region    |
|--------------|-------------|-----------------|
| **Direct**   | **✓**       | **✓**           |
| **Inverse**  | **✓**       | **X**           |

Implemented Optimizers include:
- `BFGS`
- `DavidonFletcherPowell`
- `Brodyen`

#### Line Search vs. Trust Region
[Line search methods](https://en.wikipedia.org/wiki/Line_search) determine step direction first,
then (attempt to) determine the optimal step size, usually in an inexact fashion. This inexact
optimality is usually qualified by conditions such as the
[Armijo-Goldstein conditon](https://sites.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf)
or the [(Strong) Wolfe Conditions](https://en.wikipedia.org/wiki/Wolfe_conditions). Both are supported
here; the latter is preferred for QN methods.  This implementation supports both a backtracking line
search as well as the "zoom" line search as implemented by [SciPy](https://github.com/scipy/scipy)
in their optimization toolbox, adapted to work with PyTorch. The latter is recommended and is
enabled by providing a curvature condition in the line search specification.

[Trust region methods](https://en.wikipedia.org/wiki/Trust_region) first set a step size, then attempt
to determine the optimal step direction. This is typically accomplished by modeling the function
locally as a simpler function (e.g. a quadratic), solving for the minimum directly, then comparing the expected improvement to the actual improvement. Based upon this result, the step size is either
increased (if a "good" approximation), kept the same size, or shrunk (a "bad" approximation).
These require a solver for the so-called "trust region subproblem" -- only the
[Conjugate-Gradient Steihaug](https://optimization.cbe.cornell.edu/index.php?title=Trust-region_methods)
method has been implemented, although stubs exist for the Cauchy Point and Dogleg methods.

#### Direct vs. Inverse Quasi-Newton Methods
Direct QN methods "directly" form the Hessian $B$ and then solve it via an appropriate Krylov
solver, subject to the properties of the Hessian (e.g. hermitian or not, positive definite or not,
etc.).

Inverse methods form the inverse Hessian $H=B^{-1}$ and require no solver to invert the Hessian.


#### Direct Line Search (`QuasiNewton`)
Arguments:
- `params`: The model parameters
- `lr`: The learning rate/maximum stepsize. If no line search is used, it functions as a damping coefficient. If one is used, it is the starting/maximum value of the backtracking line search
- `max_newton`: The maximum number of newton steps that may be taken per optimizer step.
- `max_krylov`: The maximum number of Krylov iterations that can be performed. Note that an inexact solution may still be "good enough" and convergence of the solver isn't required.
- `abs_newton_tol`: The maximum absolute value of the grad norm before the optimizer is considered converged
- `rel_newton_tol`: The relative absolute value of the grad norm before the optimizer is considered converged
- `krylov_tol`: The absolute value of the residual in the Krylov solver before it is considered converged
- `matrix_free_memory`: (Optional) The number of prior steps to retain. If None, the memory will grow without bound (not recommended).
- `line_search_spec`: (Optional) A dataclass defining the specifications for the line search algorithm.
- `mu`: The value to use for the finite difference in the Hessian-vector approximation.
- `verbose`: Enables some additional print statements under invalid conditions.

#### Direct Trust Region (`QuasiNewtonTrust`)
Arguments:
- `params`: The model parameters
- `trust_region`: A TrustRegionSpec object dictating the trust region specifications
- `lr`: A damping factor for your step size (recommended to leave at 1)
- `max_newton`: The maximum number of newton steps that may be taken per optimizer step.
- `abs_newton_tol`: The maximum absolute value of the grad norm before the optimizer is considered converged
- `rel_newton_tol`: The relative absolute value of the grad norm before the optimizer is considered converged
- `matrix_free_memory`: (Optional) The number of prior steps to retain. If None, the memory will grow without bound (not recommended).
- `mu`: The value to use for the finite difference in the Hessian-vector approximation.

#### Inverse Line Search and Trust Region (`InverseQuasiNewton`)
Arguments:
- `params`: The model parameters
- `lr`: The learning rate/maximum stepsize. If no line search is used, it functions as a damping coefficient. If one is used, it is the starting/maximum value of the backtracking line search
- `max_newton`: The maximum number of newton steps that may be taken per optimizer step.
- `abs_newton_tol`: The maximum absolute value of the grad norm before the optimizer is considered converged
- `rel_newton_tol`: The relative absolute value of the grad norm before the optimizer is considered converged
- `matrix_free_memory`: (Optional) The number of prior steps to retain. If None, the memory will grow without bound (not recommended).
- `line_search_spec`: (Optional) A dataclass defining the specifications for the line search algorithm.
- `trust_region`: A TrustRegionSpec object dictating the trust region specifications
- `verbose`: Enables some additional print statements under invalid conditions.

**Note**: breaking with the other classes, this class can accept either a `LineSearchSpec` or `TrustRegionSpec` (but not both!). However, as the trust region is not implemented for this class, it is functionally only a line search optimizer.

## Other Classes of Note
### `LineSearchSpec`
A frozen dataclass with parameters:
- `max_searches`
- `extrapolation_factor` (note this should be $<1$)
- `sufficient_decrease`
- `curvature_constant` (Optional)

### `TrustRegionSpec`
A frozen dataclass with parameters:
- `initial_radius`: Initial trust region radius
- `max_radius`: Maximum trust region radius
- `nabla0`: see below
- `nabla1`: see below
- `nabla2`: see below
- `shrink_factor`: Factor by which the radius shrinks (should be $∈(0,1)$)
- `growth_factor`: Factor by which the radius grows (should be $∈(1,∞)$)
- `trust_region_subproblem_solver`: Which subproblem solver to use (only `cg` is currently valid)
- `trust_region_subproblem_tol`: The tolerance for subproblem convergence
- `trust_region_subproblem_iter`: A limit on how many iterations the solver can use (Optional)

#### Nabla values
These terms are actually misnamed and should be called "eta".

These set the limits for whether the quadratic model of the method is "good". `nabla0` is the
value below which an abject failure occurs and the step is rejected entirely.
`nabla1` is the value below which the step is accepted but the radius is decreased. `nabla2` is the value above which the step is accepted and the radius is increased.

Note that $0\leq\eta_0 \leq \eta_1 \leq \eta_2 \leq 1$; in practice, it suggested that
$\eta_0 \ll 1$ ("very small", on the order of 1e-4) and that $\eta_0 < \eta_1 < \eta_2 < 1$.

### Solver
Linear system solvers (i.e. $Ax=b$) that support functor behavior with inputs:
- `A`: a matrix(-like) object supporting matvec multiplication
- `x0`: The initial guess
- `b`: the `b` vector

Arguments:
- `max_iter`: Maximum iterations the solver can take
- `tol`: an absolute tolerance for convergence

#### Implemented Solvers
- Conjugate Gradient
- Conjugate Residual

Other solvers (such as MINRES, GMRES, Biconjugate Gradient, etc.) could be added if desired.

### (Inverse) Matrix-Free Operators
The basis of the QN methods, this object acts like a matrix and accepts updates using the
current gradient, the last gradient, and the change in $x$. Also supports constructing a full
matrix by multiplying by each basis vector and stacking the results (for debugging purposes).


`MatrixFreeOperator` Arguments:
- `B0p` A callable accepting a tensor and returning a tensor. Should be the closure used by the second order methods if possible. If this is not available
at construction, add it later using the `change_B0p` method.
- `n`: the size of the memory. If none, it is unlimited.

`InverseMatrixFreeOperator` Arguments:
- `n`: the size of the memory. If none, it is unlimited.

There are also matrix operators, which actually form the full matrix. These are not suggested except for small optimization problems.

#### Implemented Operators:
- BFGS
- SR1
- DFP

Note that, technically, the SR1 Dual is also available -- its inverse **is** its dual.

## Possible Improvements
- [ ] Implement Trust region inverse QN method(s).
- [ ] Refactor QN classes to reduce duplicated code. Consider Multiple inheritance to match the 2x2 matrix of line search vs. trust region and direct vs. inverse
- [ ] Rename the `nabla` variables to `eta`; the author was lacking sleep when identifying the greek letter used in his sources
- [ ] Update the Matrix Free operators to use the newer `__matmul__` dunder method instead of just `__mul__`

## Acknowledgements
This work was supported by Pacific Northwest National Lab, the University of Washington, and
Schweitzer Engineering Laboratories. A special thanks to Dr. Andrew Lumsdaine and Dr. Tony Chiang
for their time, support, feedback, and patience on this project.

![Sponsor Logos](imgs/logos/logos.png)