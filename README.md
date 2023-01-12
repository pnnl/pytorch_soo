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

### Line Search vs. Trust Region
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

### Direct vs. Inverse Quasi-Newton Methods
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
- `mu`: the value to use for the finite difference in the Hessian-vector approximation.
- `verbose`: Enables some additional print statements under invalid conditions.

#### Direct Trust Region

## Acknowledgements
This work was supported by Pacific Northwest National Lab, the University of Washington, and
Schweitzer Engineering Laboratories. A special thanks to Dr. Andrew Lumsdaine and Dr. Tony Chiang
for their time, support, feedback, and patience on this project.

![Sponsor Logos](imgs/logos/logos.png)