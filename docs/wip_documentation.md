# Scalable Second Order Optimizers in PyTorch
This project is intended to investigate the use of second-order methods to speed the convergence of
Machine Learning problems, or for general optimization problems.

## Implemented Optimizers
### Nonlinear Conjugate Gradient Methdods
Conjugate gradient methods, in essence, search in directions that are said to be "conjugate" to each
other; that is, given two gradients u and v, and some matrix A, u and v are conjugate IFF u^T A v =
0. For a nonlinear problem, this matrix A is assumed to be the Hessian, even if its not explicitly
calculated. The "vanilla" conjugate gradient method assumes a linear system with a positive
definite Hessian. Nonlinear conjugate gradient methods make no such assumption about the Hessian
matrix, and can therefore solve/optimize nonlinear functions. Even better, NLCG methods typically
"learn" the Hessian, rather than calculating it explicitly.  Conveniently, the various methods only
vary by their calculation of the parameter Beta at each step. We have exploited this in our software
design; if you wish to extend our NLCG optimizer, simply inherit and define a custom overload for
the beta calculation. Methods implemented include:

    - Fletcher-Reeves
    - Polak-Ribiere
    - Hestenes-Stiefel
    - Dai-Yuan

We have additionally created a Daniels method interface, but as this requires evaluating the Hessian
directly, we have not implemented it. An interesting avenue of research would be determining the
best approximation for this (e.g. Quasi-Newton updates, Taylor Series expansion, etc.)

We provide a search reset of `max(beta, 0.0)` in the algorithm.

## Newton's Method (and its relatives)
Newton's method hinges on the notion of determining the search direction by scaling the
gradient with the inverse Hessian. This is equivalent to solving a linear system, and so we can
utilize solvers such as Conjugate-Gradient or Conjugate-Residual to solve them

### Matrix Free methods
Representing the Hessian requires O(n^2) memory, which is infeasible. All hope is not lost, however!
We do not typically care about the Hessian itself -- that is, most Newton-based algorithms do not
examine the individual members of the Hessian (there is one notable exception, discussed later). We
only care about the product of the Hessian with a vector, typically the gradient.

So, we use an alternative: approximation.

### Matrix-Free Conjugate Residual Optimizer
We can approximate this matrix-vector product in one fashion as follows:
TODO: Images of the F(x+mu d) ~= F(x) + muDF(x)d equation

The Taylor Series expansion provides an approximation of the Hessian, but not its inverse. So, we
must solve a linear system. The solver depends on the properties of the Hessian: if we can guarantee
it is positive-definite, we can use the Conjugate-Gradient method. If it is not, but is symmetric
(which the Taylor approximation guarantees), we can use the similar Conjugate-Residual method. TODO:
CR algo listing

Putting it all together:
TODO: Full ALGO listing

### Quasi-Newton/Secant Methods
Whereas Matrix-Free methods approximate via a specific finite-difference equation, Quasi-Newton
methods (traditionally) form a full matrix that approximates the true Hessian. They do this by
forming a finite-difference approximation of the Hessian as a Secant line:
TODO: Equation for Secant condition
which we call the **secant condition**. This takes two forms, the *direct* and *dual* 
Secant condition. However, this equation is underdetermined in dimensions higher than 1, so we must
impose constraints to solve it. The various methods are differentiated by which constraints they
impose:

    - Broyden: The update should be as small as possible (as measured by the Frobenius norm).
      No guarantee of symmetry or posite-definiteness.
    - Symmetric Rank-One: The update should be symmetric, and will be a rank-one update
    - Davidon-Fletcher-Powell: The update should be symmetric and positive-definite

Again, we do not care about the exact matrix or its contents --- only the result of the
matrix-vector product. So, rather than forming the matrix explicitly, we store the vectors used to
calculate the updates and, at each point, calculate each update times the vector of interest and
then sum the results. The key here is that we can calculate these as inner products and
vector-scalar products, no outer products (producing a matrix) are needed.

This does, however, leave the open question: what about the first matrix? We can form an initial
positive-definite approximation, such as:

    - The aforementioned Taylor series method
    - A (scaled) Identity Matrix
    - Some other diagonal matrix
    - An outer product of the initial gradient

All of these have a trivially calculatable matrix-vector product that doesn't require forming the
full matrix.

At each step, we can calculate new information about the curvature and use this to update our list
of relevant vectors to (hopefully) better approximate the true Hessian.

### Levenberg(-Marquardt)
Another method, proposed by Levenberg (TODO citation...), is a ``trust-region'' method that adjusts
between gradient descent and Newton's method by varying a parameter λ as the optimization
progresses, predicated around whether or not the optimization step was "good" (lowered the loss) or
not. 

TODO image of Levenberg's method

Some in the literature use the average of outer products of the gradient at each step to approximate
the Hessian -- we have opted to use the Matrix-Free conjugate residual method instead. It may be
interesting to consider, instead, the use of a Quasi-Newton method update or similar.

Marquardt modified this method to always use some of the information available in the Hessian by
use the diagonal of it:

TODO image of Levenberg-Marquardt

Unfortunately, there is no obvious way to implement this in our matrix-free methods and so we have
elected to not pursue it.