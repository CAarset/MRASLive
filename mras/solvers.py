"""Linear algebra shared by the MRAS step and the forward model.

What is constant over a run is assembled and factorised once; what changes
from step to step is assembled alone and solved with the constant factors as
preconditioner, iterated to round-off.  Nothing here approximates: a Krylov
solve that does not reach round-off falls back to a direct factorisation of
the full matrix.
"""

import functools

import numpy as np
from ngsolve import ET, BilinearForm, IntegrationRule, Norm, Projector, ds, dx
from ngsolve.krylovspace import CGSolver, GMResSolver

TOL = 1e-13     # relative residual of every Krylov solve
MAXIT = 200
CHECK = 1e-9    # true relative residual above which a solve is redone directly


def is_symmetric(mat, trials=2, tol=1e-12, seed=0):
    """Whether ``mat`` equals its transpose, probed on random vectors."""
    rng = np.random.default_rng(seed)
    x = mat.CreateColVector()
    for _ in range(trials):
        x.FV().NumPy()[:] = rng.standard_normal(len(x))
        ax = (mat * x).Evaluate()
        diff = (ax - mat.T * x).Evaluate()
        if Norm(diff) > tol * max(Norm(ax), 1e-300):
            return False
    return True


def factorize(mat, freedofs=None, symmetric=None):
    """Sparse Cholesky for a symmetric matrix, sparse LU otherwise."""
    if symmetric is None:
        symmetric = is_symmetric(mat)
    return mat.Inverse(freedofs, inverse="sparsecholesky" if symmetric
                       else "umfpack")


@functools.lru_cache(maxsize=None)
def measures(order, boundary=None):
    """``dx`` and ``ds`` with one explicit quadrature of ``order``; ``ds``
    on the boundaries named ``boundary`` (a name or regular expression of
    whichever mesh assembles the form) only, when given.

    NGSolve otherwise picks the quadrature from the space a form lives on,
    so the same integrand would be integrated differently on U_h, on Q_h
    and on U_h x Q_h; every form the driver builds uses these instead.
    (Restricting a measure afterwards, ``ds(intrules=...)(definedon=...)``,
    drops its quadrature: the restriction is built here, with it.)
    """
    volume = {et: IntegrationRule(et, order)
              for et in (ET.TRIG, ET.QUAD, ET.TET, ET.HEX)}
    edges = {et: IntegrationRule(et, order)
             for et in (ET.SEGM, ET.TRIG, ET.QUAD)}
    surface = (ds(intrules=edges) if boundary is None
               else ds(definedon=boundary, intrules=edges))
    return dx(intrules=volume), surface


def mass(space, measure=dx):
    """The assembled L2 mass matrix of ``space``."""
    u, v = space.TnT()
    form = BilinearForm(space, symmetric=False)
    form += u * v * measure
    form.Assemble()
    return form.mat


class Solve:
    """Solve ``op x = b`` on ``freedofs`` to round-off.

    ``pre`` is a factorised matrix close to ``op`` (the constant part of a
    step), ``symmetric`` picks CG over GMRES.  When the iteration does not
    reach round-off, ``direct()`` -- a callable returning ``op`` as an
    assembled matrix -- is factorised instead; without one, ``converged``
    is False and the caller must solve another way.  ``iterations`` keeps
    the count of the last solve for diagnostics.
    """

    def __init__(self, pre, freedofs=None, symmetric=False):
        self.pre = pre
        self.freedofs = freedofs
        self.symmetric = symmetric
        self.iterations = 0
        self.fallbacks = 0
        self.converged = True

    def __call__(self, op, b, x, direct=None, tol=TOL):
        """``tol`` is the relative residual to reach; looser than round-off
        only where an outer iteration (Newton's) corrects the rest."""
        if self.freedofs is not None:
            proj = Projector(self.freedofs, True)
            b = (proj * b).Evaluate()
        kind = CGSolver if self.symmetric else GMResSolver
        solver = kind(mat=op, pre=self.pre, tol=tol, maxiter=MAXIT,
                      printrates=False)
        x.data = solver * b
        self.iterations = solver.iterations
        residual = (b - op * x).Evaluate()
        if self.freedofs is not None:
            residual = (proj * residual).Evaluate()
        self.converged = bool(
            Norm(residual) <= max(CHECK, 10 * tol) * max(Norm(b), 1e-300))
        if self.converged or direct is None:
            return x
        self.fallbacks += 1
        x.data = direct().Inverse(self.freedofs, inverse="umfpack") * b
        return x
