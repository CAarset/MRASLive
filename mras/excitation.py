"""Persistent excitation of the data: the operators, and where the data excite.

For f affine in q, ``B(t) = f_q'(q~, z(t)) : H -> U*`` does not depend on
q~, and the data are persistently exciting (Baumeister et al. 1997, as
Kaltenbacher-Nguyen 2021, Remark 1.6, recall it) if for a window length l

    sup_{t0 <= t1 < t2 <= t0 + l} || int_t1^t2 B(s) p ds ||_U*  >=  mu ||p||_H

for all t0 and p.  The window's Gramian ``Gamma = (1/l) int B^* J B ds``
has ``<Gamma p, p>_H = (1/l) int ||B(s) p||_U*^2 ds``, and (mu / l)^2 <=
lambda_min(Gamma).  ``mras.verify`` turns both into dense eigenproblems on
H_h; this module holds the operators they are built from, and the
pictures of where the data excite, region by region:

* ``GramianAccumulator``: G(t) = int_0^t B^T K^-1 B ds over a run's data,
  H-whitened (the mass of H_h made the identity), by the trapezoidal rule.
* ``conditional_excitation``: c_K = 1 / (G^-1)_KK in the H-orthonormal basis
  e_K = 1_K / |K|^(1/2) of P0: the least excitation of a direction with a
  unit component on cell K, all others free to compensate.
* ``weakest_direction``: the eigenvector of the smallest eigenvalue.
* ``cell_dose``: for a gradient coupling, <B(t) p, v> = int p grad z : grad v,
  Cauchy-Schwarz bounds every direction's excitation by the dose,
  <Gamma p, p> <= int p^2 D dx with D = (1/T) int_0^T |grad z|^2 dt, without
  an eigenproblem.  For each cell, lambda_min <= c_K <= G_KK <= D_K.

The U-norm is ||v||_U = ||grad v|| where U_h has a Dirichlet part (a norm by
Poincare-Friedrichs), the H1 norm where it has none: (A1), (A3) and the
persistent excitation all measure in it (mras.verify).  These quantities are
new here; neither MRAS paper computes the persistent excitation.
"""

import numpy as np
from ngsolve import (BilinearForm, GridFunction, InnerProduct, LinearForm,
                     MultiVector, TaskManager, grad)

from .solvers import factorize

RHO = 1e-13  # eigenvalues below RHO times the largest: round-off


# -- the operators ----------------------------------------------------------
def has_dirichlet(problem):
    """Whether U_h has a Dirichlet part (constrained dofs)."""
    free = problem.Ufes.FreeDofs()
    return free.NumSet() < len(free)


def u_gram(problem):
    """The U-norm on U_h, as its Gram form: ||grad v|| where U_h has a
    Dirichlet part (a norm by Poincare-Friedrichs, the paper's choice), the
    full H1 norm where it has none."""
    U = problem.Ufes
    u, v = U.TnT()
    form = BilinearForm(U)
    form += InnerProduct(grad(u), grad(v)) * problem.dx
    if not has_dirichlet(problem):
        form += InnerProduct(u, v) * problem.dx
    form.Assemble()
    return form


def dense(mat, n):
    """An assembled sparse matrix as a dense n x n array."""
    rows, cols, vals = mat.COO()
    out = np.zeros((n, n))
    np.add.at(out, (np.array(rows), np.array(cols)), np.array(vals))
    return out


def mass_whitening(problem):
    """L^-1 for the Cholesky factor L of the H_h mass matrix: eigenvalues
    relative to ||.||_H become plain symmetric eigenvalues of W G W^T."""
    Q = problem.Qfes
    p, s = Q.TnT()
    form = BilinearForm(Q)
    form += p * s * problem.dx
    form.Assemble()
    return np.linalg.inv(np.linalg.cholesky(dense(form.mat, Q.ndof)))


def linearisation_point(problem):
    """q~ for f_q'(q~, z): the truth where it is known, else the run's
    start; f affine in q does not see it."""
    for start in (problem.q_true, problem.q0):
        if start is not None:
            return problem.project_q(start)
    return problem.project_q(problem.initial_guess())


def coupling_matrix(problem, z, q_lin):
    """B = f_q'(q~, z): H_h -> U_h*, assembled."""
    U, Q = problem.Ufes, problem.Qfes
    p, v = Q.TrialFunction(), U.TestFunction()
    b = BilinearForm(trialspace=Q, testspace=U)
    b += p * problem.coupling_adjoint(q_lin, z, v) * problem.dx
    b.Assemble()
    return b.mat


def gramian_term(problem, mat, inv):
    """B^T K^-1 B on H_h, K the U-norm Gram matrix (``inv`` its factors):
    p^T (B^T K^-1 B) p = ||B p||_U*^2, dense."""
    Q = problem.Qfes
    cols = MultiVector(mat.CreateColVector(), Q.ndof)
    unit = mat.CreateRowVector()
    for j in range(Q.ndof):
        unit[:] = 0.0
        unit[j] = 1.0
        cols[j].data = mat * unit
    solved = MultiVector(mat.CreateColVector(), Q.ndof)
    solved[:] = inv * cols
    # One dense product for all n^2 inner products: the same sums as
    # InnerProduct(solved, cols), in another order (2.5x faster at n = 4398).
    a = np.array([solved[j].FV().NumPy() for j in range(Q.ndof)])
    b = np.array([cols[j].FV().NumPy() for j in range(Q.ndof)])
    return a @ b.T


# -- over a run -------------------------------------------------------------
class GramianAccumulator:
    """G(t) = int_0^t B(s)^T K^-1 B(s) ds over a run's data, H-whitened.

    ``add(t, z)`` takes the datum at time t (a GridFunction on U_h) and adds
    the trapezoidal panel from the previous sample (from t = 0, where the
    data of a run starting at rest excite nothing, for the first);
    ``integral()`` is W G(t) W^T and ``mean()`` that over t, the Gramian of
    the window [0, t].  Dense on H_h: for dim H_h up to a few thousand.
    """

    def __init__(self, problem):
        self.problem = problem
        self.whiten = mass_whitening(problem)
        self._inv = factorize(u_gram(problem).mat, problem.Ufes.FreeDofs(),
                              symmetric=True)
        self._q_lin = linearisation_point(problem)
        n = problem.Qfes.ndof
        self._total = np.zeros((n, n))
        self._previous = None
        self.t = 0.0

    def add(self, t, z):
        gram = gramian_term(self.problem,
                            coupling_matrix(self.problem, z, self._q_lin),
                            self._inv)
        gram = 0.5 * (gram + gram.T)
        width = t - self.t
        if self._previous is None:
            self._total += 0.5 * width * gram
        else:
            self._total += 0.5 * width * (gram + self._previous)
        self._previous, self.t = gram, t

    def integral(self):
        return self.whiten @ self._total @ self.whiten.T

    def mean(self):
        return self.integral() / self.t


def cell_dose(problem, frames):
    """D_K = (1/T) int_0^T (1/|K|) int_K |grad z|^2 dx dt per cell of H_h
    (P0), by the trapezoidal rule over ``frames`` -- (t, coefficients of z)
    pairs in increasing t, the first panel from t = 0 -- and the cells'
    areas.  For a gradient coupling it bounds the excitation of every cell:
    G_KK <= D_K for the window mean G of the same samples."""
    s = problem.Qfes.TestFunction()
    z = GridFunction(problem.Ufes)
    form = LinearForm(problem.Qfes)
    form += 1 * s * problem.dx
    form.Assemble()
    areas = form.vec.FV().NumPy().copy()
    total = np.zeros(problem.Qfes.ndof)
    previous, t_prev = None, 0.0
    with TaskManager():
        for t, values in frames:
            z.vec.FV().NumPy()[:] = values
            form = LinearForm(problem.Qfes)
            form += InnerProduct(grad(z), grad(z)) * s * problem.dx
            form.Assemble()
            now = form.vec.FV().NumPy() / areas
            if previous is None:
                total += 0.5 * (t - t_prev) * now
            else:
                total += 0.5 * (t - t_prev) * (now + previous)
            previous, t_prev = now, t
    return total / t_prev, areas


def conditional_excitation(gram, rho=RHO):
    """c_K = 1 / (G^-1)_KK for a whitened Gramian G (P0: the basis e_K is
    H-orthonormal): min { <G p, p> : (p, e_K)_H = 1 }.  Through the
    eigenpairs, eigenvalues below ``rho`` times the largest -- round-off --
    raised to that level.  Returns (c, eigenvalues, eigenvectors)."""
    lam, vec = np.linalg.eigh(0.5 * (gram + gram.T))
    floor = rho * lam[-1]
    inv_diag = (vec ** 2) @ (1.0 / np.maximum(lam, floor))
    return 1.0 / inv_diag, lam, vec


def weakest_direction(vec, whiten):
    """The eigenvector of the smallest eigenvalue as values on the cells
    (P0, W diagonal), scaled to max |p| = 1."""
    p = whiten.T @ vec[:, 0]
    return p / np.abs(p).max()


def sublevel(values, eps):
    """The cells where ``values`` is at most ``eps``: a sublevel set."""
    return np.asarray(values) <= eps
