"""Blank example: copy this file, rename it, and edit the four TODO blocks.

The worked-out model is recovery of a reaction coefficient in

    Dt u - Laplace u + q(x) u = g(x, t),   u = 0 on the boundary, u(0) = 0.

Run it with ``python -m mras.live blank_reaction --quick``: it identifies q
from the noise-free datum and from a noise level 5% datum
(``noise_level=0.05``), prints the verdicts, and writes figures to
results/blank_reaction/.

A problem declares its physics -- nothing else.  The model operator
f(q, u) of ``Dt u + f(q, u) = g`` is written once, split in two:

1. ``model(u, w)``      the q-free part <f(0, u), w> as a density (here
                        grad u . grad w, the Laplacian);
2. ``coupling(q, u, w)`` and ``coupling_adjoint(q, u, w)``  the q-dependent
                        part <f(q, u) - f(0, u), w> (here q u w) and the
                        density of its q-derivative's adjoint (here u w).

Densities carry no ``* dx``: the driver integrates every form with one
quadrature.  From these hooks the driver builds the data generator (the
model solved at ``q_true`` in the estimator's own discretisation), the MRAS step of the chosen
time scheme (``scheme``: implicit, semi-implicit, explicit) and its
solves, and the checklist probes the same coupling pair.  Comment out
``coupling`` below and the constructor raises ``TheorySpecError`` naming
exactly what is missing.
"""

from netgen.geom2d import SplineGeometry
from ngsolve import H1, L2, Mesh, exp, grad, pi, sin, x as xc, y as yc

from mras.domains import Domains
from mras.scheme import MRASProblem

BOX = ((0.0, 0.0), (10.0, 10.0))  # the square this demo solves on


# --- TODO 1: mesh and trial pair ---------------------------------------------
def plate(maxh, order=3, order_q=1):
    """Mesh on a square plate, homogeneous Dirichlet on U_h, L2 for Q_h.

    Any netgen/NGSolve mesh works.  Q_h is discontinuous of order
    ``order_q``: P1 for a smooth parameter like this one (best fit 0.22 % at
    h = 0.5; P0 4.7 %), P0 for a piecewise constant one on a mesh that
    follows its jumps.  A wide plate keeps the first Dirichlet eigenvalue
    small, which is what makes the gradient flow of scheme (4) converge
    within a short demo.
    """
    geo = SplineGeometry()
    geo.AddRectangle(*BOX, bcs=("wall",) * 4)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    Ufes = H1(mesh, order=order, dirichlet="wall")
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


class ReactionProblem(MRASProblem):
    """Identify the reaction coefficient q from state data."""

    name = "blank_reaction"
    sigma = 0.0    # f is affine in q here (the checklist proves it by probe):
                   # sigma = 0 converges through the persistent excitation,
                   # sigma = 1 through (A2); a nonlinear f needs sigma = 1
    # Display hints: g >= 0 and u0 = 0 keep the state positive (maximum
    # principle), so the pictures need not be symmetric about zero.
    positive_source = positive_state = True

    def __init__(self, h=0.5, dt=0.05, tend=10.0, k_u=3, k_q=1, seed=0):
        super().__init__(plate(h, k_u, k_q), dt=dt, tend=tend, k_u=k_u,
                         seed=seed)

        # --- TODO 2: the true parameter q_true(x) ----------------------------
        self.q_true = 0.3 + 2.0 * exp(-((xc - 3.5) ** 2 + (yc - 6.0) ** 2) / 5.0)
        self.u0 = 0.0  # initial state; the estimate starts at initial_guess()

    # --- TODO 3: the source g(x, t) -------------------------------------------
    def source(self, t):
        # Nonnegative and never switched off: persistent excitation of the
        # parameter is what makes q identifiable from the datum.
        bump = exp(-((xc - 5.0) ** 2 + (yc - 5.0) ** 2) / 8.0)
        return 2.0 * (1.0 + 0.5 * sin(2 * pi * t)) * bump

    # --- TODO 4: the model, f(q, u) = -Laplace u + q u ------------------------
    def model(self, u, w):
        """<f(0, u), w>: the q-free part, here the Laplacian."""
        return grad(u) * grad(w)

    def coupling(self, q, u, w):
        """<f(q, u) - f(0, u), w> = int q u w dx.  f is affine in q, and the
        (A2) pairing int (qa - qb)^2 u is >= 0 because the datum u >= 0."""
        return q * u * w

    def coupling_adjoint(self, q, u, w):
        """<f_q(q, u) p, w> = int p u w dx, transcribed independently."""
        return u * w


# --- runner registration (mras.live reads these markers) ----------------------
# No CHECK entry: the template is the dogfood path (the runner builds and
# runs it), not an audited ladder verdict; sizes stay small for --quick.

PROBLEM = ReactionProblem

DEFAULTS = {"blank_reaction": dict(tend=100.0)}  # a limit: runs stop at their target

QUICK = {"blank_reaction": dict(tend=2.0)}
