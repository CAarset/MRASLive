"""Nonlinear advection-diffusion benchmark (paper Section 5.2, Figures 7-8).

The reference state u^dagger is a steady nonlinear vorticity speed field
decaying linearly in time toward a positive offset; the parameter is
q^dagger = pi - r on the disk of radius pi.
"""

import math

import numpy as np
from ngsolve import (BilinearForm, GridFunction, H1, IfPos, Integrate,
                     LinearForm, Norm, Projector, dx, grad, sqrt, x as xc,
                     y as yc)
from ngsolve.krylovspace import GMResSolver

from mras.domains import potential
from mras.fields import l2norm
from mras.solvers import factorize

from mras.scheme import MRASProblem

NU = 0.1                    # vorticity-solve diffusion coefficient
DECAY_T = 6.0               # time scale of the linear decay (DECAY_T - t) / DECAY_T
OFFSET = 6.0                # positive offset required by the analysis
Z_MIN, Z_MAX = 0.1, 100.0   # state bounds entering the stabilisation bound C(.)
# Picard: residual reduction to reach, iteration cap, damping of the update
PICARD_TOL, PICARD_MAXIT, DAMPING = 1e-10, 200, 0.7


def reference_speed(mesh, seed=2):
    """Interior-masked speed field of the nonlinear vorticity reference state.

    The steady vorticity ``-nu Laplacian(omega) + (u . grad) omega = f``
    with streamfunction velocity ``u = (psi_y, -psi_x)``, ``-Laplacian(psi)
    = omega``, and a seeded Gaussian source, solved by damped Picard
    iteration until the residual on the free dofs has dropped by
    ``PICARD_TOL``.  The Laplacian does not change and is factorised once:
    it gives the streamfunction, and -- scaled by ``1 / nu`` -- preconditions
    the vorticity solve.  Each Picard correction needs only a few digits,
    since the exact residual of the next iteration corrects the rest, so
    GMRES stops at 1 %.
    """
    fes = H1(mesh, order=3, dirichlet=".*")
    trial, test = fes.TnT()
    free = fes.FreeDofs()
    keep = Projector(free, True)

    rng = np.random.default_rng(seed)
    f = GridFunction(fes)
    f.vec.FV().NumPy()[:] = rng.normal(size=fes.ndof)
    load = LinearForm(fes)
    load += f * test * dx
    load.Assemble()
    laplace = BilinearForm(fes, symmetric=True)
    laplace += grad(trial) * grad(test) * dx
    laplace.Assemble()
    laplace_inv = factorize(laplace.mat, free, symmetric=True)
    mass = BilinearForm(fes, symmetric=True)
    mass += trial * test * dx
    mass.Assemble()

    omega, psi = GridFunction(fes), GridFunction(fes)
    pre, first = (1 / NU) * laplace_inv, None
    for _ in range(PICARD_MAXIT):
        psi.vec.data = laplace_inv * (mass.mat * omega.vec)
        u_vel, v_vel = grad(psi)[1], -grad(psi)[0]
        a = BilinearForm(fes)
        a += NU * grad(trial) * grad(test) * dx
        a += (u_vel * grad(trial)[0] + v_vel * grad(trial)[1]) * test * dx
        a.Assemble()
        res = (keep * (load.vec - a.mat * omega.vec)).Evaluate()
        first = first or Norm(res)
        if Norm(res) <= PICARD_TOL * first:
            break
        solver = GMResSolver(mat=a.mat, pre=pre, tol=1e-2, maxiter=100,
                             printrates=False)
        omega.vec.data += DAMPING * (solver * res)
    else:
        raise RuntimeError(
            f"the reference vorticity did not converge (residual "
            f"{Norm(res) / first:.1e} of its start after {PICARD_MAXIT} "
            f"Picard steps): a mesh this coarse has no fixed point to "
            f"converge to; h = 0.25 or finer does")

    psi.vec.data = laplace_inv * (mass.mat * omega.vec)
    velocity = grad(psi)
    r = sqrt(xc * xc + yc * yc)
    mask = IfPos(0.5 - (math.pi - r), math.pi - r, 0.5)
    return mask * Norm(velocity)


class PotentialProblem(MRASProblem):
    """Identify q in the nonlinear potential model of Example 5.2.

    f(q, u) = -Laplace u + q u + q |q|^{2/3} u (Prop. 6.1), nonlinear in q,
    so sigma = 1; the analytic reference state fixes the source,
    g = u^dagger_t + f(q^dagger, u^dagger), and C(||q||) is the paper's
    norm-dependent stabilisation.  q^dagger = pi - r is continuous, so
    Q_h is discontinuous P1 (best fit 0.03% at h = 0.12; P0 2%).
    """

    name = "potential"
    sigma = 1.0
    C_scale = None  # C(||q||) = c(||q||) (-Laplace), see tracking_scale
    manufactured_source = True
    decaying = True  # the reference relaxes linearly toward its offset
    positive_state = True  # u_wot >= 0 under an offset of 6

    def __init__(self, h=0.1, dt=1e-3, tend=5.0, k_u=3, k_q=1, seed=2):
        super().__init__(potential(maxh=h, order_u=k_u, order_q=k_q), dt=dt,
                         tend=tend, k_u=k_u, seed=seed)
        self.q_true = math.pi - sqrt(xc * xc + yc * yc)
        self.u_wot = 25 * reference_speed(self.mesh, seed=seed)
        # ||q^dagger||_{L2}^{2/3} + ||q^dagger||_{L3}^{2/3}, with q^dagger >= 0
        self._c_offset = (l2norm(self.q_true, self.mesh) ** (2 / 3)
                          + float(Integrate(self.q_true ** 3, self.mesh))
                          ** (2 / 9))

    def truth(self, t):
        """u^dagger = u_wot (DECAY_T - t) / DECAY_T + OFFSET."""
        return self.u_wot * (DECAY_T - t) / DECAY_T + OFFSET

    def truth_rate(self, t):
        return -self.u_wot / DECAY_T

    def c_coe(self):
        """(A2), paper (30): q -> q + q |q|^{2/3} is monotone with slope at
        least 1, so the pairing is >= int u^dagger (q - p)^2, and C_coe =
        ess inf u^dagger.  u_wot >= 0, so u^dagger >= OFFSET until DECAY_T;
        past it the decay term turns negative."""
        if self.tend <= DECAY_T:
            return OFFSET
        peak = GridFunction(self.Ufes)
        peak.Set(self.u_wot)
        return OFFSET - ((self.tend - DECAY_T) / DECAY_T
                         * float(np.max(peak.vec.FV().NumPy())))

    def tracking_scale(self, q):
        """c(||q||) of the paper's stabilisation bound."""
        core = Z_MAX / Z_MIN * l2norm(q, self.mesh) ** (2 / 3) + self._c_offset
        return 2 + core ** 2

    def dirichlet(self, t):
        """The state is prescribed on the boundary (paper (26)): the
        reference state's values, so u - z vanishes there."""
        return {"circle": self.truth(t)}

    def model(self, u, w):
        return grad(u) * grad(w)

    def coupling(self, q, u, w):
        """q u + q |q|^{2/3} u: monotone in q wherever u >= 0, the (A2)
        bound resting on the datum's positive lower bound."""
        return (q + q * sqrt(q * q) ** (2 / 3)) * u * w

    def coupling_adjoint(self, q, u, w):
        """d/dq [q + q |q|^{2/3}] = 1 + (5/3) |q|^{2/3}, transcribed
        independently."""
        return (1 + (5 / 3) * sqrt(q * q) ** (2 / 3)) * u * w


# --- runner registration (mras.live reads these markers) ----------------------

PROBLEM = PotentialProblem

# The time limit is DECAY_T, where the reference has decayed to its offset.
DEFAULTS = {"potential": dict(h=0.12, dt=0.02, tend=DECAY_T)}
CHECK = {"potential": dict(h=0.25, dt=0.05, tend=0.5)}
QUICK = CHECK
