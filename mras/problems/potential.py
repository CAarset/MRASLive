"""Nonlinear advection-diffusion benchmark (paper Section 5.2, Figures 7-8).

The reference state u^dagger is a steady nonlinear vorticity speed field with
a linear time decay and a positive offset; the parameter is q^dagger = pi - r.
"""

import math

import numpy as np
from ngsolve import (BilinearForm, Grad, GridFunction, H1, IfPos, Integrate,
                     LinearForm, Norm, sqrt, dx, ds, x as xc, y as yc)
from ngsolve import grad

from ..domains import potential
from ..fields import l2norm, real_root
from ..noise import GaussianFEMNoise
from ..scheme import Problem

NU = 0.1                    # vorticity-solve diffusion coefficient
Z_MIN, Z_MAX = 0.1, 100.0   # datum offset / bounds entering C(.)
PICARD_ITERS, PICARD_TOL, DAMPING = 15, 1e-9, 0.7


def reference_speed(mesh, seed=2):
    """Interior-masked speed field of the nonlinear vorticity reference state.

    Picard iteration for ``-nu Laplacian(omega) + (u . grad) omega = f`` with
    streamfunction velocity ``u = (psi_y, -psi_x)``, a deterministic Gaussian
    source and damped updates.
    """
    fes_psi = H1(mesh, order=3, dirichlet=".*")
    fes_omega = H1(mesh, order=3, dirichlet=".*")
    psi, vpsi = fes_psi.TnT()
    omega, vomega = fes_omega.TnT()

    rng = np.random.default_rng(seed)
    f = GridFunction(fes_omega)
    f.vec.FV().NumPy()[:] = rng.normal(size=fes_omega.ndof)
    gf_omega = GridFunction(fes_omega, "omega")

    def streamfunction():
        a = BilinearForm(fes_psi)
        a += grad(psi) * grad(vpsi) * dx
        a.Assemble()
        l = LinearForm(fes_psi)
        l += gf_omega * vpsi * dx
        l.Assemble()
        gf = GridFunction(fes_psi)
        gf.vec.data = a.mat.Inverse(fes_psi.FreeDofs()) * l.vec
        return gf

    for _ in range(PICARD_ITERS):
        gf_psi = streamfunction()
        u_vel, v_vel = grad(gf_psi)[1], -grad(gf_psi)[0]
        a = BilinearForm(fes_omega)
        a += NU * grad(omega) * grad(vomega) * dx
        a += (u_vel * grad(omega)[0] + v_vel * grad(omega)[1]) * vomega * dx
        a.Assemble()
        l = LinearForm(fes_omega)
        l += f * vomega * dx
        l.Assemble()
        res = l.vec - a.mat * gf_omega.vec
        if Norm(res) < PICARD_TOL:
            break
        gf_omega.vec.data += DAMPING * a.mat.Inverse(fes_omega.FreeDofs()) * res

    velocity = grad(streamfunction())
    r = sqrt(xc * xc + yc * yc)
    mask = IfPos(0.5 - (math.pi - r), math.pi - r, 0.5)
    return mask * Norm(velocity)


class PotentialProblem(Problem):
    """Identify q in the nonlinear advection-diffusion model of Example 5.2.

    The joint step treats the nonlinear diffusion and reaction explicitly
    around the datum.  For noisy data, integration-by-parts pairs move the
    Laplacian of the reference state and of the datum into the volume; the
    accompanying ``ds`` fluxes act through the state (H1) test function, the
    only one with a trace on the boundary.

    The noise is re-seeded before every draw, so one frozen pattern is reused
    at all time steps of a level.  ``z_old`` and the reference state carry
    over from the end of the previous noise level, as in the paper runs.
    """

    name = "potential"
    box = ((-math.pi, -math.pi), (math.pi, math.pi))
    counter_start = 0
    incremental = True
    stop_margin = 2.0
    seed_initial_errors = False
    sigma = 1.0
    L = 1.0

    def __init__(self, h=0.1, dt=1e-3, tend=6.0, k_u=3, seed=2):
        super().__init__(potential(maxh=h, order_u=k_u), dt=dt, tend=tend,
                         k_u=k_u)
        self.series_counters = frozenset(
            [1, 5, 10, 25, 50, 75, 100, 200, 500, 1500, 4500, 5000,
             self.tsteps - 2])
        self.noise = GaussianFEMNoise(self.Ufes, seed=int(seed))
        self.q_true = math.pi - sqrt(xc * xc + yc * yc)
        self.u_wot = 25 * reference_speed(self.mesh, seed=seed)
        self._u_exact = GridFunction(self.Ufes)
        self._u_exact_old = GridFunction(self.Ufes)
        self._ut = GridFunction(self.Ufes)
        self._c_offset = (l2norm(self.q_true, self.mesh) ** (2 / 3)
                          + real_root(float(Integrate(self.q_true ** 3, self.mesh)), 3)
                          ** (2 / 3))

    # -- driver hooks --------------------------------------------------------
    def init_level(self, state):
        state.gfu.Set(self._u_exact)
        state.gfq.Set(0)

    def on_level_start(self, state, prev_state):
        if prev_state is not None:
            state.z_old.vec.data = prev_state.z.vec.Copy()

    def data(self, state):
        self._u_exact_old.vec.data = self._u_exact.vec.Copy()
        decay = (self.tend - (state.t.Get() + self.dt)) / self.tend
        self._u_exact.Set(self.u_wot * decay + Z_MIN)
        self._ut.vec.data = self._u_exact.vec.Copy()
        self._ut.vec.data -= self._u_exact_old.vec.Copy()
        self._ut.vec.data /= self.dt
        state.u_true.vec.data = self._u_exact.vec.Copy()
        state.set_z(self.noise.draw(self._u_exact, state.level, state.counter))
        state.ut = self._ut

    def c_value(self, gfq):
        """Explicit stabilisation coefficient C(a) from the paper's estimate."""
        core = Z_MAX / Z_MIN * l2norm(gfq, self.mesh) ** (2 / 3) + self._c_offset
        return 2 + core ** 2

    def forms(self, state):
        dt, sigma = self.dt, self.sigma
        gfq, z, z_old, zt, ut = state.gfq, state.z, state.z_old, state.zt, state.ut
        noisy = state.level > 0
        absq = sqrt(gfq * gfq)
        absq23 = absq ** (2 / 3)
        qex = self.q_true
        qex23 = sqrt(qex * qex) ** (2 / 3)
        C = self.c_value(gfq)
        a = BilinearForm(self.X)
        res = LinearForm(self.X)

        # parameter row
        a += self.q * self.s * dx
        a += sigma * dt * (absq23 * self.q * z + self.q * z) * self.s * dx
        res += sigma * dt * ut * self.s * dx
        res += sigma * dt * qex23 * qex * state.u_true * self.s * dx
        if noisy:
            res += sigma * dt * Grad(state.u_true) * Grad(self.s) * dx
        res += sigma * dt * qex * z * self.s * dx
        res += dt * (1 + 0 * (5 / 3) * absq23) * z_old * (state.gfu - z_old) * self.s * dx
        res += -sigma * dt * (zt + absq23 * gfq * z + gfq * z) * self.s * dx
        if noisy:
            res += -sigma * dt * Grad(z) * Grad(self.s) * dx

        # state row
        a += self.u * self.v * dx
        a += dt * C * Grad(self.u) * Grad(self.v) * dx
        a += dt * gfq * z * self.v * dx
        a += dt * absq23 * gfq * z * self.v * dx
        res += -dt * absq23 * gfq * z * self.v * dx
        res += -dt * gfq * z * self.v * dx
        res += dt * qex23 * qex * state.u_true * self.v * dx
        res += dt * zt * self.v * dx
        if noisy:
            res += dt * Grad(state.u_true) * Grad(self.v) * dx
            res += -dt * Grad(state.u_true) * state.n * self.v * ds
        res += dt * qex * state.u_true * self.v * dx
        if noisy:
            res += -dt * Grad(z) * Grad(self.v) * dx
            res += dt * Grad(z) * state.n * self.v * ds
        res += -dt * C * Grad(state.gfu) * Grad(self.v) * dx
        res += dt * C * Grad(z) * Grad(self.v) * dx
        return a, res
