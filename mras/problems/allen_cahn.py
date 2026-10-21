"""Modified Allen-Cahn benchmark (paper Section 5.3, Figures 9-11).

The reference state is a separable sine profile with a linear time decay and
a positive offset; the true parameter is piecewise constant on a three-piece
domain partition.  Both the datum offset (``z_min``) and the noise model
match the paper runs; see docs/PAPER_TARGETS.md for notes.
"""

import numpy as np
from ngsolve import (BilinearForm, Grad, GridFunction, IfPos, Integrate,
                     LinearForm, sqrt, sin, pi, dx, ds, x as xc, y as yc)

from ..domains import allen_cahn
from ..fields import l2norm, real_root
from ..noise import NoiseModel
from ..scheme import Problem

Z_MIN, Z_MAX = 0.2, 1.0     # datum offset / bound entering C(.)
V_BG, V_INC1, V_INC2 = 1.0, 4.0, 2.0   # inclusion values of q^dagger


def piecewise_constant_field(mesh):
    """q^dagger: background ``V_BG`` with ``V_INC1`` / ``V_INC2`` inclusion pieces."""
    r1 = IfPos(xc + yc - 1, 1, 0)
    r2 = IfPos(xc - 2 * yc + 0.4, 1, 0) * IfPos(xc + yc - 1, 0, 1)
    return V_BG * (1 - r1) * (1 - r2) + V_INC1 * r1 + V_INC2 * r2


class AllenCahnProblem(Problem):
    """Identify q in the cubic reaction-diffusion model of Example 5.3.

    As in the potential-flow example, noisy data enter through
    integration-by-parts pairs whose ``ds`` fluxes act through the state (H1)
    test function only.  The parameter enters both rows through cubic data
    powers (z^3) as in the paper's modified Allen-Cahn setting.
    """

    name = "allen_cahn"
    counter_start = 0
    incremental = True
    stop_margin = 1.0         # AC loop runs while t < tend - dt
    seed_initial_errors = False
    sigma = 1.0
    L = 1.0

    def __init__(self, h=0.1, dt=1e-3, tend=10.0, k_u=3, noise_peaks=20):
        super().__init__(allen_cahn(maxh=h, order_u=k_u), dt=dt, tend=tend,
                         k_u=k_u)
        self.series_counters = frozenset(
            [1, 5, 25, 50, 75, 100, 200, 500, 1500, 2500, 5000,
             self.tsteps - 2])
        self.noise = NoiseModel(peaks=noise_peaks,
                                box=((-2.0, -1.0), (2.0, 1.0)))
        self.q_true = piecewise_constant_field(self.mesh)
        self.u_wot = sin(0.25 * pi * (xc - 2)) * sin(0.5 * pi * (yc - 1))
        self._u_exact = GridFunction(self.Ufes)
        self._u_exact_old = GridFunction(self.Ufes)
        self._ut = GridFunction(self.Ufes)
        self._c_base = (l2norm(self.q_true, self.mesh) ** (2 / 3)
                        + real_root(float(Integrate(self.q_true ** 3, self.mesh)), 3)
                        ** (2 / 3))

    # -- driver hooks --------------------------------------------------------
    def init_level(self, state):
        decay = (self.tend - (self.tstart + self.dt)) / (self.tend - self.tstart)
        state.gfu.Set(self.u_wot * decay + Z_MIN)
        state.gfq.Set(0)

    def on_level_start(self, state, prev_state):
        # The legacy loop keeps u_exact and z_old alive across noise levels.
        if prev_state is not None:
            self._u_exact.vec.data = prev_state.u_true.vec.Copy()
            state.z_old.vec.data = prev_state.z.vec.Copy()

    def data(self, state):
        self._u_exact_old.vec.data = self._u_exact.vec.Copy()
        decay = (self.tend - (state.t.Get() + self.dt)) / (self.tend - self.tstart)
        self._u_exact.Set(self.u_wot * decay + Z_MIN)
        self._ut.vec.data = self._u_exact.vec.Copy()
        self._ut.vec.data -= self._u_exact_old.vec.Copy()
        self._ut.vec.data /= self.dt
        state.u_true.vec.data = self._u_exact.vec.Copy()
        state.set_z(self.noise.draw(self._u_exact, state.level, state.mesh))
        state.ut = self._ut

    def c_value(self, gfq):
        core = (l2norm(gfq, self.mesh) ** (2 / 3) + self._c_base)
        return 1 + 2 * (Z_MAX ** 2 / Z_MIN) * core ** 2

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
        a += sigma * dt * self.q * z * z * z * self.s * dx
        a += sigma * dt * (absq23 * self.q * z + self.q * z) * self.s * dx
        res += sigma * dt * ut * self.s * dx
        res += sigma * dt * qex23 * qex * state.u_true * self.s * dx
        if noisy:
            res += sigma * dt * Grad(state.u_true) * Grad(self.s) * dx
        res += sigma * dt * qex * z * self.s * dx
        res += dt * (1 + 0 * (5 / 3) * absq23) * z_old * (state.gfu - z_old) * self.s * dx
        res += -sigma * dt * (zt + absq23 * gfq * z + gfq * z * z * z) * self.s * dx
        if noisy:
            res += -sigma * dt * Grad(z) * Grad(self.s) * dx

        # state row
        a += self.u * self.v * dx
        a += dt * C * Grad(self.u) * Grad(self.v) * dx
        a += dt * self.q * z * z * z * self.v * dx
        a += dt * absq23 * self.q * z * self.v * dx
        res += dt * qex23 * qex * state.u_true * self.v * dx
        res += dt * zt * self.v * dx
        if noisy:
            res += dt * Grad(state.u_true) * Grad(self.v) * dx
            res += -dt * Grad(state.u_true) * state.n * self.v * ds
        res += dt * qex * state.u_true * self.v * dx
        res += -dt * absq23 * gfq * z * self.v * dx
        res += -dt * gfq * z * z * z * self.v * dx
        if noisy:
            res += -dt * Grad(z) * Grad(self.v) * dx
            res += dt * Grad(z) * state.n * self.v * ds
        res += -dt * C * Grad(state.gfu) * Grad(self.v) * dx
        res += dt * C * Grad(z) * Grad(self.v) * dx
        return a, res
