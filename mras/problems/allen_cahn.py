"""Modified Allen-Cahn benchmark (paper Section 5.3, Figures 9-11).

The reference state is a separable sine profile decaying linearly in time
toward a positive offset; the true parameter is piecewise constant on a
three-piece domain partition.  The measurement noise is the seeded Gaussian
FEM noise of :mod:`mras.noise`.
"""

from ngsolve import (BilinearForm, Grad, GridFunction, IfPos, Integrate,
                     LinearForm, sqrt, sin, pi, dx, ds, x as xc, y as yc)

from ..domains import allen_cahn
from ..fields import l2norm

from ..scheme import Problem

DECAY_T = 10.0              # time scale of the linear decay (DECAY_T - t) / DECAY_T
OFFSET = 1.0                # positive offset required by the analysis
Z_MIN, Z_MAX = 0.2, 1.0     # state bounds entering the stabilisation bound C(.)
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
    box = ((-2.0, -1.0), (2.0, 1.0))
    counter_start = 1
    incremental = True
    stop_margin = 2.0
    sigma = 1.0

    def __init__(self, h=0.1, dt=1e-3, tend=5.0, k_u=3, seed=0):
        super().__init__(allen_cahn(maxh=h, order_u=k_u), dt=dt, tend=tend,
                         k_u=k_u, seed=seed)
        self.series_counters = frozenset(
            [1, 5, 25, 50, 75, 100, 200, 500, 1500, 2500, 5000,
             self.tsteps - 2])
        self.q_true = piecewise_constant_field(self.mesh)
        self.u_wot = sin(0.25 * pi * (xc - 2)) * sin(0.5 * pi * (yc - 1))
        self._u_exact = GridFunction(self.Ufes)
        # d/dt of the linear decay:  u^dagger = u_wot (DECAY_T - t) / DECAY_T + OFFSET
        self._ut = -self.u_wot / DECAY_T
        # ||q^dagger||_{L2}^{2/3} + ||q^dagger||_{L3}^{2/3}, with q^dagger > 0
        self._c_base = (l2norm(self.q_true, self.mesh) ** (2 / 3)
                        + float(Integrate(self.q_true ** 3, self.mesh)) ** (2 / 9))

    # -- driver hooks --------------------------------------------------------
    def init_run(self, state):
        # Self-contained fresh start: the reference state at t = 0 (decay
        # factor 1) is both the initial estimate and the time-zero reference.
        self._u_exact.Set(self.u_wot + OFFSET)
        state.gfu.Set(self._u_exact)
        state.u_true.Set(self._u_exact)
        state.gfq.Set(0)

    def data(self, state):
        decay = (DECAY_T - (state.t.Get() + self.dt)) / DECAY_T
        self._u_exact.Set(self.u_wot * decay + OFFSET)
        state.u_true.vec.data = self._u_exact.vec.Copy()
        state.set_z(self.noise.draw(self._u_exact, state.noise_level, state.counter))
        state.ut = self._ut

    def c_value(self, gfq):
        core = (l2norm(gfq, self.mesh) ** (2 / 3) + self._c_base)
        return 1 + 2 * (Z_MAX ** 2 / Z_MIN) * core ** 2

    adjoint_deviation = ("paper-literal c~ = 0 linearisation of the discrete "
                         "multiplier, DEVIATIONS.md")

    def coupling(self, q, z, s):
        """<f(q, z), s> = int phi(q, z) s dx for f(q, z) = q z^3 +
        3/5 q |q|^{2/3} z: monotone in q wherever z >= 0.  The datum
        u^dagger = u_wot (DECAY_T - t) / DECAY_T + OFFSET has a positive
        a.e. lower bound (its nodal P3 values ring below zero at sine
        extrema, but the cell means stay >= 1 - |u_wot| > 0), so (A2) holds
        on the actual space-time range."""
        return (q * z ** 3 + 0.6 * q * sqrt(q * q) ** (2 / 3) * z) * s

    def coupling_adjoint(self, q, z, s):
        """d/dq [q z^3 + 3/5 q |q|^{2/3} z] = (z^3 + |q|^{2/3} z),
        transcribed independently."""
        return (z ** 3 + sqrt(q * q) ** (2 / 3) * z) * s

    def forms(self, state):
        dt, sigma = self.dt, self.sigma
        gfq, z, z_old, zt, ut = state.gfq, state.z, state.z_old, state.zt, state.ut
        noisy = state.noise_level > 0
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
        a += sigma * dt * absq23 * self.q * z * self.s * dx
        res += sigma * dt * ut * self.s * dx
        res += sigma * dt * qex23 * qex * state.u_true * self.s * dx
        if noisy:
            res += sigma * dt * Grad(state.u_true) * Grad(self.s) * dx
        res += sigma * dt * qex * z * z * z * self.s * dx
        # tracking term z^n (u^n - z^n): the factor 1 + (5/3)|c~|^{2/3}
        # equals 1 at the linearisation point c~ = 0 chosen in Example 5.3
        res += dt * z_old * (state.gfu - z_old) * self.s * dx
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
        res += dt * qex * z * z * z * self.v * dx
        res += -dt * absq23 * gfq * z * self.v * dx
        res += -dt * gfq * z * z * z * self.v * dx
        if noisy:
            res += -dt * Grad(z) * Grad(self.v) * dx
            res += dt * Grad(z) * state.n * self.v * ds
        res += -dt * C * Grad(state.gfu) * Grad(self.v) * dx
        res += dt * C * Grad(z) * Grad(self.v) * dx
        return a, res
