"""Semilinear reaction-diffusion benchmark (Fisher-type; paper Figures 5-6)."""

import numpy as np
from ngsolve import (BilinearForm, CF, GridFunction, Grad, IfPos,
                     LinearForm, cos, dx, exp, x as xc, y as yc)

from ..domains import fisher
from ..fields import to_gridfunc
from ..noise import NoiseModel
from ..scheme import Problem
from ..stepper import ImplicitStepper

R_IN, R_OUT = 0.5, 0.9      # ring radii of the true permeability
Q_IN, Q_OUT = 1.0, 0.25     # ring / exterior coefficient values
G_AMP = 3.0                 # source amplitude
G_WIDTH = 0.4               # Gaussian width in the source exponent


class FisherProblem(Problem):
    """Identification of q in u_t - div(q grad u) = g + f(u), f(u) = u^2 - u.

    The ground truth uses a lagged logistic bilinear form re-assembled every
    step; the joint step linearises the reaction explicitly around the datum
    z: the residual carries ``dt * (g - z + z*z) * v``.

    The pixel noise lives on the unit square inside this domain; outside the
    unit box the mapped field continues by edge extrapolation.
    """

    name = "fisher"
    box = ((-1.25, -1.25), (1.25, 1.25))
    counter_start = 1
    stop_margin = 2.0
    seed_initial_errors = False
    seed_saved = False

    def __init__(self, h=0.1, dt=1e-3, tend=10.0, k_u=3, seed=None,
                 noise_peaks=12, data_h=0.09, data_order=4):
        super().__init__(fisher(maxh=h, order_u=k_u), dt=dt, tend=tend, k_u=k_u)
        self.series_counters = frozenset(
            [1, 5, 10, 25, 50, 75, 100, 200, 500, 1500, 5000, self.tsteps - 2])
        self.noise = NoiseModel(peaks=noise_peaks,
                                rng=np.random.default_rng(seed) if seed is not None else None)
        self.data_h = data_h
        self.data_order = data_order
        self.q_true = self._ring()
        self.source_shape = (128, 128)
        self._generate()

    @staticmethod
    def _ring():
        r_sq = xc * xc + yc * yc
        return (IfPos(r_sq - R_IN ** 2, Q_IN, Q_OUT)
                * IfPos(R_OUT ** 2 - r_sq, Q_IN, Q_OUT))

    @staticmethod
    def _g(time):
        """The Gaussian forcing g(t) = 3 exp(-(x^2 + y^2)/0.4) cos(2 t)."""
        return G_AMP * exp(-(xc * xc + yc * yc) / G_WIDTH) * cos(2 * time)

    # -- ground truth ------------------------------------------------------
    def _generate(self):
        fine = fisher(maxh=self.data_h, order_u=self.data_order)
        V = fine.Ufes
        u, v = V.TnT()
        if self.noise.rng is not None:
            self.u0_pixels = self.noise.rng.random(self.source_shape)
        else:
            self.u0_pixels = np.random.rand(*self.source_shape)
        gf = to_gridfunc(self.u0_pixels, V)
        self.u_exact = np.empty((self.tsteps, self.Ufes.ndof), dtype=np.float32)
        proj = GridFunction(self.Ufes)
        t = 0.0
        for counter in range(self.tsteps):
            a = BilinearForm(V)
            a += self._ring() * Grad(u) * Grad(v) * dx
            a += u * v * dx
            a += -gf * u * v * dx
            a.Assemble()
            stepper = ImplicitStepper(self.dt, a)
            stepper.step(gf, self._g(t))
            proj.Set(gf)
            self.u_exact[counter] = proj.vec.FV().NumPy()
            t += self.dt

    def u_at(self, counter):
        gf = GridFunction(self.Ufes)
        gf.vec.data = np.asarray(self.u_exact[counter], dtype=np.float64)
        return gf

    # -- driver hooks --------------------------------------------------------
    def init_level(self, state):
        state.gfu.Set(self.u_at(0))
        state.gfq.Set(IfPos(xc + 1.15, 1, 0) * IfPos(1.15 - xc, 1, 0)
                      * IfPos(yc + 1.15, 1, 0) * IfPos(1.15 - yc, 1, 0))
        state.z_old.Set(CF(1))

    def data(self, state):
        state.u_true.vec.data = np.asarray(self.u_exact[state.counter],
                                           dtype=np.float64)
        state.set_z(self.noise.draw(state.u_true, state.level, state.mesh))
        state.g = GridFunction(self.Ufes)
        state.g.Set(self._g(state.t.Get()))

    def forms(self, state):
        dt = self.dt
        a = BilinearForm(self.X)
        res = LinearForm(self.X)
        a += self.q * self.s * dx
        res += state.gfq * self.s * dx
        res += dt * Grad(state.z_old) * (Grad(state.gfu) - Grad(state.z_old)) * self.s * dx
        a += self.u * self.v * dx
        a += dt * Grad(self.u) * Grad(self.v) * dx
        a += dt * self.q * Grad(state.z) * Grad(self.v) * dx
        res += state.gfu * self.v * dx
        res += dt * (state.g - state.z + state.z * state.z) * self.v * dx
        res += dt * Grad(state.z) * Grad(self.v) * dx
        return a, res
