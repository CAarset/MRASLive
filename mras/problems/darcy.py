"""Time-dependent Darcy flow benchmark (paper Section 4, Figures 4 and 12)."""

import numpy as np
from ngsolve import (BilinearForm, GridFunction, Grad, LinearForm, dx, pi, sin,
                     x as xc, y as yc)

from ..data.pdebench import load_nu_grid
from ..domains import darcy
from ..fields import to_gridfunc, to_gridfunc0
from ..noise import seeded_rng
from ..scheme import Problem
from ..stepper import ImplicitStepper

DATA_H = 0.03       # ground-truth mesh width (finer than the design mesh)
DATA_ORDER = 4      # ground-truth polynomial order
GRID = 128          # pixel size of the source field
RADIUS = 0.42       # radius of the disk initial guess a_0

U0 = sin(pi * xc) * sin(pi * yc)  # deterministic initial state


class DarcyProblem(Problem):
    """Identification of q = nu in  u_t - div(nu grad u) = g(x,t).

    The reference model is q~ = 0 (plain diffusion with unit coefficient); the
    joint step therefore couples the trial q only through ``q grad(z).grad(v)``.
    Ground truth u^dagger starts from the smooth state ``U0`` and follows
    implicit Euler on a finer mesh, driven by a fresh i.i.d. N(0, 1) pixel
    source at every step; the data feed reuses the source array that generated
    the matching truth snapshot.

    Parameters
    ----------
    h, k_u : design mesh width / polynomial order of U_h.
    sample, hdf5 : which PDEBench permeability field to use.
    seed : pixel source stream and datum noise (both seeded).
    data_h, data_order, grid : ground-truth resolution knobs.
    """

    name = "darcy"
    counter_start = 1
    stop_margin = 2.0

    def __init__(self, h=0.04, dt=1e-3, tend=5.0, k_u=3, sample=0,
                 hdf5=None, seed=0, data_h=DATA_H, data_order=DATA_ORDER,
                 grid=GRID):
        super().__init__(darcy(maxh=h, order_u=k_u), dt=dt, tend=tend,
                         k_u=k_u, seed=seed)
        self.series_counters = frozenset(
            [1, 5, 10, 25, 50, 75, 100, 200, 500, 1500, 5000, self.tsteps - 2])
        self.data_h = data_h
        self.data_order = data_order
        nu = np.asarray(load_nu_grid(hdf5, sample=sample))
        self.grid = grid if nu.shape[0] == grid else nu.shape[0]
        self.source_shape = (self.grid, self.grid)
        self.nu = nu
        self.q_true = to_gridfunc0(nu, self.Qfes)
        self.u0 = GridFunction(self.Ufes)
        self.u0.Set(U0)
        self._generate()

    # -- ground truth ----------------------------------------------------
    def _generate(self):
        """Backward-Euler ground truth plus the driving pixel stream."""
        fine = darcy(maxh=self.data_h, order_u=self.data_order)
        pixels = seeded_rng(self.seed).standard_normal(
            (self.tsteps, *self.source_shape))
        self.sources = pixels.astype(np.float32)
        q_fine = to_gridfunc0(self.nu, fine.Qfes)
        u, v = fine.Ufes.TnT()
        a = BilinearForm(fine.Ufes)
        a += q_fine * Grad(u) * Grad(v) * dx
        a.Assemble()
        stepper = ImplicitStepper(self.dt, a)
        gf = GridFunction(fine.Ufes)
        gf.Set(U0)
        proj = GridFunction(self.Ufes)
        self.u_exact = np.empty((self.tsteps, self.Ufes.ndof), dtype=np.float32)
        for counter in range(self.tsteps):
            stepper.step(gf, to_gridfunc(self.sources[counter], fine.Ufes))
            proj.Set(gf)
            self.u_exact[counter] = proj.vec.FV().NumPy()

    # -- driver hooks ------------------------------------------------------
    def init_level(self, state):
        state.gfu.Set(self.u0)
        state.u_true.Set(self.u0)
        state.gfq.Set(self.initial_guess())

    def initial_guess(self):
        """Disk indicator a_0 for the parameter initialisation."""
        from ngsolve import IfPos
        return IfPos(RADIUS ** 2 - ((xc - 0.5) ** 2 + (yc - 0.5) ** 2), 1, 0)

    def data(self, state):
        counter = state.counter
        state.u_true.vec.data = np.asarray(self.u_exact[counter - 1],
                                           dtype=np.float64)
        state.g = to_gridfunc(self.sources[counter - 1], self.Ufes)
        state.set_z(self.noise.draw(state.u_true, state.level, counter))

    def forms(self, state):
        dt = self.dt
        a = BilinearForm(self.X)
        res = LinearForm(self.X)
        a += self.u * self.v * dx
        a += dt * Grad(self.u) * Grad(self.v) * dx
        a += dt * self.q * Grad(state.z) * Grad(self.v) * dx
        a += self.q * self.s * dx
        res += state.gfu * self.v * dx
        res += dt * state.g * self.v * dx
        res += dt * Grad(state.z) * Grad(self.v) * dx
        res += state.gfq * self.s * dx
        res += dt * Grad(state.z_old) * (Grad(state.gfu) - Grad(state.z_old)) * self.s * dx
        return a, res
