"""The MRAS time loop: data feed, joint (u, q) step, error tracking."""

import time

import numpy as np
from ngsolve import (FESpace, GridFunction, Parameter, Preconditioner,
                     specialcf)
from ngsolve.solvers import BVP

from .fields import l2norm


class State:
    """Mutable objects shared by the driver and a problem's hooks."""

    def __init__(self, problem, level):
        self.problem = problem
        self.level = level
        self.mesh = problem.mesh
        self.Ufes, self.Qfes = problem.Ufes, problem.Qfes
        self.X = problem.X
        self.u, self.q = problem.u, problem.q
        self.v, self.s = problem.v, problem.s
        self.n = specialcf.normal(self.mesh.dim)
        self.dt = problem.dt
        self.t = Parameter(problem.tstart)
        self.counter = problem.counter_start
        self.gf = GridFunction(problem.X)
        self.aux = GridFunction(problem.X)
        self.gfu = GridFunction(problem.Ufes)
        self.gfq = GridFunction(problem.Qfes)
        self.z = GridFunction(problem.Ufes)
        self.z_old = GridFunction(problem.Ufes)
        self.u_true = GridFunction(problem.Ufes)
        self.zt = None  # problems set z, zt, (ut) inside data()

    def set_z(self, z):
        """Copy a datum into the persistent z and refresh zt = (z - z_old)/dt."""
        self.z.vec.data = z.vec.Copy()
        self.zt = GridFunction(self.Ufes)
        self.zt.vec.data = self.z.vec.Copy()
        self.zt.vec.data -= self.z_old.vec.Copy()
        self.zt.vec.data /= self.dt


def l2diff(u, v, mesh):
    """L2 norm of a difference; ``v`` may be a coefficient or grid function."""
    return l2norm(u - v, mesh)


class Problem:
    """Base class: what a benchmark must provide for the generic driver.

    Required hooks
    --------------
    ``data(state)``        set the datum ``state.z`` (via ``state.set_z``), the
                           reference ``state.u_true`` and any extra step data.
    ``forms(state)``       return the joint (BilinearForm, LinearForm) pair at
                           the current step.

    Optional configuration (class attributes)
    -----------------------------------------
    ``incremental``        solve for the increment into ``aux`` and add it to
                           ``gf`` (True) or solve for the new state directly
                           (False).
    ``stop_margin``        loop runs while ``t < tend - stop_margin*dt``.
    ``counter_start``      first step index of the identification loop.
    ``seed_initial_errors``prepend the t=0 errors to the error histories.
    ``series_counters``    step indices where solution snapshots are kept.
    """

    name = "problem"
    box = ((0.0, 0.0), (1.0, 1.0))  # domain bounding box, for field figures
    tstart = 0.0
    counter_start = 1
    incremental = False
    stop_margin = 2.0
    seed_initial_errors = False
    seed_saved = True         # include the t=0 snapshots in saved_* histories
    saved_every = 100
    series_counters = ()
    sigma = 1.0
    L = 1.0

    def __init__(self, domains, dt=1e-3, tend=5.0, k_u=3):
        self.mesh, self.Ufes, self.Qfes = domains
        self.dt = dt
        self.tend = tend
        self.tsteps = int(round((tend - self.tstart) / dt))
        self.X = FESpace([self.Ufes, self.Qfes])
        self.u, self.q = self.X.TrialFunction()
        self.v, self.s = self.X.TestFunction()
        self.noise = None      # NoiseModel, set by subclasses
        self.q_true = None     # exact parameter coefficient

    # -- hooks ----------------------------------------------------------
    def init_level(self, state):
        """Set gfu / gfq / z_old initial values for one noise level."""

    def data(self, state):
        raise NotImplementedError

    def forms(self, state):
        raise NotImplementedError

    def seed_errors(self, state):
        """Error histories at t = 0 (empty unless the benchmark seeds them)."""
        return [], []

    def on_level_start(self, state, prev_state):
        """Carry data between noise levels (``prev_state`` is None for the first)."""

    # -- driver ---------------------------------------------------------
    def solve(self, a, res, target):
        pre = Preconditioner(a, "local")
        pre.Update()
        BVP(bf=a, lf=res, gf=target, pre=pre, print=False)

    def run_level(self, level, progress=True, prev_state=None):
        from .results import LevelResult
        started = time.perf_counter()
        state = State(self, level)
        self.init_level(state)
        self.on_level_start(state, prev_state)
        error_u, error_q = (list(e) for e in self.seed_errors(state))
        saved_u = [state.gfu.vec.FV().NumPy().copy()] if self.seed_saved else []
        saved_q = [state.gfq.vec.FV().NumPy().copy()] if self.seed_saved else []
        series_u = [state.gfu.vec.FV().NumPy().copy()]
        series_q = [state.gfq.vec.FV().NumPy().copy()]

        while state.t.Get() < self.tend - self.stop_margin * self.dt:
            if progress:
                print(f"\r{self.name} level={level}: t={state.t.Get():.3f}"
                      f"/{self.tend}", end="", flush=True)
            self.data(state)
            a, res = self.forms(state)
            a.Assemble()
            res.Assemble()
            target = state.aux if self.incremental else state.gf
            self.solve(a, res, target)
            if self.incremental:
                state.gf.vec.data += state.aux.vec.data

            state.gfu.Set(state.gf.components[0])
            error_q.append(l2diff(state.gfq, self.q_true, state.mesh))
            if state.counter % self.saved_every == 0:
                saved_q.append(state.gfq.vec.FV().NumPy().copy())
            state.gfq.Set(state.gf.components[1])
            error_u.append(l2diff(state.gfu, state.u_true, state.mesh))
            if state.counter % self.saved_every == 0:
                saved_u.append(state.gfu.vec.FV().NumPy().copy())

            state.z_old.vec.data = state.z.vec.Copy()
            state.t.Set(state.t.Get() + self.dt)
            state.counter += 1
            if state.counter in self.series_counters:
                series_u.append(state.gfu.vec.FV().NumPy().copy())
                series_q.append(state.gfq.vec.FV().NumPy().copy())

        if progress:
            print()
        result = LevelResult(
            problem=self.name, noise_level=level,
            error_u=np.array(error_u), error_q=np.array(error_q),
            saved_gfu=np.array(saved_u), saved_gfq=np.array(saved_q),
            series_gfu=np.array(series_u), series_gfq=np.array(series_q),
            state=state,
            meta={"dt": self.dt, "tend": self.tend,
                  "dofs_u": len(state.gfu.vec.FV().NumPy()),
                  "dofs_q": len(state.gfq.vec.FV().NumPy()),
                  "steps": len(error_u),
                  "seconds": round(time.perf_counter() - started, 1)})
        return result

    def run(self, levels, progress=True):
        results, prev = [], None
        for level in levels:
            result = self.run_level(level, progress, prev_state=prev)
            results.append(result)
            prev = result.state
        return results
