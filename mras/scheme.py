"""The MRAS time loop: data feed, joint (u, q) step, error tracking."""

import time
from collections import namedtuple

import numpy as np
from ngsolve import (FESpace, GridFunction, Parameter, Preconditioner,
                     specialcf)
from ngsolve.solvers import BVP

from .fields import l2norm
from .noise import GaussianFEMNoise


class DataStep(namedtuple("DataStep", "fes u v u_prev t g dt")):
    """One step of the forward (data-generating) solve.

    ``fes`` is the data finite element space with trial/test functions
    ``u``/``v``, ``u_prev`` the state at the previous time, ``t`` a
    ``Parameter`` set to the new time, ``g`` the source from ``source(t)``
    and ``dt`` the step size.
    """


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
        self.z_started = False
        self.has_truth = True  # data() clears it for purely observational runs

    def set_z(self, z):
        """Copy a datum into the persistent z and refresh zt = (z - z_old)/dt.

        The first datum of a level also initialises ``z_old``, so ``zt``
        starts at zero and every noise level is a fresh statistical start.
        """
        if not self.z_started:
            self.z_old.vec.data = z.vec.Copy()
            self.z_started = True
        self.z.vec.data = z.vec.Copy()
        self.zt = GridFunction(self.Ufes)
        self.zt.vec.data = self.z.vec.Copy()
        self.zt.vec.data -= self.z_old.vec.Copy()
        self.zt.vec.data /= self.dt


def l2diff(u, v, mesh):
    """L2 norm of a difference; ``v`` may be a coefficient or grid function."""
    return l2norm(u - v, mesh)


class Problem:
    """Base class: what a problem must provide for the generic driver.

    Required hook
    -------------
    ``forms(state)``       return the joint (BilinearForm, LinearForm) pair at
                           the current step (the MRAS weak form of the model).

    Data hooks (the base ``data()`` resolves them in this order)
    ------------------------------------------------------------
    ``observations(state)``  known data as a coefficient or grid function —
                             use for measured data that already contains its
                             own noise (run at level 0).
    ``truth(state)``         ground-truth state ``u^dagger`` at ``state.t``;
                             observations are synthesized as truth plus
                             Gaussian FEM noise at the requested level.
    ``forward_forms(step)``  one backward-Euler-type step of the model at
                             ``step.t`` with the parameter frozen at
                             ``q_true``; used when neither of the above is
                             given, so NGSolve generates the data.

    The driving source ``source(t)`` (a coefficient) is optional and is
    placed in ``state.g`` for every step of the identification loop.

    Optional configuration (class or instance attributes)
    -----------------------------------------------------
    ``u0``                   initial state (coefficient or grid function).
    ``q_true``               true parameter (coefficient or grid function);
                             error_q is reported only when it is known.
    ``data_domains``         optional finer ``Domains`` for data generation.
    ``incremental``          solve for the increment into ``aux`` and add it to
                             ``gf`` (True) or solve for the new state directly
                             (False).
    ``stop_margin``          loop runs while ``t < tend - stop_margin*dt``.
    ``counter_start``        first step index of the identification loop.
    ``series_counters``      step indices where solution snapshots are kept.

    ``init_level`` sets the starting ``gfu``, ``gfq`` (from ``initial_guess``)
    and the reference ``u_true`` at ``tstart``; the driver records the t=0
    errors and snapshots before the first step. Runs without ground truth
    report the state misfit ``||u - z||`` instead of error histories.
    """

    name = "problem"
    box = ((0.0, 0.0), (1.0, 1.0))  # domain bounding box, for field figures
    tstart = 0.0
    counter_start = 1
    incremental = False
    stop_margin = 2.0
    saved_every = 100
    series_counters = ()
    sigma = 1.0
    L = 1.0
    u0 = None
    q_true = None
    data_domains = None

    def __init__(self, domains, dt=1e-3, tend=5.0, k_u=3, seed=0):
        self.mesh, self.Ufes, self.Qfes = domains
        self.dt = dt
        self.tend = tend
        self.seed = int(seed)
        self.tsteps = int(round((tend - self.tstart) / dt))
        self.X = FESpace([self.Ufes, self.Qfes])
        self.u, self.q = self.X.TrialFunction()
        self.v, self.s = self.X.TestFunction()
        self.noise = GaussianFEMNoise(self.Ufes, seed=self.seed)
        self._snaps = None

    # -- hooks ----------------------------------------------------------
    def init_level(self, state):
        """Set gfu / gfq / u_true initial values for one noise level.

        The driver mirrors gfu and gfq into the joint solver function, so
        problems never touch state.gf directly.
        """
        if self.u0 is None:
            raise NotImplementedError(
                f"{self.name}: set u0 or override init_level")
        state.gfu.Set(self.u0)
        state.u_true.Set(self.u0)
        state.gfq.Set(self.initial_guess())

    def initial_guess(self):
        """Starting value of the parameter estimate."""
        return 0.0

    def observations(self, state):
        """Known data at this step (coefficient or grid function)."""

    def truth(self, state):
        """Ground-truth state at ``state.t`` (coefficient or grid function)."""

    def source(self, t):
        """Source coefficient of the state equation at time ``t``."""

    def forward_forms(self, step):
        """Weak form of one data-generation step; see ``DataStep``."""

    def data(self, state):
        """Feed one step: resolve data, refresh u_true, g, and z."""
        source = self.source(state.t)
        if source is not None:
            state.g = source
        obs = self.observations(state)
        truth = self.truth(state)
        if obs is not None:
            state.has_truth = truth is not None
            if truth is not None:
                self._set_reference(state, truth)
            state.set_z(self._as_gf(obs))
            return
        if truth is not None:
            self._set_reference(state, truth)
        elif self._forward_data() is None:
            raise NotImplementedError(
                f"{self.name}: supply observations(), truth(), or "
                "forward_forms() to generate data")
        else:
            state.u_true.vec.data = np.asarray(
                self._snaps[state.counter - 1], dtype=np.float64)
        state.set_z(self.noise.draw(state.u_true, state.level, state.counter))

    def forms(self, state):
        raise NotImplementedError

    # -- forward (data-generating) model ---------------------------------
    def _set_reference(self, state, value):
        if isinstance(value, GridFunction):
            state.u_true.vec.data = value.vec.Copy()
        else:
            state.u_true.Set(value)

    def _as_gf(self, value):
        if isinstance(value, GridFunction):
            return value
        gf = GridFunction(self.Ufes)
        gf.Set(value)
        return gf

    def _forward_data(self):
        """Generate observations by stepping the model at q = q_true."""
        if self._snaps is None and (
                type(self).forward_forms is not Problem.forward_forms):
            fes = (self.data_domains.Ufes if self.data_domains is not None
                   else self.Ufes)
            u, v = fes.TnT()
            t = Parameter(self.tstart)
            gf = GridFunction(fes)
            gf.Set(self.u0)
            snaps = np.empty((self.tsteps, self.Ufes.ndof), dtype=np.float32)
            snap = GridFunction(self.Ufes)
            for k in range(self.tsteps):
                t.Set(self.tstart + (k + 1) * self.dt)
                step = DataStep(fes, u, v, gf, t, self.source(t) or 0, self.dt)
                a, res = self.forward_forms(step)
                a.Assemble()
                res.Assemble()
                nxt = GridFunction(fes)
                self.solve(a, res, nxt)
                gf.vec.data = nxt.vec.Copy()
                snap.Set(gf)
                snaps[k] = snap.vec.FV().NumPy()
            self._snaps = snaps
        return self._snaps

    # -- driver ---------------------------------------------------------
    symmetric = True  # False: nonsymmetric weak form, solved by restarted GMRES

    def solve(self, a, res, target):
        pre = Preconditioner(a, "local")
        pre.Update()
        if self.symmetric:
            BVP(bf=a, lf=res, gf=target, pre=pre, print=False)
        else:
            from ngsolve.solvers import GMRES
            GMRES(A=a.mat, b=res.vec, pre=pre, x=target.vec,
                  maxsteps=1000, tol=1e-12, printrates=False)

    def run_level(self, level, progress=True):
        from .results import LevelResult
        started = time.perf_counter()
        state = State(self, level)
        self.init_level(state)
        # Mirror the initial estimate into the joint solver function: the
        # incremental update is carried in state.gf, and BVP also uses it as
        # the initial guess of the direct solve.
        state.gf.components[0].Set(state.gfu)
        state.gf.components[1].Set(state.gfq)
        truth_known = self._truth_known()
        error_u = [l2diff(state.gfu, state.u_true, state.mesh)] \
            if truth_known else None
        error_q = [l2diff(state.gfq, self.q_true, state.mesh)] \
            if truth_known and self.q_true is not None else None
        misfit_u = []
        saved_u = [state.gfu.vec.FV().NumPy().copy()]
        saved_q = [state.gfq.vec.FV().NumPy().copy()]
        series_u = [state.gfu.vec.FV().NumPy().copy()]
        series_q = [state.gfq.vec.FV().NumPy().copy()]

        while state.t.Get() < self.tend - self.stop_margin * self.dt:
            self.data(state)
            misfit_u.append(l2diff(state.gfu, state.z, state.mesh))
            if progress:
                headline = (error_u[-1] if error_u is not None
                            else misfit_u[-1])
                print(f"\r{self.name} level={level}: t={state.t.Get():.3f}"
                      f"/{self.tend}  err={headline:.3e}", end="", flush=True)
            a, res = self.forms(state)
            a.Assemble()
            res.Assemble()
            target = state.aux if self.incremental else state.gf
            self.solve(a, res, target)
            if self.incremental:
                state.gf.vec.data += state.aux.vec.data

            state.gfu.Set(state.gf.components[0])
            if error_u is not None and state.has_truth:
                error_u.append(l2diff(state.gfu, state.u_true, state.mesh))
            state.gfq.Set(state.gf.components[1])
            if error_q is not None and state.has_truth:
                error_q.append(l2diff(state.gfq, self.q_true, state.mesh))
            if state.counter % self.saved_every == 0:
                saved_q.append(state.gfq.vec.FV().NumPy().copy())
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
            error_u=np.array(error_u) if error_u is not None else None,
            error_q=np.array(error_q) if error_q is not None else None,
            misfit_u=np.array(misfit_u),
            saved_gfu=np.array(saved_u), saved_gfq=np.array(saved_q),
            series_gfu=np.array(series_u), series_gfq=np.array(series_q),
            state=state,
            meta={"dt": self.dt, "tend": self.tend,
                  "dofs_u": len(state.gfu.vec.FV().NumPy()),
                  "dofs_q": len(state.gfq.vec.FV().NumPy()),
                  "steps": len(misfit_u),
                  "seconds": round(time.perf_counter() - started, 1)})
        return result

    def _truth_known(self):
        """True when some data path supplies a reference state."""
        return (type(self).data is not Problem.data
                or type(self).truth is not Problem.truth
                or type(self).forward_forms is not Problem.forward_forms)

    def run(self, levels, progress=True):
        """One run per noise level; levels are statistically independent."""
        return [self.run_level(level, progress) for level in levels]
