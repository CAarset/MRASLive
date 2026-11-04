"""The forward (data-generating) model, built from the problem's own physics.

``u_t + f(q_true, u) = g`` with ``f`` the very hooks the MRAS step uses
(``model`` and ``coupling``), stepped on demand so data generation runs in
lockstep with the estimator.  Terms of f that are linear in u are found by
probing and assembled once, with the mass matrix, into the constant part of
the step matrix; a linear model is then one direct solve per step with
factors computed once, and a nonlinear one takes Newton steps whose Jacobian
solves are preconditioned by those factors and iterated to round-off.
"""

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
from ngsolve import (BND, BilinearForm, CoefficientFunction, GridFunction,
                     InnerProduct, LinearForm, Norm, Parameter, Projector)

from .solvers import TOL, Solve, factorize, mass, measures

PROBE = 0.1      # amplitude of the random states the linearity probe uses
REFRESH = 30     # Krylov iterations past which the preconditioner is renewed


def _is_linear(fes, term, seed=0, tol=1e-10):
    """Whether the weak form ``term`` is linear in its trial function."""
    form = BilinearForm(fes)
    form += term
    rng = np.random.default_rng(seed)
    x, y = (GridFunction(fes).vec for _ in range(2))
    x.FV().NumPy()[:] = PROBE * rng.standard_normal(len(x))
    y.FV().NumPy()[:] = PROBE * rng.standard_normal(len(y))
    out = [x.CreateVector() for _ in range(4)]
    zero = x.CreateVector()
    zero[:] = 0.0
    both = (x + y).Evaluate()
    for vec, res in zip((x, y, both, zero), out):
        form.Apply(vec, res)
    scale = Norm(out[0]) + Norm(out[1]) + 1e-300
    return (Norm(out[3]) <= tol * scale
            and Norm((out[2] - out[0] - out[1]).Evaluate()) <= tol * scale)


class BoundaryData:
    """A problem's Dirichlet data on one space: ``lift(t)`` is the grid
    function carrying them on the constrained dofs, zero on the free ones.

    The problem declares them per boundary, ``dirichlet(t) -> {boundary:
    value}``: each boundary a name (or regular expression) of the mesh's
    boundaries, all on the Dirichlet part of the space, whose other
    boundaries are held at zero.  Each is set as NGSolve sets boundary
    values, ``Set(value, BND, definedon=...)`` -- the L2 projection of the
    value onto the traces on that boundary -- and where two meet, the later
    one's value is kept.  The boundaries' dofs are found once.
    """

    def __init__(self, problem, fes):
        self.problem, self.fes = problem, fes
        self._regions = {}
        self._part = None

    def lift(self, t):
        gf = GridFunction(self.fes)
        data = self.problem.dirichlet(t)
        if not data:
            return gf
        if not isinstance(data, Mapping):
            raise TypeError(
                f"{self.problem.name}: dirichlet(t) returns {{boundary: "
                f"value}} (or None), not {type(data).__name__}")
        if self._part is None:
            self._part = GridFunction(self.fes)
        values = gf.vec.FV().NumPy()
        for name, value in data.items():
            region, dofs = self._region(name)
            self._part.Set(value, BND, definedon=region)
            values[dofs] = self._part.vec.FV().NumPy()[dofs]
        return gf

    def _region(self, name):
        if name not in self._regions:
            mesh = self.fes.mesh
            region = mesh.Boundaries(name)
            dofs = np.array(self.fes.GetDofs(region), dtype=bool)
            if not dofs.any():
                raise ValueError(
                    f"{self.problem.name}: dirichlet(t) names {name!r}, which "
                    f"matches none of the boundaries "
                    f"{sorted(set(mesh.GetBoundaries()))}")
            if (dofs & np.array(self.fes.FreeDofs(), dtype=bool)).any():
                raise ValueError(
                    f"{self.problem.name}: dirichlet(t) names {name!r}, which "
                    f"is not on the Dirichlet part of U_h: data there would "
                    f"be a natural condition, not imposed")
            self._regions[name] = (region, dofs)
        return self._regions[name]


class ForwardRun:
    """The forward model of one run, stepped on demand.

    Each run owns one, so datum ``k`` is solved for when the estimator first
    asks for it and nothing is precomputed or stored.  ``step()`` advances
    the model one step at ``q_true`` with the problem's ``forward_scheme``
    and returns the new state.  The model is the estimator's own: the same
    mesh and U_h, only the parameter is the true one, on its own mesh
    (``q_domains``) -- a physical quantity, where the state, the datum and
    the estimate are one consistent discretisation.
    """

    def __init__(self, problem):
        p = self.problem = problem
        fes = self.fes = p.Ufes
        u, self.v = fes.TnT()
        self.free = fes.FreeDofs()
        self.t = Parameter(p.tstart)
        self.gf = GridFunction(fes)
        self.gf.Set(p.u0 if p.u0 is not None else p.truth(self.t))
        self.boundary = BoundaryData(p, fes)
        # The initial state takes the Dirichlet data exactly; interpolation
        # alone leaves it off by the projection error on the boundary.
        self.gf.vec.data = (Projector(self.free, True) * self.gf.vec
                            + self.boundary.lift(self.t).vec)
        self.prev = GridFunction(fes)      # u^n
        self.before = GridFunction(fes)    # u^{n-1}, for Newton's start
        self.steps = 0
        order = 2 * fes.globalorder + 2
        self.dx = measures(order)[0]

        linear, nonlinear = [], []
        for term in p.f_terms(p.q_true, u, self.v, order):
            (linear if _is_linear(fes, term) else nonlinear).append(term)
        # The constant part of the step matrix, M / dt + (linear part of f).
        self.const = BilinearForm(fes)
        self.const += (1 / p.dt) * InnerProduct(u, self.v) * self.dx
        for term in linear:
            self.const += term
        self.const.Assemble()
        self.mass = mass(fes, self.dx)
        self.nonlinear = None
        if nonlinear:
            self.nonlinear = BilinearForm(fes)
            for term in nonlinear:
                self.nonlinear += term
            self._precondition()
        else:
            self.inv = factorize(self.const.mat, self.free)
        self.krylov = 0
        self._mass_inv = None

    def restore(self, restart):
        """Put the forward model where a checkpoint left it."""
        self.steps = int(restart["forward_steps"])
        self.t.Set(self.problem.tstart + self.steps * self.problem.dt)
        for name in ("gf", "prev", "before"):
            getattr(self, name).vec.FV().NumPy()[:] = restart[f"forward_{name}"]
        if self.nonlinear is not None:
            self._precondition()

    def _precondition(self):
        """Factorise the step matrix with f linearised at the current state."""
        self.nonlinear.AssembleLinearization(self.gf.vec)
        jac = self._jacobian_matrix()
        self.inv = factorize(jac, self.free)
        self.solve = Solve(self.inv, self.free)

    def _jacobian_matrix(self):
        """const + linearised nonlinear part, as one assembled matrix."""
        jac = BilinearForm(self.fes)
        u, v = self.fes.TnT()
        jac += 0 * InnerProduct(u, v) * self.dx
        jac.Assemble()
        jac.mat.AsVector().data = (self.const.mat.AsVector()
                                   + self.nonlinear.mat.AsVector())
        return jac.mat

    def _residual(self, rhs):
        """rhs - (const u + N(u)) at the current state, constrained dofs zeroed."""
        r = (rhs - self.const.mat * self.gf.vec).Evaluate()
        if self.nonlinear is not None:
            out = r.CreateVector()
            self.nonlinear.Apply(self.gf.vec, out)
            r -= out
        return (Projector(self.free, True) * r).Evaluate()

    def step(self):
        p = self.problem
        self.steps += 1
        self.t.Set(p.tstart + self.steps * p.dt)
        self.before.vec.data = (self.prev if self.steps > 1 else self.gf).vec
        self.prev.vec.data = self.gf.vec
        load = LinearForm(self.fes)
        source = p.source(self.t)
        if source is not None:
            load += InnerProduct(CoefficientFunction(source), self.v) * self.dx
        load.Assemble()
        rhs = (load.vec + (1 / p.dt) * (self.mass * self.prev.vec)).Evaluate()
        boundary = self.boundary.lift(self.t).vec
        keep = Projector(self.free, True)

        if p.forward_scheme == "explicit":
            self._explicit(rhs, keep, boundary)
        elif self.nonlinear is None:
            self.gf.vec.data = keep * self.gf.vec + boundary
            self.gf.vec.data += self.inv * self._residual(rhs)
        else:
            self.gf.vec.data = keep * self.gf.vec + boundary
            self._newton(rhs, 1 if p.forward_scheme == "semi-implicit"
                         else p.newton_maxit)
        return self.gf

    def _newton(self, rhs, iterations):
        """Backward Euler by Newton's method.

        Converged (``iterations > 1``), the iteration starts from the linear
        extrapolation 2 u^n - u^{n-1} and stops once the residual is below
        ``newton_tol`` relative to the right-hand side, and each Jacobian
        solve is iterated only as far as Newton's method needs (inexact
        Newton, forcing term min(1e-2, relative residual)) -- the limit is
        the same backward Euler solution, reached in fewer solves.  One
        step (the linearly implicit scheme) starts from u^n and solves to
        round-off.
        """
        p = self.problem
        delta = self.gf.vec.CreateVector()
        converge = iterations > 1
        if converge and self.steps > 1:
            keep = Projector(self.free, True)
            self.gf.vec.data += keep * (self.prev.vec - self.before.vec)
        # The scale of the step: its load, or -- for a flow driven only by
        # boundary data -- the step operator acting on the lifted iterate.
        free = Projector(self.free, True)
        scale = max(Norm((free * rhs).Evaluate()),
                    Norm((free * (self.const.mat * self.gf.vec)).Evaluate()),
                    1e-300)
        for _ in range(iterations):
            residual = self._residual(rhs)
            relative = Norm(residual) / scale
            if converge and relative <= p.newton_tol:
                return
            self.nonlinear.AssembleLinearization(self.gf.vec)
            jac = self.const.mat + self.nonlinear.mat
            tol = max(TOL, min(1e-2, relative)) if converge else TOL
            self.solve(jac, residual, delta, direct=self._jacobian_matrix,
                       tol=tol)
            self.krylov = self.solve.iterations
            if self.krylov > REFRESH:
                self._precondition()
            self.gf.vec.data += delta
            if converge and (Norm(delta) <= p.newton_tol
                             * max(Norm(self.gf.vec), 1e-300)):
                return
        if converge:
            raise RuntimeError(
                f"{p.name}: the forward Newton iteration did not converge in "
                f"{iterations} steps at t = {self.t.Get():g}")

    def _explicit(self, rhs, keep, boundary):
        """Forward Euler: M (u - u_prev) / dt = g - f(q_true, u_prev)."""
        if self._mass_inv is None:
            self._mass_inv = factorize(self.mass, self.free, symmetric=True)
        step = self._residual(rhs)  # at u = u_prev
        self.gf.vec.data += self.problem.dt * (self._mass_inv * step)
        self.gf.vec.data = keep * self.gf.vec + boundary


class DataCache:
    """The noise-free data of one example, kept for its next runs.

    Runs at several noise levels see the same noise-free data -- the noise
    is drawn on top of them -- so the first run steps the forward model and
    the snapshots on U_h are kept, in an HDF5 file at ``path``; later runs
    read them back, and step the same forward model on where they run
    longer than any before.  The runs must share the discretization: each
    problem attached is checked against the first (dofs of U_h, the mesh's
    vertices, the time step).  ``close()`` deletes the file.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.forward = None
        self.stored = 0
        self._file = self._rows = self._key = self._checked = None

    def datum(self, problem, k):
        """Snapshot ``k`` (from 1) of the noise-free data, as an array."""
        if problem is not self._checked:
            self._check(problem)
        while self.stored < k:
            if self.forward is None:
                self.forward = ForwardRun(problem)
            self._append(self.forward.step().vec.FV().NumPy())
        return self._rows[k - 1]

    def _check(self, problem):
        vertices = np.array([v.point for v in problem.mesh.vertices])
        key = (problem.Ufes.ndof, problem.dt, problem.tstart,
               tuple(vertices.sum(axis=0)), tuple(np.abs(vertices).sum(axis=0)))
        if self._key is None:
            self._key = key
        elif key != self._key:
            raise ValueError(
                f"{problem.name}: this data cache holds the data of another "
                "discretization")
        self._checked = problem

    def _append(self, row):
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = h5py.File(self.path, "w")
            self._rows = self._file.create_dataset(
                "data", shape=(0, row.size), maxshape=(None, row.size),
                dtype="f8", chunks=(1, row.size))
        self._rows.resize(self.stored + 1, axis=0)
        self._rows[self.stored] = row
        self.stored += 1

    def close(self):
        """Delete the cached data."""
        if self._file is not None:
            self._file.close()
            self._file = self._rows = None
        self.path.unlink(missing_ok=True)
        self.forward, self.stored = None, 0
