"""The model fit: where the estimator plateaus, estimated as it runs.

The L2 floor (``MRASProblem.q_floor``) is the closest any estimate in Q_h
comes to q.  The estimator, though, sees q only through the state equation,
so it settles near the q_h in Q_h that best explains the data in its own
discrete model.  In an element a jump of q cuts through, the data pin a
gradient-weighted average of q there, not its area average; the part of q
outside Q_h leaves a residual of its own.  That q_h minimises the residual of the datum
in the estimator's state equation,

    r_n(q) = M (z^{n+1} - z^n) / dt + F(q, z^{n+1}) - g^{n+1}   (free dofs),

over the steps seen, weighted as the estimator's error equation weights it,
by the inverse of the tracking operator C (with a mass term where C has no
Dirichlet part to make it definite): when the parameter moves slowly, the
error settles at e = -C^-1 r, and the parameter flow then descends
sum |r|^2_{C^-1}.  Each update solves that least-squares problem by
preconditioned conjugate gradients, linearised at the previous update's fit
-- exact for an f affine in q, a Gauss-Newton iteration across the updates
otherwise -- with a faint pull toward the L2 best fit for the directions no
step has excited.  The fit's error is an estimate of the plateau, not a
bound -- the estimator, which does not settle exactly, may end a little
below it, and a semi-converging one passes well below it on its way -- and
the pictures do not draw it.  An error at or below it that has stopped
falling has most likely gone as far as these data and this mesh allow (the
run's ``"plateau"`` stop).

The fit samples a step every ``every`` steps (from EVERY), keeping at most
CAP of them -- beyond, every other one is dropped and ``every`` doubles --
and updates at step FIRST, then at twice the steps each time, so the
updates cost about twice the last one.
"""

import types

import numpy as np
from ngsolve import (BilinearForm, GridFunction, InnerProduct, LinearForm,
                     Parameter, Projector, grad)
from ngsolve.krylovspace import CGSolver
from ngsolve.la import BaseMatrix

from .solvers import factorize, mass

EVERY = 5     # steps between two sampled steps, to begin with
CAP = 100     # sampled steps kept
FIRST = 50    # the first update's step; each later one at twice the steps
STALL = 0.02  # an error that fell by less than this since the last update
PULL = 1e-10  # the pull toward the L2 best fit, relative to the equations


class ModelFit:
    """The estimator's model fit over a run (see the module docstring)."""

    def __init__(self, problem):
        p = self.problem = problem
        self.reference = p._true_parts()["reference"]  # the L2 best fit
        self.point = GridFunction(p.Qfes)       # the latest fit
        self.point.vec.data = self.reference.vec
        self.every, self.kept = EVERY, []       # (t^{n+1}, z^{n+1}, zt)
        self.next_update = FIRST
        self.value = None                       # the latest fit's error
        self.values = []                        # (t, value) at every update
        self._error = None                      # the error at the last update
        self.stalled = False
        free = np.array(p.Ufes.FreeDofs(), dtype=bool)
        self._keep = Projector(p.Ufes.FreeDofs(), True)
        self._root = np.sqrt(np.where(free, 1 / _diagonal(mass(p.Ufes, p.dx)),
                                      0.0))
        u, v = p.Ufes.TnT()
        weight = BilinearForm(p.Ufes, symmetric=True)
        scale = p.tracking_scale(self.reference)
        weight += scale * InnerProduct(grad(u), grad(v)) * p.dx
        for term in p._on_boundary(p.tracking_boundary(u, v), p.order):
            weight += scale * term
        if free.all():  # no Dirichlet part: C alone is only semidefinite
            weight += InnerProduct(u, v) * p.dx
        weight.Assemble()
        self._weight_inv = factorize(weight.mat, p.Ufes.FreeDofs(),
                                     symmetric=True)
        self._q_mass = mass(p.Qfes, p.dx)
        self._scratch = types.SimpleNamespace(
            t_next=Parameter(p.tstart), g=None, z=GridFunction(p.Ufes),
            zt=GridFunction(p.Ufes), u_true=GridFunction(p.Ufes))

    # -- during the run -------------------------------------------------------
    def offer(self, state):
        """Sample the step that has just read the datum ``state.z``."""
        if (state.step + 1) % self.every:
            return
        self.kept.append((state.t_next.Get(),
                          state.z.vec.FV().NumPy().copy(),
                          state.zt.vec.FV().NumPy().copy()))
        if len(self.kept) > CAP:
            self.kept = self.kept[1::2]
            self.every *= 2

    def due(self, step):
        return bool(self.kept) and step >= self.next_update

    def update(self, step, t, error):
        """Fit the steps sampled so far; record whether the error has
        stalled since the last update."""
        self.value = self._fit()
        self.values.append((t, self.value))
        self.stalled = (self._error is not None
                        and error >= (1 - STALL) * self._error)
        self._error = error
        self.next_update = 2 * max(step, FIRST // 2)
        return self.value

    # -- the fit --------------------------------------------------------------
    def _fit(self):
        p, qk = self.problem, self.point
        blocks, loads = [], []
        for t, z, zt in self.kept:
            jacobian, residual = self._equations(t, z, zt, qk)
            blocks.append(jacobian)
            loads.append(residual)
        weight_inv, keep = self._weight_inv, self._keep

        def apply(x, out):
            out[:] = 0.0
            for jacobian in blocks:
                out.data += jacobian.T * (weight_inv * (keep * (jacobian * x))
                                          .Evaluate()).Evaluate()
        rhs = qk.vec.CreateVector()
        rhs[:] = 0.0
        for jacobian, residual in zip(blocks, loads):
            rhs.data -= jacobian.T * (weight_inv * (keep * residual)
                                      .Evaluate()).Evaluate()
        probe = qk.vec.CreateVector()
        probe.FV().NumPy()[:] = 1.0
        image = probe.CreateVector()
        apply(probe, image)
        pull = PULL * max(InnerProduct(image, probe), 1e-300) / max(
            InnerProduct(self._q_mass * probe, probe), 1e-300)
        offset = (qk.vec - self.reference.vec).Evaluate()
        rhs.data -= pull * (self._q_mass * offset)
        normal = _Operator(p.Qfes, apply, self._q_mass, pull)
        solver = CGSolver(mat=normal, pre=self._preconditioner(blocks),
                          tol=1e-6, maxiter=200, printrates=False)
        fit = GridFunction(p.Qfes)
        fit.vec.data = qk.vec + solver * rhs
        self.point.vec.data = fit.vec
        return p.q_error(fit)

    def _equations(self, t, z, zt, qk):
        """The Jacobian and residual of a sampled step's equations at qk."""
        p, s = self.problem, self._scratch
        s.t_next.Set(t)
        s.z.vec.FV().NumPy()[:] = z
        s.zt.vec.FV().NumPy()[:] = zt
        s.g = p.source(s.t_next)
        if p.manufactured_source:
            s.u_true.Set(p.truth(s.t_next))
        q, v = p.Qfes.TrialFunction(), p.Ufes.TestFunction()
        jacobian = BilinearForm(trialspace=p.Qfes, testspace=p.Ufes)
        jacobian += q * p.coupling_adjoint(qk, s.z, v) * p.dx
        jacobian.Assemble()
        residual = LinearForm(p.Ufes)
        residual += InnerProduct(s.zt, v) * p.dx
        for term in p.model_terms(s.z, v):
            residual += term
        residual += p.coupling(qk, s.z, v) * p.dx
        for term in p._source_terms(s, v):
            residual += (-1) * term
        residual.Assemble()
        out = residual.vec.CreateVector()
        out.data = residual.vec
        return jacobian.mat, out

    def _preconditioner(self, blocks):
        """The same fit weighted by the inverse diagonal of the mass matrix
        instead: assembled, and factorised."""
        normal = None
        for jacobian in blocks:
            scaled = jacobian.CreateMatrix()
            scaled.AsVector().data = jacobian.AsVector()
            values, _, rows = scaled.CSR()
            np.asarray(values)[:] *= self._root[_entry_rows(rows)]
            product = scaled.CreateTranspose() @ scaled
            if normal is None:
                normal = product
            else:
                normal.AsVector().data += product.AsVector()
        where, values = self._pull_entries(normal)
        scale = float(np.sum(_diagonal(normal))) / max(float(np.sum(values)),
                                                       1e-300)
        normal.AsVector().FV().NumPy()[where] += (1e-8 * max(scale, 1e-300)
                                                  * values)
        return factorize(normal, symmetric=True)

    def _pull_entries(self, normal):
        """Where the mass matrix of Q_h sits in the normal matrix's pattern
        (it does: the dofs of one element share the element's state dofs),
        and its values."""
        m_values, m_cols, m_rows = (np.asarray(a) for a in self._q_mass.CSR())
        _, g_cols, g_rows = (np.asarray(a) for a in normal.CSR())
        where = []
        for row in range(len(m_rows) - 1):
            start, stop = int(g_rows[row]), int(g_rows[row + 1])
            columns = {int(c): start + k
                       for k, c in enumerate(g_cols[start:stop])}
            where += [columns[int(c)]
                      for c in m_cols[m_rows[row]:m_rows[row + 1]]]
        return np.array(where), m_values.copy()


class _Operator(BaseMatrix):
    """The weighted normal operator, applied without being assembled."""

    def __init__(self, space, apply, q_mass, pull):
        super().__init__()
        self.space, self.apply, self.q_mass, self.pull = (space, apply,
                                                          q_mass, pull)

    def IsComplex(self):
        return False

    def Height(self):
        return self.space.ndof

    def Width(self):
        return self.space.ndof

    def CreateColVector(self):
        return GridFunction(self.space).vec.CreateVector()

    def CreateRowVector(self):
        return GridFunction(self.space).vec.CreateVector()

    def Mult(self, x, y):
        self.apply(x, y)
        y.data += self.pull * (self.q_mass * x)


def _entry_rows(rows):
    """The row of every stored entry, from a CSR row pointer (NGSolve's is
    unsigned)."""
    rows = np.asarray(rows).astype(np.int64)
    return np.repeat(np.arange(len(rows) - 1), np.diff(rows))


def _diagonal(matrix):
    """The diagonal of an assembled sparse matrix, as an array."""
    values, cols, rows = (np.asarray(a) for a in matrix.CSR())
    entry_rows = _entry_rows(rows)
    on = cols.astype(np.int64) == entry_rows
    out = np.zeros(len(rows) - 1)
    out[entry_rows[on]] = values[on]
    return out
