"""Adjoint-pairing verification for the coupled MRAS step forms.

The joint step assembled by ``Problem.forms`` splits into a state row
(tested by ``v``) and a parameter row (tested by ``s``).  The state row
couples to the parameter trial through the bilinear form, while the
parameter row sees the state trial through the lagged iterate
``state.gfu`` in the residual, so the linear step operator on joint
increments reads

    A = [[M_uu, K    ],        K x_q = [M x]_state  for x = (0, x_q),
         [N,    M_qq ]]        N y_u = [res(gfu <- y_u) - res(gfu <- 0)]_param.

Symmetry of A in the coefficient pairing ``<A x, y> = x . (A y)`` is the
discrete signature of the adjoint identity behind Assumption 2.2: the
cross blocks must be transposes of one another (the paper's
``f'_q(q, z)^*`` appearing in the parameter row is exactly the adjoint of
the ``f'_q(q, z)`` coupling of the state row), while the diagonal blocks
(mass, diffusion, parameter mass, datum-weighted parameter mass) are
symmetric by construction.  Pairing draws that live on one block at a
time cancels the diagonal blocks identically and isolates ``K - N^T``.

With the alignment ``z_old := z`` (one fresh, single-level datum, i.e. the
operator the continuous scheme assigns to the step) the mismatch must
vanish to machine precision for every problem that implements the
abstract scheme.  On an untouched step taken mid-assimilation the lagged
datum differs from ``z`` by ``O(dt)``, which would leave a first-order
deviation of the same relative size as the datum movement itself; it is
reported separately and is a discretisation property, not an adjointness
failure.  Note that the driver stores ``z_old := z`` at the end of every
step, so the last step's forms — the ones a post-run probe reads — are
always aligned, and the lagged column degenerates to the aligned one
(for the stationary Darcy datum it is degenerate at every step).
"""

import numpy as np
from ngsolve import GridFunction


def _ratio(t1, t2):
    denom = abs(t1) + abs(t2)
    return abs(t1 - t2) / denom if denom > 1e-30 else 0.0


def _residual(problem, state):
    _, res = problem.forms(state)
    res.Assemble()
    return np.array(res.vec.FV().NumPy())


def _probe(problem, state, draws, seed):
    nu = problem.Ufes.ndof
    a, _ = problem.forms(state)
    a.Assemble()
    M = a.mat
    carrier = GridFunction(problem.X)

    def apply(x):
        carrier.vec.data = x
        out = np.asarray(M.matvec(carrier.vec)).copy()
        base = np.array(state.gfu.vec.FV().NumPy())
        state.gfu.vec.data = x[:nu]
        hit = _residual(problem, state)[nu:]
        state.gfu.vec.data = np.zeros(nu)
        off = _residual(problem, state)[nu:]
        state.gfu.vec.data = base
        out[nu:] += hit - off
        return out

    rng = np.random.default_rng(seed)
    full_ratios, full_activity = [], 0.0
    cross_ratios, cross_activity = [], 0.0
    for _ in range(draws):
        x = rng.standard_normal(M.shape[1])
        y = rng.standard_normal(M.shape[1])
        t1, t2 = np.dot(y, apply(x)), np.dot(x, apply(y))
        full_ratios.append(_ratio(t1, t2))
        full_activity = max(full_activity, abs(t1) + abs(t2))
        # Block-separated pairs: only K and N can contribute.
        x_q, y_u = np.zeros(M.shape[1]), np.zeros(M.shape[1])
        x_q[nu:] = rng.standard_normal(M.shape[1] - nu)
        y_u[:nu] = rng.standard_normal(nu)
        t1, t2 = np.dot(y_u, apply(x_q)), np.dot(x_q, apply(y_u))
        cross_ratios.append(_ratio(t1, t2))
        cross_activity = max(cross_activity, abs(t1) + abs(t2))
    return {
        "max_ratio": max(cross_ratios),
        "max_ratio_full": max(full_ratios),
        "cross_activity": cross_activity,
        # K and N are both zero (the parameter never touches the state row
        # and vice versa): pairing holds trivially and proves nothing.
        "vacuous": cross_activity < 1e-10 * max(full_activity, 1e-300),
    }


def adjoint_mismatch(problem, state, draws=6, seed=0):
    """Randomised adjoint-pairing report for one frozen step form.

    Returns ``{"aligned": ..., "lagged": ..., "lag": ...}`` where each
    subreport is the probe output for the step form with ``z_old := z``
    and for the untouched step form, and ``lag`` is the sup-norm of
    ``z - z_old`` left over by the last assimilation step.
    """
    lagged = _probe(problem, state, draws, seed)
    original = np.array(state.z_old.vec.FV().NumPy())
    lag = float(np.max(np.abs(
        original - np.array(state.z.vec.FV().NumPy()))))
    try:
        state.z_old.vec.data = np.array(state.z.vec.FV().NumPy())
        aligned = _probe(problem, state, draws, seed)
    finally:
        state.z_old.vec.data = original
    return {"aligned": aligned, "lagged": lagged, "lag": lag}
