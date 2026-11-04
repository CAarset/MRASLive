"""Adjoint-pairing verification for the coupled MRAS step forms.

The joint step assembled by ``MRASProblem.forms`` splits into a state row
(tested by ``v``) and a parameter row (tested by ``s``).  The state row
couples to the parameter trial through the bilinear form.  The parameter
row sees the state through the tracking term f_q^*(u - z): in the
semi-implicit and explicit schemes at the lagged iterate ``state.gfu``, in
the residual, and in the implicit scheme at the new state, in the matrix,
where it stands on the left with the opposite sign.  The linear step
operator on joint increments, with the tracking term read as a right-hand
side either way, is

    A = [[M_uu, K    ],        K x_q = [M x]_state  for x = (0, x_q),
         [N,    M_qq ]]        N y_u = [res(gfu <- y_u) - res(gfu <- 0)]_param
                                       - [M y]_param  for y = (y_u, 0).

Symmetry of A in the coefficient pairing ``<A x, y> = x . (A y)`` is the
discrete signature of the adjoint identity behind Assumption 2.2: the
cross blocks must be transposes of one another (the paper's
``f'_q(q, z)^*`` appearing in the parameter row is exactly the adjoint of
the ``f'_q(q, z)`` coupling of the state row), while the diagonal blocks
(mass, diffusion, parameter mass, datum-weighted parameter mass) are
symmetric by construction.  Pairing draws that live on one block at a
time cancels the diagonal blocks identically and isolates ``K - N^T``.

With the alignment ``z_old := z`` (one fresh datum at a single noise level, i.e. the
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


def _ratio(t1, t2, scale):
    """The mismatch of a pairing against its Cauchy-Schwarz scale, the size
    round-off is measured on: a random pairing can cancel to a small sum of
    large terms, and |t1| + |t2| would then count the cancellation."""
    return abs(t1 - t2) / scale if scale > 1e-30 else 0.0


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
        # An implicit tracking block sits in the matrix as -N: flip it.
        state_part = np.array(x)
        state_part[nu:] = 0.0
        carrier.vec.data = state_part
        out[nu:] -= 2 * np.asarray(M.matvec(carrier.vec))[nu:]
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
    norm = np.linalg.norm
    for _ in range(draws):
        x = rng.standard_normal(M.shape[1])
        y = rng.standard_normal(M.shape[1])
        ax, ay = apply(x), apply(y)
        scale = norm(y) * norm(ax) + norm(x) * norm(ay)
        full_ratios.append(_ratio(np.dot(y, ax), np.dot(x, ay), scale))
        full_activity = max(full_activity, scale)
        # Block-separated pairs: only K and N can contribute.
        x_q, y_u = np.zeros(M.shape[1]), np.zeros(M.shape[1])
        x_q[nu:] = rng.standard_normal(M.shape[1] - nu)
        y_u[:nu] = rng.standard_normal(nu)
        k_x, n_y = apply(x_q)[:nu], apply(y_u)[nu:]  # K x_q and N y_u
        scale = norm(y_u) * norm(k_x) + norm(x_q) * norm(n_y)
        cross_ratios.append(_ratio(np.dot(y_u[:nu], k_x),
                                   np.dot(x_q[nu:], n_y), scale))
        cross_activity = max(cross_activity, scale)
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
