"""Where the data excite (mras.excitation): the operators of the persistent
excitation, the accumulated Gramian, the dose and the conditional
excitation, against what they are claimed to be."""

import numpy as np
import pytest
from ngsolve import TaskManager

from mras.excitation import (GramianAccumulator, cell_dose,
                             conditional_excitation, sublevel,
                             weakest_direction)
from mras.forward import ForwardRun
from mras.live import build


def test_conditional_excitation_is_the_constrained_minimum():
    # c_K = min { p^T G p : p_K = 1 }, by eliminating p_K: the Schur
    # complement G_KK - g^T G_rest^-1 g.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((7, 7))
    gram = a @ a.T + 1e-3 * np.eye(7)
    c, lam, _ = conditional_excitation(gram)
    for k in range(7):
        rest = [j for j in range(7) if j != k]
        g = gram[rest, k]
        schur = gram[k, k] - g @ np.linalg.solve(gram[np.ix_(rest, rest)], g)
        assert c[k] == pytest.approx(schur, rel=1e-9)
    assert np.all(lam[0] <= c * (1 + 1e-12))
    assert np.all(c <= np.diag(gram) * (1 + 1e-12))


def test_round_off_eigenvalues_are_floored_not_inverted():
    gram = np.diag([1.0, 1e-20, -1e-18])  # a kernel, seen through round-off
    c, _, _ = conditional_excitation(gram, rho=1e-13)
    assert np.all(np.isfinite(c)) and np.all(c > 0)
    assert c[0] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def burgers_samples():
    """The clean 2D Burgers data at a few instants, on a coarse mesh."""
    p = build("burgers_2d", h=0.5, dt=0.05, tend=6.0)
    frames = []
    with TaskManager():
        run = ForwardRun(p)
        for step in range(1, p.tsteps + 1):
            z = run.step()
            if step % 40 == 0:
                frames.append((step * p.dt, z.vec.FV().NumPy().copy()))
    return p, frames


def test_the_chain_from_the_weakest_direction_to_the_dose(burgers_samples):
    # lambda_min <= c_K <= G_KK <= D_K, cell by cell: the dose bounds every
    # cell's excitation (Cauchy-Schwarz, a gradient coupling), the
    # conditional excitation lies between it and the weakest direction.
    from ngsolve import GridFunction
    p, frames = burgers_samples
    acc = GramianAccumulator(p)
    z = GridFunction(p.Ufes)
    for t, values in frames:
        z.vec.FV().NumPy()[:] = values
        acc.add(t, z)
    gram = acc.mean()
    dose, _ = cell_dose(p, frames)
    c, lam, vec = conditional_excitation(gram)
    diag = np.diag(gram)
    assert np.all(lam[0] <= c * (1 + 1e-9))
    assert np.all(c <= diag * (1 + 1e-9))
    assert np.all(diag <= dose * (1 + 1e-9))
    # ... and for any direction, whitened: p^T G p <= sum_K p_K^2 D_K
    rng = np.random.default_rng(1)
    for _ in range(5):
        hat = rng.standard_normal(gram.shape[0])
        assert hat @ gram @ hat <= np.sum(hat ** 2 * dose) * (1 + 1e-9)
    p1 = weakest_direction(vec, acc.whiten)
    assert np.abs(p1).max() == pytest.approx(1.0)
    assert sublevel(dose, np.median(dose)).sum() >= dose.size // 2


def test_the_accumulator_agrees_with_the_checklists_gramian(burgers_samples):
    # One sample, one panel: the accumulator's mean is the checklist's
    # window mean of B^T K^-1 B, whitened alike.
    from ngsolve import GridFunction
    from mras.excitation import (coupling_matrix, gramian_term,
                                 linearisation_point, mass_whitening, u_gram)
    from mras.solvers import factorize
    p, frames = burgers_samples
    t, values = frames[-1]
    z = GridFunction(p.Ufes)
    z.vec.FV().NumPy()[:] = values
    acc = GramianAccumulator(p)
    acc.add(t, z)
    inv = factorize(u_gram(p).mat, p.Ufes.FreeDofs(), symmetric=True)
    direct = gramian_term(p, coupling_matrix(p, z, linearisation_point(p)), inv)
    whiten = mass_whitening(p)
    expected = 0.5 * whiten @ (0.5 * (direct + direct.T)) @ whiten.T
    np.testing.assert_allclose(acc.mean(), expected, rtol=1e-10,
                               atol=1e-14 * np.abs(expected).max())
