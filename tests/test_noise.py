"""GaussianFEMNoise contract.

A draw is z = gf + e where e is a Gaussian field in U_h with

* exact relative amplitude:  ||e||_L2 == noise_level * ||gf||_L2,
* FEM white-noise covariance: E[e e^T] == (noise_level^2 ||gf||^2 / d) M^{-1},
  i.e. E[(e, phi)(e, psi)] = (phi, psi)_L2 for all phi, psi in U_h (the discrete
  analogue of Gaussian white noise),
* order-independent seeding: the stream is fixed by (seed, noise_level, counter) alone.
"""

import numpy as np
import pytest
from netgen.geom2d import SplineGeometry
from ngsolve import CoefficientFunction, GridFunction, H1, Mesh, dx

from mras.fields import l2norm
from mras.noise import GaussianFEMNoise, seeded_rng


@pytest.fixture(scope="module")
def unit_square():
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bcs=("wall", "wall", "wall", "wall"))
    return Mesh(geo.GenerateMesh(maxh=0.5))


@pytest.fixture(scope="module")
def coarse_fes(unit_square):
    return H1(unit_square, order=1)


@pytest.fixture(scope="module")
def base_field(coarse_fes):
    gf = GridFunction(coarse_fes)
    gf.Set(CoefficientFunction(1.0))
    return gf


def _dense_mass(fes):
    """Dense free-dof block of the mass matrix, by mass-matrix matvecs."""
    import ngsolve

    u, v = fes.TnT()
    m = ngsolve.BilinearForm(fes)
    m += u * v * dx
    m.Assemble()
    idx = np.flatnonzero(np.fromiter(fes.FreeDofs(), dtype=bool))
    A = np.zeros((idx.size, idx.size))
    e = np.zeros(fes.ndof)
    for j, jj in enumerate(idx):
        e[jj] = 1.0
        A[:, j] = np.array(m.mat.matvec(e))[idx]
        e[jj] = 0.0
    return A


def test_draw_is_gridfunction_in_space(coarse_fes, base_field):
    noise = GaussianFEMNoise(coarse_fes, seed=0)
    z = noise.draw(base_field, 0.05, 1)
    assert isinstance(z, GridFunction)
    assert z.space is coarse_fes


@pytest.mark.parametrize("noise_level", [0.01, 0.05, 0.2])
def test_level_is_exact(coarse_fes, unit_square, base_field, noise_level):
    noise = GaussianFEMNoise(coarse_fes, seed=1)
    z = noise.draw(base_field, noise_level, 3)
    e = z - base_field
    assert l2norm(e, unit_square) == pytest.approx(noise_level * l2norm(base_field, unit_square),
                                                   rel=1e-10)


def test_level_zero_is_exact_copy(coarse_fes, base_field):
    noise = GaussianFEMNoise(coarse_fes, seed=1)
    z = noise.draw(base_field, 0.0, 7)
    assert np.array_equal(np.array(z.vec), np.array(base_field.vec))


def test_same_keys_reproduce_anywhere(coarse_fes, base_field):
    a = GaussianFEMNoise(coarse_fes, seed=5).draw(base_field, 0.05, 7)
    # intervening draws on the same object must not shift the stream
    other = GaussianFEMNoise(coarse_fes, seed=5)
    other.draw(base_field, 0.1, 9)
    other.draw(base_field, 0.05, 2)
    b = other.draw(base_field, 0.05, 7)
    assert np.array_equal(np.array(a.vec), np.array(b.vec))


def test_counter_seed_and_level_change_the_draw(coarse_fes, base_field):
    ref = np.array(GaussianFEMNoise(coarse_fes, seed=2).draw(base_field, 0.05, 1).vec)
    assert not np.array_equal(np.array(GaussianFEMNoise(coarse_fes, seed=2).draw(base_field, 0.05, 2).vec), ref)
    assert not np.array_equal(np.array(GaussianFEMNoise(coarse_fes, seed=3).draw(base_field, 0.05, 1).vec), ref)
    assert not np.array_equal(np.array(GaussianFEMNoise(coarse_fes, seed=2).draw(base_field, 0.06, 1).vec), ref)


def test_negative_level_rejected(coarse_fes, base_field):
    noise = GaussianFEMNoise(coarse_fes, seed=0)
    with pytest.raises(ValueError):
        noise.draw(base_field, -0.01, 1)


def test_noise_vanishes_on_dirichlet_dofs(unit_square, base_field):
    fes = H1(unit_square, order=2, dirichlet="wall")
    gf = GridFunction(fes)
    gf.Set(CoefficientFunction(1.0))
    z = GaussianFEMNoise(fes, seed=4).draw(gf, 0.1, 2)
    free = np.fromiter(fes.FreeDofs(), dtype=bool)
    diff = np.array(z.vec) - np.array(gf.vec)
    assert diff[~free] == pytest.approx(0.0)
    assert np.abs(diff[free]).max() > 0.0


def test_white_noise_covariance_and_mean(unit_square, coarse_fes, base_field):
    """Known-answer check of the defining white-noise identity.

    With d free dofs, exact normalization gives
    E[e e^T] = noise_level^2 ||gf||^2 M^{-1} / d  and  E[e] = 0.
    """
    noise_level = 0.1
    noise = GaussianFEMNoise(coarse_fes, seed=11)
    idx = np.flatnonzero(np.fromiter(coarse_fes.FreeDofs(), dtype=bool))
    M = _dense_mass(coarse_fes)
    Minv = np.linalg.inv(M)
    target = noise_level ** 2 * l2norm(base_field, unit_square) ** 2 / idx.size * Minv

    n = 3000
    acc = np.zeros((idx.size, idx.size))
    mean = np.zeros(idx.size)
    basev = np.array(base_field.vec)[idx]
    for k in range(1, n + 1):
        v = np.array(noise.draw(base_field, noise_level, k).vec)[idx] - basev
        acc += np.outer(v, v)
        mean += v
    cov = acc / n
    mean /= n

    rel = np.linalg.norm(cov - target) / np.linalg.norm(target)
    assert rel < 0.12
    # mean must be zero well inside Monte-Carlo error: E||mean||_M = scale/sqrt(n)
    scale = noise_level * l2norm(base_field, unit_square)
    m_M = np.sqrt(float(mean @ (M @ mean)))
    assert m_M < 0.1 * scale  # ~6 sigma


def test_streams_are_independent_of_construction():
    a = seeded_rng(3, 1, 2)
    b = seeded_rng(3, 1, 2)
    assert np.array_equal(a.standard_normal(5), b.standard_normal(5))
    c = seeded_rng(3, 1, 3)
    assert not np.array_equal(c.standard_normal(5), a.standard_normal(5))
