"""Noise model: exact relative L2 amplitude, rng conventions, frozen draws."""

import numpy as np
import pytest
from ngsolve import CoefficientFunction, GridFunction

from mras.fields import l2norm
from mras.noise import NoiseModel


def _unit_field(small_darcy):
    gf = GridFunction(small_darcy.Ufes)
    gf.Set(CoefficientFunction(1.0))
    return gf


@pytest.mark.parametrize("level", [0.01, 0.05, 0.2])
def test_noise_level_is_exact(small_darcy, level):
    gf = _unit_field(small_darcy)
    model = NoiseModel(peaks=8, rng=np.random.default_rng(0))
    z = model.draw(gf, level, small_darcy.mesh)
    assert l2norm(z - gf, small_darcy.mesh) == pytest.approx(level, rel=1e-9)


def test_level_zero_is_identity_but_consumes_rng(small_darcy):
    gf = _unit_field(small_darcy)
    rng = np.random.default_rng(0)
    model = NoiseModel(peaks=8, rng=rng)
    z = model.draw(gf, 0.0, small_darcy.mesh)
    assert np.array_equal(np.array(z.vec), np.array(gf.vec))
    fresh = np.random.default_rng(0)
    assert not np.allclose(rng.random(4), fresh.random(4))


def test_reseed_each_draw_freezes_pattern(small_darcy):
    gf = _unit_field(small_darcy)
    model = NoiseModel(peaks=6, reseed_each_draw=True)
    z1 = model.draw(gf, 0.05, small_darcy.mesh)
    z2 = model.draw(gf, 0.05, small_darcy.mesh)
    assert np.array_equal(np.array(z1.vec), np.array(z2.vec))


def test_global_rng_seeding_reproduces():
    model = NoiseModel(peaks=6)
    np.random.seed(7)
    a1 = model.amplitudes()
    np.random.seed(7)
    a2 = model.amplitudes()
    assert np.allclose(a1, a2)


def test_gaussian_distribution(small_darcy):
    gf = _unit_field(small_darcy)
    model = NoiseModel(peaks=8, rng=np.random.default_rng(1),
                       distribution="gaussian")
    z = model.draw(gf, 0.1, small_darcy.mesh)
    assert l2norm(z - gf, small_darcy.mesh) == pytest.approx(0.1, rel=1e-9)
