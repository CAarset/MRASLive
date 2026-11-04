"""Pixel-array field helpers and norms."""

import numpy as np
import pytest
from ngsolve import CoefficientFunction, GridFunction

from mras.domains import pixel_square
from mras.fields import l2norm, pixel_field, to_gridfunc


def test_to_gridfunc_constant(small_darcy):
    fes = small_darcy.Ufes
    gf = to_gridfunc(np.full((6, 6), 2.0), fes)
    assert l2norm(gf - CoefficientFunction(2.0), small_darcy.mesh) < 1e-12


def test_to_gridfunc_subbox_extrapolates(small_darcy):
    # VoxelCoefficient keeps the edge value outside the box, so a constant
    # array restricted to a sub-box is still constant on the whole mesh.
    fes = small_darcy.Ufes
    gf = to_gridfunc(np.full((6, 6), 3.0), fes, box=((0.25, 0.25), (0.75, 0.75)))
    assert l2norm(gf - CoefficientFunction(3.0), small_darcy.mesh) < 1e-12


def test_pixel_constant_field_lies_in_p0_of_an_aligned_grid():
    # A field constant on the pixels, on a grid of the pixels halved: every
    # element lies in one pixel, so the P0 projection is the field itself.
    values = np.random.default_rng(0).random((8, 8))
    domains = pixel_square(16, order_u=1, order_q=0)
    field = pixel_field(values, linear=False)
    gf = GridFunction(domains.Qfes)
    gf.Set(field)
    assert l2norm(gf - field, domains.mesh) < 1e-12
    assert gf(0.3, 0.7) == pytest.approx(values[5, 2])  # rows run along y


def test_l2norm_matches_area_scaling(small_darcy):
    mesh = small_darcy.mesh
    assert l2norm(CoefficientFunction(1.0), mesh) == pytest.approx(1.0, rel=1e-10)
    assert l2norm(CoefficientFunction(3.0), mesh) == pytest.approx(3.0, rel=1e-10)
