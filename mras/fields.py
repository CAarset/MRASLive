"""Grid-function helpers: pixel-array fields and L2 norms."""

import numpy as np
from ngsolve import (GridFunction, IfPos, Integrate, VoxelCoefficient,
                     x as xc, y as yc)

UNIT_BOX = ((0.0, 0.0), (1.0, 1.0))


def to_gridfunc(values, fes, box=UNIT_BOX, linear=True):
    """Map a 2D pixel array onto a finite element space.

    ``values`` is interpreted as an image filling ``box``; ``linear`` selects
    bilinear interpolation with edge extrapolation outside the box.
    """
    lo, hi = box
    coef = VoxelCoefficient((lo[0], lo[1]), (hi[0], hi[1]),
                            np.asarray(values, dtype=np.float64), linear=linear)
    gf = GridFunction(fes)
    gf.Set(coef)
    return gf


def to_gridfunc0(values, fes, box=UNIT_BOX, band=0.05, linear=True):
    """``to_gridfunc`` additionally zeroed in a strip along the box boundary."""
    lo, hi = box
    ramp = (IfPos(xc - (lo[0] + band), 1, (xc - lo[0]) / band)
            * IfPos(yc - (lo[1] + band), 1, (yc - lo[1]) / band)
            * IfPos((hi[0] - band) - xc, 1, (hi[0] - xc) / band)
            * IfPos((hi[1] - band) - yc, 1, (hi[1] - yc) / band))
    coef = VoxelCoefficient((lo[0], lo[1]), (hi[0], hi[1]),
                            np.asarray(values, dtype=np.float64), linear=linear)
    gf = GridFunction(fes)
    gf.Set(coef * ramp)
    return gf


def l2norm(cf, mesh):
    """L2 norm of a coefficient over the mesh."""
    return float(Integrate(cf * cf, mesh)) ** 0.5
