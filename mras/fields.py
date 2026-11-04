"""Grid-function helpers: pixel-array fields and L2 norms."""

import numpy as np
from ngsolve import L2, GridFunction, Integrate, VoxelCoefficient

UNIT_BOX = ((0.0, 0.0), (1.0, 1.0))


def pixel_field(values, box=UNIT_BOX, linear=True):
    """A 2D pixel array as a coefficient over ``box``.

    ``linear`` selects bilinear interpolation between the pixel centres,
    with edge extrapolation outside the box; otherwise the field is
    constant on each pixel.
    """
    lo, hi = box
    return VoxelCoefficient((lo[0], lo[1]), (hi[0], hi[1]),
                            np.asarray(values, dtype=np.float64), linear=linear)


def subdomain_field(mesh, values, default=0.0):
    """A field constant on the subdomains of ``mesh``: ``values`` by
    subdomain name, ``default`` elsewhere, as a P0 grid function -- exact,
    element by element.  Other meshes of the same geometry evaluate it by
    point search, so it is the true parameter for meshes that do not follow
    its jumps."""
    gf = GridFunction(L2(mesh, order=0))
    out = gf.vec.FV().NumPy()
    for element in mesh.Elements():
        out[element.nr] = values.get(element.mat, default)
    return gf


def to_gridfunc(values, fes, box=UNIT_BOX, linear=True):
    """Map a 2D pixel array onto a finite element space (see pixel_field)."""
    gf = GridFunction(fes)
    gf.Set(pixel_field(values, box, linear))
    return gf


def l2norm(cf, mesh):
    """L2 norm of a coefficient over the mesh."""
    return float(Integrate(cf * cf, mesh)) ** 0.5
