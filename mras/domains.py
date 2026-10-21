"""Domain constructors: mesh plus the (U_h, Q_h) trial pair for each benchmark."""

from collections import namedtuple

from netgen.geom2d import SplineGeometry
from netgen.occ import X, Y, OCCGeometry, Rectangle
from ngsolve import H1, L2, Mesh

Domains = namedtuple("Domains", "mesh Ufes Qfes")


def _occ_rectangle(bcs, maxh, order_u, corners=((0, 0), (1, 1))):
    (x0, y0), (x1, y1) = corners
    shape = Rectangle(x1 - x0, y1 - y0).Face()
    shape.edges.Min(X).name = bcs[0]
    shape.edges.Max(X).name = bcs[1]
    shape.edges.Min(Y).name = bcs[2]
    shape.edges.Max(Y).name = bcs[3]
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
    return mesh


def darcy(maxh=0.04, order_u=3, order_q=0):
    """Unit square, homogeneous Dirichlet on U_h; Q_h is unconstrained."""
    mesh = _occ_rectangle(("left", "right", "bottom", "top"), maxh, order_u)
    Ufes = H1(mesh, order=order_u, dirichlet="top|bottom|left|right")
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def fisher(maxh=0.1, order_u=3, order_q=0):
    """Square [-1.25, 1.25]^2, homogeneous Dirichlet on U_h; Q_h free.

    The reference state decays like a Gaussian, so the homogeneous boundary
    condition is essentially exact for the paper's setting.
    """
    geo = SplineGeometry()
    geo.AddRectangle((-1.25, -1.25), (1.25, 1.25), bcs=("wall",) * 4)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    Ufes = H1(mesh, order=order_u, dirichlet="wall")
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def potential(maxh=0.1, order_u=3, order_q=0):
    """Disk of radius pi; U_h and Q_h unconstrained.

    The reference state does not vanish on the boundary, so imposing
    homogeneous Dirichlet conditions on U_h would preclude convergence to
    the true state (see docs/DEVIATIONS.md).
    """
    import math
    geo = SplineGeometry()
    geo.AddCircle(c=(0, 0), r=math.pi, bc="circle", leftdomain=1, rightdomain=0)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    Ufes = H1(mesh, order=order_u)
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def allen_cahn(maxh=0.1, order_u=3, order_q=0):
    """Rectangle [-2, 2] x [-1, 1]; U_h and Q_h unconstrained.

    As in the potential-flow benchmark, the reference state carries a
    positive offset and is not representable in a Dirichlet space.
    """
    geo = SplineGeometry()
    geo.AddRectangle((-2, -1), (2, 1), bcs=("wall",) * 4)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    Ufes = H1(mesh, order=order_u)
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def burgers_pipe(maxh=0.15, order_u=3, order_q=0, length=6.0, height=1.2,
                 hole_xs=(1.5, 3.0, 4.5), hole_r=0.18):
    """Pipe: rectangle with circular holes, homogeneous Dirichlet everywhere.

    A channel of size ``length x height`` from which disks of radius
    ``hole_r`` are cut at the given x-coordinates (mid-height).  Boundaries:
    inlet/outlet at x = 0/length, wall top and bottom, hole on each circle.
    """
    import netgen.libngpy._meshing as ngm
    from netgen.geom2d import CSG2d, Circle, Rectangle
    Pnt2 = getattr(ngm, "Point<2>")
    geo = CSG2d()
    duct = Rectangle(Pnt2(0, 0), Pnt2(length, height),
                     left="inlet", right="outlet", bottom="wall", top="wall")
    for cx in hole_xs:
        duct = duct - Circle(center=Pnt2(cx, height / 2), radius=hole_r,
                             bc="hole")
    geo.Add(duct)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    Ufes = H1(mesh, order=order_u, dirichlet="inlet|outlet|wall|hole")
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)
