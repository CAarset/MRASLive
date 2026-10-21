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
    """Unit square, homogeneous Dirichlet on U_h and constrained Q_h."""
    mesh = _occ_rectangle(("left", "right", "bottom", "top"), maxh, order_u)
    Ufes = H1(mesh, order=order_u, dirichlet="top|bottom|left|right")
    Qfes = L2(mesh, order=order_q, dirichlet="top|bottom|left|right")
    return Domains(mesh, Ufes, Qfes)


def fisher(maxh=0.1, order_u=3, order_q=0):
    """Square [-1.25, 1.25]^2; no Dirichlet conditions (pure Neumann-type feed)."""
    geo = SplineGeometry()
    geo.AddRectangle((-1.25, -1.25), (1.25, 1.25),
                     bcs=("wall", "outlet", "wall", "inlet"))
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    Ufes = H1(mesh, order=order_u)
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def potential(maxh=0.1, order_u=3, order_q=0):
    """Disk of radius pi, homogeneous Dirichlet on U_h; Q_h unconstrained."""
    import math
    geo = SplineGeometry()
    geo.AddCircle(c=(0, 0), r=math.pi, bc="circle", leftdomain=1, rightdomain=0)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    Ufes = H1(mesh, order=order_u, dirichlet="circle")
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def allen_cahn(maxh=0.1, order_u=3, order_q=0):
    """Rectangle [-2, 2] x [-1, 1], homogeneous Dirichlet on U_h."""
    geo = SplineGeometry()
    geo.AddRectangle((-2, -1), (2, 1), bcs=("wall", "wall", "wall", "wall"))
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    Ufes = H1(mesh, order=order_u, dirichlet="wall")
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)
