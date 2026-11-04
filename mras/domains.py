"""Domain constructors: mesh plus the (U_h, Q_h) trial pair for each benchmark."""

import math
from collections import namedtuple

from netgen.geom2d import SplineGeometry
from netgen.occ import (X, Y, Face, Glue, OCCGeometry, Pnt, Rectangle,
                        Segment, SplineInterpolation, Wire)
from ngsolve import H1, L2, Mesh, VectorH1, exp, pi, sin, x, y

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


def darcy(maxh=0.04, order_u=3, order_q=0, size=1.0, mesh=None):
    """Square [0, size]^2, homogeneous Dirichlet on U_h; Q_h is unconstrained.

    ``mesh`` reuses a mesh of the square (say a refinement, ``refined``)
    instead of generating one."""
    if mesh is None:
        mesh = _occ_rectangle(("left", "right", "bottom", "top"), maxh,
                              order_u, corners=((0, 0), (size, size)))
    Ufes = H1(mesh, order=order_u, dirichlet="top|bottom|left|right")
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def pixel_square(cells, order_u=3, order_q=0):
    """The unit square as a structured grid of ``cells x cells`` squares,
    each cut in two; homogeneous Dirichlet on U_h, Q_h free.

    With ``cells`` a multiple of a pixel grid's width every element lies in
    one pixel, so a field constant on the pixels lies in P0 exactly.
    """
    from ngsolve.meshes import MakeStructured2DMesh
    mesh = MakeStructured2DMesh(quads=False, nx=cells, ny=cells)
    Ufes = H1(mesh, order=order_u, dirichlet="bottom|right|top|left")
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def fisher(maxh=0.1, order_u=3, order_q=0, rings=(0.5, 0.9), fitted=False):
    """Square [-1.25, 1.25]^2, homogeneous Dirichlet on U_h; Q_h free.

    ``fitted`` makes the circles of radii ``rings`` -- the jumps of the true
    ring parameter -- interfaces of the mesh, curved to the state order, so
    a parameter constant on the rings lies in P0: the mesh the true
    parameter is defined on.  Subdomains, outside in: "outside", "ring",
    "core" (for two rings).
    """
    geo = SplineGeometry()
    geo.AddRectangle((-1.25, -1.25), (1.25, 1.25), bcs=("wall",) * 4,
                     leftdomain=1, rightdomain=0)
    if fitted:
        for k, radius in enumerate(sorted(rings, reverse=True)):
            geo.AddCircle(c=(0, 0), r=radius, leftdomain=k + 2,
                          rightdomain=k + 1, bc="interface")
        names = ("outside", "ring", "core")
        for k in range(len(rings) + 1):
            geo.SetMaterial(k + 1,
                            names[k] if k < len(names) else f"inner{k}")
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(order_u)
    Ufes = H1(mesh, order=order_u, dirichlet="wall")
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def potential(maxh=0.1, order_u=3, order_q=0):
    """Disk of radius pi, curved to the state order; Dirichlet on the circle
    for U_h, Q_h unconstrained.

    The paper's problem (26) prescribes the state on the boundary, u = h
    there: the example imposes the reference state's boundary values (they
    do not vanish), which makes u - z vanish on the circle and C = c
    (-Laplace) coercive.
    """
    geo = SplineGeometry()
    geo.AddCircle(c=(0, 0), r=math.pi, bc="circle", leftdomain=1, rightdomain=0)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    mesh.Curve(order_u)
    Ufes = H1(mesh, order=order_u, dirichlet="circle")
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def allen_cahn(maxh=0.1, order_u=3, order_q=0, fitted=False):
    """Rectangle [-2, 2] x [-1, 1]; Dirichlet on the walls for U_h, Q_h
    unconstrained.

    As in the potential-flow benchmark, the state is prescribed on the
    boundary, where the reference state carries its positive offset.
    ``fitted`` makes the jumps of the true parameter, the lines x + y = 1
    and x - 2y + 0.4 = 0 (left of the first), interfaces of the mesh, so the
    piecewise constant parameter lies in P0: the mesh the true parameter is
    defined on.  Subdomains: "background", "upper" (beyond x + y = 1) and
    "lower" (below the second line).
    """
    if not fitted:
        geo = SplineGeometry()
        geo.AddRectangle((-2, -1), (2, 1), bcs=("wall",) * 4)
        mesh = Mesh(geo.GenerateMesh(maxh=maxh))
        return Domains(mesh, H1(mesh, order=order_u, dirichlet="wall"),
                       L2(mesh, order=order_q))
    geo = SplineGeometry()
    pts = {name: geo.AppendPoint(*xy) for name, xy in {
        "A": (-2, -1), "B": (2, -1), "C": (2, 1), "D": (0, 1),
        "E": (-2, 1), "F": (-2, -0.8), "X": (8 / 15, 7 / 15)}.items()}
    for a, b, left, right, bc in (
            ("A", "B", 3, 0, "wall"), ("B", "C", 2, 0, "wall"),
            ("C", "D", 2, 0, "wall"), ("D", "E", 1, 0, "wall"),
            ("E", "F", 1, 0, "wall"), ("F", "A", 3, 0, "wall"),
            ("D", "X", 2, 1, "interface"), ("X", "B", 2, 3, "interface"),
            ("F", "X", 1, 3, "interface")):
        geo.Append(["line", pts[a], pts[b]], leftdomain=left,
                   rightdomain=right, bc=bc)
    for k, name in enumerate(("background", "upper", "lower")):
        geo.SetMaterial(k + 1, name)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    Ufes = H1(mesh, order=order_u, dirichlet="wall")
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def burgers_pipe(maxh=0.15, order_u=3, order_q=0, length=6.0, height=1.2,
                 hole_xs=(1.5, 3.0, 4.5), hole_r=0.18, vector=False,
                 dirichlet="inlet|outlet|wall|hole"):
    """Pipe: rectangle with circular holes, Dirichlet on ``dirichlet``.

    A channel of size ``length x height`` from which disks of radius
    ``hole_r`` are cut at the given x-coordinates (mid-height).  Boundaries:
    inlet/outlet at x = 0/length, wall top and bottom, hole on each circle.
    The mesh is curved to order ``order_u``, so meshes of any width share
    one geometry.  ``vector`` makes U_h a velocity space (VectorH1).
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
    # Isoparametric holes: a straight-sided mesh cuts a polygon out of the
    # pipe, different at every width; curved, every mesh has the true holes
    # (docs/DEVIATIONS.md).
    mesh.Curve(order_u)
    space = VectorH1 if vector else H1
    Ufes = space(mesh, order=order_u, dirichlet=dirichlet)
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)


def pipe_wake_field(length=6.0, q_min=0.12, q_max=0.55):
    """The pipe examples' true diffusion: two smooth low-diffusion wake bands.

    The bands straddle the hole row (y = 0.4 and y = 0.8), lightly modulated
    along x; the field is bounded in [q_min, q_max] -- in fact it spans
    about 0.33 to 0.55, since wake / (1 + wake) stays below about 1/2 --
    and has no sharp boundaries: abrupt disks leave most of the L2 error in
    the kink layer rather than in the wake.
    """
    wake = exp(-((y - 0.4) ** 2) / 0.03) + exp(-((y - 0.8) ** 2) / 0.03)
    wake *= 0.75 + 0.25 * sin(2 * pi * x / length)  # mild streamwise modulation
    return q_min + (q_max - q_min) * (1.0 - wake / (1.0 + wake))


# -- the 2D Burgers channel ----------------------------------------------------

CHANNEL_WIDTH, CHANNEL_HEIGHT = 3.2, 5.0
# Inward bulges of the walls: (wall, height, depth, half-width), wall 0 the
# left (x = 0), 1 the right (x = CHANNEL_WIDTH).
CHANNEL_BULGES = ((0, 1.45, 0.30, 0.45), (0, 3.75, 0.22, 0.40),
                  (1, 2.45, 0.28, 0.50), (1, 4.20, 0.16, 0.35))
# Barriers, irregular rounded shapes r(theta) = radius (1 + sum a cos(m
# (theta - turn) + phase)): (centre x as a fraction of the width, centre y,
# radius, ((m, a, phase), ...), turn) -- an elongated tilted one, a
# teardrop, a bean.  Their positions scale with CHANNEL_WIDTH, their sizes
# do not.
CHANNEL_BARRIERS = (
    (0.6375, 1.25, 0.17, ((2, 0.45, 0.0),), 0.5),
    (0.4125, 2.55, 0.16, ((1, 0.30, 0.0), (2, 0.15, 0.6)), -1.2),
    (0.625, 3.70, 0.15, ((2, 0.25, 0.0), (3, 0.12, 0.4)), 0.3),
)
# Subdomains the fitted mesh follows, where the true viscosity differs:
# (shape, barrier, side), a shape beside a barrier at its height, in the lane
# between it and the wall on ``side`` (-1 left, +1 right).  The shapes on the
# left share one x, their lanes' mean centre rounded to 0.1; those on the
# right sit mirrored about the channel's middle (as if the walls were
# straight).  All three fill bounding squares of side CHANNEL_SHAPE_SIDE.
CHANNEL_INCLUSIONS = (("star", 0, -1), ("heart", 1, 1), ("diamond", 2, -1))
CHANNEL_SHAPE_SIDE = 1.05


def _bulge(wall, height):
    return sum(depth * math.exp(-((height - at) / width) ** 2)
               for side, at, depth, width in CHANNEL_BULGES if side == wall)


def _barriers():
    """The barriers, with their centres placed in the channel's width."""
    return [(fraction * CHANNEL_WIDTH, *rest)
            for fraction, *rest in CHANNEL_BARRIERS]


def _barrier_radius(spec, theta):
    _, _, radius, modes, turn = spec
    return radius * (1 + sum(a * math.cos(m * (theta - turn) + phase)
                             for m, a, phase in modes))


def _barrier(cx, cy, radius, modes, turn, n=48):
    spec = (cx, cy, radius, modes, turn)
    points = []
    for k in range(n):
        theta = 2 * math.pi * k / n
        r = _barrier_radius(spec, theta)
        points.append(Pnt(cx + r * math.cos(theta), cy + r * math.sin(theta), 0))
    return points + points[:1]


def _corners(shape, cx, cy, size):
    """The corners of a straight-sided shape of ``size`` on (cx, cy): a
    five-pointed star of outer radius ``size``, or a diamond ``size`` wide
    and 1.4 times as tall."""
    if shape == "diamond":
        return [(cx, cy + 0.7 * size), (cx - size / 2, cy),
                (cx, cy - 0.7 * size), (cx + size / 2, cy)]
    return [(cx + r * math.cos(angle), cy + r * math.sin(angle))
            for k in range(10)
            for r, angle in [(size if k % 2 == 0 else 0.433 * size,
                              math.pi / 2 + math.pi * k / 5)]]  # a point up


def _heart_half(cx, cy, width, side, n=40):
    """One half of the heart x = 16 sin^3 s, y = 13 cos s - 5 cos 2s -
    2 cos 3s - cos 4s, ``width`` wide and centred on (cx, cy), from the cusp
    (s = 0) to the tip (s = pi): the halves meet there, and both stay
    sharp."""
    scale = width / 32
    return [(cx + side * scale * 16 * math.sin(s) ** 3,
             cy + scale * (13 * math.cos(s) - 5 * math.cos(2 * s)
                           - 2 * math.cos(3 * s) - math.cos(4 * s) + 2.5))
            for s in (math.pi * k / n for k in range(n + 1))]


def _outline(shape, cx, cy, size):
    """The shape's outline as a closed polygon, finely sampled."""
    if shape == "heart":
        return (_heart_half(cx, cy, size, 1)
                + _heart_half(cx, cy, size, -1)[::-1][1:-1])
    corners = _corners(shape, cx, cy, size)
    return [(a[0] + (b[0] - a[0]) * k / 8, a[1] + (b[1] - a[1]) * k / 8)
            for a, b in zip(corners, corners[1:] + corners[:1])
            for k in range(8)]


def _face(shape, cx, cy, size):
    if shape == "heart":
        return Face(Wire([SplineInterpolation([Pnt(x_, y_, 0) for x_, y_ in
                                               _heart_half(cx, cy, size, s)])
                          for s in (1, -1)]))
    corners = [Pnt(x_, y_, 0) for x_, y_ in _corners(shape, cx, cy, size)]
    return Face(Wire([Segment(a, b) for a, b in
                      zip(corners, corners[1:] + corners[:1])]))


def _inside(point, polygon):
    """Whether ``point`` lies inside ``polygon`` (ray casting)."""
    px, py = point
    inside = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        if (ay > py) != (by > py) and px < ax + (py - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def _fits(outline, placed):
    """Whether a shape's outline keeps off the walls, the barriers and the
    shapes ``placed`` before it."""
    for x_, y_ in outline:
        if not (0 < y_ < CHANNEL_HEIGHT
                and _bulge(0, y_) < x_ < CHANNEL_WIDTH - _bulge(1, y_)):
            return False
        for spec in _barriers():
            dx_, dy_ = x_ - spec[0], y_ - spec[1]
            if math.hypot(dx_, dy_) <= _barrier_radius(spec, math.atan2(dy_, dx_)):
                return False
    others = [[(p.x, p.y) for p in _barrier(*spec)] for spec in _barriers()]
    for other in others + placed:
        if any(_inside(p, outline) for p in other):
            return False
    return all(not _inside(p, other) for other in placed for p in outline)


def _square(shape, side):
    """The size of ``shape`` whose bounding square has sides ``side``."""
    outline = _outline(shape, 0.0, 0.0, 1.0)
    xs, ys = [p[0] for p in outline], [p[1] for p in outline]
    return side / max(max(xs) - min(xs), max(ys) - min(ys))


def _inclusions():
    """The shapes, {name: (cx, cy, size)}, placed and sized as
    CHANNEL_INCLUSIONS says."""
    barriers = _barriers()
    centres = {}
    for shape, index, side in CHANNEL_INCLUSIONS:
        spec = barriers[index]
        edge = spec[0] + side * _barrier_radius(spec, 0 if side > 0 else math.pi)
        wall = (CHANNEL_WIDTH - _bulge(1, spec[1]) if side > 0
                else _bulge(0, spec[1]))
        centres[shape] = (edge + wall) / 2
    left = round(sum(c for (shape, _, side) in CHANNEL_INCLUSIONS
                     for c in [centres[shape]] if side < 0)
                 / sum(1 for *_, side in CHANNEL_INCLUSIONS if side < 0), 1)
    out, placed = {}, []
    for shape, index, side in CHANNEL_INCLUSIONS:
        cx = left if side < 0 else CHANNEL_WIDTH - left
        out[shape] = (cx, barriers[index][1],
                      _square(shape, CHANNEL_SHAPE_SIDE))
        outline = _outline(shape, *out[shape])
        if not _fits(outline, placed):
            raise ValueError(f"no room for the {shape} in the channel: "
                             f"it touches a wall, a barrier or a shape")
        placed.append(outline)
    return out


def burgers_channel(maxh=0.12, order_u=3, order_q=0, vector=True,
                    dirichlet="inlet|wall|barrier", fitted=False):
    """A vertical channel: inlet at the bottom (y = 0), outlet at the top.

    The walls bulge inward at staggered heights (``CHANNEL_BULGES``), and
    three irregular barriers (``CHANNEL_BARRIERS``) stand in the lanes the
    bulges leave.  Boundaries: "inlet", "outlet", "wall", "barrier".
    ``fitted`` adds the subdomains "star", "heart" and "diamond", beside
    the barriers (``CHANNEL_INCLUSIONS``), which the mesh then follows, in
    the "background", with "interface" boundaries between: the mesh the
    true viscosity is defined on.  The mesh is curved to order ``order_u``; ``vector`` makes
    U_h a velocity space (VectorH1).
    """
    heights = [CHANNEL_HEIGHT * k / 60 for k in range(61)]
    left = [Pnt(_bulge(0, t), t, 0) for t in heights]
    right = [Pnt(CHANNEL_WIDTH - _bulge(1, t), t, 0) for t in heights]
    channel = Face(Wire([
        Segment(left[0], right[0]), SplineInterpolation(right),
        Segment(right[-1], left[-1]), SplineInterpolation(left[::-1])]))
    channel.edges.name = "wall"
    channel.edges.Min(Y).name = "inlet"
    channel.edges.Max(Y).name = "outlet"
    for spec in _barriers():
        barrier = Face(Wire([SplineInterpolation(_barrier(*spec))]))
        barrier.edges.name = "barrier"
        channel = channel - barrier
    channel.faces.name = "background"
    shape = channel
    if fitted:
        inclusions = {name: _face(name, *placement)
                      for name, placement in _inclusions().items()}
        for name, inclusion in inclusions.items():
            inclusion.faces.name = name
            inclusion.edges.name = "interface"
            channel = channel - inclusion
        shape = Glue([channel, *inclusions.values()])
    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
    mesh.Curve(order_u)
    space = VectorH1 if vector else H1
    Ufes = space(mesh, order=order_u, dirichlet=dirichlet)
    Qfes = L2(mesh, order=order_q)
    return Domains(mesh, Ufes, Qfes)
