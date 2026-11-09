#!/usr/bin/env python
"""Is an example's forward model resolved at its discretization?

Steps the data model (mras.forward.ForwardRun, at q_true) at the example's
DEFAULTS, with half the time step, and on a coarser mesh (twice h), and
compares the states at the marked times: relative differences in L2 and in
the gradient's L2 norm, evaluated on the coarser of the two meshes.  It also
reports each state's pointwise range against its boundary data's -- a flow
obeying a maximum principle (Burgers) stays within it -- and, for named
boundaries, the flux int u . n ds.

    .venv/bin/python scripts/flow_check.py burgers_2d --tend 50 --marks 20 40 50 \\
        --flux inlet outlet

Writes ``results/flow_check/<name>.json``.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ngsolve  # noqa: E402
import numpy as np  # noqa: E402
from ngsolve import (BND, VOL, GridFunction, InnerProduct, Integrate,  # noqa: E402
                     IntegrationRule, TaskManager, grad, specialcf)

from mras.forward import ForwardRun  # noqa: E402
from mras.live import build  # noqa: E402


def run(name, h, dt, tend, marks, threads):
    """The forward states at the marks: {t: coefficient vector}, and the
    problem."""
    ngsolve.SetNumThreads(threads)
    p = build(name, h=h, dt=dt, tend=tend)
    steps = {round(t / dt): t for t in marks}
    states = {}
    with TaskManager():
        forward = ForwardRun(p)
        for step in range(1, p.tsteps + 1):
            z = forward.step()
            if step in steps:
                states[steps[step]] = z.vec.FV().NumPy().copy()
    return p, states


def field(p, values):
    gf = GridFunction(p.Ufes)
    gf.vec.FV().NumPy()[:] = values
    return gf


def differences(coarse, fine, mesh):
    """Relative L2 and gradient differences, integrated on ``mesh``."""
    d = fine - coarse
    l2 = (Integrate(InnerProduct(d, d), mesh, order=10)
          / Integrate(InnerProduct(fine, fine), mesh, order=10)) ** 0.5
    dg = grad(fine) - grad(coarse)
    h1 = (Integrate(InnerProduct(dg, dg), mesh, order=8)
          / Integrate(InnerProduct(grad(fine), grad(fine)), mesh, order=8)) ** 0.5
    return float(l2), float(h1)


def ranges(p, z, t):
    """The state's pointwise range per component, and its boundary data's."""
    shape = ngsolve.TRIG if p.mesh.dim == 2 else ngsolve.TET
    points = p.mesh.MapToAllElements(IntegrationRule(shape, 6), VOL)
    values = np.array(z(points)).reshape(len(points), -1)
    lift = p.boundary_data.lift(ngsolve.Parameter(t))
    free = np.fromiter(p.Ufes.FreeDofs(), dtype=bool)
    data = lift.vec.FV().NumPy()[~free]
    return {"state_min": values.min(axis=0).tolist(),
            "state_max": values.max(axis=0).tolist(),
            "boundary_coefficients": [float(data.min()), float(data.max())]
            if data.size else None}


def fluxes(p, z, names):
    n = specialcf.normal(p.mesh.dim)
    return {name: float(Integrate(InnerProduct(z, n), p.mesh, BND,
                                  definedon=p.mesh.Boundaries(name)))
            for name in names}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("--tend", type=float, required=True)
    ap.add_argument("--marks", type=float, nargs="+", required=True)
    ap.add_argument("--flux", nargs="*", default=[])
    ap.add_argument("--h", type=float, help="the mesh width (DEFAULTS')")
    ap.add_argument("--dt", type=float, help="the time step (DEFAULTS')")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("results/flow_check"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    from mras.live import _locate
    module, _ = _locate(args.name)
    sizes = getattr(module, "DEFAULTS", {}).get(args.name, {})
    h, dt = args.h or sizes["h"], args.dt or sizes["dt"]
    variants = {"base": (h, dt), "half_dt": (h, dt / 2), "double_h": (2 * h, dt)}
    states, problems = {}, {}
    for label, (hh, dd) in variants.items():
        problems[label], states[label] = run(args.name, hh, dd, args.tend,
                                             args.marks, args.threads)
        print(f"{label}: h = {hh:g}, dt = {dd:g} done", flush=True)
    report = {"name": args.name, "variants": variants, "marks": {}}
    base = problems["base"]
    for t in args.marks:
        z = field(base, states["base"][t])
        row = {"half_dt": differences(field(base, states["half_dt"][t]), z,
                                      base.mesh),
               "double_h": differences(
                   field(problems["double_h"], states["double_h"][t]), z,
                   problems["double_h"].mesh),
               "range": ranges(base, z, t)}
        if args.flux:
            row["flux"] = fluxes(base, z, args.flux)
        report["marks"][f"{t:g}"] = row
        print(f"t = {t:g}: {json.dumps(row)}", flush=True)
    (args.out / f"{args.name}.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
