#!/usr/bin/env python
"""Where is a reconstruction's error?  Its share by region, from a checkpoint.

Splits ||q - P_h q^dagger||_H^2 (the part of the error the estimator can
still remove; the rest, ||P_h q^dagger - q^dagger||_H, is the best
approximation's) over regions of P0 cells: the cells cut by a jump of a
piecewise constant q^dagger (where P_h q^dagger takes none of its values),
bands at a distance from them, bands near named boundaries, and the rest.
The problem is rebuilt from the checkpoint's own arguments.

    .venv/bin/python scripts/error_regions.py \\
        results/burgers_2d/burgers_2d_noise0.h5 --best --near inlet:0.5 outlet:0.5

Writes ``results/error_regions/<checkpoint>.json``.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from ngsolve import (CoefficientFunction, GridFunction, Integrate,  # noqa: E402
                     LinearForm, x, y)

from mras.live import build  # noqa: E402

BANDS = (0.15, 0.4)  # distances from the cut cells


def frame(h5, best, t):
    """The trajectory frame to read: the saved iterate of least error, the
    one nearest ``t``, or the last."""
    steps = h5["trajectory/step"][:]
    times = h5["trajectory/t"][:]
    if best:
        err = h5["history/error_q"][:][steps]
        k = int(np.argmin(err))
    elif t is not None:
        k = int(np.argmin(np.abs(times - t)))
    else:
        k = len(times) - 1
    return k, float(times[k])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint", type=pathlib.Path)
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--t", type=float)
    ap.add_argument("--near", nargs="*", default=[],
                    help="named boundary:distance, e.g. inlet:0.5")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("results/error_regions"))
    args = ap.parse_args(argv)
    h5 = h5py.File(args.checkpoint, "r")
    kwargs = json.loads(h5.attrs["arguments"])["kwargs"]
    p = build(json.loads(h5.attrs["problem"]) if h5.attrs["problem"].startswith('"')
              else h5.attrs["problem"], **kwargs)
    k, t = frame(h5, args.best, args.t)
    q = h5["trajectory/q"][k]
    if p.Qfes.globalorder != 0:
        raise SystemExit("regions are cells of a P0 parameter space")

    s = p.Qfes.TestFunction()
    form = LinearForm(p.Qfes)
    form += 1 * s * p.dx
    form.Assemble()
    area = form.vec.FV().NumPy().copy()
    reference = p.project_q(p.q_true).vec.FV().NumPy().copy()
    order = p.order + 4
    norm = float(Integrate(CoefficientFunction(p.q_true) ** 2, p.mesh,
                           order=order)) ** 0.5
    floor = p.q_floor()
    gap = q - reference
    gap2 = area * gap ** 2
    total = float(gap2.sum())
    error = (total + (floor * norm) ** 2) ** 0.5 / norm
    centre = []
    for coordinate in (x, y):
        gf = GridFunction(p.Qfes)
        gf.Set(coordinate)
        centre.append(gf.vec.FV().NumPy().copy())
    X, Y = centre

    regions = []
    values = (np.unique(np.round(p.q_true.vec.FV().NumPy(), 12))
              if isinstance(p.q_true, GridFunction) else None)
    taken = np.zeros(q.size, dtype=bool)
    if values is not None and values.size <= 8:
        cut = np.min(np.abs(reference[:, None] - values[None, :]), axis=1) > 1e-6
        idx = np.flatnonzero(cut)
        dist = np.min(np.hypot(X[:, None] - X[idx][None, :],
                               Y[:, None] - Y[idx][None, :]), axis=1)
        regions.append(("cut cells", cut))
        lo = 0.0
        for hi in BANDS:
            regions.append((f"within {lo:g}-{hi:g} of a cut cell",
                            ~cut & (dist > lo) & (dist <= hi)))
            lo = hi
        taken = cut | (dist <= BANDS[-1])
    for spec in args.near:
        name, distance = spec.split(":")
        points = np.array([p.mesh[v].point for el in
                           p.mesh.Boundaries(name).Elements() for v in el.vertices])
        near = np.min(np.hypot(X[:, None] - points[None, :, 0],
                               Y[:, None] - points[None, :, 1]), axis=1)
        mask = ~taken & (near <= float(distance))
        regions.append((f"within {distance} of {name}", mask))
        taken |= mask
    regions.append(("the rest", ~taken))

    rows = []
    for name, mask in regions:
        a = area[mask].sum()
        rows.append({"region": name,
                     "area_share": float(a / area.sum()),
                     "share_of_gap2": float(gap2[mask].sum() / total),
                     "rms_gap": float((gap2[mask].sum() / a) ** 0.5) if a else 0.0,
                     "rms_gap_relative": float((gap2[mask].sum() / a) ** 0.5
                                               / (norm / area.sum() ** 0.5))
                     if a else 0.0,
                     "max_abs_gap": float(np.abs(gap[mask]).max()) if mask.any() else 0.0})
    report = {"checkpoint": str(args.checkpoint), "t": t, "error": error,
              "floor": floor, "removable": total ** 0.5 / norm, "regions": rows}
    print(f"{args.checkpoint.name} at t = {t:g}: error {error:.4f}, floor "
          f"{floor:.4f}, removable {total ** 0.5 / norm:.4f}")
    print(f"{'region':34s} {'area':>7s} {'gap^2':>7s} {'rms':>8s} {'rms/q':>8s} {'max':>7s}")
    for row in rows:
        print(f"{row['region']:34s} {row['area_share']:7.3f} {row['share_of_gap2']:7.3f} "
              f"{row['rms_gap']:8.4f} {row['rms_gap_relative']:8.4f} {row['max_abs_gap']:7.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.checkpoint.stem}.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
