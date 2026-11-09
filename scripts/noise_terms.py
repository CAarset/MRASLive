#!/usr/bin/env python
"""Which term of the model carries the data's noise?  From a noisy checkpoint.

The data's residual in the model, D_t z + f(q, z) - g, takes up the noise
e = z - u^dagger of the datum the estimator saw (smoothed, as it saw it)
through each of its terms.  For saved frames of a noisy run (which store
the clean datum ``u_true`` beside the datum ``z``), the U* norm of each
term's part, in ||v||_U = ||grad v|| (as the verification measures), over
the parameter's own term ||f(P_h q^dagger, z) - f(0, z)||_U*:

  tracking     C_0 e, C_0 = -Laplace (how the noise reaches the state)
  coupling     f(P_h q^dagger, z) - f(0, z) of e: the parameter's term on it
  model        f(0, z) - f(0, u^dagger) over the domain (convection, ...)
  boundary     f's boundary densities, by name (a stiff resistance, ...)
  time         D_t e over a saved stride (a lower bound of the step's own)

    .venv/bin/python scripts/noise_terms.py results/burgers_2d/burgers_2d_noise0.01.h5 \\
        --times 10 20 40 60

Writes ``results/noise_terms/<checkpoint>.json``.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from ngsolve import GridFunction, InnerProduct, LinearForm, grad  # noqa: E402

from mras.excitation import u_gram  # noqa: E402
from mras.live import build  # noqa: E402
from mras.robust import dual2  # noqa: E402
from mras.solvers import factorize  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint", type=pathlib.Path)
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("results/noise_terms"))
    args = ap.parse_args(argv)
    h5 = h5py.File(args.checkpoint, "r")
    if "u_true" not in h5["trajectory"]:
        raise SystemExit("a noisy run's checkpoint stores u_true; this one does not")
    p = build(h5.attrs["problem"], **json.loads(h5.attrs["arguments"])["kwargs"])
    inv = factorize(u_gram(p).mat, p.Ufes.FreeDofs(), symmetric=True)
    reference = p.project_q(p.q_true)
    w = p.Ufes.TestFunction()
    times = h5["trajectory/t"][:]

    def gf(values):
        g = GridFunction(p.Ufes)
        g.vec.FV().NumPy()[:] = values
        return g

    def dual(terms):
        form = LinearForm(p.Ufes)
        for term in terms:
            form += term
        form.Assemble()
        return dual2(inv, form.vec) ** 0.5

    densities = p.model_boundary(gf(h5["trajectory/z"][0]), w)
    named = (sorted(densities) if isinstance(densities, dict)
             else ([None] if densities is not None else []))
    rows = []
    for target in args.times:
        k = int(np.argmin(np.abs(times - target)))
        if k == 0:
            continue
        z, clean = gf(h5["trajectory/z"][k]), gf(h5["trajectory/u_true"][k])
        z_prev = gf(h5["trajectory/z"][k - 1])
        clean_prev = gf(h5["trajectory/u_true"][k - 1])
        e = gf(z.vec.FV().NumPy() - clean.vec.FV().NumPy())
        stride = float(times[k] - times[k - 1])
        de = gf((z.vec.FV().NumPy() - z_prev.vec.FV().NumPy()
                 - clean.vec.FV().NumPy() + clean_prev.vec.FV().NumPy()) / stride)
        signal = dual([p.coupling(reference, clean, w) * p.dx])
        row = {"t": float(times[k]),
               "noise_L2_relative": float(np.sqrt(
                   _l2(p, e) / max(_l2(p, clean), 1e-300))),
               "tracking": dual([InnerProduct(grad(e), grad(w)) * p.dx]) / signal,
               "coupling": dual([p.coupling(reference, e, w) * p.dx]) / signal,
               "model": dual(p.model_terms(z, w, boundary=False)
                             + [-1 * term for term in
                                p.model_terms(clean, w, boundary=False)]) / signal,
               "time": dual([InnerProduct(de, w) * p.dx]) / signal}
        for name in named:
            density = p.model_boundary(e, w)
            density = density[name] if isinstance(density, dict) else density
            row[f"boundary {name}"] = dual(p._on_boundary({name: density},
                                                          p.order)) / signal
        rows.append(row)
        print(json.dumps({key: (round(v, 4) if isinstance(v, float) else v)
                          for key, v in row.items()}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.checkpoint.stem}.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "relative_to":
         "||f(P_h q^dagger, u^dagger) - f(0, u^dagger)||_U*", "rows": rows},
        indent=1))
    return rows


def _l2(p, g):
    from ngsolve import Integrate
    return float(Integrate(InnerProduct(g, g), p.mesh, order=p.order))


if __name__ == "__main__":
    main()
