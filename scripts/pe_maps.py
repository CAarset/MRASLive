#!/usr/bin/env python
"""Persistent excitation of an example's clean data, as data for pictures.

The checklist's Gramian (mras.verify.persistent_excitation), accumulated
over a whole run instead of two windows:

    G(t) = int_0^t B(s)^T K^-1 B(s) ds,    B(s) = f_q'(q~, z(s)) : Q_h -> U_h*,

K the Gram matrix of ||v||_U = ||grad v|| on U_h, so that p^T G(t) p =
int_0^t ||f_q'(q~, z(s)) p||_U*^2 ds, and its quotient by ||p||_H^2 is the
excitation of the direction p up to time t.  The datum is sampled every
``--every`` time units (the trapezoidal rule in time); G is written,
H-whitened (the Q_h mass made the identity: W = L^-1, M_q = L L^T, so
eigenvalues of W G W^T are the Rayleigh quotients' extremes), every
``--save`` time units, in float64 (the weakest directions are some 1e-10 of
the strongest), under ``--out``.  Q_h must fit the dense
eigenproblem (mras.verify.DENSE_MAX_DOFS); the example is built at ``--h``.

    .venv/bin/python scripts/pe_maps.py burgers_2d --h 0.1 --tend 100 --every 1 --save 5
"""

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ngsolve  # noqa: E402
import numpy as np  # noqa: E402
from ngsolve import TaskManager  # noqa: E402

from mras.excitation import GramianAccumulator  # noqa: E402
from mras.forward import ForwardRun  # noqa: E402
from mras.live import build  # noqa: E402
from mras.verify import DENSE_MAX_DOFS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("--h", type=float, default=0.1)
    ap.add_argument("--tend", type=float, default=100.0)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--save", type=float, default=5.0,
                    help="a multiple of --every")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("results/pe"))
    args = ap.parse_args()
    ngsolve.SetNumThreads(args.threads)
    out = args.out / f"{args.name}_h{args.h:g}"
    out.mkdir(parents=True, exist_ok=True)
    p = build(args.name, h=args.h, tend=args.tend)
    if p.Qfes.ndof > DENSE_MAX_DOFS:
        raise SystemExit(f"Q_h has {p.Qfes.ndof} dofs, beyond the dense "
                         f"eigenproblem's {DENSE_MAX_DOFS}: take a coarser h")
    print(f"{args.name}: h = {args.h}, Q_h {p.Qfes.ndof}, U_h {p.Ufes.ndof} "
          f"dofs", flush=True)
    acc = GramianAccumulator(p)
    np.save(out / "whiten.npy", acc.whiten)
    every = max(1, round(args.every / p.dt))
    save = every * max(1, round(args.save / (every * p.dt)))  # on the samples
    started = time.time()
    meta = {"name": args.name, "h": args.h, "dt": p.dt, "every": every * p.dt,
            "saved": [], "q_dofs": p.Qfes.ndof, "u_dofs": p.Ufes.ndof}
    with TaskManager():
        run = ForwardRun(p)
        for step in range(1, p.tsteps + 1):
            z = run.step()
            if step % every:
                continue
            acc.add(step * p.dt, z)
            if step % save == 0:
                t = round(step * p.dt, 6)
                np.save(out / f"G_{t:07.2f}.npy", acc.integral())
                meta["saved"].append(t)
                (out / "meta.json").write_text(json.dumps(meta, indent=1))
                print(f"t = {t:6.2f}: saved ({time.time() - started:.0f} s)",
                      flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
