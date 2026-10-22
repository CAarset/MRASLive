#!/usr/bin/env python
"""Run any paper benchmark end to end: data feed, MRAS loop, figures.

    .venv/bin/python examples/paper_problems.py --problem fisher --quick
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from mras.plotting import error_figure, evolution_figure
from mras.problems.allen_cahn import AllenCahnProblem
from mras.problems.darcy import DarcyProblem
from mras.problems.fisher import FisherProblem
from mras.problems.potential import PotentialProblem

PROBLEMS = {
    "darcy": DarcyProblem,
    "fisher": FisherProblem,
    "potential": PotentialProblem,
    "allen_cahn": AllenCahnProblem,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", choices=sorted(PROBLEMS), default="darcy")
    ap.add_argument("--h", type=float)
    ap.add_argument("--dt", type=float)
    ap.add_argument("--tend", type=float)
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.05])
    ap.add_argument("--quick", action="store_true",
                    help="h=0.2, short horizon: a wiring check in seconds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    overrides = {k: v for k, v in
                 (("h", args.h), ("dt", args.dt), ("tend", args.tend))
                 if v is not None}
    if args.quick:
        overrides.setdefault("h", 0.2)
        overrides.setdefault("tend", 0.01)
    out = Path(args.out or f"results/{args.problem}")
    out.mkdir(parents=True, exist_ok=True)

    prob = PROBLEMS[args.problem](**overrides)
    results = prob.run(args.noise_levels)
    for res in results:
        res.save(out)
        print(f"noise_level={res.noise_level:g}: err_u={res.final_error_u:.4f} "
              f"err_q={res.final_error_q:.4f}")

    error_figure(results, prob.dt, fname=out / "errors.png")
    evolution_figure([[r.state.gfu for r in results],
                      [r.state.gfq for r in results]],
                     prob.box,
                     col_labels=[f"noise_level={r.noise_level:g}" for r in results],
                     row_labels=["u", "q"], fname=out / "final_fields.png")
    print(f"figures and npz files in {out}")


if __name__ == "__main__":
    main()
