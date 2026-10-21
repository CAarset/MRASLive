#!/usr/bin/env python
"""Assemble figures and the table summary from campaign output directories.

Reads <in>/table.json plus the per-level npz files written by
scripts/darcy_reproduce.py or scripts/reproduce.py, and writes

    <in>/errors.png              error histories, one row per level
    <in>/fields_<level>.png      u and q snapshots at the paper time instants
    <in>/table.md                markdown table of final errors

    .venv/bin/python scripts/make_figures.py --problem fisher --in results/fisher
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
from ngsolve import GridFunction

from mras.plotting import error_figure, evolution_figure
from mras.problems.allen_cahn import AllenCahnProblem
from mras.problems.darcy import DarcyProblem
from mras.problems.fisher import FisherProblem
from mras.problems.potential import PotentialProblem

PROBLEMS = {
    "darcy": {"cls": DarcyProblem, "times": [0.001, 0.05, 0.1, 0.5, 1.5, 5.0]},
    "fisher": {"cls": FisherProblem, "times": [0.001, 0.05, 0.1, 0.5, 1.5, 5.0]},
    "potential": {"cls": PotentialProblem,
                  "times": [0.001, 0.05, 0.1, 0.5, 1.5, 5.0]},
    "allen_cahn": {"cls": AllenCahnProblem,
                   "times": [0.001, 0.05, 0.075, 0.2, 0.5, 1.5, 5.0]},
}


class _Curves:
    """Duck-typed LevelResult: just what error_figure reads."""

    def __init__(self, problem, level, error_u, error_q):
        self.problem = problem
        self.noise_level = level
        self.error_u = error_u
        self.error_q = error_q


def rebuild_field(fes, series, index):
    """Rebuild a grid function from a saved free-dof snapshot row."""
    gf = GridFunction(fes)
    gf.vec.FV().NumPy()[:] = series[index]
    return gf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    ap.add_argument("--in", dest="indir", default=None)
    args = ap.parse_args()

    indir = Path(args.indir or f"results/{args.problem}")
    spec = PROBLEMS[args.problem]
    with open(indir / "table.json") as fh:
        table = json.load(fh)
    dt = table["dt"]
    tsteps = round(table["tend"] / dt)
    times = [0.0] + [t for t in spec["times"] if round(t / dt) <= tsteps]
    counters = [0] + [round(t / dt) for t in times[1:]]

    levels = sorted(float(name[len(args.problem) + 1:].replace("p", "."))
                    for name in (f.stem for f in indir.glob("*.json"))
                    if name.startswith(args.problem))
    if not levels:
        raise SystemExit(f"no {args.problem}_<level>.json files in {indir}")

    curves, rows_u, rows_q = [], [], []
    for level in levels:
        stem = f"{args.problem}_{level:g}".replace(".", "p")
        data = np.load(indir / f"{stem}.npz")
        curves.append(_Curves(args.problem, level, data["error_u"],
                              data["error_q"]))
        if "series_gfu" in data:
            if not rows_u:  # one problem instance provides the FE spaces
                prob = spec["cls"](h=table["h"], dt=dt, tend=table["tend"])
            n = min(len(counters), len(data["series_gfu"]))
            rows_u.append([rebuild_field(prob.Ufes, data["series_gfu"], i)
                           for i in range(n)])
            rows_q.append([rebuild_field(prob.Qfes, data["series_gfq"], i)
                           for i in range(n)])
            # pad shorter rows so the grid stays rectangular
            pad_u, pad_q = rows_u[-1], rows_q[-1]
            while len(pad_u) < len(times):
                pad_u.append(pad_u[-1])
                pad_q.append(pad_q[-1])

    error_figure(curves, dt, fname=indir / "errors.png")

    if rows_u:
        labels = [f"{t:g}" if i else "t=0" for i, t in enumerate(times)]
        for level, ru, rq in zip(levels, rows_u, rows_q):
            evolution_figure([ru, rq], prob.box, col_labels=labels,
                             row_labels=["u", "q"],
                             fname=indir
                             / (f"fields_{level:g}".replace(".", "p") + ".png"))

    lines = [f"# {args.problem} — h={table['h']:g}, dt={dt:g}, "
             f"T={table['tend']:g}", "",
             "| noise | err_u | err_q |", "|---|---|---|"]
    for c in curves:
        lines.append(f"| {c.noise_level:g} | {c.error_u[-1]:.4f} "
                     f"| {c.error_q[-1]:.4f} |")
    (indir / "table.md").write_text("\n".join(lines) + "\n")
    print("wrote errors.png, table.md"
          + (", fields_*.png" if rows_u else "") + f" in {indir}")


if __name__ == "__main__":
    main()
