#!/usr/bin/env python
"""Reconstruction demo for the Arrhenius pair (sixth/seventh examples).

Runs both members of the pair at the shipped configuration, noise level 0% (noise-free), and
writes results/arrhenius_pair.png: q_true, the converged estimate, the
error field, and the error history.

    .venv/bin/python scripts/demo_arrhenius.py [--tend 30]
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mras.problems.arrhenius import ArrheniusLinearProblem, ArrheniusProblem
from mras.plotting import draw_field


def q_panels(problem, qhat, axes):
    box, mesh = problem.box, problem.mesh
    for ax, gf, title, kw in (
            (axes[0], problem.q_true, "$q^\\dagger$",
             dict(cmap="inferno", mesh=mesh)),
            (axes[1], qhat, "$\\hat q$", dict(cmap="inferno")),
            (axes[2], qhat - problem.q_true, "$\\hat q - q^\\dagger$",
             dict(cmap="bwr", vmin=-0.5, vmax=0.5, colorbar=False,
                  mesh=mesh))):
        draw_field(ax, gf, box, **kw)
        ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tend", type=float, default=None)
    ap.add_argument("--h", type=float, default=None)
    args = ap.parse_args()

    out = pathlib.Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    for row, cls in enumerate((ArrheniusLinearProblem, ArrheniusProblem)):
        kw = {}
        if args.tend is not None:
            kw["tend"] = args.tend
        if args.h is not None:
            kw["h"] = args.h
        problem = cls(**kw)
        res = problem.run_level(0, progress=False)
        qhat = res.state.gfq
        tag = "linearised $r(u)=1+u$" if "Linear" in cls.__name__ \
            else "nonlinear $r(u)=e^u$"
        q_panels(problem, qhat, axes[row, :3])
        axes[row, 0].set_ylabel(tag, fontsize=9)
        ax = axes[row, 3]
        t = np.arange(len(res.error_q)) * problem.dt
        ax.plot(t, res.error_q, lw=1.2)
        ax.set_yscale("log")
        ax.axhline(0.10, color="grey", ls=":", lw=0.8)
        ax.set_title(f"rel. err. $q$: final {res.error_q[-1]:.1%}",
                     fontsize=9)
        ax.set_xlabel("$t$")
        print(f"[demo] {tag}: final error_q = {res.error_q[-1]:.4f}",
              flush=True)
    fig.tight_layout()
    fig.savefig(out / "arrhenius_pair.png", dpi=140)
    print(f"[demo] wrote {out / 'arrhenius_pair.png'}", flush=True)


if __name__ == "__main__":
    main()
