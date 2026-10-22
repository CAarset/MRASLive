#!/usr/bin/env python
"""Decaying Darcy demo: sequential implicit updates at the stepper level.

Identifies the mobility q in  u_t = div((1 - q) grad u) + g  from noisy data
of a reference solution generated with the true permeability.  The estimator
alternates an explicit L2 gradient step for q with an implicit step for u —
a sequential scheme, unlike the joint ``mras.Problem`` solve of the paper's
benchmarks.  No (z - z_old)/dt term appears: the sigma coupling it would
feed is zero here.

    .venv/bin/python examples/darcy_decaying.py --quick
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ngsolve import (BilinearForm, GridFunction, IfPos, LinearForm, dx, grad,
                     x, y)

from mras.data.pdebench import load_nu_grid
from mras.domains import darcy
from mras.fields import l2norm, to_gridfunc, to_gridfunc0
from mras.noise import GaussianFEMNoise
from mras.plotting import draw_field
from mras.stepper import ImplicitStepper


def reference(dt, tsteps, q_true, Ufes, V, grids):
    """Noisy-data truth: implicit fine-mesh solves with the true permeability."""
    u, v = V.TrialFunction(), V.TestFunction()
    a = BilinearForm(V)
    a += q_true * grad(u) * grad(v) * dx
    a.Assemble()
    stepper = ImplicitStepper(dt, a)
    gf = GridFunction(V)
    gf.Set(to_gridfunc(grids[0], V))
    u_exact = []
    for counter in range(tsteps):
        stepper.step(gf, to_gridfunc(grids[counter], V))
        snap = GridFunction(Ufes)
        snap.Set(gf)
        u_exact.append(snap)
    return u_exact


def identify(dt, tsteps, noise_levels, mesh, Ufes, Qfes, q_true, u_exact, u_init,
             source, noise):
    """Alternate q-gradient and implicit u steps; yields error histories."""
    s = Qfes.TrialFunction()
    mass_q = BilinearForm(Qfes)
    mass_q += s * Qfes.TestFunction() * dx
    mass_q.Assemble()
    inv_q = mass_q.mat.Inverse(Qfes.FreeDofs())

    u, v = Ufes.TrialFunction(), Ufes.TestFunction()
    stiff = BilinearForm(Ufes)
    stiff += grad(u) * grad(v) * dx
    stiff.Assemble()
    stepper_u = ImplicitStepper(dt, stiff)

    r = (x - 0.5) ** 2 + (y - 0.5) ** 2
    gfq = GridFunction(Qfes)
    gfu = GridFunction(Ufes)
    z = GridFunction(Ufes)
    for noise_level in noise_levels:
        gfu.Set(u_init)
        gfq.Set(IfPos(0.42 ** 2 - r, 1, 0))
        error_u = [l2norm(gfu - u_exact[0], mesh)]
        error_q = [l2norm(gfq - q_true, mesh)]
        for i in range(tsteps - 1):
            z.Set(u_exact[i])
            z.Set(noise.draw(z, noise_level, i + 1))
            # explicit L2-gradient step on the tracking functional
            f_deriv = grad(z) * (grad(gfu) - grad(z))
            rhs = LinearForm(Qfes)
            rhs += dt * f_deriv * Qfes.TestFunction() * dx
            rhs.Assemble()
            gfq.vec.data += inv_q * rhs.vec
            error_q.append(l2norm(gfq - q_true, mesh))
            # implicit u step with mobility (1 - q), starting from the datum
            gfu.vec.data = z.vec.Copy()
            stepper_u.step(gfu, source, rhs_gradv_factor=1 - gfq,
                           rhs_gradv=z)
            error_u.append(l2norm(gfu - u_exact[i + 1], mesh))
        yield noise_level, np.array(error_u), np.array(error_q)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h", type=float, default=0.06)
    ap.add_argument("--tend", type=float, default=1.5)
    ap.add_argument("--tsteps", type=int, default=100)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.03])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="small sizes, seconds")
    ap.add_argument("--out", default="results/darcy_decaying")
    args = ap.parse_args()
    if args.quick:
        args.h, args.tend, args.tsteps = 0.2, 0.05, 20
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    grids = [rng.random((128, 128)) for _ in range(args.tsteps + 1)]

    dom = darcy(maxh=args.h, order_u=3, order_q=0)
    mesh, Ufes, Qfes = dom.mesh, dom.Ufes, dom.Qfes
    noise = GaussianFEMNoise(Ufes, seed=args.seed)
    dt = args.tend / args.tsteps
    q_true = to_gridfunc0(load_nu_grid(sample=args.sample), Qfes)
    V = darcy(maxh=args.h, order_u=2, order_q=0).Ufes
    u_exact = reference(dt, args.tsteps, q_true, Ufes, V, grids)
    # The data loop leaves the source at its last random grid; the
    # identification then drives the estimator with that constant field.
    source = to_gridfunc(grids[-1], Ufes)
    u_init = to_gridfunc(grids[0], Ufes)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for noise_level, eu, eq in identify(dt, args.tsteps, args.noise_levels, mesh, Ufes,
                                  Qfes, q_true, u_exact, u_init, source,
                                  noise):
        t = dt * np.arange(len(eu))
        axes[0].plot(t, eu, label=f"noise_level={noise_level:g}")
        axes[1].plot(t, eq, label=f"noise_level={noise_level:g}")
        np.savez_compressed(out / f"level_{noise_level:g}".replace(".", "p"),
                            error_u=eu, error_q=eq)
        print(f"noise_level={noise_level:g}: err_u={eu[-1]:.4f} err_q={eq[-1]:.4f}")
    for ax, name in zip(axes, ["u", "q"]):
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel(rf"$\|e_{{{name}}}\|$")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "errors.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    draw_field(axes[0], u_exact[-1], ((0, 0), (1, 1)), title="u(T)")
    draw_field(axes[1], q_true, ((0, 0), (1, 1)), title="q true")
    fig.tight_layout()
    fig.savefig(out / "fields.png", dpi=150, bbox_inches="tight")
    print(f"figures and npz files in {out}")


if __name__ == "__main__":
    main()
