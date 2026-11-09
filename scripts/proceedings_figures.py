#!/usr/bin/env python
"""Figures of the 2D Burgers example for the proceedings (proceedings/figures).

    .venv/bin/python scripts/proceedings_figures.py [reconstruction] [dose] [gramian]

reconstruction  q^dagger, the best-found estimate at each noise level and
                its error, and the error histories: from the production
                checkpoints in results/burgers_2d (the best-found estimate is
                the saved iterate of smallest error, an oracle choice that
                reads q^dagger: the runs had no stopping rule for noise).
dose            the dose D_K(T) = (1/T) int_0^T (1/|K|) int_K |grad z|^2 dx dt
                of the clean datum on the production mesh, an upper bound of
                every cell's excitation, and its sublevel set.
gramian         the Gramian of the persistent excitation on a coarser mesh
                (scripts/pe_maps.py): its weakest direction, the conditional
                excitation of each cell and its sublevel set, the spectrum.

Fields are painted by ngsplot, as the movies are; outlines of sublevel sets
are drawn on top, through the panel's own camera.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import ngsolve  # noqa: E402
import numpy as np  # noqa: E402
from ngsolve import L2, CoefficientFunction, GridFunction, Norm, TaskManager  # noqa: E402
from ngsplot import Plot, frame, render_data  # noqa: E402
from ngsplot.pyplot import colorbar as ngs_colorbar  # noqa: E402

from mras.excitation import (RHO, cell_dose, conditional_excitation,  # noqa: E402
                             sublevel, weakest_direction)
from mras.live import build  # noqa: E402

FIGURES = ROOT / "proceedings" / "figures"
RESULTS = ROOT / "results" / "burgers_2d"
LEVELS = (0.0, 0.01, 0.05)
PANEL = 520          # pixels of a panel's width (print: about 3 cm at 400 dpi)
Q_RANGE = (0.475, 1.025)  # q^dagger's values 0.5 and 1, padded 5 % as in the movies
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8,
                     "mathtext.fontset": "stix", "font.family": "STIXGeneral"})


def checkpoint(level):
    return h5py.File(RESULTS / f"burgers_2d_noise{level:g}.h5", "r")


def best(level):
    """The saved iterate of smallest error: (t, error, q coefficients)."""
    h5 = checkpoint(level)
    t, err = h5["history/t"][:], h5["history/error_q"][:]
    frames, steps = h5["trajectory/t"][:], h5["trajectory/step"][:]
    errs = err[steps]
    k = int(np.argmin(errs))
    return float(frames[k]), float(errs[k]), h5["trajectory/q"][k], \
        float(t[np.argmin(err)]), float(err.min())


def view_of(payload):
    return frame(payload, width=PANEL, height=None, fit="box")


def paint(ax, field, mesh, view, **kw):
    payload = render_data(field) if mesh is None else render_data(field, mesh)
    image = Plot(payload, ax=ax, view=view, wireframe=False, **kw)
    ax.set_axis_off()
    return image


def outline(ax, view, mesh, inside, **kw):
    """Draw the boundary of a union of cells: the facets between a cell in
    the set and one outside it (or the domain's boundary), projected by the
    panel's camera."""
    segments = []
    for facet in mesh.facets:
        cells = [el.nr for el in facet.elements]
        flags = [bool(inside[c]) for c in cells]
        if len(cells) == 2 and flags[0] != flags[1] or len(cells) == 1 and flags[0]:
            points = np.array([list(mesh[v].point) + [0.0] for v in facet.vertices])
            segments.append(points)
    if not segments:
        return
    cam = view.camera if hasattr(view, "camera") else view
    for points in segments:
        pix = cam.project(points)
        ax.plot(pix[:, 0], pix[:, 1], **kw)


def bar(fig, ax_list, image, label, orientation="vertical"):
    """ngsplot's bar for ``image`` (the picture holds colours, not values),
    in a slot beside -- or under -- the axes ``ax_list``."""
    boxes = [ax.get_position() for ax in np.atleast_1d(ax_list).ravel()]
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    y0, y1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    if orientation == "vertical":
        cax = fig.add_axes([x1 + 0.012, y0 + 0.08 * (y1 - y0), 0.014,
                            0.84 * (y1 - y0)])
    else:
        cax = fig.add_axes([x0 + 0.05 * (x1 - x0), y0 - 0.035,
                            0.9 * (x1 - x0), 0.014])
    cb = ngs_colorbar(image, cax=cax, orientation=orientation)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=6)
    return cb


# -- reconstruction ---------------------------------------------------------
def reconstruction():
    p = build("burgers_2d")
    mesh = p.mesh
    qhat = GridFunction(p.Qfes)
    shown_q = GridFunction(L2(p.q_domains.mesh, order=0))
    shown_q.Set(p.q_true)
    view = view_of(render_data(qhat))
    picks = [best(level) for level in LEVELS]
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 4.6))
    fig.subplots_adjust(left=0.01, right=0.9, wspace=0.06, hspace=0.18)
    image_q = paint(axes[0, 0], shown_q, None, view, autoscale=False,
                    vmin=Q_RANGE[0], vmax=Q_RANGE[1])
    axes[0, 0].set_title(r"$q^\dagger$")
    # the clean flow at the end of the clean run, as speed |z|
    h5 = checkpoint(0.0)
    z = GridFunction(p.Ufes)
    z.vec.FV().NumPy()[:] = h5["trajectory/z"][-1]
    t_z = float(h5["trajectory/t"][-1])
    image_z = paint(axes[1, 0], Norm(z), mesh, view)
    axes[1, 0].set_title(rf"$|z|$ at $t={t_z:g}$")
    bar(fig, axes[1, 0], image_z, r"$|z|$", orientation="horizontal")
    errors = []
    for col, (level, (t, e, q, t_min, e_min)) in enumerate(zip(LEVELS, picks), 1):
        qhat.vec.FV().NumPy()[:] = q
        paint(axes[0, col], qhat, None, view, autoscale=False,
              vmin=Q_RANGE[0], vmax=Q_RANGE[1])
        axes[0, col].set_title(rf"$\delta={100 * level:g}\%$, $t={t:g}$")
        image_e = paint(axes[1, col], qhat - CoefficientFunction(p.q_true),
                        mesh, view, autoscale=False, vmin=-0.5, vmax=0.5)
        axes[1, col].set_title(rf"error {100 * e:.1f}$\%$")
        errors.append((level, t, e, t_min, e_min))
    bar(fig, axes[0, :], image_q, r"$q$")
    bar(fig, axes[1, 1:], image_e, r"$q-q^\dagger$")
    fig.savefig(FIGURES / "burgers2d_reconstruction.pdf", dpi=400,
                bbox_inches="tight")
    plt.close(fig)

    # the error histories
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    for level, colour in zip(LEVELS, ("C0", "C1", "C3")):
        h5 = checkpoint(level)
        t, err = h5["history/t"][:], h5["history/error_q"][:]
        ax.semilogy(t, err, color=colour, lw=1.0,
                    label=rf"$\delta={100 * level:g}\%$")
        k = int(np.argmin(err))
        ax.plot(t[k], err[k], "o", color=colour, ms=3.5)
    floor = p.q_floor()
    ax.axhline(floor, color="0.4", ls=":", lw=0.9, label=r"$e_h$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|q(t)-q^\dagger\|_H/\|q^\dagger\|_H$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False)
    fig.savefig(FIGURES / "burgers2d_history.pdf", bbox_inches="tight")
    plt.close(fig)
    (FIGURES / "burgers2d_reconstruction.json").write_text(json.dumps(
        {"floor": floor, "picks": [dict(zip(("level", "t", "error", "t_min",
                                             "error_min"), row))
                                   for row in errors]}, indent=1))
    print("reconstruction:", errors, "floor", floor, flush=True)


# -- dose -------------------------------------------------------------------


def dose(t_end=100.0):
    p = build("burgers_2d")
    h5 = checkpoint(0.01)  # the clean datum, to t = 100, every fourth step
    ts = h5["trajectory/t"][:]
    keep = ts <= t_end + 1e-9
    frames = zip(ts[keep][1:], h5["trajectory/u_true"][1:][keep[1:]])
    D, areas = cell_dose(p, frames)
    np.savez(FIGURES / "burgers2d_dose.npz", D=D, areas=areas, t_end=t_end)
    print("dose: min %.3g, median %.3g, max %.3g" % (D.min(), np.median(D), D.max()),
          flush=True)
    return p, D, areas


# -- gramian ---------------------------------------------------------------
PE_DIR = ROOT / "results" / "pe" / "burgers_2d_h0.1"


def gramian_at(t):
    """The H-whitened Gramian mean (1/t) int_0^t B^T K^-1 B ds (scripts/pe_maps.py)."""
    return np.load(PE_DIR / f"G_{t:07.2f}.npy") / t




def coarse_dose(p, t_end=100.0, every=1.0):
    """D_K on the coarse mesh, from the same samples as the Gramian."""
    from mras.forward import ForwardRun
    stride = round(every / p.dt)
    frames = []
    with TaskManager():
        run = ForwardRun(p)
        for step in range(1, round(t_end / p.dt) + 1):
            z = run.step()
            if step % stride == 0:
                frames.append((step * p.dt, z.vec.FV().NumPy().copy()))
    return cell_dose(p, frames)


def gramian(t_end=100.0):
    meta = json.loads((PE_DIR / "meta.json").read_text())
    pc = build("burgers_2d", h=meta["h"], tend=t_end)
    whiten = np.load(PE_DIR / "whiten.npy")
    G = gramian_at(t_end)
    c, lam, vec = conditional_excitation(G)
    p1 = weakest_direction(vec, whiten)
    Dc, areas_c = coarse_dose(pc, t_end)
    chain = {"lambda_1": float(lam[0]), "lambda_max": float(lam[-1]),
             "negative": int((lam < 0).sum()),
             "below_rho": int((lam < RHO * lam[-1]).sum()),
             "lambda1_le_c": bool(np.all(lam[0] <= c * (1 + 1e-6))),
             "c_le_diag": bool(np.all(c <= np.diag(G) * (1 + 1e-6))),
             "diag_le_D": bool(np.all(np.diag(G) <= Dc * (1 + 1e-6))),
             "worst_diag_over_D": float(np.max(np.diag(G) / Dc)),
             "c_quantiles": np.quantile(c, [0.01, 0.05, 0.25, 0.5]).tolist(),
             "diag_quantiles": np.quantile(np.diag(G), [0.01, 0.05, 0.25, 0.5]).tolist(),
             "D_quantiles": np.quantile(Dc, [0.01, 0.05, 0.25, 0.5]).tolist()}
    print(json.dumps(chain, indent=1), flush=True)
    # over time: the excitation each run had by its best-found time
    times = [t for t in meta["saved"] if t <= t_end]
    windows = {}
    for t in times:
        Gt = gramian_at(t)
        windows[t] = float(np.linalg.eigvalsh(0.5 * (Gt + Gt.T))[0])
    sliding = {}
    length = 20.0
    for t in times:
        if t - length in meta["saved"]:
            Gw = (np.load(PE_DIR / f"G_{t:07.2f}.npy")
                  - np.load(PE_DIR / f"G_{t - length:07.2f}.npy")) / length
            sliding[t] = float(np.linalg.eigvalsh(0.5 * (Gw + Gw.T))[0])
    np.savez(FIGURES / "burgers2d_gramian.npz", c=c, lam=lam, p1=p1, Dc=Dc,
             diag=np.diag(G), areas=areas_c, t_end=t_end)
    (FIGURES / "burgers2d_gramian.json").write_text(json.dumps(
        {"chain": chain, "cumulative_lambda_min": windows,
         "sliding_lambda_min": sliding, "window": length, "h": meta["h"],
         "q_dofs": meta["q_dofs"], "every": meta["every"]}, indent=1))
    return pc, c, lam, p1, Dc


# -- excitation pictures ---------------------------------------------------
DOSE_LEVELS = (1e-2, 1e-1)   # sublevel sets of the dose: these times its median
COND_LEVELS = (1e-2, 1e-1)   # ... of the conditional excitation: times its median


def p0_field(p, values):
    gf = GridFunction(p.Qfes)
    gf.vec.FV().NumPy()[:] = values
    return gf


def excitation(t_end=100.0):
    p = build("burgers_2d")
    dose_data = np.load(FIGURES / "burgers2d_dose.npz")
    D = dose_data["D"]
    meta = json.loads((PE_DIR / "meta.json").read_text())
    pc = build("burgers_2d", h=meta["h"], tend=t_end)
    g = np.load(FIGURES / "burgers2d_gramian.npz")
    c, p1 = g["c"], g["p1"]
    view = view_of(render_data(GridFunction(p.Qfes)))
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.75))
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.28, bottom=0.2)
    styles = (dict(color="white", lw=0.8), dict(color="black", lw=0.8))

    logD = np.log10(np.maximum(D, 1e-300))
    lo, hi = np.floor(np.quantile(logD, 0.005)), np.ceil(np.quantile(logD, 0.995))
    image = paint(axes[0], p0_field(p, logD), None, view, autoscale=False,
                  vmin=lo, vmax=hi)
    for eps, style in zip(DOSE_LEVELS, styles):
        outline(axes[0], view, p.mesh, sublevel(D, eps * np.median(D)), **style)
    axes[0].set_title(r"(a) $\log_{10}\bar D_T$, $h=0.05$")
    bar(fig, axes[0], image, "", orientation="horizontal")

    # the weakest direction is a few cells' spike: its logarithm shows its
    # reach
    image = paint(axes[1], p0_field(pc, np.log10(np.maximum(np.abs(p1), 1e-6))),
                  None, view, autoscale=False, vmin=-4.0, vmax=0.0)
    axes[1].set_title(r"(b) $\log_{10}|p_1|$, $h=0.1$")
    bar(fig, axes[1], image, "", orientation="horizontal")

    logc = np.log10(np.maximum(c, 1e-300))
    lo, hi = np.floor(np.quantile(logc, 0.005)), np.ceil(np.quantile(logc, 0.995))
    image = paint(axes[2], p0_field(pc, logc), None, view, autoscale=False,
                  vmin=lo, vmax=hi)
    for eps, style in zip(COND_LEVELS, styles):
        outline(axes[2], view, pc.mesh, sublevel(c, eps * np.median(c)), **style)
    axes[2].set_title(r"(c) $\log_{10}c_K$, $h=0.1$")
    bar(fig, axes[2], image, "", orientation="horizontal")

    t_best, e_best, q_best = best(0.0)[:3]
    err = np.abs(q_best - p.project_q(p.q_true).vec.FV().NumPy())
    image = paint(axes[3], p0_field(p, err), None, view, autoscale=False,
                  vmin=0.0, vmax=0.25)
    outline(axes[3], view, p.mesh, D <= DOSE_LEVELS[1] * np.median(D),
            color="white", lw=0.8)
    outline(axes[3], view, pc.mesh, c <= COND_LEVELS[1] * np.median(c),
            color="black", lw=0.8, ls="--")
    axes[3].set_title(r"(d) $|q(t^*)-P_hq^\dagger|$, $\delta=0$")
    bar(fig, axes[3], image, "", orientation="horizontal")
    fig.savefig(FIGURES / "burgers2d_excitation.pdf", dpi=400,
                bbox_inches="tight")
    plt.close(fig)

    # the excitation up to the best iterates' times, and over time
    info = json.loads((FIGURES / "burgers2d_gramian.json").read_text())
    times = (10.0, 20.0, 50.0)  # saved every 5: near the best iterates
    fig = plt.figure(figsize=(6.3, 2.75))
    axes = [fig.add_axes([0.0 + 0.2 * k, 0.2, 0.19, 0.68]) for k in range(3)]
    images = []
    cs = [conditional_excitation(gramian_at(t))[0] for t in times]
    logs = [np.log10(np.maximum(ck, 1e-300)) for ck in cs]
    lo = np.floor(min(np.quantile(lk, 0.005) for lk in logs))
    hi = np.ceil(max(np.quantile(lk, 0.995) for lk in logs))
    for ax, t, ck, lk in zip(axes, times, cs, logs):
        images.append(paint(ax, p0_field(pc, lk), None, view, autoscale=False,
                            vmin=lo, vmax=hi))
        outline(ax, view, pc.mesh, ck <= COND_LEVELS[1] * np.median(c),
                color="black", lw=0.7)
        ax.set_title(rf"$\log_{{10}}c_K$ up to $t={t:g}$")
    bar(fig, axes, images[-1], "", orientation="horizontal")
    ax = fig.add_axes([0.66, 0.24, 0.32, 0.62])
    cum = {float(k): v for k, v in info["cumulative_lambda_min"].items()}
    sld = {float(k): v for k, v in info["sliding_lambda_min"].items()}
    ax.semilogy(list(cum), list(cum.values()), "C0-", lw=1,
                label=r"$\lambda_{\min}\Gamma_h(0,t)$")
    ax.semilogy(list(sld), list(sld.values()), "C3--", lw=1,
                label=rf"$\lambda_{{\min}}\Gamma_h(t-{info['window']:g},{info['window']:g})$")
    ax.set_xlabel(r"$t$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, loc="lower right")
    ax.tick_params(labelsize=6)
    fig.savefig(FIGURES / "burgers2d_excitation_time.pdf", dpi=400,
                bbox_inches="tight")
    plt.close(fig)

    # the spectrum
    lam = g["lam"]
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.semilogy(np.arange(1, lam.size + 1), np.maximum(lam, RHO * lam[-1]) / lam[-1],
                lw=1)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\lambda_k/\lambda_{\max}$ of $\Gamma_h(0,T)$")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(FIGURES / "burgers2d_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)
    print("excitation figures written", flush=True)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    ngsolve.SetNumThreads(8)
    jobs = sys.argv[1:] or ["reconstruction", "dose"]
    if "reconstruction" in jobs:
        reconstruction()
    if "dose" in jobs:
        dose()
    if "gramian" in jobs:
        gramian()
    if "excitation" in jobs:
        excitation()
