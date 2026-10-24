"""Rendering of MRAS results: ngsplot FEM panels, matplotlib error curves.

Fields are painted with ``ngsplot.Plot`` -- the finite element renderer that
reproduces the webgui's shader -- so a figure shows the FEM representation
itself: an L2 field of cellwise constants appears in its true bands, not as
a vertex-interpolated smear, and nothing is point-sampled outside the mesh.
"""

import numpy as np
import matplotlib.pyplot as plt
from ngsplot import Plot

__all__ = ["draw_field", "evolution_figure", "error_figure"]


def draw_field(field, ax=None, mesh=None, cmap="viridis", vmin=None,
               vmax=None, title=None, colorbar=True, **kwargs):
    """Draw one field into ``ax`` and return the axes.

    ``field`` is a GridFunction, or a CoefficientFunction plus the ``mesh``
    it lives on.  Extra keywords go to ``ngsplot.Plot`` (``width``,
    ``height``, view angles, ...).
    """
    ax = plt.gca() if ax is None else ax
    rng = ({"autoscale": False, "vmin": vmin, "vmax": vmax}
           if vmin is not None else {"autoscale": True})
    Plot(field, ax=ax, mesh=mesh, cmap=cmap, colorbar=colorbar, **rng,
         **kwargs)
    if title:
        ax.set_title(title)
    return ax


def evolution_figure(rows, col_labels=None, row_labels=None, fname=None,
                     mesh=None, cmap="viridis", figsize_per=(4.8, 3.6),
                     **kwargs):
    """One panel per field; ``rows`` is a list of lists of grid functions.

    Colours are normalised per row, matching the parameter/state panel
    style of the paper figures. Returns the matplotlib figure.
    """
    nrows, ncols = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols,
                                      figsize_per[1] * nrows),
                             squeeze=False)
    for i, row in enumerate(rows):
        values = [np.asarray(gf.vec.FV().NumPy()) for gf in row]
        vmin = min(float(v.min()) for v in values)
        vmax = max(float(v.max()) for v in values)
        for j, gf in enumerate(row):
            label = None
            if row_labels and col_labels:
                label = f"{row_labels[i]}  {col_labels[j]}"
            elif col_labels:
                label = col_labels[j]
            elif row_labels:
                label = row_labels[i]
            draw_field(gf, ax=axes[i][j], mesh=mesh, cmap=cmap,
                       vmin=vmin, vmax=vmax, title=label,
                       colorbar=(j == ncols - 1), **kwargs)
        for j in range(len(row), ncols):
            axes[i][j].set_visible(False)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    return fig


def error_figure(results, dt, fname=None, log=True, labels=None):
    """L2 error histories of ``results`` (one LevelResult per noise level).

    Left column state u, right column parameter q; the first history entry
    is the t = 0 error, so the time axis is ``dt * arange(len(error))``.
    Returns the figure.
    """
    nrows = len(results)
    fig, axes = plt.subplots(nrows, 2, figsize=(8, 2.6 * nrows),
                             squeeze=False)
    for i, res in enumerate(results):
        t = dt * np.arange(len(res.error_u))
        for j, (err, name) in enumerate([(res.error_u, "u"), (res.error_q, "q")]):
            ax = axes[i][j]
            ax.plot(t, err)
            if log:
                ax.set_yscale("log")
            ax.set_xlabel("t")
            ax.set_ylabel(rf"$\|e_{{{name}}}\|$")
            tag = labels[i] if labels else f"noise_level={res.noise_level:g}"
            ax.set_title(f"{res.problem} {tag}")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    return fig
