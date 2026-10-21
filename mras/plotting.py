"""Matplotlib rendering of MRAS results: field panels and error curves.

Fields are evaluated on a pixel grid and drawn with ``imshow``, which is
resolution-independent and needs no ngsolve plotting windows.
"""

import numpy as np
import matplotlib.pyplot as plt
from netgen.libngpy._meshing import NgException

__all__ = ["sample_field", "draw_field", "evolution_figure", "error_figure"]


def sample_field(gf, box, n=128, mesh=None):
    """Evaluate a field on an n x n grid over ``box`` -> (x, y, Z).

    Accepts a GridFunction or any CoefficientFunction built from one.
    Grid points outside the mesh (e.g. the corners around a disk) are NaN
    where point evaluation fails, which imshow renders transparent; note
    that a CoefficientFunction expression evaluates analytically outside
    the mesh instead of failing, so pass it only over a box-shaped mesh.
    A pure CoefficientFunction (no grid function at all) exposes no mesh
    attribute, so pass ``mesh`` for it.
    """
    lo, hi = box
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    points_on = getattr(gf, "mesh", mesh)

    def at(x, y):
        try:
            # a plain grid function returns a float here; an expression
            # with a CoefficientFunction (e.g. an analytic q_true) returns
            # a point-evaluation functional instead, which the MeshPoint
            # overload below evaluates
            return float(gf(float(x), float(y)))
        except NgException:
            return np.nan
        except TypeError:
            if points_on is None:
                raise TypeError(
                    "sample_field: this field needs a mesh for point "
                    "evaluation, pass mesh=...") from None
            return float(gf(points_on(float(x), float(y))))

    Z = np.array([[at(x, y) for x in xs] for y in ys])
    return xs, ys, Z


def draw_field(ax, gf, box, n=128, cmap="viridis", vmin=None, vmax=None,
               title=None, colorbar=True, mesh=None):
    """Draw one field into ``ax`` and return the mappable."""
    xs, ys, Z = sample_field(gf, box, n, mesh=mesh)
    im = ax.imshow(Z, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]],
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    if title:
        ax.set_title(title)
    ax.set_xticks(())
    ax.set_yticks(())
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def evolution_figure(rows, box, col_labels=None, row_labels=None,
                     fname=None, n=128, cmap="viridis",
                     figsize_per=(3.2, 2.6), mesh=None):
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
        vals = [np.asarray(sample_field(gf, box, 32, mesh=mesh)[2])
                for gf in row]
        vmin = min(np.nanmin(v) for v in vals)
        vmax = max(np.nanmax(v) for v in vals)
        for j, gf in enumerate(row):
            label = None
            if row_labels and col_labels:
                label = f"{row_labels[i]}  {col_labels[j]}"
            elif col_labels:
                label = col_labels[j]
            elif row_labels:
                label = row_labels[i]
            draw_field(axes[i][j], gf, box, n=n, cmap=cmap,
                       vmin=vmin, vmax=vmax, title=label,
                       colorbar=(j == ncols - 1), mesh=mesh)
        for j in range(len(row), ncols):
            axes[i][j].axis("off")
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
            tag = labels[i] if labels else f"level {res.noise_level:g}"
            ax.set_title(f"{res.problem} {tag}")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    return fig
