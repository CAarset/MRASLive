"""The example runner and the notebook view: one entry point for everything.

Run any example from the command line -- the example files themselves
carry no argparse code::

    .venv/bin/python -m mras.live --help          # the catalog
    .venv/bin/python -m mras.live burgers_1d --quick
    .venv/bin/python -m mras.live darcy --config configs/darcy.yaml
    .venv/bin/python -m mras.live fisher --tend 1.0 --noise-levels 0.0 0.05

Each ``examples/*.py`` module registers its problems with markers --
``PROBLEM`` (one class) or ``PROBLEMS`` (a name-to-class mapping) -- plus
optional ``DEFAULTS`` (notebook-demo sizes), ``QUICK`` (the ``--quick``
sizes) and ``CHECK`` (the audit settings mras.verify consumes), all keyed
by the problem's ``name``.  Values layer in this order, later winning:
constructor signature, module ``DEFAULTS``, ``QUICK`` (under ``--quick``),
YAML ``defaults``, YAML run entry, command-line flags.

In a notebook, :func:`run_live` assimilates one noise level while a figure
of ngsplot panels follows the solve step for step, and always ends with an
MP4 movie of the same figure and an HDF5 checkpoint of the run
(:mod:`mras.checkpoint`):

    from mras.live import run_live
    result = run_live("burgers_1d", noise_level=0.05, live=True)

``scripts/run_examples.py`` does the same for every example, or a chosen few,
from the command line.

Under the forcing g, panels read estimate | error | reference -- state u,
data error u - z, datum z over estimate qhat, reconstruction error
qhat - q, true parameter q -- above a strip of the relative q error against
time, painted by ``ngsplot.Plot``, so what you see is the finite element
representation itself, elementwise for the L2 parameter.
"""

import argparse
import importlib.util
import sys
import threading
from collections import deque
from io import BytesIO
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from ngsolve import (L2, CoefficientFunction, GridFunction, InnerProduct,
                     Integrate, Norm, Parameter)
from ngsplot import Plot, frame, render_data

from .config import RunConfig, load_yaml, build as problem_from_config
from .plotting import error_figure, evolution_figure
from .scheme import MRASProblem

ROOT = Path(__file__).resolve().parent.parent
#: an error panel spans this many root mean squares of the error either way
ERROR_SPREAD = 2.0
# The colour ranges' time constants, as fractions of the horizon: moving
# averages over RANGE_FRACTION of it, pinned ranges patient for PIN_FRACTION.
RANGE_FRACTION, PIN_FRACTION = 1 / 250, 0.05
# q and qhat share q's range, padded at each end by this fraction of its
# span, so q's own values stay clear of the colour bar's ends.
Q_PAD = 0.05
EXAMPLES_DIR = ROOT / "examples"


# -- the example catalog --------------------------------------------------

def load_example(stem):
    """Import ``examples/<stem>.py`` by path; the package never needs to own them."""
    key = f"mras_example_{stem}"
    if key in sys.modules:
        return sys.modules[key]
    path = EXAMPLES_DIR / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no example module at {path}")
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module  # before exec: re-entrant imports resolve
    spec.loader.exec_module(module)
    return module


def _modules():
    return [load_example(path.stem) for path in sorted(EXAMPLES_DIR.glob("*.py"))]


def _problems(module):
    """The problems one module registers: PROBLEM (one) or PROBLEMS (many)."""
    named = getattr(module, "PROBLEMS", None)
    if named is not None:
        return dict(named)
    problem = getattr(module, "PROBLEM", None)
    return {} if problem is None else {problem.name: problem}


def list_examples():
    """{name: problem class} for every example module that registers problems."""
    catalog = {}
    for module in _modules():
        catalog.update(_problems(module))
    return catalog


def _locate(name):
    """(module, class) of the registered problem called ``name``."""
    for module in _modules():
        cls = _problems(module).get(name)
        if cls is not None:
            return module, cls
    raise ValueError(f"unknown example {name!r}; choose from "
                     f"{', '.join(sorted(list_examples()))}")


def build(name, **kwargs):
    """Instantiate a registered example: its DEFAULTS, overridden by kwargs.

    ``name`` is a key of :func:`list_examples`; the keywords (``h``, ``dt``,
    ``tend``, ``seed``, ...) go straight to the constructor.
    """
    module, cls = _locate(name)
    defaults = getattr(module, "DEFAULTS", {}).get(name, {})
    return cls(**{**defaults, **kwargs})


def example_registry():
    """{name: (class, settings)} for the examples with audit settings (CHECK).

    This is ``mras.verify``'s view of the examples: the class plus the
    discretization the checklist runs it at, DEFAULTS topped up with CHECK.
    """
    registry = {}
    for module in _modules():
        checks = getattr(module, "CHECK", {})
        defaults = getattr(module, "DEFAULTS", {})
        for name, cls in _problems(module).items():
            if name in checks:
                registry[name] = (cls, {**defaults.get(name, {}),
                                        **checks[name]})
    return registry


# -- the command-line runner -----------------------------------------------

def run_example(name, config=None, quick=False, **cli):
    """Run one example end to end; one LevelResult per noise level.

    ``config`` is a YAML path (sweeps run in order); the keyword overrides
    are the command-line flags.  Each run saves its npz error histories,
    errors.png and final_fields.png under its ``out`` directory (by
    default ``results/cli/<name>``, beside -- not among -- the runs of
    ``scripts/run_examples.py`` and the notebook in ``results/<name>``) and
    prints the final errors per noise level.
    """
    module, cls = _locate(name)
    settings = dict(getattr(module, "DEFAULTS", {}).get(name, {}))
    if quick:
        settings.update(getattr(module, "QUICK", {}).get(name, {}))
    yaml_defaults, yaml_runs = {}, [{}]
    if config is not None:
        yaml_defaults, yaml_runs = load_yaml(config)
        if not yaml_runs:
            yaml_runs = [{}]
    overrides = {key: value for key, value in cli.items() if value is not None}

    results = []
    for entry in yaml_runs:
        cfg = RunConfig(**{**settings, **yaml_defaults, **entry, **overrides})
        prob = problem_from_config(cls, cfg)
        noise_levels = cfg.noise_levels or [0.0, 0.05]
        out = Path(cfg.out) if cfg.out else ROOT / "results" / "cli" / prob.name
        out.mkdir(parents=True, exist_ok=True)
        level_results = [prob.run_level(noise_level)
                         for noise_level in noise_levels]
        for res in level_results:
            res.save(out)
            print(f"noise_level={res.noise_level:g}: "
                  f"err_u={res.final_error_u:.4f} "
                  f"err_q={res.final_error_q:.4f}")
        error_figure(level_results, prob.dt, fname=out / "errors.png")
        evolution_figure(
            [[res.state.gfu for res in level_results],
             [res.state.gfq for res in level_results]],
            col_labels=[f"noise_level={res.noise_level:g}"
                        for res in level_results],
            row_labels=["u", "q"], fname=out / "final_fields.png")
        results.extend(level_results)
    return results


def main(argv=None):
    """The CLI entry point; ``python -m mras.live NAME [options]``."""
    ap = argparse.ArgumentParser(
        prog="python -m mras.live",
        description="Run one MRAS example; the example's own defaults apply "
                    "unless the YAML or these flags say otherwise.")
    ap.add_argument("name", nargs="?",
                    help=f"the example to run: {', '.join(sorted(list_examples()))}")
    ap.add_argument("--config", type=Path,
                    help="YAML run configuration; sweeps run every entry")
    ap.add_argument("--quick", action="store_true",
                    help="the example's QUICK check sizes instead of its defaults")
    ap.add_argument("--h", type=float, help="design mesh width")
    ap.add_argument("--dt", type=float, help="time step")
    ap.add_argument("--tend", type=float, help="final time")
    ap.add_argument("--k_u", type=int, help="polynomial order of U_h")
    ap.add_argument("--k_q", type=int,
                    help="polynomial order of Q_h, for examples that take it")
    ap.add_argument("--seed", type=int, help="noise seed")
    ap.add_argument("--noise-levels", dest="noise_levels", type=float,
                    nargs="+", help="relative noise levels as fractions "
                                    "(0.05 is a noise level of 5%%)")
    ap.add_argument("--out", help="output directory (default results/cli/NAME)")
    args = vars(ap.parse_args(argv))
    name = args.pop("name")
    if name is None:
        ap.error("give an example name; choose from "
                 + ", ".join(sorted(list_examples())))
    config = args.pop("config")
    quick = args.pop("quick")
    return run_example(name, config=config, quick=quick, **args)


# -- the live figure ---------------------------------------------------------

class Panels:
    """The figure: seven ngsplot panels over an error strip.

    Under the forcing, panels read estimate | error | reference::

                       | forcing g                     |
        state u        | data error u - z              | datum z
        estimate qhat  | reconstruction error qhat - q | true parameter q

    and the strip below plots ||qhat - q|| / ||q|| against model time (the
    data misfit ||u - z|| when q is unknown); the title reads t / tend.  The
    strip spans the curve, from its lowest value so far to its largest (1
    for an estimate starting at zero), so the plot moves down with it; the
    best fit in Q_h -- the L2 floor, below which no estimate can go -- is
    drawn dotted, and the axis reaches down to it, once the curve comes
    within a factor two of it.  The forcing row
    is left out when the problem has no source.  A vector state is painted
    as its length.

    Colour ranges: q and qhat share one range, which never moves: q's
    extremes, padded at each end by ``Q_PAD`` of its span, so q stays clear
    of the ends of the colour bar and an estimate that over- or
    undershoots shows as the saturated colour of an end -- converged, qhat
    looks like q, colour for colour.  Every other panel follows its field: its ends are moving exponential
    averages over ``range_time`` units of model time (a 250th of the
    horizon unless given) of the field's extremes -- or, for the two
    errors, of ``ERROR_SPREAD`` times their root mean square, so the bulk
    of an error spans the bands and its spikes saturate red and blue.  u
    and z share one range.  The ranges of the forcing, the state and the
    datum are symmetric about zero, as the errors' are, unless the problem
    guarantees the field positive (``positive_source``,
    ``positive_state``); q is fixed as above.  For a problem that does not
    decay (``decaying = False``, the default) the forcing, state and datum
    ranges are pinned to their largest values so far instead, and follow a
    field down -- gently, over 5 % of the horizon -- only after nothing has
    come within 90 % of them for 5 % of the horizon.

    The horizon is the run's length when it is known (``horizon``, a
    movie of a finished run) and the time so far otherwise -- a run stops
    at its target error, at a time no one knows in advance: the strip's
    time axis then reaches the time shown, and the ranges' time constants
    grow with the time so far.

    The mesh lines are left out unless ``wireframe=True``: at the meshes
    the examples run on, elements are a few pixels across and the lines
    would cover the field.

    :meth:`job` turns a state -- the solver's, or one stored in a
    checkpoint -- into what :meth:`paint` paints; the live view
    (:class:`LivePanels`) and the movie (:mod:`mras.movie`) share both.
    """

    def __init__(self, prob, panel_width=400, range_time=None,
                 wireframe=False, horizon=None):
        self._range_time = range_time
        self.horizon = horizon
        self.t_end = None if horizon is None else prob.tstart + horizon
        range_time, window = self._spans(horizon or 20 * prob.dt)
        self.prob = prob
        self.wireframe = bool(wireframe)
        Ufes, Qfes = prob.Ufes, prob.Qfes

        self.u = GridFunction(Ufes)
        self.err_u = GridFunction(Ufes)
        self.z = GridFunction(Ufes)
        self.qhat = GridFunction(Qfes)
        # q and its error are drawn exactly, not projected onto Q_h: the
        # projection over- and undershoots q near steep layers, which would
        # stretch the fixed range and paint a fitting error as an estimate.
        self.q_true = prob.q_true
        if self.q_true is not None and not isinstance(self.q_true,
                                                      CoefficientFunction):
            self.q_true = CoefficientFunction(self.q_true)
        self.floor = prob.q_floor()
        # The error panel evaluates q where the estimate lives; the q panel
        # paints it on a mesh that follows its jumps -- its own, or the
        # estimator's when that does (a floor of zero).
        self.err_q = None if self.q_true is None else self.qhat - self.q_true
        own = prob.q_domains
        if self.q_true is not None and own is not None:
            if not (isinstance(self.q_true, GridFunction)
                    and self.q_true.space.mesh is own.mesh):
                shown = GridFunction(L2(own.mesh, order=0))
                shown.Set(self.q_true)
                self.q_true = shown
        elif self.floor is not None and self.floor < 1e-9:
            # q lies in Q_h (a mesh that follows its jumps): its projection
            # is q itself, painted element by element.  Evaluated pointwise,
            # q would be ambiguous at the vertices on its jumps, and paint
            # each element there with a neighbour's value.
            self.q_true = prob.project_q(self.q_true)
        self._q_span = (None if self.q_true is None else
                        _padded(*_payload_range(self._payload(self.q_true))))
        self._area = Integrate(CoefficientFunction(1.0), prob.mesh)
        # The forcing on a time of the panels' own, rebuilt for every
        # picture: a source may be data indexed by time (Darcy's pixel
        # fields), not a function of t alone.
        self._t = Parameter(prob.tstart)
        self.g = (None if prob.source(self._t) is None
                  else _Forcing(prob, self._t, GridFunction(Ufes).dim > 1))
        # One camera for every panel and frame: the mesh never moves.
        self._view = _frame(self._payload(self.u), panel_width)
        self._w, self._h = self._view.width, self._view.height

        def moving(symmetric=False):
            return _MovingRange(range_time, prob.dt, symmetric)

        def field(symmetric):
            """The range of a field: pinned to its largest values for a
            problem that does not decay (``decaying``), moving otherwise."""
            if prob.decaying:
                return moving(symmetric)
            return _PinnedRange(window, window, prob.dt, symmetric,
                                fast=range_time)
        # Fields of either sign are painted symmetrically about zero, so
        # zero is always the middle band; a field the problem guarantees
        # positive keeps the range of its own extremes, and a vector state
        # is painted as its length, which is never negative.
        scalar = self.u.dim == 1
        self._ranges = {
            "forcing": field(scalar and not prob.positive_source),
            "state": field(scalar and not prob.positive_state),
            "state_err": moving(scalar),
            "param_err": moving(True)}
        if self.q_true is None:
            self._ranges["param"] = moving()
        self._bars = {}
        self._build_figure()

    # -- layout ---------------------------------------------------------------
    def _build_figure(self):
        # A vector state is painted as its length, which the titles say:
        # one component alone would be a different picture of the flow.
        vector = self.u.dim > 1

        def shown(gf, label):
            if not vector:
                return gf, rf"${label}$"
            return Norm(gf), rf"$|{label}|$"

        rows = []
        if self.g is not None:
            label = r"$|g|$" if vector else r"$g$"
            rows.append([None, (self.g, f"forcing {label}", "forcing"), None])
        u, label_u = shown(self.u, "u")
        err, label_err = shown(self.err_u, "u - z")
        z, label_z = shown(self.z, "z")
        rows.append([(u, f"state {label_u}", "state"),
                     (err, f"data error {label_err}", "state_err"),
                     (z, f"datum {label_z}", "state")])
        if self.q_true is None:
            rows.append([(self.qhat, r"estimate $\hat q$", "param")])
        else:
            rows.append([(self.qhat, r"estimate $\hat q$", "estimate"),
                         (self.err_q, r"reconstruction error $\hat q - q$",
                          "param_err"),
                         (self.q_true, r"true parameter $q$", "param")])
            self._q_kw = _range_kwargs(*self._q_span)

        # Laid out in pixels at dpi 100.  Each picture's axes is exactly its
        # canvas, so it lands one to one (no resampling of the mesh lines, no
        # raster wasted), and its colourbar gets a fixed slot of its own:
        # under the picture for a wide domain, where a bar beside it would be
        # as short as the domain is thin, beside it otherwise.
        w, h = self._w, self._h
        wide = w / h >= 1.8
        cell_w, cell_h = (w, h + 46) if wide else (w + 86, h)
        gap, top, title, row_gap = 34, 52, 24, 36
        strip_gap, strip_h, bottom = 46, 190, 52
        width = 3 * cell_w + 4 * gap
        height = (top + len(rows) * (title + cell_h) + (len(rows) - 1) * row_gap
                  + strip_gap + strip_h + bottom)
        self.figure = Figure(figsize=(width / 100, height / 100), dpi=100)
        FigureCanvasAgg(self.figure)

        def rect(x, y, rw, rh):  # pixels from the top left -> figure fractions
            return (x / width, 1 - (y + rh) / height, rw / width, rh / height)

        self._animated, self._static = [], []
        for row, panels in enumerate(rows):
            for col, panel in enumerate(panels):
                if panel is None:
                    continue
                gf, label, kind = panel
                x = gap + col * (cell_w + gap)
                y = top + title + row * (title + cell_h + row_gap)
                ax = self.figure.add_axes(rect(x, y, w, h))
                ax.set_title(label, fontsize=9)
                if wide:
                    bar = rect(x + 0.04 * w, y + h + 8, 0.92 * w, 10)
                else:
                    bar = rect(x + w + 6, y + 0.04 * h, 12, 0.92 * h)
                self._bars[ax] = {"cax": self.figure.add_axes(bar),
                                  "orientation": "horizontal" if wide
                                  else "vertical"}
                bucket = self._static if gf is self.q_true else self._animated
                bucket.append((gf, ax, kind))
        for gf, ax, kind in self._static:
            self._plot(ax, self._payload(gf), self._kwargs(kind))

        ax = self._strip = self.figure.add_axes(
            rect(70, height - bottom - strip_h, width - 70 - gap, strip_h))
        if self.q_true is not None:
            label = r"$\|\hat q - q\| \,/\, \|q\|$"
        else:
            label = r"data misfit $\|u - z\|$"
        (self._line,) = ax.semilogy([], [], color="C0", lw=1.4, label=label)
        # The L2 floor, below which no estimate in Q_h goes, dotted.
        self._floors = {}
        if self.floor:
            self._floors["l2"] = ax.axhline(
                self.floor, color="0.55", ls=":", lw=0.9,
                label=r"best fit in $Q_h$", visible=False)
        ax.set_xlim(self.prob.tstart, self._axis_end(self.prob.tstart))
        ax.set_xlabel("t", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", alpha=0.3)
        self._legend()
        self._title = self.figure.suptitle("", fontsize=11)

    def _spans(self, horizon):
        """The moving averages' time and the pinned ranges' window, for a
        run of length ``horizon``."""
        return (self._range_time or RANGE_FRACTION * horizon,
                PIN_FRACTION * horizon)

    def _rescale(self, horizon):
        """The ranges' time constants for a run of length ``horizon``."""
        moving, window = self._spans(horizon)
        for track in self._ranges.values():
            if isinstance(track, _MovingRange):
                track.tau = moving
            elif isinstance(track, _PinnedRange):
                track.patience = track.tau = window
                track.fast = track._low.tau = moving

    def _legend(self):
        handles = [self._line] + [line for line in self._floors.values()
                                  if line.get_visible()]
        self._strip.legend(handles=handles, loc="upper right", fontsize=8)

    def _payload(self, field):
        if isinstance(field, _Forcing):
            field = field.field()
        if isinstance(field, GridFunction):
            return render_data(field)
        return render_data(field, self.prob.mesh)

    def _plot(self, ax, payload, kw):
        """Paint one panel; its colourbar, on the panel's own slot, follows."""
        Plot(payload, ax=ax, view=self._view, wireframe=self.wireframe,
             colorbar=self._bars[ax], **kw)
        self._bars[ax]["cax"].tick_params(labelsize=7)

    def _kwargs(self, kind):
        if kind in ("param", "estimate") and self.q_true is not None:
            return self._q_kw
        return self._ranges[kind].kwargs()

    # -- a state into a picture ---------------------------------------------
    def fill(self, t, u, z, q):
        """Put a state into the panels' grid functions: the coefficient
        vectors of u, z on U_h and q on Q_h at time ``t``."""
        self.u.vec.FV().NumPy()[:] = u
        self.z.vec.FV().NumPy()[:] = z
        self.err_u.vec.FV().NumPy()[:] = u - z
        self.qhat.vec.FV().NumPy()[:] = q
        if self.g is not None:
            self._t.Set(t)
            self.g.update()

    def extremes(self):
        """The payloads of the animated panels, and the span each kind of
        panel is to be painted over (u and z share one).

        A field's span is its extremes; an error's is ``ERROR_SPREAD``
        times its root mean square instead -- an error's extremes are
        spikes (a corner, an overshoot at a hole), and a range set by them
        leaves the rest of the field one colour.
        """
        payloads = [self._payload(gf) for gf, _, _ in self._animated]
        errors = {"state_err": self.err_u, "param_err": self.err_q}
        seen = {}
        for payload, (_, _, kind) in zip(payloads, self._animated):
            if kind in errors:
                field = errors[kind]
                rms = (Integrate(InnerProduct(field, field), self.prob.mesh,
                                 order=2 * self.prob.order) / self._area) ** 0.5
                lo, hi = (-ERROR_SPREAD * rms, ERROR_SPREAD * rms)
                if self._ranges[kind].symmetric is False:  # a length
                    lo = 0.0
            else:
                lo, hi = _payload_range(payload)
            if kind in seen:
                lo, hi = min(lo, seen[kind][0]), max(hi, seen[kind][1])
            seen[kind] = (lo, hi)
        return payloads, {kind: span for kind, span in seen.items()
                          if kind in self._ranges}

    def job(self, t, history, offset, ranges=None, payloads=None):
        """Everything :meth:`paint` needs for the filled state at time t.

        ``history`` is the error history up to t (the misfit, from step 1,
        with ``offset`` 1, when q is unknown).  Without ``ranges`` the
        moving colour ranges take this state in; with them (kind -> the
        keywords of :func:`_range_kwargs`) they are used as given -- the
        movie computes them for every frame in order first.
        """
        if payloads is None:
            payloads, seen = self.extremes()
            if ranges is None:
                if self.horizon is None:  # the run's length is not known yet
                    self._rescale(max(t - self.prob.tstart,
                                      20 * self.prob.dt))
                for kind, (lo, hi) in seen.items():
                    self._ranges[kind].add(t, lo, hi)
        history = np.asarray(history)
        title = f"{self.prob.name}    t = {t:.2f}"
        if self.t_end is not None:
            title += f" / {self.t_end:g}"
        if self.q_true is not None and offset == 0 and len(history):
            title += rf"    $\|\hat q - q\| \,/\, \|q\|$ = {history[-1]:.4f}"

        def kwargs(kind):
            if ranges is not None and kind in ranges:
                return dict(ranges[kind])
            return dict(self._kwargs(kind))
        return {"t": t, "title": title,
                "panels": [(ax, payload, kwargs(kind))
                           for payload, (_, ax, kind)
                           in zip(payloads, self._animated)],
                "times": self.prob.tstart
                + (np.arange(len(history)) + offset) * self.prob.dt,
                "history": history}

    def paint(self, job):
        """Paint a job into the figure; return it as PNG bytes."""
        for ax, payload, kw in job["panels"]:
            self._plot(ax, payload, kw)
        times, history = job["times"], job["history"]
        stride = max(1, len(history) // 4000)  # plenty for a strip this wide
        times, history = times[::stride], history[::stride]
        self._line.set_data(times, history)
        self._strip.set_xlim(self.prob.tstart, self._axis_end(job["t"]))
        shown = history[history > 0]
        if shown.size:
            # The axis spans the curve, from the lowest error so far to the
            # largest; once the curve comes within a factor two of a line,
            # the line shows and the axis reaches down to it (just below,
            # so it clears the frame).
            low, high = shown.min(), shown.max()
            values = {"l2": self.floor}
            changed = False
            bottom = low
            for kind, line in self._floors.items():
                value = values[kind]
                near = value is not None and bool(low <= 2 * value)
                if near:
                    line.set_ydata([value, value])
                    bottom = min(bottom, value / 1.03)
                if near != line.get_visible():
                    line.set_visible(near)
                    changed = True
            if high <= bottom:  # a single point
                high = bottom * 1.03
            self._strip.set_ylim(bottom, high)
            if changed:
                self._legend()
        self._title.set_text(job["title"])
        return self.png()

    def _axis_end(self, t):
        """The strip's time axis: to the run's end when it is known, else
        to the time ``t`` the picture shows."""
        if self.t_end is not None:
            return self.t_end
        return max(t, self.prob.tstart + self.prob.dt)

    def png(self):
        buffer = BytesIO()
        self.figure.savefig(buffer, format="png")
        return buffer.getvalue()


class LivePanels(Panels):
    """The figure, following a run: the run's on_step hook.

    Painting never holds up the solve.  :meth:`update` copies the fields
    into the panels' grid functions and builds their payloads on the
    solver's thread (a few milliseconds), and a background thread
    rasterises them -- NGSolve releases the GIL while it assembles and
    solves, so the two run side by side.  A new picture is queued whenever
    the painter is idle, so the view shows the newest step it can.  The
    movie is not painted here: it is rendered from the run's checkpoint
    afterwards (:mod:`mras.movie`).
    """

    def __init__(self, prob, **kwargs):
        super().__init__(prob, **kwargs)
        self._jobs = deque()
        self._cond = threading.Condition()
        self._busy = self._closing = False
        self._worker = None
        self._error = None
        self._error_raised = False
        self._newest_png = None
        self._image = None
        self.painted = 0  # pictures painted so far

    def update(self, state):
        """The run's on_step hook: queue this step when the painter is idle."""
        if self._error is not None and not self._error_raised:
            self._error_raised = True
            raise RuntimeError("the live view's painter failed") from self._error
        self._show_newest()
        with self._cond:
            idle = not self._jobs and not self._busy
        if idle:
            self._submit(self._state_job(state))

    def _state_job(self, state):
        z = state.z if state.z_started else state.u_true  # no datum before step 1
        self.fill(state.t.Get(), state.gfu.vec.FV().NumPy(),
                  z.vec.FV().NumPy(), state.gfq.vec.FV().NumPy())
        if state.error_q is not None:
            history, offset = state.error_q, 0
        else:
            history, offset = state.misfit_u, 1
        return self.job(state.t.Get(), list(history), offset)

    def _submit(self, job):
        with self._cond:
            self._jobs.append(job)
            self._cond.notify()
        if self._worker is None:
            self._worker = threading.Thread(target=self._work, daemon=True,
                                            name="mras-live-painter")
            self._worker.start()

    def _show_newest(self):
        with self._cond:
            png, self._newest_png = self._newest_png, None
        if png is not None and self._image is not None:
            self._image.value = png

    # -- the painter thread ---------------------------------------------------
    def _work(self):
        while True:
            with self._cond:
                while not self._jobs and not self._closing:
                    self._cond.wait()
                if not self._jobs:
                    return
                job = self._jobs.popleft()
                self._busy = True
            try:
                png = self.paint(job)
            except BaseException as err:  # surfaced on the solver's thread
                with self._cond:
                    self._error = err
                    self._jobs.clear()
                    self._busy = False
                return
            with self._cond:
                self._newest_png = png
                self.painted += 1
                self._busy = False

    # -- notebook display and shutdown ---------------------------------------
    def show(self):
        """Display the figure inline; later pictures replace it as they land."""
        try:
            from IPython.display import display
            from ipywidgets import Image
        except ImportError:
            return
        self._image = Image(value=self.png(), format="png")
        display(self._image)

    def close(self, state=None, drain=True):
        """Paint the final state and stop the painter.

        ``drain=False`` (an interrupted run) drops the pictures still queued
        instead of painting them.
        """
        if state is not None and self._error is None:
            self._submit(self._state_job(state))
        with self._cond:
            if not drain:
                self._jobs.clear()
            self._closing = True
            self._cond.notify_all()
        if self._worker is not None:
            self._worker.join()
        self._show_newest()
        if self._error is not None and not self._error_raised:
            self._error_raised = True
            raise RuntimeError("the live view's painter failed") from self._error


class _Forcing:
    """The source at the panels' time, painted as the coefficient itself.

    Projected onto U_h, a source with sharp features -- a beam's edge, a
    shadow -- would be painted with the projection's over- and undershoots
    (``GridFunction.Set`` is a local L2 projection: a positive source showed
    -4 against a peak of 30), and the solver never uses such a projection:
    it integrates the coefficient.  Sampled where it is painted, the source
    is shown as it is.
    """

    def __init__(self, prob, t, vector):
        self.prob, self.t, self.vector = prob, t, vector
        self.cf = None

    def update(self):
        self.cf = CoefficientFunction(self.prob.source(self.t))

    def field(self):
        return Norm(self.cf) if self.vector else self.cf


def _frame(payload, width, max_height=300):
    """The camera of a panel framed on the box of the mesh, and its canvas.

    ``width`` wide and as tall as the box makes it, unless that is taller
    than ``max_height``: then ``max_height`` tall and as wide as the box
    makes it.  Nothing is painted to find out.
    """
    view = frame(payload, width=width, height=None, fit="box")
    if view.height > max_height:
        view = frame(payload, width=None, height=max_height, fit="box")
    return view


class _MovingRange:
    """A colour range that follows a field: moving averages of its extremes.

    Each sampled minimum and maximum is weighted by the model time it stands
    for and discounted by exp(-age / tau), and the sums are normalised by the
    total weight, so the range starts at the first extremes rather than
    being pulled up from zero.  A flat field (a state at rest) says nothing
    about the range and is skipped; until a sample arrives the panel
    autoscales.
    """

    def __init__(self, tau, step, symmetric=False):
        self.tau, self.step, self.symmetric = tau, step, symmetric
        self._sums = np.zeros(2)
        self._weight = 0.0
        self._t = None

    def add(self, t, lo, hi):
        if self.symmetric:
            hi = max(abs(lo), abs(hi))
            lo = -hi
        if not hi - lo > 1e-12 * max(1.0, abs(hi), abs(lo)):
            return
        span = self.step if self._t is None else max(t - self._t, self.step)
        keep = np.exp(-span / self.tau)
        self._sums = keep * self._sums + (1.0 - keep) * np.array([lo, hi])
        self._weight = keep * self._weight + (1.0 - keep)
        self._t = t

    def kwargs(self):
        if self._weight == 0.0:
            return {"autoscale": True}
        return _range_kwargs(*(self._sums / self._weight))

    def fresh(self):
        """The same range, unstarted."""
        return _MovingRange(self.tau, self.step, self.symmetric)


class _PinnedRange:
    """A colour range pinned to the largest values seen so far.

    For a field that does not decay: the upper end -- both ends, for a range
    symmetric about zero -- is the running maximum, and it starts to follow
    the field down, as a moving exponential average over ``tau``, only once
    no value has come within ``near`` of it for ``patience`` units of model
    time; a new maximum pins it again.  The lower end of a range that is not
    symmetric follows the field's minimum with a :class:`_MovingRange` over
    ``fast``.
    """

    def __init__(self, patience, tau, step, symmetric=False, near=0.9,
                 fast=None):
        self.patience, self.tau, self.step = patience, tau, step
        self.symmetric, self.near = symmetric, near
        self.fast = fast if fast is not None else tau
        self.high = None
        self._near_at = self._t = None
        self._low = _MovingRange(self.fast, step)

    def add(self, t, lo, hi):
        if self.symmetric:
            hi = max(abs(lo), abs(hi))
        if not hi > 1e-12 * max(1.0, abs(lo)) and self.high is None:
            return  # a field at rest says nothing yet
        if self.high is None or hi >= self.high:
            self.high, self._near_at = hi, t
        elif hi >= self.near * self.high:
            self._near_at = t
        elif t - self._near_at > self.patience:
            keep = np.exp(-max(t - self._t, self.step) / self.tau)
            self.high = keep * self.high + (1.0 - keep) * hi
        self._t = t
        if not self.symmetric:
            self._low.add(t, lo, max(hi, lo + 1e-300))

    def kwargs(self):
        if self.high is None:
            return {"autoscale": True}
        if self.symmetric:
            return _range_kwargs(-self.high, self.high)
        low = self._low.kwargs()
        return _range_kwargs(low.get("vmin", 0.0), self.high)

    def fresh(self):
        """The same range, unstarted."""
        return _PinnedRange(self.patience, self.tau, self.step,
                            self.symmetric, self.near, self.fast)


def _payload_range(payload):
    return float(payload["funcmin"]), float(payload["funcmax"])


def _padded(lo, hi):
    """The range of q, padded at each end by ``Q_PAD`` of its span (of its
    size, for a constant q)."""
    pad = Q_PAD * ((hi - lo) or max(abs(hi), 1.0))
    return lo - pad, hi + pad


def _range_kwargs(lo, hi):
    return {"autoscale": False, "vmin": lo, "vmax": hi}


def _evenly(n, k):
    """Up to k indices spread evenly over range(n), first and last kept."""
    return sorted({round(i) for i in np.linspace(0, n - 1, min(n, k))})


def _in_notebook():
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config


def _display_movie(path):
    """Show the movie inline in a notebook, embedded so that it survives
    re-opening the notebook without the file."""
    print(f"movie: {path}")
    if _in_notebook():
        from IPython.display import Video, display
        display(Video(str(path), embed=True, html_attributes="controls loop"))


def run_live(example, noise_level=0.0, live=False, dt=None, tend=None,
             target=None, q0=None, fps=None, path=None, checkpoint=True,
             budget=256.0, movie=True, every=None, workers=None, **kwargs):
    """Assimilate one noise level; write its checkpoint and movie.

    ``example`` is a name from :func:`list_examples` (or an already-built
    ``MRASProblem``); ``noise_level`` is the relative L2 noise of the data
    (``0.05`` = a noise level of 5 %); ``dt`` and ``tend`` are the step
    size and the time limit of the assimilation, which stops earlier once
    the reconstruction error reaches ``target`` (max(1 %, 1.25 times the
    floor) unless given; 0 runs to ``tend``); ``q0`` is the initial
    estimate qhat_0 (a number, coefficient or grid function), the
    example's own unless given (0 for most).  Data are generated in
    lockstep with the estimator, so with ``live=True`` the view opens at
    t = 0 and follows the solve from its first step, repainting whenever
    the painter is free.

    The run is written to an HDF5 checkpoint (:mod:`mras.checkpoint`:
    settings, histories, the trajectory within ``budget`` megabytes --
    every step with ``budget=None`` -- and a restart point), and the movie
    is rendered from it afterwards (:mod:`mras.movie`): a frame for every
    stored step, up to 1200, lasting 4 to 10 seconds (``every`` and ``fps``
    override the pace), painted by ``workers`` processes, and shown inline
    in a notebook.  ``path`` is the stem of both files,
    ``results/<name>/<name>_noise<p>-notebook`` unless given -- beside the
    run of ``scripts/run_examples.py`` in the same folder; with
    ``checkpoint=False`` the checkpoint is a temporary file, deleted once
    the movie is made.  Returns the :class:`~mras.results.LevelResult`.
    """
    import tempfile
    from .checkpoint import Checkpoint
    from .movie import render_movie
    if dt is not None:
        kwargs["dt"] = dt
    if tend is not None:
        kwargs["tend"] = tend
    prob = (example if isinstance(example, MRASProblem)
            else build(example, **kwargs))
    if q0 is not None:
        prob.q0 = q0
    if target is not None:
        prob.target = target
    stem = Path(path) if path is not None else (
        ROOT / "results" / prob.name
        / f"{prob.name}_noise{noise_level:g}-notebook"
        )
    scratch = None
    if checkpoint:
        file = Path(f"{stem}.h5")
    else:
        scratch = tempfile.TemporaryDirectory()
        file = Path(scratch.name) / "run.h5"

    writer = Checkpoint(prob, file, noise_level, budget)
    panels = LivePanels(prob) if live else None
    hooks = [writer.update] + ([panels.update] if panels is not None else [])
    if panels is not None:
        panels.show()
    prob.on_step = lambda state: [hook(state) for hook in hooks]

    try:
        result = prob.run_level(noise_level, progress=panels is None)
    except BaseException:
        if panels is not None:
            panels.close(drain=False)
        raise
    finally:
        prob.on_step = None

    writer.close(result.state)
    if panels is not None:
        panels.close(result.state)

    if checkpoint:
        print(f"checkpoint: {file}")
    if movie:
        movie_file = Path(f"{stem}.mp4")
        _display_movie(render_movie(file, movie_file, fps=fps,
                                    every=every, workers=workers,
                                    problem=prob))
    if scratch is not None:
        scratch.cleanup()

    return result


if __name__ == "__main__":
    main()
