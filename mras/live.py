"""The example runner and the notebook view: one entry point for everything.

Run any example from the command line -- the example files themselves
carry no argparse code::

    .venv/bin/python -m mras.live --help          # the catalog
    .venv/bin/python -m mras.live burgers_pipe --quick
    .venv/bin/python -m mras.live darcy --config configs/darcy.yaml
    .venv/bin/python -m mras.live fisher --tend 1.0 --noise-levels 0.0 0.05

Each ``examples/*.py`` module registers its problems with markers --
``PROBLEM`` (one class) or ``PROBLEMS`` (a name-to-class mapping) -- plus
optional ``DEFAULTS`` (notebook-demo sizes), ``QUICK`` (the ``--quick``
sizes) and ``CHECK`` (the audit settings mras.verify consumes), all keyed
by the problem's ``name``.  Values layer in this order, later winning:
constructor signature, module ``DEFAULTS``, ``QUICK`` (under ``--quick``),
YAML ``defaults``, YAML run entry, command-line flags.

In a notebook, :func:`run_live` assimilates one noise level while a six-panel
ngsplot figure streams the solve-in-progress, and always ends with an
animated HTML movie of the same figure:

    from mras.live import run_live
    result = run_live("burgers_pipe", noise_level=0.05, live=True)

Panels follow the paper layout -- datum z, data error u - z, ground-truth
state, estimate qhat, reconstruction error qhat - q, true parameter q --
painted by ``ngsplot.Plot``, so what you see is the finite element
representation itself, cellwise for the L2 parameter.
"""

import argparse
import importlib.util
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from ngsolve import GridFunction
from ngsplot import Plot, pyplot

from .config import RunConfig, load_yaml, build as problem_from_config
from .plotting import error_figure, evolution_figure
from .scheme import Problem, State

ROOT = Path(__file__).resolve().parent.parent
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
    default ``results/<name>``) and prints the final errors per noise level.
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
        out = Path(cfg.out) if cfg.out else ROOT / "results" / prob.name
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
    ap.add_argument("--seed", type=int, help="noise seed")
    ap.add_argument("--noise-levels", dest="noise_levels", type=float,
                    nargs="+", help="relative noise levels as fractions "
                                    "(0.05 is a noise level of 5%%)")
    ap.add_argument("--out", help="output directory (default results/NAME)")
    args = vars(ap.parse_args(argv))
    name = args.pop("name")
    if name is None:
        ap.error("give an example name; choose from "
                 + ", ".join(sorted(list_examples())))
    config = args.pop("config")
    quick = args.pop("quick")
    return run_example(name, config=config, quick=quick, **args)


# -- the live figure ---------------------------------------------------------

def _minmax(gf):
    values = gf.vec.FV().NumPy()
    return float(values.min()), float(values.max())


class LivePanels:
    """The 2x3 ngsplot figure, the step callback, and the recorded frames.

    Animated panels (datum z, data error u - z, estimate qhat, and the
    reconstruction error when the true parameter is known) are repainted by
    :meth:`render`; the truth panels are painted once, except the final
    state when the problem does not expose it up front and it is tracked
    from the data layer instead.  Color ranges are fixed from the truth
    fields, so colors stay comparable across frames.

    Every redrawn frame is recorded; :func:`make_movie` stitches them into
    the animated HTML.  ``live=True`` also streams the figure as a notebook
    widget (:meth:`show`).
    """

    def __init__(self, prob, every=5, live=False, panel_size=(480, 360)):
        if every < 1:
            raise ValueError("every must be a positive step count")
        self.prob = prob
        self.every = int(every)
        self.live = bool(live)
        self._n = 0
        self._image = None
        self._w, self._h = panel_size
        Ufes, Qfes = prob.Ufes, prob.Qfes

        self.z = GridFunction(Ufes)
        self.err_u = GridFunction(Ufes)
        self.u_final = GridFunction(Ufes)
        self.qhat = GridFunction(Qfes)
        self.err_q = GridFunction(Qfes)
        self.q_true = None
        if prob.q_true is not None:
            self.q_true = GridFunction(Qfes)
            self.q_true.Set(prob.q_true)

        # Final-time ground truth: precomputable when the data layer exposes
        # it (forward snapshots or a truth hook); otherwise the problem's own
        # data() supplies u_true step by step and the panel tracks it, so the
        # last frame is the final-time truth.
        self._final_tracked = not self._resolve_final_u()

        self.frames = []
        self._build_figure()

    def _resolve_final_u(self):
        """Fill u_final from the data layer; True when resolved upfront."""
        snaps = self.prob._forward_data()  # generates when forward_forms given
        if snaps is not None:
            self.u_final.vec.data = np.asarray(snaps[-1], dtype=np.float64)
            return True
        if type(self.prob).truth is not Problem.truth:
            state = State(self.prob, 0.0)
            state.t.Set(self.prob.tend)
            state.counter = self.prob.counter_start + self.prob.tsteps - 1
            value = self.prob.truth(state)
            if value is not None:
                self.prob._set_reference(state, value)
                self.u_final.vec.data = state.u_true.vec.Copy()
                return True
        return False

    def _range_kwargs(self, lo, hi):
        return {"autoscale": False, "vmin": lo, "vmax": hi} if lo is not None \
            else {"autoscale": True}

    def _build_figure(self):
        if self._final_tracked:
            state_kw = err_kw = {"autoscale": True}
        else:
            lo, hi = _minmax(self.u_final)
            state_kw = self._range_kwargs(lo, hi)
            sym = 0.25 * (hi - lo) + 1e-12
            err_kw = self._range_kwargs(-sym, sym)
        if self.q_true is None:
            q_kw = q_err_kw = {"autoscale": True}
        else:
            lo, hi = _minmax(self.q_true)
            q_kw = self._range_kwargs(lo, hi)
            sym = 0.25 * (hi - lo) + 1e-12
            q_err_kw = self._range_kwargs(-sym, sym)

        # (gf, title, range kwargs, animated?) in panel order
        spec = [(self.z, "datum z", state_kw, True),
                (self.err_u, "data error u - z", err_kw, True),
                (self.u_final, "ground truth u at final time", state_kw,
                 self._final_tracked),
                (self.qhat, "estimate qhat", q_kw, True)]
        if self.q_true is not None:
            spec += [
                (self.err_q, "reconstruction error qhat - q", q_err_kw, True),
                (self.q_true, "true parameter q", q_kw, False)]

        self.figure = Figure(figsize=(4.8 * 3 + 1.2, 3.6 * 2 + 1.4),
                             dpi=100)
        FigureCanvasAgg(self.figure)
        self.figure.subplots_adjust(left=0.02, right=0.98, top=0.91,
                                    bottom=0.03, hspace=0.45, wspace=0.55)
        axes = self.figure.subplots(2, 3)
        self._animated, self._static = [], []
        for ax, (gf, title, kw, animated) in zip(axes.ravel(), spec):
            ax.set_title(title, fontsize=9)
            bucket = self._animated if animated else self._static
            bucket.append((gf, ax, kw))
        for ax in axes.ravel()[len(spec):]:
            ax.set_visible(False)
        self.render()
        for gf, ax, kw in self._static:
            Plot(gf, ax=ax, colorbar=True, width=self._w, height=self._h,
                 **kw)

    def render(self):
        """Repaint the animated panels from the current grid functions."""
        for gf, ax, kw in self._animated:
            Plot(gf, ax=ax, colorbar=True, width=self._w, height=self._h,
                 **kw)

    def _png(self):
        buffer = BytesIO()
        self.figure.savefig(buffer, format="png")
        return buffer.getvalue()

    def show(self):
        """Display the panel figure inline, streaming updates while solving."""
        try:
            from IPython.display import display
            from ipywidgets import Image
        except ImportError:
            return
        self.render()
        self._image = Image(value=self._png(), format="png")
        display(self._image)

    def update(self, state, force=False):
        """Refresh all panels from the current scheme state (throttled)."""
        self._n += 1
        if not force and self._n % self.every:
            return
        self.z.vec.data = state.z.vec.Copy()
        self.err_u.vec.data = state.gfu.vec.Copy()
        self.err_u.vec.data -= state.z.vec.Copy()
        self.qhat.vec.data = state.gfq.vec.Copy()
        if self.q_true is not None:
            self.err_q.vec.data = state.gfq.vec.Copy()
            self.err_q.vec.data -= self.q_true.vec.Copy()
        if self._final_tracked and state.has_truth:
            self.u_final.vec.data = state.u_true.vec.Copy()
        snap = lambda gf: np.asarray(gf.vec.data).copy()
        self.frames.append(tuple(snap(gf) for gf, _, _ in self._animated))
        if self.live:
            self.render()
            if self._image is not None:
                self._image.value = self._png()


def make_movie(panels, path, fps=24.0, max_frames=120):
    """Stitch the recorded frames into an animated HTML and return the path.

    The frames replay the panel figure through ngsplot -- the same FEM
    rendering as the live view -- with the fixed color ranges set at
    construction, so every frame covers the same amount of model time and
    colors are comparable across it.  Frames are thinned to ``max_frames``
    and the last one is always kept.
    """
    if not panels.frames:
        raise ValueError("make_movie needs recorded frames; run_live records "
                         "them automatically")
    from matplotlib.animation import FuncAnimation

    n = len(panels.frames)
    stride = -(-n // max_frames)  # ceil
    keep = list(range(0, n, stride))
    if keep[-1] != n - 1:
        keep.append(n - 1)
    gfs = [gf for gf, _, _ in panels._animated]

    def draw(i):
        for gf, arr in zip(gfs, panels.frames[keep[i]]):
            gf.vec.data = np.asarray(arr, dtype=np.float64)
        panels.render()

    anim = FuncAnimation(panels.figure, draw, frames=range(len(keep)),
                         init_func=lambda: None, cache_frame_data=True)
    html = anim.to_jshtml(fps=fps, default_mode="loop")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    pyplot.close(panels.figure)
    return path


def _display_movie(path):
    """Show the animated movie inline when a notebook front-end is present."""
    print(f"movie: {path}")
    try:
        from IPython.display import HTML, display
    except ImportError:
        return
    display(HTML(Path(path).read_text()))


def run_live(example, noise_level=0.0, every=5, live=False, fps=24.0,
             path=None, max_frames=120, **kwargs):
    """Assimilate one noise level with the six-panel ngsplot view.

    ``example`` is a name from :func:`list_examples` (or an already-built
    ``Problem``); ``noise_level`` is the relative L2 noise of the data
    (``0.05`` = a noise level of 5 %).  The frames are always recorded and
    stitched into an animated HTML movie (``results/<name>_noise<p>.html``
    unless ``path`` says otherwise), displayed inline; ``live=True`` adds
    the streaming figure while the solve runs.  Returns the
    :class:`~mras.results.LevelResult`.
    """
    prob = example if isinstance(example, Problem) else build(example, **kwargs)
    panels = LivePanels(prob, every=every, live=live)
    if live:
        panels.show()
    prob.on_step = panels.update
    try:
        result = prob.run_level(noise_level, progress=False)
    finally:
        prob.on_step = None
    panels.update(result.state, force=True)
    if path is None:
        path = ROOT / "results" / f"{prob.name}_noise{noise_level:g}.html"
    movie = make_movie(panels, path, fps=fps, max_frames=max_frames)
    _display_movie(movie)
    return result


if __name__ == "__main__":
    main()
