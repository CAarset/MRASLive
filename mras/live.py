"""Live visualization for one MRAS run, in two display modes.

``run_live`` runs one noise level and stitches the frames into an mp4 with
a fixed frame rate, displayed inline in the notebook (the default).  The
movie re-renders the six panels with matplotlib and *static* colorbars:
each range (state, data error, parameter, reconstruction error) is set by
the extremes over all frames, so colors are comparable across time::

    datum z        |  data error u - z  |  ground truth u at final time
    estimate qhat  |  error qhat - q    |  true parameter q

``live=True`` keeps the interactive mode: six side-by-side
``ngsolve.webgui`` canvases in two rows, redrawn while the scheme runs
(``every``-th step).  The notebook-facing API is :func:`run_live`: pass an
example name (see :data:`EXAMPLES`) and a noise level.  All wiring — the
``on_step`` callback, the panel grid functions, the fixed color ranges,
the final-truth resolution, the ffmpeg stitch — lives here.
"""

import importlib.util
from pathlib import Path

import numpy as np
from ngsolve import GridFunction
from ngsolve.webgui import Draw

from .problems import (AllenCahnProblem, ArrheniusLinearProblem,
                       ArrheniusProblem, DarcyProblem, FisherProblem,
                       PotentialProblem)
from .scheme import Problem, State

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _burgers_pipe():
    """Load the Burgers template from examples/ (kept out of the package)."""
    path = _EXAMPLES_DIR / "burgers_pipe.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"the Burgers demo lives in the repository at {path}; run "
            "live notebooks from a repository checkout")
    spec = importlib.util.spec_from_file_location("burgers_pipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BurgersPipe


def _burgers(**kwargs):
    return _burgers_pipe()(**kwargs)


# Each entry: factory plus the demo discretization that keeps a full
# assimilation live-visualizable at notebook speed.
EXAMPLES = {
    "burgers": (_burgers, dict(h=0.35, dt=0.02, tend=100.0)),
    "darcy": (DarcyProblem, dict(h=0.08, dt=0.02, tend=2.0,
                                 data_h=0.06, data_order=2)),
    "fisher": (FisherProblem, dict(h=0.1, dt=0.02, tend=3.0)),
    "potential": (PotentialProblem, dict(h=0.12, dt=0.02, tend=3.0)),
    "allen_cahn": (AllenCahnProblem, dict(h=0.12, dt=0.02, tend=3.0)),
    "arrhenius": (ArrheniusProblem,
                  dict(h=0.3, dt=0.02, tend=10.0, data_h=0.15)),
    "arrhenius_linear": (ArrheniusLinearProblem,
                         dict(h=0.3, dt=0.02, tend=10.0, data_h=0.15)),
}


def build(name, **kwargs):
    """Instantiate a bundled example at notebook-demo size.

    ``name`` is a key of :data:`EXAMPLES`; keyword arguments override the
    demo discretization (``h``, ``dt``, ``tend``, ``seed``, ...).
    """
    if name not in EXAMPLES:
        raise ValueError(f"unknown example {name!r}; choose from "
                         f"{', '.join(EXAMPLES)}")
    factory, demo = EXAMPLES[name]
    return factory(**{**demo, **kwargs})


def _panel(gf, title, lo, hi):
    """A drawn scene with a caption; color range fixed unless lo/hi are None."""
    colormap = ({"autoscale": True} if lo is None else
                {"autoscale": False, "min": float(lo), "max": float(hi)})
    scene = Draw(gf, show=False, height="13rem",
                 settings={"Colormap": colormap})
    scene.caption = title
    return scene


def _minmax(gf):
    values = gf.vec.FV().NumPy()
    return float(values.min()), float(values.max())


def _symmetric(gf):
    lo, hi = _minmax(gf)
    return 1e-12 + max(abs(lo), abs(hi))


class LivePanels:
    """The six panel grid functions, their scenes, and the step callback.

    ``scenes=False`` skips the webgui canvases (movie mode renders with
    matplotlib instead); ``record=True`` snapshots every redrawn frame for
    the movie stitch.
    """

    def __init__(self, prob, every=5, scenes=True, record=False):
        if every < 1:
            raise ValueError("every must be a positive step count")
        self.prob = prob
        self.every = int(every)
        self._n = 0
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

        self.frames = [] if record else None
        self.top, self.bottom = [], []
        if not scenes:
            return

        if self._final_tracked:
            state_lo = state_hi = err_sym = None
        else:
            state_lo, state_hi = _minmax(self.u_final)
            err_sym = 0.25 * (state_hi - state_lo) + 1e-12
        if self.q_true is None:
            q_lo = q_hi = q_err_sym = None
        else:
            q_lo, q_hi = _minmax(self.q_true)
            q_err_sym = 0.25 * (q_hi - q_lo) + 1e-12

        self.top = [_panel(self.z, "datum z", state_lo, state_hi),
                    _panel(self.err_u, "data error u - z",
                           None if err_sym is None else -err_sym, err_sym),
                    _panel(self.u_final, "ground truth u at final time",
                           state_lo, state_hi)]
        self.bottom = [_panel(self.qhat, "estimate qhat", q_lo, q_hi)]
        if self.q_true is not None:
            self.bottom += [
                _panel(self.err_q, "reconstruction error qhat - q",
                       None if q_err_sym is None else -q_err_sym, q_err_sym),
                _panel(self.q_true, "true parameter q", q_lo, q_hi)]

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

    def show(self):
        """Render both panel rows (a no-op outside a notebook)."""
        try:
            from IPython.display import display
            from ipywidgets import HTML, HBox, Layout, VBox
        except ImportError:
            return
        def cell(scene):
            label = HTML(f"<p style='text-align:center;margin:0'>"
                         f"{scene.caption}</p>")
            return VBox([label, scene.widget], layout=Layout(width="33%"))
        row = lambda scenes: HBox([cell(s) for s in scenes],
                                  layout=Layout(width="100%"))
        display(VBox([row(self.top), row(self.bottom)]))

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
        if self.frames is not None:
            snap = lambda gf: np.asarray(gf.vec.data).copy()
            self.frames.append((snap(self.z), snap(self.err_u), snap(self.qhat),
                                None if self.q_true is None
                                else snap(self.err_q)))
        for scene in self.top + self.bottom:
            scene.Redraw()


def _view_space(mesh):
    """A P1 view space with its matplotlib triangulation and vertex coords.

    Every panel field (P3 state, P0 parameter) is interpolated onto this
    space, so one Gouraud renderer covers all six panels; the triangulation
    comes from the P1 element vertices directly, with no mesh plotting.
    """
    from matplotlib.tri import Triangulation
    from ngsolve import H1, x as xcf, y as ycf
    p1 = H1(mesh, order=1)
    xc, yc = GridFunction(p1), GridFunction(p1)
    xc.Set(xcf)
    yc.Set(ycf)
    tri = np.array([p1.GetDofNrs(el) for el in mesh.Elements()])
    triangulation = Triangulation(np.asarray(xc.vec.data),
                                  np.asarray(yc.vec.data), tri)
    return GridFunction(p1), triangulation


def _to_view(frames, fes, view):
    """Convert raw dof vectors to P1-view arrays (one shared view buffer)."""
    gf = GridFunction(fes)
    out = []
    for arr in frames:
        gf.vec.data = np.asarray(arr, dtype=np.float64)
        view.Set(gf)
        out.append(np.asarray(view.vec.data).copy())
    return out


def make_movie(panels, path, fps=24.0, max_frames=240):
    """Stitch the recorded frames into an mp4 and return the path.

    The 2x3 panel grid mirrors the live view (datum z | data error u - z |
    ground-truth state, then estimate qhat | reconstruction error qhat - q |
    true parameter q).  No panel chrome: bare Gouraud plots with titles.
    Each colorbar is static, set by the extremes over all frames of its
    category — state panels share [min, max], error panels are symmetric
    about zero, parameter panels share [min, max] — and the playback frame
    rate is constant, so every frame covers the same amount of model time.
    """
    if panels.frames is None:
        raise ValueError("make_movie needs a LivePanels(record=True)")
    from matplotlib.animation import FFMpegWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    mesh = panels.prob.mesh
    view, triangulation = _view_space(mesh)
    has_q = panels.q_true is not None
    cols = list(zip(*panels.frames)) if panels.frames else [()] * 4
    z = _to_view(cols[0], panels.z.space, view)
    err_u = _to_view(cols[1], panels.err_u.space, view)
    qhat = _to_view(cols[2], panels.qhat.space, view)
    err_q = (_to_view(cols[3], panels.qhat.space, view)
             if has_q and cols[3] else [])
    u_final = _to_view([panels.u_final.vec.data], panels.u_final.space,
                       view)[0]
    q_true = (_to_view([panels.q_true.vec.data], panels.q_true.space, view)[0]
              if has_q else None)

    def _lo_hi(*series):
        lo = min(float(np.min(s)) for s in series)
        hi = max(float(np.max(s)) for s in series)
        return lo, hi + 1e-12 * (hi - lo + 1e-300)

    def _sym(*series):
        return 1e-12 + max(float(np.max(np.abs(s))) for s in series)

    state_rng = _lo_hi(z, [u_final])
    q_rng = _lo_hi(qhat, [q_true]) if has_q else None
    spec = [
        ("datum z", z, *state_rng),
        ("data error u - z", err_u, -_sym(*err_u), _sym(*err_u)),
        ("ground truth u at final time", [u_final], *state_rng),
        ("estimate qhat", qhat, *(q_rng or _lo_hi(qhat))),
        ("reconstruction error qhat - q", err_q,
         -_sym(*err_q), _sym(*err_q)) if err_q else None,
        ("true parameter q", [q_true], *(q_rng or (0.0, 1.0))) if has_q else None,
    ]

    n = max([len(p[1]) for p in spec if p is not None] or [1])
    stride = -(-n // max_frames)  # ceil
    keep = list(range(0, n, stride))

    fig = Figure(figsize=(12.0, 5.2), dpi=100)
    FigureCanvasAgg(fig)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.89, bottom=0.06,
                        hspace=0.55, wspace=0.35)
    axes = fig.subplots(2, 3)
    mappers = []
    for ax, panel in zip(axes.ravel(), spec):
        if panel is None:
            ax.set_visible(False)
            continue
        title, series, lo, hi = panel
        mesh_plot = ax.tripcolor(triangulation, series[keep[0]],
                                 shading="gouraud", cmap="viridis",
                                 vmin=lo, vmax=hi)
        fig.colorbar(mesh_plot, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="datalim")
        if len(series) > 1:
            mappers.append((mesh_plot, series))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    with writer.saving(fig, str(path), dpi=100):
        for k in keep:
            for mapper, series in mappers:
                mapper.set_array(series[k])
            writer.grab_frame()
    return path


def _display_movie(path):
    """Show the mp4 inline when a notebook front-end is present."""
    print(f"movie: {path}")
    try:
        from IPython.display import Video, display
    except ImportError:
        return
    display(Video(str(path), embed=True, mimetype="video/mp4"))


def run_live(example, noise_level=0.0, every=5, live=False, fps=24.0, path=None,
             **kwargs):
    """Assimilate one noise level with the six-panel view.

    ``example`` is a name from :data:`EXAMPLES` (or an already-built
    ``Problem``); ``noise_level`` is the relative L2 noise of the data
    (``0.05`` = 5 %).  By default the run is stitched into an mp4 with a
    fixed frame rate (frames every ``every`` steps, static colorbars) and
    displayed; ``path`` overrides ``results/<name>_noise<n>.mp4``.  With
    ``live=True`` the panels stream live in the notebook instead.  Returns
    the :class:`~mras.results.LevelResult`.
    """
    prob = example if isinstance(example, Problem) else build(example, **kwargs)
    if live:
        panels = LivePanels(prob, every=every)
        panels.show()
        prob.on_step = panels.update
        try:
            result = prob.run_level(noise_level)
        finally:
            prob.on_step = None
        panels.update(result.state, force=True)
        return result
    panels = LivePanels(prob, every=every, scenes=False, record=True)
    prob.on_step = panels.update
    try:
        result = prob.run_level(noise_level)
    finally:
        prob.on_step = None
    if path is None:
        path = Path("results") / f"{prob.name}_noise{noise_level:g}.mp4"
    movie = make_movie(panels, path, fps=fps)
    _display_movie(movie)
    return result
