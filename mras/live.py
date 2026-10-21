"""Live webgui panels for one MRAS run.

Six side-by-side ``ngsolve.webgui`` canvases in two rows, updated while the
scheme runs::

    datum z        |  data error u - z  |  ground truth u at final time
    estimate qhat  |  error qhat - q    |  true parameter q

The notebook-facing API is :func:`run_live`: pass an example name (see
:data:`EXAMPLES`) and a noise level, and it builds a demo-sized problem,
draws the panels, and assimilates, updating every ``every``-th step.  All
wiring — the ``on_step`` callback, the panel grid functions, the fixed color
ranges, the final-truth resolution — lives here.
"""

import importlib.util
from pathlib import Path

import numpy as np
from ngsolve import GridFunction
from ngsolve.webgui import Draw

from .problems import AllenCahnProblem, DarcyProblem, FisherProblem, PotentialProblem
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
    "burgers": (_burgers, dict(h=0.35, dt=0.02, tend=5.0)),
    "darcy": (DarcyProblem, dict(h=0.08, dt=0.02, tend=2.0,
                                 data_h=0.06, data_order=2)),
    "fisher": (FisherProblem, dict(h=0.1, dt=0.02, tend=3.0)),
    "potential": (PotentialProblem, dict(h=0.12, dt=0.02, tend=3.0)),
    "allen_cahn": (AllenCahnProblem, dict(h=0.12, dt=0.02, tend=3.0)),
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
    """The six panel grid functions, their scenes, and the step callback."""

    def __init__(self, prob, every=5):
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
        for scene in self.top + self.bottom:
            scene.Redraw()


def run_live(example, level=0.0, every=5, **kwargs):
    """Assimilate one noise level with the live six-panel view.

    ``example`` is a name from :data:`EXAMPLES` (or an already-built
    ``Problem``); ``level`` is the relative L2 noise of the data
    (``0.05`` = 5 %).  Returns the :class:`~mras.results.LevelResult`.
    """
    prob = example if isinstance(example, Problem) else build(example, **kwargs)
    panels = LivePanels(prob, every=every)
    panels.show()
    prob.on_step = panels.update
    try:
        result = prob.run_level(level)
    finally:
        prob.on_step = None
    panels.update(result.state, force=True)
    return result
