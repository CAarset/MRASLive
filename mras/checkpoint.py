"""Run checkpoints: one HDF5 file per run, from the initial state to the end.

HDF5 is the container scientific data ship in (the PDEBench data this
package reads among them), readable without NGSolve; the fields are stored
as the coefficient vectors of their finite element spaces, which the
problem -- built again from the recorded constructor arguments -- turns
back into grid functions.  The layout::

    /              attributes: problem name and class, constructor arguments
                   (JSON), scheme, forward scheme, noise level, seed, dt,
                   tend, ngsolve and mras versions, git revision, created,
                   completed
    /history/      t, error_q, error_u, misfit_u -- every step; model_fit,
                   the model fit's (t, error) at its updates (mras.floor),
                   the last of them also the attribute ``model_fit``
    /trajectory/   t, step and the coefficient vectors u, q, z (and u_true
                   when the data are noisy) every ``stride`` steps from the
                   initial state, in single precision.  Every step is kept
                   until they fill ``budget`` megabytes; then every other
                   one is dropped and ``stride`` doubles, as often as
                   needed, so a run of any length -- one that stops at its
                   target -- keeps as many evenly spaced steps as the budget
                   holds (``budget=None``: every step)
    /restart/      the double-precision state of the last checkpointed step
                   -- estimate, datum, reference, histories, and the forward
                   model where it stood -- which :func:`resume` continues
                   from exactly
    /spaces/       the dimensions of U_h and Q_h

Like the checkpoint of a training run, the restart point is rewritten every
``every`` steps (a twentieth of the run unless given) and at the end, so a
run that is interrupted loses at most that much.

    from mras.checkpoint import load, resume
    ckpt = load("results/burgers_1d/burgers_1d_noise0.h5")
    q = ckpt.field("q", -1)            # the last stored estimate, a GridFunction
    result = resume("results/burgers_1d/burgers_1d_noise0.h5", tend=400)
"""

import json
import subprocess
import time
from importlib import metadata
from pathlib import Path

import h5py
import numpy as np
import ngsolve
from ngsolve import GridFunction

ROOT = Path(__file__).resolve().parent.parent


def _git_revision():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return rev.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def _version():
    try:
        return metadata.version("mras")
    except metadata.PackageNotFoundError:
        return "unknown"


def _plain(value):
    """numpy scalars as Python numbers; anything else is no plain value."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not a plain value")


def _arguments(problem):
    """The recorded constructor arguments as JSON, or None if they are not
    plain values (the problem cannot then be built again from the file)."""
    args, kwargs = getattr(problem, "init_args", ((), {}))
    try:
        return json.dumps({"args": list(args), "kwargs": kwargs},
                          default=_plain)
    except TypeError:
        return None


class Checkpoint:
    """Writes the checkpoint of one run; ``update`` is its on_step hook.

    ``problem`` and ``noise_level`` are the run's, ``path`` the file.  With
    ``restart`` (the restart point being resumed from) the file is continued:
    the trajectory after the restart point is dropped, then extended.
    """

    def __init__(self, problem, path, noise_level, budget=256.0, every=None,
                 restart=None):
        self.problem = problem
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        steps = problem.tsteps + 1
        self.every = max(1, int(every or -(-steps // 20)))
        noisy = noise_level > 0
        self.fields = ("u", "q", "z", "u_true") if noisy else ("u", "q", "z")
        nu, nq = problem.Ufes.ndof, problem.Qfes.ndof
        per = 4 * (nu * (len(self.fields) - 1) + nq)
        self.capacity = (None if budget is None
                         else max(2, int(budget * 2**20) // per))
        self.stride = 1
        self.noise_level = noise_level
        self._last_restart = None
        if restart is None:
            self._create()
        else:
            self._continue(restart)

    def _create(self):
        p = self.problem
        with h5py.File(self.path, "w") as f:
            f.attrs.update({
                "problem": p.name, "class": f"{type(p).__module__}."
                f"{type(p).__qualname__}",
                "arguments": _arguments(p) or "", "scheme": p.scheme,
                "forward_scheme": p.forward_scheme,
                "noise_level": self.noise_level, "seed": p.seed, "dt": p.dt,
                # the noise and the safeguards, set after construction
                **{name: json.dumps(getattr(p, name))
                   for name in _SETTINGS},
                "tend": p.tend, "stride": self.stride, "every": self.every,
                "capacity": self.capacity or 0,
                "ngsolve": ngsolve.__version__, "mras": _version(),
                "git": _git_revision(),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "completed": False})
            spaces = f.create_group("spaces")
            spaces.attrs.update({"U": p.Ufes.ndof, "Q": p.Qfes.ndof})
            traj = f.create_group("trajectory")
            for name in ("t", "step"):
                traj.create_dataset(name, shape=(0,), maxshape=(None,),
                                    dtype="f8" if name == "t" else "i8")
            for name in self.fields:
                n = p.Qfes.ndof if name == "q" else p.Ufes.ndof
                traj.create_dataset(name, shape=(0, n), maxshape=(None, n),
                                    dtype="f4", chunks=(1, n),
                                    compression="gzip", compression_opts=1,
                                    shuffle=True)
            f.create_group("history")

    def _continue(self, restart):
        with h5py.File(self.path, "a") as f:
            self.stride = int(f.attrs["stride"])
            self.every = int(f.attrs["every"])
            self.capacity = (None if f.attrs.get("capacity", 0) <= 0
                             else int(f.attrs["capacity"]))
            # The resumed run writes the restart step again.
            keep = int(np.searchsorted(f["trajectory/step"][()],
                                       restart["step"], side="left"))
            for name in f["trajectory"]:
                ds = f["trajectory"][name]
                ds.resize(keep, axis=0)
            f.attrs["tend"] = self.problem.tend
            f.attrs["completed"] = False

    # -- the on_step hook ----------------------------------------------------
    def update(self, state):
        step = state.step
        if step % self.stride == 0:
            stored = self._snapshot(state, step)
            if self.capacity is not None and stored > self.capacity:
                self._thin()
        if step and step % self.every == 0:
            self._restart(state, step)

    def _thin(self):
        """Drop every other stored step and double the stride."""
        with h5py.File(self.path, "a") as f:
            traj = f["trajectory"]
            keep = traj["step"][()] % (2 * self.stride) == 0
            for name in traj:
                values = traj[name][()][keep]
                traj[name].resize(len(values), axis=0)
                traj[name][...] = values
            self.stride *= 2
            f.attrs["stride"] = self.stride

    def close(self, state):
        """The final snapshot and restart point, and the run marked done."""
        step = state.step
        with h5py.File(self.path, "r") as f:
            steps = f["trajectory/step"][()]
        if not len(steps) or steps[-1] != step:
            self._snapshot(state, step)
        if self._last_restart != step:
            self._restart(state, step)
        with h5py.File(self.path, "a") as f:
            f.attrs["completed"] = True
            fit = getattr(state, "model_fit", None)
            if fit is not None and fit.value is not None:
                # the whole run's model fit (mras.floor), and its updates
                f.attrs["model_fit"] = fit.value
                if "model_fit" in f["history"]:
                    del f["history/model_fit"]
                f["history/model_fit"] = np.asarray(fit.values, dtype="f8")

    def _snapshot(self, state, step):
        values = {"u": state.gfu, "q": state.gfq, "z": state.z,
                  "u_true": state.u_true}
        with h5py.File(self.path, "a") as f:
            traj = f["trajectory"]
            n = traj["t"].shape[0]
            for name in ("t", "step", *self.fields):
                traj[name].resize(n + 1, axis=0)
            traj["t"][n] = state.t.Get()
            traj["step"][n] = step
            for name in self.fields:
                traj[name][n] = values[name].vec.FV().NumPy()
        return n + 1

    def _restart(self, state, step):
        histories = {"error_q": state.error_q, "error_u": state.error_u,
                     "misfit_u": state.misfit_u,
                     "effective_noise": state.effective_noise,
                     "residual": state.residual}
        dt, t0 = self.problem.dt, self.problem.tstart
        with h5py.File(self.path, "a") as f:
            history = f["history"]
            for name, values in histories.items():
                if name in history:
                    del history[name]
                if values is not None:
                    history[name] = np.asarray(values, dtype="f8")
            if "t" in history:
                del history["t"]
            # error histories start at t^0; the misfit at t^1
            history["t"] = t0 + dt * np.arange(step + 1)
            if "restart" in f:
                del f["restart"]
            restart = f.create_group("restart")
            restart.attrs.update({"t": state.t.Get(), "step": step,
                                  "has_truth": state.has_truth})
            for name in ("gfu", "gfq", "z", "z_old", "u_true"):
                restart[name] = getattr(state, name).vec.FV().NumPy()
            forward = state.forward
            if forward is not None:
                restart.attrs["forward_steps"] = forward.steps
                for name in ("gf", "prev", "before"):
                    restart[f"forward_{name}"] = \
                        getattr(forward, name).vec.FV().NumPy()
            # the noise's random state and the noisy-data machinery
            # (mras.robust), to go on exactly
            if state.noise_level:
                rng = self.problem.noise.rng_state()
                if rng is not None:
                    restart["noise_rng"] = rng
            if getattr(state, "q_start", None) is not None:
                restart["q_start"] = state.q_start.vec.FV().NumPy()
            for part in (state.smoother, state.watch):
                if part is not None:
                    arrays, attrs = part.export()
                    for name, values in arrays.items():
                        restart[name] = values
                    restart.attrs.update(attrs)
        self._last_restart = step


# Problem settings that a run may change after construction and a resumed
# run must take again: the data noise and the safeguards (mras.robust).
_SETTINGS = ("noise_model", "noise_length", "noise_nu", "smooth_data",
             "smooth_time", "smooth_space", "discrepancy", "projection",
             "q_bounds", "leakage")


class Stored:
    """A checkpoint opened for reading (see :func:`load`)."""

    def __init__(self, path, problem=None):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.attrs = dict(f.attrs)
            self.history = {name: f["history"][name][()]
                            for name in f["history"]}
            self.times = f["trajectory/t"][()]
            self.steps = f["trajectory/step"][()]
            self.spaces = dict(f["spaces"].attrs)
        self._problem = problem

    @property
    def problem(self):
        """The problem, built again from the recorded constructor arguments
        (or the one handed to :func:`load`)."""
        if self._problem is None:
            if not self.attrs["arguments"]:
                raise ValueError(
                    f"{self.path}: the constructor arguments were not plain "
                    "values; pass the problem to load(path, problem=...)")
            from .live import _locate
            _, cls = _locate(self.attrs["problem"])
            arguments = json.loads(self.attrs["arguments"])
            self._problem = cls(*arguments["args"], **arguments["kwargs"])
            # Settings changed after construction travel in the attributes.
            self._problem.scheme = str(self.attrs["scheme"])
            self._problem.forward_scheme = str(self.attrs["forward_scheme"])
            for name in _SETTINGS:
                if name in self.attrs:
                    value = json.loads(self.attrs[name])
                    setattr(self._problem, name,
                            tuple(value) if name == "q_bounds" else value)
        p = self._problem
        if (p.Ufes.ndof, p.Qfes.ndof) != (self.spaces["U"], self.spaces["Q"]):
            raise ValueError(f"{self.path}: the problem's spaces do not "
                             "match the checkpoint's")
        return p

    def snapshot(self, k):
        """Stored step ``k``: its time, step and coefficient vectors."""
        with h5py.File(self.path, "r") as f:
            traj = f["trajectory"]
            out = {name: traj[name][k] for name in traj}
        return out

    def field(self, name, k):
        """Snapshot ``k`` of field ``name`` ("u", "q", "z", "u_true") as a
        grid function on the problem's space."""
        space = self.problem.Qfes if name == "q" else self.problem.Ufes
        gf = GridFunction(space)
        with h5py.File(self.path, "r") as f:
            gf.vec.FV().NumPy()[:] = f["trajectory"][name][k]
        return gf

    def restart(self):
        """The restart point, as :meth:`MRASProblem.run_level` reads it."""
        with h5py.File(self.path, "r") as f:
            group = f["restart"]
            out = {name: group[name][()] for name in group}
            out.update(group.attrs)
        for name in ("error_q", "error_u", "misfit_u", "effective_noise",
                     "residual"):
            out[name] = self.history.get(name)
        return out


def load(path, problem=None):
    """Open a checkpoint; ``problem`` is needed only when the file cannot
    build it again (an unregistered class, or non-plain arguments)."""
    return Stored(path, problem)


def resume(path, tend=None, problem=None, progress=False, budget=256.0):
    """Continue the run of a checkpoint from its restart point, to ``tend``
    (the original horizon unless given), extending the same file.  Returns
    the :class:`~mras.results.LevelResult` of the whole run's histories."""
    stored = load(path, problem)
    p = stored.problem
    if tend is not None:
        p.tend = tend
        p.tsteps = int(round((tend - p.tstart) / p.dt))
    restart = stored.restart()
    noise_level = float(stored.attrs["noise_level"])
    writer = Checkpoint(p, path, noise_level, budget, restart=restart)
    previous = p.on_step

    def hook(state):
        if previous is not None:
            previous(state)
        writer.update(state)
    p.on_step = hook
    try:
        result = p.run_level(noise_level, progress=progress, resume=restart)
    finally:
        p.on_step = previous
    writer.close(result.state)
    return result
