#!/usr/bin/env python
"""Speed smoke: where an example's time goes, per step and per phase.

Runs a few steps of an example at its DEFAULTS (the production mesh; any
setting can be overridden) without movie, and prints the wall time per
step of each phase: the forward model that generates the data (its Newton
solves apart), the MRAS step, the model fit, the error norms, the noise,
the smoothing, the residual watch, the checkpoint.  The build (meshes,
spaces) is timed on its own.  A run refits its model fit at its end, which
a short smoke counts in full: a long run refits only at steps 50, 100,
200, ...

``--production`` times a noisy level as ``scripts/run_examples.py`` runs
it after the noise-free one: the noise-free data read back from a
DataCache (filled by a noise-free run of the same steps first, timed
apart) and a checkpoint written every step.  ``--profile N`` profiles the
timed run with cProfile as well and prints the N functions with the most
time of their own, so that nothing is left unattributed.

    .venv/bin/python scripts/speed.py                     # burgers_2d, 10 steps
    .venv/bin/python scripts/speed.py burgers_2d --steps 5 --set dt=0.01
    .venv/bin/python scripts/speed.py fisher --noise 0.01
    .venv/bin/python scripts/speed.py burgers_2d --noise 0.01 --production --profile 30
"""

import argparse
import collections
import cProfile
import functools
import io
import pathlib
import pstats
import sys
import tempfile
import time
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mras.checkpoint as checkpoint  # noqa: E402
import mras.fields as fields  # noqa: E402
import mras.floor as floor  # noqa: E402
import mras.forward as forward  # noqa: E402
import mras.noise as noise  # noqa: E402
import mras.robust as robust  # noqa: E402
import mras.scheme as scheme  # noqa: E402
from mras.live import build, list_examples  # noqa: E402

# (owner, method, label): the phases, each timed where it is called.
PHASES = [
    (forward.ForwardRun, "step", "data: forward step"),
    (forward.ForwardRun, "_newton", "  of which Newton"),
    (scheme.MRASProblem, "_step", "MRAS step"),
    (floor.ModelFit, "update", "model fit: refit"),
    (floor.ModelFit, "offer", "model fit: sample"),
    (scheme.MRASProblem, "q_error", "error norms: q"),
    (scheme.MRASProblem, "u_distance", "error norms: u"),
    (fields, "l2norm", "norms (fields)"),
    (noise.GaussianFEMNoise, "draw", "noise draw (white)"),
    (noise.MaternFEMNoise, "draw", "noise draw (Matern)"),
    (robust.Smoother, "__call__", "smoothing"),
    (scheme.MRASProblem, "_equation_base", "residual watch: base"),
    (scheme.MRASProblem, "_equation_residual", "residual watch: residuals"),
    (scheme.MRASProblem, "_project", "projection"),
    (forward.DataCache, "datum", "data cache read"),
    (checkpoint.Checkpoint, "update", "checkpoint write"),
]


def instrument():
    spent, calls = collections.defaultdict(float), collections.Counter()
    for owner, attr, label in PHASES:
        fn = getattr(owner, attr)

        @functools.wraps(fn)
        def timed(*args, _fn=fn, _label=label, **kwargs):
            started = time.perf_counter()
            try:
                return _fn(*args, **kwargs)
            finally:
                spent[_label] += time.perf_counter() - started
                calls[_label] += 1
        setattr(owner, attr, timed)
    return spent, calls


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", nargs="?", default="burgers_2d",
                    choices=sorted(list_examples()))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="settings over the DEFAULTS, e.g. h=0.1 dt=0.01")
    ap.add_argument("--production", action="store_true",
                    help="as run_examples runs a noisy level: data from a "
                    "DataCache, a checkpoint written every step")
    ap.add_argument("--profile", type=int, default=0, metavar="N",
                    help="profile the timed run with cProfile too and print "
                    "its N most expensive functions (own time)")
    args = ap.parse_args(argv)
    overrides = {}
    for item in args.set:
        key, value = item.split("=")
        overrides[key] = float(value) if "." in value or "e" in value \
            else int(value)
    warnings.simplefilter("ignore")
    if args.noise:
        # The noise imports torch and ngsrng on its first draw: import them
        # here, outside the timed run.
        import ngsrng  # noqa: F401
        import torch  # noqa: F401
    spent, calls = instrument()

    def problem():
        prob = build(args.name, **overrides)
        prob.tsteps = args.steps  # as a checkpoint's resume sets a new end
        prob.tend = prob.tstart + args.steps * prob.dt
        prob.target = 0.0
        return prob

    scratch = tempfile.TemporaryDirectory()
    cache = writer = None
    if args.production:
        cache = forward.DataCache(pathlib.Path(scratch.name) / "data.h5")
        started = time.perf_counter()
        first = problem()
        first.data_cache = cache
        first.run_level(0.0, progress=False)
        print(f"noise-free level filling the data cache: "
              f"{time.perf_counter() - started:.1f} s (not counted below)")
        spent.clear()
        calls.clear()
    started = time.perf_counter()
    prob = problem()
    built = time.perf_counter() - started
    print(f"{args.name}: build {built:.1f} s; U_h {prob.Ufes.ndof} dofs, "
          f"Q_h {prob.Qfes.ndof}")
    if args.production:
        prob.data_cache = cache
        writer = checkpoint.Checkpoint(
            prob, pathlib.Path(scratch.name) / "run.h5", args.noise, 1024.0)
        prob.on_step = writer.update
    profiler = cProfile.Profile() if args.profile else None
    started = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    result = prob.run_level(args.noise, progress=False)
    if profiler is not None:
        profiler.disable()
    total = time.perf_counter() - started
    if writer is not None:
        writer.close(result.state)
    if cache is not None:
        cache.close()
    scratch.cleanup()
    steps = prob.tsteps
    print(f"{steps} steps in {total:.1f} s: {total / steps:.2f} s per step")
    accounted = 0.0
    for label, seconds in sorted(spent.items(), key=lambda kv: -kv[1]):
        print(f"   {label:22s} {seconds / steps:7.3f} s per step "
              f"({100 * seconds / total:4.1f} %, {calls[label]} calls)")
        if not label.startswith("  "):
            accounted += seconds
    print(f"   {'rest':22s} {(total - accounted) / steps:7.3f} s per step")
    if profiler is not None:
        out = io.StringIO()
        stats = pstats.Stats(profiler, stream=out)
        stats.sort_stats("tottime").print_stats(args.profile)
        print(out.getvalue())


if __name__ == "__main__":
    main()
