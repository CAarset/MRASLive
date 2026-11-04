#!/usr/bin/env python
"""Run the examples: every registered one by default, or those named.

Each example runs at its own DEFAULTS (``--quick``: its QUICK sizes),
noise-free unless ``--noise-levels`` says otherwise, until its
reconstruction error reaches ``--target`` (max(1 %, 1.25 times the
floor) unless given; 0 runs to the time limit), with ``--discrepancy
TAU`` its residual TAU times the effective noise level (which counts the
data's mismatch with the estimator's model too, so noise-free runs have
one), or the time limit ``--tend`` (100 for most) is reached.  Noisy data enter smoothed, and q is projected onto its bounds
(mras.robust; ``--no-smoothing``, ``--no-projection`` switch them off;
``--leakage`` adds a leakage toward q_0, which the MRAS theory does not
support).
Every run writes, under
``--out/<name>/``:

    <name>_noise<p>.h5    the HDF5 checkpoint (mras.checkpoint): settings,
                          error histories, the trajectory -- every step, as
                          far as ``--budget`` megabytes go (1024 by
                          default) -- and a restart point
    <name>_noise<p>.mp4   the movie, rendered from the checkpoint once the run
                          is done (mras.movie): a frame for every stored step,
                          up to 1200, lasting 4 to 10 seconds (``--fps`` and
                          ``--every`` override the pace), painted by
                          ``--workers`` processes (``--no-movie`` skips it)

and ``--out/summary.json`` collects the final errors of all runs, printed as a
table at the end (a run replaces only its own example and noise level's row,
so invocations side by side add up).  An example's noise levels run one after another on the
same noise-free data: the first run generates them, the next read them
back (mras.forward.DataCache, a temporary ``.<name>_data.h5`` beside the
runs, deleted after the example's last level).  After an example's runs, the persistent excitation of
its data (Baumeister's mu) is checked over the horizon they took (the
longest), at its audit discretization (mras.verify.check_pe_after;
``--no-pe`` skips it).  The notebook writes its runs beside them, as
``<name>_noise<p>-notebook.h5`` and ``.mp4``.  ``--delete-results`` empties
``--out`` first -- every file and folder in it, so keep notes elsewhere
(``archive/``).  ``--resume FILE`` continues an interrupted (or finished)
run from its checkpoint, to ``--tend`` if given, and ``--movie-from FILE``
renders the movie of a checkpoint again.

    .venv/bin/python scripts/run_examples.py                       # all
    .venv/bin/python scripts/run_examples.py --delete-results      # all, afresh
    .venv/bin/python scripts/run_examples.py burgers_1d fisher --quick
    .venv/bin/python scripts/run_examples.py darcy --noise-levels 0 0.05
    .venv/bin/python scripts/run_examples.py --resume results/burgers_1d/burgers_1d_noise0.h5 --tend 400
    .venv/bin/python scripts/run_examples.py --movie-from results/burgers_1d/burgers_1d_noise0.h5
    .venv/bin/python scripts/run_examples.py --list
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mras.checkpoint import resume  # noqa: E402
from mras.forward import DataCache  # noqa: E402
from mras.live import _locate, build, list_examples, run_live  # noqa: E402
from mras.movie import render_movie  # noqa: E402
from mras.verify import FAIL, OK, check_pe_after  # noqa: E402


def settings(name, quick, overrides):
    """The example's DEFAULTS (QUICK under ``quick``) under the overrides."""
    module, _ = _locate(name)
    out = dict(getattr(module, "DEFAULTS", {}).get(name, {}))
    if quick:
        out.update(getattr(module, "QUICK", {}).get(name, {}))
    out.update({key: value for key, value in overrides.items()
                if value is not None})
    return out


def delete_results(out):
    """Empty the results folder ``out``: every file and folder in it."""
    out = out.resolve()
    if out == ROOT or out in ROOT.parents or (out / ".git").exists():
        raise SystemExit(f"--delete-results: {out} holds the repository; "
                         "give --out a results folder")
    if not out.is_dir():
        return
    entries = sorted(out.iterdir())
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    print(f"deleted {len(entries)} entries from {out}", flush=True)


def row(name, noise_level, result, seconds, stem):
    meta = result.meta
    return {"example": name, "noise_level": noise_level,
            "error_q": result.final_error_q, "q_floor": meta.get("q_floor"),
            "model_fit": meta.get("model_fit"),
            "error_u": result.final_error_u, "steps": meta["steps"],
            "t_end": meta.get("t_end"), "stopped": meta.get("stopped"),
            "effective_noise": meta.get("effective_noise"),
            "residual": meta.get("residual"),
            "scheme": meta["scheme"], "seconds": round(seconds, 1),
            # not with_suffix: the stem of noise level 0.01 ends in ".01"
            "checkpoint": f"{stem}.h5",
            "movie": f"{stem}.mp4"}


def safeguards(prob, args):
    """The noisy-data safeguards (mras.robust) the command line switches."""
    if args.discrepancy is not None:
        prob.discrepancy = args.discrepancy or None
    if args.noise_model is not None:
        prob.noise_model = args.noise_model
    if args.noise_length is not None:
        prob.noise_length = args.noise_length
    for flag, attribute in (("no_smoothing", "smooth_data"),
                            ("no_projection", "projection")):
        if getattr(args, flag):
            setattr(prob, attribute, False)
    if args.leakage:
        prob.leakage = True


def excitation(name, rows):
    """Persistent excitation of the data over the horizon the runs took
    (the longest): computed after them, since a run's end is known only
    then.  Recorded in each row as ``pe``."""
    ends = [r["t_end"] for r in rows if r.get("t_end") is not None]
    if not ends:
        return
    t_end = max(ends)
    print(f"== {name}: persistent excitation up to t = {t_end:g}",
          flush=True)
    check = check_pe_after(name, t_end)
    if check is None:
        print("   no audit settings (CHECK or QUICK): skipped", flush=True)
        return
    print(f"   {check.status}: {check.detail}", flush=True)
    for r in rows:
        r["pe"] = {"status": check.status, "mu": check.value,
                   "t_end": t_end, "detail": check.detail}


def _pe_cell(pe):
    if not pe:
        return f"{'-':>8s}"
    if pe["status"] == OK:
        return f"{pe.get('mu', pe.get('gamma')):8.1e}"
    return f"{'fail' if pe['status'] == FAIL else '?':>8s}"


def table(rows):
    """The runs as a table, its first column as wide as the longest name."""
    name = max([len("example")] + [len(r["example"]) for r in rows])
    lines = [f"{'example':{name}s} {'noise':>6s} {'error_q':>9s} "
             f"{'fit':>8s} {'floor':>8s} {'error_u':>9s} {'eff':>8s} "
             f"{'steps':>6s} {'end':>7s} {'stop':>11s} {'pe':>8s} "
             f"{'time':>8s}"]

    def number(value, width, fmt):
        return f"{'-':>{width}s}" if value is None else f"{value:{width}{fmt}}"
    for r in rows:
        lines.append(f"{r['example']:{name}s} {r['noise_level']:6g} "
                     f"{number(r['error_q'], 9, '.4f')} "
                     f"{number(r.get('model_fit'), 8, '.4f')} "
                     f"{number(r['q_floor'], 8, '.4f')} "
                     f"{number(r['error_u'], 9, '.2e')} "
                     f"{number(r.get('effective_noise'), 8, '.4f')} "
                     f"{r['steps']:6d} "
                     f"{number(r.get('t_end'), 7, '.2f')} "
                     f"{str(r.get('stopped') or '-'):>11s} "
                     f"{_pe_cell(r.get('pe'))} "
                     f"{r['seconds']:7.0f}s")
    return "\n".join(lines)


def main(argv=None):
    catalog = sorted(list_examples())
    ap = argparse.ArgumentParser(
        description="Run the MRAS examples with movies and checkpoints.")
    ap.add_argument("names", nargs="*", metavar="example",
                    help=f"examples to run (default: all): {', '.join(catalog)}")
    ap.add_argument("--list", action="store_true", help="list the examples")
    ap.add_argument("--quick", action="store_true",
                    help="each example's QUICK sizes instead of its DEFAULTS")
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.0],
                    help="relative noise levels (0.05 = 5%%); default 0")
    ap.add_argument("--scheme", choices=["implicit", "semi-implicit",
                                         "explicit"],
                    help="time scheme of the MRAS step (default: each "
                         "example's own)")
    ap.add_argument("--h", type=float, help="design mesh width")
    ap.add_argument("--dt", type=float, help="time step")
    ap.add_argument("--tend", type=float,
                    help="the time limit (default: each example's, 100 for "
                         "most)")
    ap.add_argument("--target", type=float,
                    help="the reconstruction error to stop at (default: "
                         "max(1%%, 1.25 times the floor); 0 runs to --tend)")
    ap.add_argument("--q0", type=float,
                    help="a constant initial estimate qhat_0 (default: each "
                         "example's own, 0 for most)")
    ap.add_argument("--out", type=Path, default=ROOT / "results",
                    help="output directory (default: results/)")
    ap.add_argument("--budget", type=float, default=1024.0,
                    help="megabytes the stored trajectory may take per run "
                         "(default 1024)")
    ap.add_argument("--no-movie", action="store_true",
                    help="skip the movie (faster)")
    ap.add_argument("--fps", type=float,
                    help="frames per second of the movie (default: as many "
                         "frames as stored, up to 1200, over 4 to 10 s)")
    ap.add_argument("--every", type=int,
                    help="a frame every this many stored steps (default: "
                         "every one, up to 1200 frames)")
    ap.add_argument("--workers", type=int,
                    help="processes painting the movie (default: half the "
                         "cores)")
    ap.add_argument("--discrepancy", type=float,
                    help="tau of the discrepancy stop, at tau times the "
                         "effective noise level (default: no stop; 1.1 is "
                         "the usual start)")
    ap.add_argument("--noise-model", choices=["matern", "white"],
                    help="the data noise: Matern-correlated (default) or "
                         "the paper's white noise")
    ap.add_argument("--noise-length", type=float,
                    help="the Matern noise's correlation length (default: "
                         "5%% of the domain's diagonal)")
    ap.add_argument("--no-smoothing", action="store_true",
                    help="noisy data enter as they are, unsmoothed")
    ap.add_argument("--no-projection", action="store_true",
                    help="no projection of q onto its bounds")
    ap.add_argument("--leakage", action="store_true",
                    help="a leakage of q toward q_0 for noisy data (biased: "
                         "not supported by the MRAS theory)")
    ap.add_argument("--no-pe", action="store_true",
                    help="skip the persistent-excitation check after each "
                         "example's runs")
    ap.add_argument("--delete-results", action="store_true",
                    help="empty --out (every file and folder in it) before "
                         "running")
    ap.add_argument("--resume", type=Path, metavar="FILE",
                    help="continue the run of a checkpoint (to --tend)")
    ap.add_argument("--movie-from", type=Path, metavar="FILE",
                    help="render the movie of a checkpoint again")
    args = ap.parse_args(argv)

    if args.list:
        print("\n".join(catalog))
        return []
    if args.delete_results and (args.resume or args.movie_from):
        ap.error("--delete-results would delete the checkpoint that "
                 "--resume / --movie-from reads")
    if args.movie_from is not None:
        path = render_movie(args.movie_from, fps=args.fps, every=args.every,
                            workers=args.workers)
        print(f"movie: {path}")
        return []
    if args.resume is not None:
        started = time.perf_counter()
        result = resume(args.resume, tend=args.tend, budget=args.budget,
                        progress=True)
        rows = [row(result.problem, result.noise_level, result,
                    time.perf_counter() - started,
                    args.resume.with_suffix(""))]
        if not args.no_pe:
            excitation(result.problem, rows)
        print(table(rows))
        return [result]

    unknown = sorted(set(args.names) - set(catalog))
    if unknown:
        ap.error(f"unknown example(s) {', '.join(unknown)}; choose from "
                 f"{', '.join(catalog)}")
    if args.delete_results:
        delete_results(args.out)
    overrides = {"h": args.h, "dt": args.dt, "tend": args.tend}
    rows = []
    for name in args.names or catalog:
        runs = []
        cache = (DataCache(args.out / name / f".{name}_data.h5")
                 if len(args.noise_levels) > 1 else None)
        try:
            for noise_level in args.noise_levels:
                stem = args.out / name / f"{name}_noise{noise_level:g}"
                print(f"== {name}, noise level {noise_level:g}", flush=True)
                started = time.perf_counter()
                prob = build(name, **settings(name, args.quick, overrides))
                prob.data_cache = cache
                safeguards(prob, args)
                if args.scheme is not None:
                    prob.scheme = args.scheme
                result = run_live(prob, noise_level, target=args.target,
                                  q0=args.q0, path=stem,
                                  budget=args.budget,
                                  movie=not args.no_movie, fps=args.fps,
                                  every=args.every, workers=args.workers)
                runs.append(row(name, noise_level, result,
                                time.perf_counter() - started, stem))
        finally:
            if cache is not None:
                cache.close()
        if not args.no_pe:
            excitation(name, runs)
        rows += runs
    args.out.mkdir(parents=True, exist_ok=True)
    merge_summary(args.out / "summary.json", rows)
    print(table(rows))
    return rows


def merge_summary(path, rows):
    """Write the runs into the summary, replacing the earlier rows of the
    same example and noise level only: invocations running side by side
    (or one after another) add up."""
    kept = []
    if path.exists():
        new = {(r["example"], r["noise_level"]) for r in rows}
        kept = [r for r in json.loads(path.read_text())
                if (r["example"], r["noise_level"]) not in new]
    path.write_text(json.dumps(kept + rows, indent=1))


if __name__ == "__main__":
    main()
