"""Movies of a run, rendered from its checkpoint.

A run writes its trajectory to its checkpoint (:mod:`mras.checkpoint`) and the
movie is rendered from that afterwards, the way movies of simulations are made
from stored time series.  Every stored step becomes one frame, so the movie is
uniform in model time and as long as the stored trajectory, up to
``MAX_FPS * DURATION[1]`` frames (beyond that, every ``every``-th stored step,
evenly), and it lasts between ``DURATION[0]`` and ``DURATION[1]`` seconds:
long runs play ten seconds at up to 120 frames a second, short ones four
(:func:`pace`).  ``fps`` and ``every`` override the pace.

Rendering is safe, and as parallel as the machine.  The run is on disk before
a frame is painted, so a failure costs a re-render, never the run.  Worker
processes paint the frames in batches, and the frames stream in order through
a pipe into ffmpeg (H.264): neither memory nor disk holds more than a batch
and the movie.  The colour ranges' moving averages run forward in time, so a
first, cheap pass reads every frame's extremes, the averages are taken in
order, and the second pass paints each frame with its own range -- the
frames the live view would have painted, had it painted every step.

    from mras.movie import render_movie
    render_movie("results/burgers_1d/burgers_1d_noise0.h5")

``layout`` picks the picture: ``"live"``, the live view's full figure
(:class:`mras.live.Panels`), or ``"pair"``, the datum beside the estimate
on fixed colour ranges (:class:`mras.live.Pair`).  ``until`` ends a movie
at that model time, its frames and colour ranges from the stored steps up
to it.  The file's suffix picks the format: ``.mp4`` (H.264), or ``.webp``
(animated WebP, which web pages and READMEs show inline), or ``.gif``.
The WebP is lossless: lossy, its encoder re-codes only the 16-pixel blocks
that change, and the seams between re-coded and kept blocks pile up into
streaks over hundreds of frames; lossless, it is exact, and these
pictures' flat colour fields keep it small.
"""

import math
import os
import subprocess
import warnings
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import matplotlib as mpl
import numpy as np

MAX_FPS = 120.0          # the smoothest a movie plays
DURATION = (4.0, 10.0)   # seconds a movie lasts, at least and at most
PACE = 30.0              # frames a second, where neither bound binds

_WORKER = {}  # the stored run and its figure, in each worker process


def pace(stored, fps=None, every=None):
    """``(every, fps)`` for a movie of ``stored`` stored steps.

    Every stored step is a frame, up to ``MAX_FPS * DURATION[1]`` frames
    (every ``every``-th beyond that), and the movie lasts ``frames / PACE``
    seconds, but at least ``DURATION[0]`` and at most ``DURATION[1]``;
    ``fps`` and ``every`` override either choice.
    """
    if every is None:
        every = max(1, math.ceil(stored / (MAX_FPS * DURATION[1])))
    frames = len(_frames(stored, every))
    if fps is None:
        seconds = min(max(frames / PACE, DURATION[0]), DURATION[1])
        fps = round(frames / seconds, 3)
    return every, fps


def _frames(stored, every):
    frames = list(range(0, stored, every))
    if frames[-1] != stored - 1:
        frames.append(stored - 1)  # the movie reaches the end
    return frames


def _horizon(stored):
    """The stored run's length: a movie of a finished run knows its end."""
    return float(stored.times[-1]) - stored.problem.tstart


def _layout(name):
    from .live import Pair, Panels
    layouts = {"live": Panels, "pair": Pair}
    if name not in layouts:
        raise ValueError(f"layout is one of {sorted(layouts)}, not {name!r}")
    return layouts[name]


def _setup(path, panel_width, problem=None, layout="live", horizon=None):
    from .checkpoint import load
    stored = load(path, problem)
    _WORKER["stored"] = stored
    _WORKER["panels"] = _layout(layout)(
        stored.problem, panel_width=panel_width,
        horizon=_horizon(stored) if horizon is None else horizon)
    _WORKER["history"] = (stored.history.get("error_q"),
                          stored.history.get("misfit_u"))


def _init(path, panel_width, layout, horizon):
    import ngsolve
    ngsolve.SetNumThreads(1)  # one process per core does the parallelism
    _setup(path, panel_width, layout=layout, horizon=horizon)


def _fill(k):
    stored, panels = _WORKER["stored"], _WORKER["panels"]
    snap = stored.snapshot(k)
    panels.fill(snap["t"], snap["u"], snap["z"], snap["q"])
    return snap


def _extremes(k):
    _fill(k)
    return _WORKER["panels"].extremes()[1]


def _paint(task):
    k, ranges = task
    snap = _fill(k)
    error_q, misfit = _WORKER["history"]
    step = int(snap["step"])
    if error_q is not None:
        history, offset = error_q[:step + 1], 0
    else:
        history, offset = misfit[:step], 1
    panels = _WORKER["panels"]
    return panels.paint(panels.job(float(snap["t"]), history, offset,
                                   ranges=ranges))


def _ranges(panels, times, seen):
    """The colour ranges of every frame: a layout's own (``frame_ranges``),
    or the moving ranges, taken in order."""
    if hasattr(panels, "frame_ranges"):
        return panels.frame_ranges(times, seen)
    tracks = {kind: track.fresh() for kind, track in panels._ranges.items()}
    out = []
    for t, extremes in zip(times, seen):
        for kind, (lo, hi) in extremes.items():
            tracks[kind].add(t, lo, hi)
        out.append({kind: track.kwargs() for kind, track in tracks.items()})
    return out


def render_movie(checkpoint, path=None, fps=None, every=None, workers=None,
                 panel_width=400, problem=None, layout="live", until=None):
    """Render the movie of a checkpointed run; return its path (the
    checkpoint's, as MP4, unless given; the suffix picks the format).

    ``every`` keeps every ``every``-th stored step and ``fps`` is the
    playback rate, both set by :func:`pace` unless given.  ``workers``
    processes paint (half the cores unless given; 1 paints in this
    process).  ``problem`` is needed only when the file cannot build the
    problem again, and then the frames are painted here.  ``layout`` is
    ``"live"`` or ``"pair"``; ``until`` ends the movie at that model time
    (the module docstring).
    """
    from .checkpoint import load
    checkpoint = Path(checkpoint)
    path = Path(path) if path is not None else checkpoint.with_suffix(".mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = load(checkpoint, problem)
    count, horizon = len(stored.steps), _horizon(stored)
    if until is not None:  # the stored steps up to ``until``, and no further
        count = int(np.searchsorted(np.asarray(stored.times),
                                    until + 1e-9 * stored.problem.dt,
                                    side="right"))
        if count == 0:
            raise ValueError(f"{checkpoint}: nothing stored up to t = {until}")
        horizon = float(stored.times[count - 1]) - stored.problem.tstart
    every, fps = pace(count, fps, every)
    frames = _frames(count, every)
    times = [float(stored.times[k]) for k in frames]
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) // 2)
    if problem is None and not stored.attrs["arguments"]:
        raise ValueError(f"{checkpoint}: pass the problem to render it")
    if problem is not None and not stored.attrs["arguments"]:
        workers = 1  # the workers could not build it again

    _layout(layout)  # an unknown layout fails before ffmpeg starts
    encoder = subprocess.Popen(_encoder(path, fps), stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    try:
        parallel = workers > 1
        if parallel:
            try:
                _render_parallel(checkpoint, frames, times, workers,
                                 panel_width, problem or stored.problem,
                                 encoder, horizon, layout)
            except BrokenProcessPool:
                # A worker that cannot start (say, under a parent run from
                # stdin, whose main module a spawned process cannot import)
                # breaks the pool before a frame is written: paint here.
                warnings.warn("the movie's worker processes failed to start; "
                              "painting in this process", stacklevel=2)
                parallel = False
        if not parallel:
            _setup(checkpoint, panel_width, problem or stored.problem,
                   layout, horizon)
            seen = [_extremes(k) for k in frames]
            for task in zip(frames, _ranges(_WORKER["panels"], times, seen)):
                encoder.stdin.write(_paint(task))
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {encoder.stderr.read().decode()}")
    except BaseException:
        encoder.kill()
        raise
    finally:
        _WORKER.clear()
    return path


def _encoder(path, fps):
    """The ffmpeg command that reads PNG frames from stdin and writes
    ``path`` in the format of its suffix."""
    head = [mpl.rcParams["animation.ffmpeg_path"], "-y", "-loglevel", "error",
            "-f", "image2pipe", "-framerate", f"{fps:g}", "-c:v", "png",
            "-i", "-"]
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        tail = ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                "-movflags", "+faststart"]
    elif suffix == ".webp":  # lossless: see the module docstring
        tail = ["-c:v", "libwebp_anim", "-lossless", "1",
                "-compression_level", "6", "-loop", "0"]
    elif suffix == ".gif":
        tail = ["-vf", "split[a][b];[a]palettegen=stats_mode=diff[p];"
                "[b][p]paletteuse=dither=sierra2_4a", "-loop", "0"]
    else:
        raise ValueError(f"{path}: a movie is .mp4, .webp or .gif")
    return head + tail + [str(path)]


def _render_parallel(checkpoint, frames, times, workers, panel_width, problem,
                     encoder, horizon, layout):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    # One thread per worker: numpy's BLAS and NGSolve would each take every
    # core otherwise.  The spawned workers inherit the environment.
    saved = {key: os.environ.get(key) for key in
             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ.update({key: "1" for key in saved})
    try:
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                   initializer=_init,
                                   initargs=(str(checkpoint), panel_width,
                                             layout, horizon))
        with pool:
            chunk = max(1, len(frames) // (8 * workers))
            seen = list(pool.map(_extremes, frames, chunksize=chunk))
            ranges = _ranges(_layout(layout)(problem,
                                             panel_width=panel_width,
                                             horizon=horizon), times, seen)
            batch = 4 * workers  # what memory holds at once
            for start in range(0, len(frames), batch):
                tasks = list(zip(frames[start:start + batch],
                                 ranges[start:start + batch]))
                for png in pool.map(_paint, tasks):
                    encoder.stdin.write(png)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
