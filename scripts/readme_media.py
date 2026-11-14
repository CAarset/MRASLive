#!/usr/bin/env python
"""The README's banner and animation: clean 2D Burgers flow, reconstruction,
truth.

A noise-free run of the ``burgers_2d`` example on a finer mesh than the
paper's (``FINE``; the paper's run is ``h = 0.05``), up to ``UNTIL``, with
every step stored, into a checkpoint of its own under ``results/readme/``
-- some hours, run once; the render reuses it.  The animation is rendered
from it as ``docs/media/burgers2d.webp`` (``layout="side"`` of
``mras.movie.render_movie``, ``mras.live.SideBySide``): the flow's speed
|z|, the reconstruction qhat and the true parameter q side by side, every
stored step at ``FPS`` frames a second.  The flow's speed is on a
logarithmic scale over ``DECADES`` decades, its top the largest speed of the
frames shown, so it reaches its top by the last one: the inflow ramps up
from rest, and on a linear scale its first hundredth -- already enough to
move qhat near the inlet -- would not show.  q and qhat are on [0.5, 1],
the range of q.  The banner, ``docs/media/banner.webp``, is the same movie
smaller (``BANNER_WIDTH`` per panel), with short labels and no time stamp,
beside the name and a line on what the package does (``HEADING``).

``FPS``: an animated image's frame lasts a whole number of milliseconds,
and browsers show a frame of 10 ms or less for 100 ms (a rule inherited
from GIF), so 90 frames a second (11 ms) is near the most it plays at;
screens refreshing at 60 Hz show 60 of them.

``--zero-slip`` does the same for ``burgers_2d_zero_slip``, the barriers
no-slip instead of slip (``examples/burgers_2d.py``), into checkpoint and
movie of their own (``..._zero_slip``, ``burgers2d_zero_slip.webp``) and
without the banner; ``--until`` sets the model time to end at.

Usage, from the repository's root:

    .venv/bin/python scripts/readme_media.py            # run (once), render
    .venv/bin/python scripts/readme_media.py --render   # render only
    .venv/bin/python scripts/readme_media.py --zero-slip --until 30
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mras.checkpoint import load, resume  # noqa: E402
from mras.live import build, run_live  # noqa: E402
from mras.movie import render_movie  # noqa: E402

UNTIL = 25.0                         # model time the run and animation end at
FINE = dict(h=0.025, dt=0.025)       # the run's mesh width and time step
STEM = ROOT / "results" / "readme" / "burgers_2d_fine_noise0"
TARGET = ROOT / "docs" / "media" / "burgers2d.webp"
BANNER = ROOT / "docs" / "media" / "banner.webp"
FPS = 90
PANEL_WIDTH = 200                    # pixels per panel; 312 tall, 672 in all
TITLES = ("Burgers flow over\nan unknown material",
          r"parameter reconstruction $\hat q$",
          r"true parameter $q^\dagger$")
Q_RANGE = (0.5, 1.0)
DECADES = 3
NCOLORS = 256  # continuous: qhat_0 = 0.75 sits on a band edge of 8 bands
BANNER_WIDTH = 130                   # pixels per panel; 862 x 239 in all
BANNER_TITLES = ("flow", r"reconstruction $\hat q$", r"truth $q^\dagger$")
HEADING = ("MRASLive", "online parameter identification\nfor time-dependent PDEs",
           "NGSolve · model reference adaptation · live")


def simulate(name, stem, until):
    """The noise-free run of example ``name``, every step stored, to
    ``until`` whatever its error; an interrupted one continues from its
    restart point."""
    path = stem.with_suffix(".h5")
    if path.exists():
        if bool(load(path).attrs["completed"]):
            return
        problem = build(name, tend=until, **FINE)
        problem.target = 0.0
        resume(path, tend=until, problem=problem, budget=None)
        return
    stem.parent.mkdir(parents=True, exist_ok=True)
    run_live(name, 0.0, tend=until, target=0.0, path=stem,
             budget=None, movie=False, **FINE)


def render(stem, until, target, banner=True):
    scale = {"q_range": Q_RANGE, "log_decades": DECADES, "ncolors": NCOLORS}
    movies = [(target, PANEL_WIDTH, {"titles": TITLES})]
    if banner:
        movies.append((BANNER, BANNER_WIDTH,
                       {"titles": BANNER_TITLES, "title_size": 10,
                        "stamp": False, "heading": HEADING}))
    for path, width, options in movies:
        path = render_movie(stem.with_suffix(".h5"), path, fps=FPS, every=1,
                            panel_width=width, layout="side", until=until,
                            layout_options={**scale, **options})
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--render", action="store_true",
                        help="render from the existing checkpoint only")
    parser.add_argument("--zero-slip", action="store_true",
                        help="no-slip barriers (burgers_2d_zero_slip), into "
                             "a checkpoint and movie of their own, no banner")
    parser.add_argument("--until", type=float, default=UNTIL,
                        help=f"model time to end at (default {UNTIL:g})")
    args = parser.parse_args()
    if args.zero_slip:
        name = "burgers_2d_zero_slip"
        stem = STEM.with_name(STEM.name + "_zero_slip")
        target = TARGET.with_name("burgers2d_zero_slip.webp")
    else:
        name, stem, target = "burgers_2d", STEM, TARGET
    if not args.render:
        simulate(name, stem, args.until)
    render(stem, args.until, target, banner=not args.zero_slip)


if __name__ == "__main__":
    main()
