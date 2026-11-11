#!/usr/bin/env python
"""The README's animation: clean 2D Burgers flow, reconstruction, truth.

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
the range of q.

``FPS``: an animated image's frame lasts a whole number of milliseconds,
and browsers show a frame of 10 ms or less for 100 ms (a rule inherited
from GIF), so 90 frames a second (11 ms) is near the most it plays at;
screens refreshing at 60 Hz show 60 of them.

Usage, from the repository's root:

    .venv/bin/python scripts/readme_media.py            # run (once), render
    .venv/bin/python scripts/readme_media.py --render   # render only
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
FPS = 90
PANEL_WIDTH = 260                    # pixels per panel; 418 tall
TITLES = ("Burgers flow over\nan unknown material",
          r"parameter reconstruction $\hat q$",
          r"true parameter $q^\dagger$")
Q_RANGE = (0.5, 1.0)
DECADES = 3
NCOLORS = 256  # continuous: qhat_0 = 0.75 sits on a band edge of 8 bands


def simulate():
    """The noise-free run, every step stored, to UNTIL whatever its error;
    an interrupted one continues from its restart point."""
    path = STEM.with_suffix(".h5")
    if path.exists():
        if bool(load(path).attrs["completed"]):
            return
        problem = build("burgers_2d", tend=UNTIL, **FINE)
        problem.target = 0.0
        resume(path, tend=UNTIL, problem=problem, budget=None)
        return
    STEM.parent.mkdir(parents=True, exist_ok=True)
    run_live("burgers_2d", 0.0, tend=UNTIL, target=0.0, path=STEM,
             budget=None, movie=False, **FINE)


def render():
    path = render_movie(STEM.with_suffix(".h5"), TARGET, fps=FPS, every=1,
                        panel_width=PANEL_WIDTH, layout="side", until=UNTIL,
                        layout_options={"titles": TITLES, "q_range": Q_RANGE,
                                        "log_decades": DECADES,
                                        "ncolors": NCOLORS})
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--render", action="store_true",
                        help="render from the existing checkpoint only")
    args = parser.parse_args()
    if not args.render:
        simulate()
    render()


if __name__ == "__main__":
    main()
