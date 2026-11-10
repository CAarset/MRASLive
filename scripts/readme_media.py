#!/usr/bin/env python
"""The README's animation: the clean 2D Burgers run, flow beside estimate.

Renders the noise-free run of ``scripts/run_examples.py burgers_2d`` from its
checkpoint as ``docs/media/burgers2d.webp``: the datum's speed |z| beside the
reconstruction qhat (``mras.live.Pair``, ``layout="pair"`` of
``mras.movie.render_movie``): the run up to t = ``UNTIL``, every stored step
(401 frames) at ``FPS`` frames a second, so 6.7 seconds.  The datum's colour
range is that of the frames shown, so it reaches its top by the last one;
the estimate's is the true parameter's.  60 frames a second is the most an
animated image plays at: browsers show a frame of 10 ms or less for 100 ms
(a rule inherited from GIF), and most screens refresh at 60 Hz.

Usage, from the repository's root:

    .venv/bin/python scripts/readme_media.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mras.movie import render_movie  # noqa: E402

CHECKPOINT = ROOT / "results" / "burgers_2d" / "burgers_2d_noise0.h5"
TARGET = ROOT / "docs" / "media" / "burgers2d.webp"
RUN = ".venv/bin/python scripts/run_examples.py burgers_2d --noise-levels 0"
UNTIL = 20.0       # model time the animation ends at
EVERY = 1          # every stored step up to it: 401 frames
FPS = 60
PANEL_WIDTH = 260  # pixels per panel; the channel makes it 418 tall


def main():
    if not CHECKPOINT.exists():
        raise SystemExit(f"missing {CHECKPOINT}; make it, from {ROOT}, with\n"
                         f"    {RUN}")
    path = render_movie(CHECKPOINT, TARGET, fps=FPS, every=EVERY,
                        panel_width=PANEL_WIDTH, layout="pair", until=UNTIL)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
