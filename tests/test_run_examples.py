"""The example runner's bookkeeping (scripts/run_examples.py)."""

from pathlib import Path
from types import SimpleNamespace

from scripts.run_examples import row


def test_a_row_names_the_files_of_its_own_noise_level():
    """The stem of noise level 0.01 ends in ".01", which Path.with_suffix
    took for a suffix: the row named the noise-free run's files."""
    result = SimpleNamespace(final_error_q=0.1, final_error_u=0.01,
                             meta={"steps": 3, "scheme": "implicit"})
    stem = Path("results/burgers_1d/burgers_1d_noise0.01")
    r = row("burgers_1d", 0.01, result, 1.0, stem)
    assert r["checkpoint"] == "results/burgers_1d/burgers_1d_noise0.01.h5"
    assert r["movie"] == "results/burgers_1d/burgers_1d_noise0.01.mp4"
