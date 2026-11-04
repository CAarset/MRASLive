"""Smoke tests: perforated pipe domain and the fully generic Burgers example."""

import importlib.util
import pathlib

import numpy as np
import pytest

from mras.config import load_yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_example():
    path = ROOT / "examples" / "burgers_1d.py"
    spec = importlib.util.spec_from_file_location("burgers_1d_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipe_domain_has_named_boundaries():
    from mras.domains import burgers_pipe
    doms = burgers_pipe(maxh=0.3)
    assert {"inlet", "outlet", "wall", "hole"} <= set(
        str(b) for b in doms.mesh.GetBoundaries())
    assert doms.Ufes.ndof > 0


def test_burgers_forward_generation_and_tracking():
    mod = _load_example()
    # Long enough for the heaters, which circle the holes, to reach the
    # pipe between them: the estimate drops by 14 % by t = 2.
    prob = mod.Burgers1D(h=0.35, dt=0.02, tend=2.0, k_u=2, seed=1)
    clean, noisy = prob.run([0.0, 0.1], progress=False)
    assert len(clean.misfit_u) == len(clean.error_u) - 1
    for res in (clean, noisy):
        # No data and no truth hooks: the driver generated the observations.
        assert res.error_u is not None and res.error_q is not None
        assert res.error_u[0] == 0.0
        # The parameter starts at the zero initial guess, so its relative
        # reconstruction error at t=0 is ||0 - q|| / ||q|| = 1 -- to
        # round-off, which threaded integration sums in a varying order.
        assert res.error_q[0] == pytest.approx(1.0, rel=1e-14)
        assert np.all(np.isfinite(res.error_u)) and np.all(np.isfinite(res.error_q))
        assert res.error_u[-1] < 0.3  # state tracks within O(dt) of the reference
        assert res.error_q[-1] < res.error_q[0]  # the estimate moves toward q
        assert 0.0 < res.misfit_u[-1] < 0.5
    # On clean data by 14 %; at a noise level of 10 % the heaters, which
    # drive the pipe away from the holes weakly at first, barely move it
    # yet (5 % by t = 2, and no further by t = 4 on this coarse mesh).
    assert clean.error_q[-1] < 0.9 * clean.error_q[0]


def test_burgers_config_yaml():
    defaults, runs = load_yaml(ROOT / "configs" / "burgers.yaml")
    assert len(runs) >= 2
    assert defaults["dt"] > 0 and defaults["tend"] > 0
    assert defaults["noise_levels"]
    assert all(r["h"] > 0 and r["seed"] == 0 for r in runs)


def test_light_does_not_pass_the_holes():
    mod = _load_example()
    from mras.domains import burgers_pipe
    mesh = burgers_pipe(maxh=0.3).mesh
    left, middle = mod.HOLES[0], mod.HOLES[1]
    mid = mod.HEIGHT / 2

    def lit(px, py):
        return mod._lit(0, mod.x - left, mod.y - mid)(mesh(px, py))

    assert lit(middle + 0.3, mid) == 0.0          # behind the middle hole
    assert lit(middle + 0.3, mid + 0.4) == 1.0    # above its shadow
    assert lit(middle - 0.3, mid) == 1.0          # in front of it
    assert lit(left - 0.4, mid) == 1.0            # the other way
