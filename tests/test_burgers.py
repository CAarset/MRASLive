"""Smoke tests: perforated pipe domain and the fully generic Burgers example."""

import importlib.util
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_example():
    path = ROOT / "examples" / "burgers_pipe.py"
    spec = importlib.util.spec_from_file_location("burgers_pipe_example", path)
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
    prob = mod.BurgersPipe(h=0.35, dt=0.02, tend=0.6, k_u=2, seed=1)
    clean, noisy = prob.run([0.0, 0.1], progress=False)
    assert len(clean.misfit_u) == len(clean.error_u) - 1
    for res in (clean, noisy):
        # No data and no truth hooks: the driver generated the observations.
        assert res.error_u is not None and res.error_q is not None
        assert res.error_u[0] == 0.0
        assert np.all(np.isfinite(res.error_u)) and np.all(np.isfinite(res.error_q))
        assert res.error_u[-1] < 0.3  # state tracks within O(dt) of the reference
        assert res.error_q[-1] < 0.95 * res.error_q[0]  # sustained excitation
        assert 0.0 < res.misfit_u[-1] < 0.5


def test_burgers_config_yaml():
    from mras.config import load_yaml
    cfgs = load_yaml(ROOT / "configs" / "burgers.yaml")
    assert len(cfgs) >= 2
    assert all(c.h > 0 and c.dt > 0 and c.levels for c in cfgs)
