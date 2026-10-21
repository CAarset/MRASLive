"""Adjoint-pairing verification, every example: <A x, y> = <x, A y>.

The harness (mras/adjoint.py) assembles the linear joint step operator A
from problem.forms(state) and randomises the pairing.  For the problems
whose parameter row is the exact adjoint coupling of the state row — the
sigma = 0 family (Darcy, Fisher, Burgers pipe) and the Arrhenius pair —
the aligned (z_old := z) mismatch must sit at machine precision.

Two committed problems deliberately do NOT reach machine precision, and
the tests pin down that status quo rather than paper over it:

* potential / allen_cahn reproduce the paper's discrete multiplier at the
  linearisation point c~ = 0 verbatim (DEVIATIONS.md), while their state
  row carries the Picard coefficient (1 + |q^n|^{2/3}) * z (and z^3 in
  Allen-Cahn) at the current estimate: the cross blocks differ by
  O(dt * z * |qhat|^{2/3}) and O(dt * z * (z^2 - 1)), respectively;
* the Helmholtz demo identifies a parameter that is absent from the truth
  model and drives it with the quadratic self-step, so no adjoint pair
  exists by construction.
"""

import importlib.util
import pathlib

import pytest

from mras.adjoint import adjoint_mismatch
from mras.problems import (AllenCahnProblem, DarcyProblem, FisherProblem,
                           PotentialProblem)
from mras.problems.arrhenius import ArrheniusLinearProblem, ArrheniusProblem

ROOT = pathlib.Path(__file__).resolve().parents[1]

BUILDERS = {
    "darcy": lambda: DarcyProblem(h=0.15, dt=0.05, tend=0.5,
                                  data_h=0.08, data_order=2),
    "fisher": lambda: FisherProblem(h=0.2, dt=0.05, tend=0.5,
                                    data_h=0.15, data_order=2),
    "burgers": lambda: _example("burgers_pipe").BurgersPipe(
        h=0.35, dt=0.05, tend=0.5),
    "arrhenius": lambda: ArrheniusProblem(h=0.4, dt=0.05, tend=0.5,
                                          data_h=0.2),
    "arrhenius_linear": lambda: ArrheniusLinearProblem(
        h=0.4, dt=0.05, tend=0.5, data_h=0.2),
    "potential": lambda: PotentialProblem(h=0.25, dt=0.05, tend=0.5),
    "allen_cahn": lambda: AllenCahnProblem(h=0.25, dt=0.05, tend=0.5),
}


def _example(stem):
    path = ROOT / "examples" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"{stem}_adjoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_adjoinable(name):
    problem = BUILDERS[name]()
    result = problem.run_level(0.0, progress=False)
    return adjoint_mismatch(problem, result.state)


@pytest.mark.parametrize("name", [
    "darcy", "fisher", "burgers", "arrhenius", "arrhenius_linear"])
def test_exact_adjoint_coupling_is_machine_precise(name):
    report = _run_adjoinable(name)
    aligned = report["aligned"]
    assert not aligned["vacuous"], (name, aligned)
    assert aligned["max_ratio"] < 1e-10, (name, report)
    assert aligned["max_ratio_full"] < 1e-10, (name, report)
    # The untouched step form of the last assimilation step: the driver
    # stores z_old := z after every step, so the final forms the probe
    # reads are aligned by construction (stationary Darcy trivially so).
    assert report["lagged"]["max_ratio"] < 1e-10, (name, report)


@pytest.mark.parametrize("name", ["potential", "allen_cahn"])
def test_paper_literal_multiplier_deviates_as_documented(name):
    """Deviation from machine precision is structural (c~ = 0 linearisation
    point of the paper's discrete rows), not a bug and not the z_old lag."""
    report = _run_adjoinable(name)
    assert not report["aligned"]["vacuous"]
    assert report["aligned"]["max_ratio"] > 1e-3, report
    # Structural, not lag-induced: aligned and lagged see the same scale.
    ratio = report["lagged"]["max_ratio"] / report["aligned"]["max_ratio"]
    assert 0.2 < ratio < 5.0, report


def test_helmholtz_self_step_has_no_adjoint_pair():
    """The Helmholtz demo is outside the abstract scheme: its parameter
    appears nowhere in the truth model and its step is the quadratic
    self-step, so the harness must flag the cross blocks as a pair that
    fails (not as a vacuous pass)."""
    problem = _example("helmholtz_in_parameter").HelmholtzProblem(
        h=0.2, tend=0.3, tsteps=12)
    result = problem.run_level(0.0, progress=False)
    report = adjoint_mismatch(problem, result.state)
    assert not report["aligned"]["vacuous"]
    assert report["aligned"]["max_ratio"] > 1e-3, report
