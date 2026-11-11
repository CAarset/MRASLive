"""The Arrhenius reaction-diffusion pair: recovery and theory witnesses.

Exponential member   Dt u - Laplace u + q(x) exp(u)  = g
Linearised member    Dt u - Laplace u + q(x) (1 + u) = g

The (A2) constant is parameter-monotonicity of the absorption term:
inf r(z) with r = exp or r = 1 + id, witnessed by the weak essential
lower bound of the generated (noise-free) data.  Recovery is measured in
the paper's metric, relative L2 error of the parameter (Section 8
tables).  The estimator flow for this zero-order coupling decays the
error mode-wise at rate ~ r^2 / lambda_k(C), so on the shipped 4 pi
vessel (lambda_1 = 1/8) the long-horizon reference numbers are
h = 0.3, dt = 0.02: error 0.13 at t = 8, 0.081 at t = 28, still
decaying (scripts/demo_arrhenius.py reproduces the curves in a minute
per member at half that horizon).  The recovery tests below run a
cheaper cut of the same design (h = 0.4, tend = 8) and pin the error
below the paper's Darcy Table 6 band (0.33).
"""

import pytest

from mras.live import build

MICRO = dict(h=0.4, dt=0.02, tend=0.4)
RECOVERY = dict(h=0.4, dt=0.05, tend=8.0)  # final errors 0.11, as at dt = 0.02


@pytest.fixture(scope="module")
def exponential():
    prob = build("arrhenius", **RECOVERY)
    return prob, prob.run_level(0.0, progress=False)


@pytest.fixture(scope="module")
def linearised():
    prob = build("arrhenius_linear", **RECOVERY)
    return prob, prob.run_level(0.0, progress=False)


def test_exponential_recovers_rate_field(exponential):
    prob, result = exponential
    final = float(result.error_q[-1])
    assert final < 0.25, (prob.data_range(), result.error_q[-10:].tolist())
    # Strict progress from the zero initial guess over the horizon.
    assert final < 0.5 * result.error_q[0]


def test_linearised_recovers_rate_field(linearised):
    prob, result = linearised
    final = float(result.error_q[-1])
    assert final < 0.25, (prob.data_range(), result.error_q[-10:].tolist())
    assert final < 0.5 * result.error_q[0]


def test_a2_witness_is_positive_and_data_range_mild(exponential):
    prob, _ = exponential
    lo, hi = prob.data_range()
    # The (A2) constant of the exponential member is inf exp(z) = exp(lo);
    # it is positive without any positivity assumption on the datum.  The
    # sampled witness varies with the horizon (0.75 at tend = 1.5, 0.55
    # at tend = 8, 0.70 at tend = 10): the datum snapshots sample the
    # heater phases at fracs of tend, so deep off-phase troughs enter or
    # leave the sample set as the horizon moves.
    assert prob.c_coe() > 0.45, (lo, hi)
    # Mild nonlinearity witness: the truth stays in a range where the
    # exponential factor is bounded, e^hi, supporting comparison-based
    # well-posedness of the data-generating problem.
    assert hi < 3.0, (lo, hi)


def test_a2_witness_linear_member(linearised):
    prob, _ = linearised
    # 1 + weak-ess-inf of the datum: positive at every sampled horizon
    # tested (0.71 at tend = 1.5, 0.32 at tend = 8, 0.60 at tend = 10).
    assert prob.c_coe() > 0.25, prob.data_range()


def test_linearisation_limit_agrees():
    """At small forcing the datum collapses towards 0, where exp(u) = 1 + u
    + O(u^2 / 2): the two members see the same inverse problem, so they
    must reach the same accuracy.  The accuracy itself is poor here by
    design - the estimator gain is the squared reaction amplitude, so the
    persistent-excitation margin vanishes with the forcing (rate ~ r^2,
    module docstring of examples/arrhenius.py).  What is pinned is
    the consistency of the linearisation, not recovery.
    """
    tiny = dict(h=0.35, dt=0.05, tend=1.0, amplitude=0.05)
    res_exp = build("arrhenius", **tiny).run_level(0.0, progress=False)
    res_lin = build("arrhenius_linear", **tiny).run_level(0.0, progress=False)
    err_exp, err_lin = float(res_exp.error_q[-1]), float(res_lin.error_q[-1])
    assert abs(err_exp - err_lin) < 0.05, (err_exp, err_lin)
    assert max(err_exp, err_lin) < 1.0, (err_exp, err_lin)


def test_feedback_decays_from_high_estimate():
    """The parameter flow q_t = r(z) (u - z) is a stable feedback: an
    estimate started above the truth decays towards it, never blows up."""
    prob = build("arrhenius", h=0.4, dt=0.05, tend=1.5)
    prob.initial_guess = lambda: 1.5
    result = prob.run_level(0.0, progress=False)
    assert result.error_q[-1] < 0.95 * result.error_q[0]
    assert result.error_q.max() < 1.25 * result.error_q[0]


def test_registered_as_live_examples():
    from mras.live import list_examples
    assert {"arrhenius", "arrhenius_linear"} <= set(list_examples())
    for name in ("arrhenius", "arrhenius_linear"):
        result = build(name, **MICRO).run_level(0.0, progress=False)
        assert result.error_q is not None
        assert result.error_q[-1] < result.error_q[0]
