"""Tests for the generic driver: data paths, schemes, solves, forward model."""

import numpy as np
import pytest
from ngsolve import (BilinearForm, GridFunction, LinearForm, dx, exp, grad,
                     pi, sin, x, y)

from mras.domains import darcy
from mras.scheme import SCHEMES, MRASProblem, TheorySpecError

BUMP = exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05)


class _Darcy(MRASProblem):
    """f(q, u) = -div(q grad u): the coupling every problem here shares."""

    sigma = 0.0

    def coupling(self, q, u, w):
        return q * grad(u) * grad(w)

    def coupling_adjoint(self, q, u, w):
        return grad(u) * grad(w)


class ForwardProblem(_Darcy):
    """u_t - div(q_true grad u) = g: no data, no truth -- the driver
    generates the observations by stepping the same physics at q_true."""

    name = "forward_gen"

    def __init__(self, h=0.2, dt=2e-2, tsteps=5, seed=0):
        super().__init__(darcy(maxh=h, order_u=2, order_q=0),
                         dt=dt, tend=dt * tsteps, k_u=2, seed=seed)
        self.u0 = BUMP
        self.q_true = 0.5 + 0.25 * (x + y)

    def source(self, t):
        return BUMP * (1.0 + 0.2 * t)


class ObservationProblem(_Darcy):
    """Known data supplied directly, ground truth unknown: misfit only."""

    name = "obs_only"

    def __init__(self, h=0.2, dt=2e-2, tsteps=4, seed=0):
        super().__init__(darcy(maxh=h, order_u=2, order_q=0),
                         dt=dt, tend=dt * tsteps, k_u=2, seed=seed)
        self.u0 = BUMP

    def observations(self, state):
        return BUMP * (1.0 + 0.1 * state.t)


class TruthProblem(_Darcy):
    """Ground truth supplied analytically: synthetic noisy data + errors."""

    name = "truth_gen"

    def __init__(self, h=0.2, dt=2e-2, tsteps=4, seed=0):
        super().__init__(darcy(maxh=h, order_u=2, order_q=0),
                         dt=dt, tend=dt * tsteps, k_u=2, seed=seed)
        self.q_true = 0.5 + 0.25 * (x + y)

    def truth(self, t):
        return BUMP * (1.0 + 0.1 * t)


class FixedPointProblem(ForwardProblem):
    """Exact data on the estimator's own mesh, started at the truth.

    The paper's MRAS (4) reduces to the model when z = u and q = q_true, so
    every discrete step must too: the forward step and the state row
    coincide at u^n = z^n, q = q_true exactly when both use g^{n+1}.
    """

    name = "fixed_point"

    def __init__(self):
        super().__init__(h=0.3, dt=0.05, tsteps=12)
        self.q_true = 0.7  # in Q_h exactly, so the start can be the truth

    def initial_guess(self):
        return self.q_true

    def source(self, t):
        return BUMP * sin(5 * t)


def test_forward_generation_produces_errors():
    prob = ForwardProblem()
    res = prob.run([0.0], progress=False)[0]
    assert res.error_u is not None and res.error_q is not None
    assert res.error_u[0] == pytest.approx(0.0)
    assert len(res.error_u) == len(res.misfit_u) + 1  # t=0 seeding
    assert np.all(np.isfinite(res.error_u))
    # Exact data: the estimator stays close to the truth and to the data.
    assert res.error_u[-1] < 0.5
    assert res.misfit_u[-1] < 0.2
    # The generated data is time-dependent, not the initial state.
    u0 = GridFunction(prob.Ufes)
    u0.Set(prob.u0)
    ref = u0.vec.FV().NumPy().copy()
    now = np.asarray(res.state.u_true.vec.FV().NumPy())
    assert np.linalg.norm(now - ref) > 1e-3 * np.linalg.norm(ref)


def test_forward_generation_is_seeded_and_independent():
    prob = ForwardProblem()
    r1 = prob.run([0.05, 0.0], progress=False)
    r2 = ForwardProblem().run([0.05, 0.0], progress=False)
    for a, b in zip(r1, r2):
        np.testing.assert_allclose(a.error_u, b.error_u)
    # A fresh, exact-data level starts from zero error.
    assert r1[1].error_u[0] == pytest.approx(0.0)


def test_data_cache_replays_the_noise_free_data(tmp_path):
    """The noise levels of one example share their noise-free data: a
    later run reads the first run's back instead of stepping the forward
    model again, with the result it would have had without the cache."""
    from mras.forward import DataCache
    plain = ForwardProblem().run_level(0.05, progress=False)
    cache = DataCache(tmp_path / "data.h5")
    first = ForwardProblem()
    first.data_cache = cache
    first.run_level(0.0, progress=False)
    stepped = cache.forward.steps
    second = ForwardProblem()
    second.data_cache = cache
    cached = second.run_level(0.05, progress=False)
    assert cache.forward.steps == stepped  # nothing generated again
    np.testing.assert_allclose(cached.error_u, plain.error_u, rtol=1e-8)
    np.testing.assert_allclose(cached.error_q, plain.error_q, rtol=1e-8)
    cache.close()
    assert not (tmp_path / "data.h5").exists()


def test_observations_only_reports_misfit_without_errors():
    prob = ObservationProblem()
    res = prob.run([0.05], progress=False)[0]
    assert res.error_u is None and res.error_q is None
    assert res.misfit_u is not None
    assert len(res.misfit_u) == prob.tsteps  # one datum per step, to tend
    assert np.all(np.isfinite(res.misfit_u))


def test_truth_hook_adds_seeded_noise_and_errors():
    res = TruthProblem().run([0.0, 0.1], progress=False)
    for r in res:
        assert r.error_u[0] == pytest.approx(0.0)
        assert np.all(np.isfinite(r.error_u))
    assert res[1].error_u[-1] > 0.0


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("split", [True, False])
def test_exact_data_are_a_fixed_point_of_every_step(scheme, split):
    prob = FixedPointProblem()
    prob.scheme, prob.split = scheme, split
    res = prob.run_level(0.0, progress=False)
    # Evaluating g at t^n instead leaves dt * g_t in every step: an error
    # of order 1e-2 here, against round-off when aligned.
    assert np.max(res.error_u) < 1e-8
    assert np.max(res.error_q) < 1e-8


@pytest.mark.parametrize("scheme", ["implicit", "semi-implicit"])
def test_solving_by_rows_matches_the_joint_solve(scheme):
    runs = []
    for split in (True, False):
        prob = ForwardProblem(tsteps=8)
        prob.initial_guess = lambda: 0.2
        prob.scheme, prob.split = scheme, split
        runs.append(prob.run_level(0.0, progress=False))
    np.testing.assert_allclose(runs[0].error_q, runs[1].error_q, atol=1e-12)
    np.testing.assert_allclose(runs[0].error_u, runs[1].error_u, atol=1e-12)


def test_the_schemes_differ_only_where_they_should():
    errors = {}
    for scheme in SCHEMES:
        prob = ForwardProblem(tsteps=8)
        prob.initial_guess = lambda: 0.2
        prob.scheme = scheme
        errors[scheme] = prob.run_level(0.0, progress=False).error_q
    # sigma = 0: explicit and semi-implicit are the same scheme ...
    np.testing.assert_allclose(errors["explicit"], errors["semi-implicit"],
                               atol=1e-13)
    # ... and backward Euler is a different one.
    assert np.max(np.abs(errors["implicit"] - errors["semi-implicit"])) > 1e-6


class CustomFormsProblem(ForwardProblem):
    """A problem with a joint step form of its own: the paper's Darcy forms,
    block triangular, which "auto" must find and solve by rows."""

    name = "custom_forms"

    def initial_guess(self):
        return 0.2

    def forms(self, state):
        dt, z = self.dt, state.z
        a = BilinearForm(self.X)
        res = LinearForm(self.X)
        a += self.u * self.v * dx + dt * grad(self.u) * grad(self.v) * dx
        a += dt * self.q * grad(z) * grad(self.v) * dx
        a += self.q * self.s * dx
        res += state.gfu * self.v * dx + dt * state.g * self.v * dx
        res += dt * grad(z) * grad(self.v) * dx
        res += state.gfq * self.s * dx
        res += dt * grad(state.z_old) * (grad(state.gfu)
                                         - grad(state.z_old)) * self.s * dx
        return a, res


def test_a_custom_block_triangular_form_is_found_and_solved_by_rows():
    runs = {}
    for split in ("auto", False):
        prob = CustomFormsProblem(tsteps=6)
        prob.split = split
        runs[split] = prob.run_level(0.0, progress=False)
        if split == "auto":
            assert prob._rows_apart is True
    np.testing.assert_allclose(runs["auto"].error_q, runs[False].error_q,
                               atol=1e-12)


class ReactionForward(ForwardProblem):
    """A model with a linear and a nonlinear term, for the forward model."""

    name = "reaction_forward"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Zero on the wall like the model's Dirichlet data: forward Euler
        # imposes them after a step, backward Euler before it.
        self.u0 = sin(pi * x) * sin(pi * y)

    def model(self, u, w):
        return [u * w, -u * u * w]


def test_forward_model_assembles_its_linear_terms_once():
    from mras.forward import ForwardRun
    linear = ForwardRun(ForwardProblem())
    assert linear.nonlinear is None  # one direct solve per step
    nonlinear = ForwardRun(ReactionForward())
    # The mass, the coupling and u*w sit in the constant matrix; only
    # -u^2 is reassembled, by Newton's method.
    assert len(list(nonlinear.nonlinear.integrators)) == 1


@pytest.mark.parametrize("scheme", SCHEMES)
def test_forward_schemes_step_the_same_model(scheme):
    # dt small enough for forward Euler, which is only conditionally stable
    # on a parabolic model (it is selectable, not recommended).
    runs = {}
    for name in (scheme, "implicit"):
        prob = ReactionForward(dt=1e-5, tsteps=4)
        prob.forward_scheme = name
        runs[name] = prob._forward_data()
    assert np.all(np.isfinite(runs[scheme]))
    gap = (np.abs(runs[scheme] - runs["implicit"]).max()
           / np.abs(runs["implicit"]).max())
    # First-order schemes a few steps of size 1e-5 apart, never more.
    assert gap < (1e-12 if scheme == "implicit" else 1e-3)


class BoundaryDataProblem(ForwardProblem):
    """Non-homogeneous, time-dependent Dirichlet data on the whole wall."""

    name = "boundary_data"

    def dirichlet(self, t):
        return {".*": (1 + t) * (x + y)}

    def initial_guess(self):
        return 0.2


@pytest.mark.parametrize("scheme", SCHEMES)
def test_dirichlet_data_reach_both_models(scheme):
    from ngsolve import BND, Integrate
    prob = BoundaryDataProblem(tsteps=6)
    prob.u0 = x + y  # consistent with the boundary data at t = 0
    prob.scheme = scheme
    res = prob.run_level(0.0, progress=False)
    t = res.state.t.Get()
    on_wall = (1 + t) * (x + y)
    for field in (res.state.u_true, res.state.gfu):
        gap = Integrate((field - on_wall) ** 2, prob.mesh, BND) ** 0.5
        assert gap < 1e-10
    assert res.error_u[-1] < 0.1 * Integrate(res.state.u_true ** 2,
                                             prob.mesh) ** 0.5


def test_boundary_data_are_imposed_by_boundary_name():
    # 2D Burgers is driven by its inflow alone: no source, the inflow
    # imposed on the inlet as Dirichlet data, the barriers no-slip -- in the
    # estimator's state and in the datum alike; the walls are slip walls.
    from ngsolve import BND, Integrate, InnerProduct, Parameter
    from mras.live import load_example
    module = load_example("burgers_2d")
    prob = module.PROBLEM(h=0.35, dt=0.05, tend=0.3)
    assert prob.source(Parameter(0.0)) is None
    res = prob.run_level(0.0, progress=False)
    inflow = module.inflow(res.state.t)
    mesh, fes = prob.mesh, prob.Ufes
    # NGSolve's own imposition: the inflow projected onto the inlet traces
    imposed = GridFunction(fes)
    imposed.Set(inflow, BND, definedon=mesh.Boundaries("inlet"))
    inlet = np.array(fes.GetDofs(mesh.Boundaries("inlet")), dtype=bool)
    no_slip = np.array(fes.GetDofs(mesh.Boundaries("barrier")), dtype=bool)
    walls = np.array(fes.GetDofs(mesh.Boundaries("wall")), dtype=bool)
    assert (walls & ~inlet & np.array(fes.FreeDofs(), dtype=bool)).any()
    want = imposed.vec.FV().NumPy()
    for field in (res.state.gfu, res.state.z):
        got = field.vec.FV().NumPy()
        np.testing.assert_allclose(got[inlet], want[inlet], rtol=1e-12,
                                   atol=1e-12 * np.abs(want).max())
        assert np.abs(got[no_slip]).max() == 0.0
        diff = field - inflow  # the trace projection's own error, small
        gap = Integrate(InnerProduct(diff, diff), mesh, BND,
                        definedon=mesh.Boundaries("inlet")) ** 0.5
        size = Integrate(InnerProduct(inflow, inflow), mesh, BND,
                         definedon=mesh.Boundaries("inlet")) ** 0.5
        assert gap < 1e-2 * size


def test_boundary_data_must_name_dirichlet_boundaries():
    from ngsolve import Parameter
    from mras.live import load_example
    prob = load_example("burgers_2d").PROBLEM(h=0.5, dt=0.05, tend=0.1)
    t = Parameter(0.0)
    for data, error in (({"nowhere": 1}, "matches none"),
                        ({"outlet": (1, 0)}, "not on the Dirichlet part")):
        prob.dirichlet = lambda t, data=data: data
        prob.boundary_data._regions.clear()
        with pytest.raises(ValueError, match=error):
            prob.boundary_data.lift(t)
    prob.dirichlet = lambda t: x
    with pytest.raises(TypeError, match="boundary: value"):
        prob.boundary_data.lift(t)


def test_forward_data_advance_in_lockstep():
    prob = ForwardProblem(tsteps=6)
    seen = []
    prob.on_step = lambda state: seen.append(
        (state.step, None if state.forward is None else state.forward.steps))
    prob.run_level(0.0, progress=False)
    # Nothing is generated before the first step, and afterwards the forward
    # model has solved exactly the data the estimator consumed.
    assert seen[0] == (0, None)
    assert all(steps == step for step, steps in seen[1:])
    assert prob._snaps is None


class NoCouplingProblem(MRASProblem):
    name = "no_coupling"

    def __init__(self):
        super().__init__(darcy(maxh=0.4, order_u=2, order_q=0),
                         dt=1e-1, tend=1e-1, k_u=2)


class NoDataProblem(_Darcy):
    name = "no_data"

    def __init__(self):
        super().__init__(darcy(maxh=0.4, order_u=2, order_q=0),
                         dt=1e-1, tend=3e-1, k_u=2)
        self.u0 = BUMP


def test_missing_physics_is_named_at_construction():
    with pytest.raises(TheorySpecError, match="coupling") as err:
        NoCouplingProblem()
    assert "coupling_adjoint" in str(err.value)


def test_missing_data_path_is_named_at_run():
    prob = NoDataProblem()  # constructs: q_true could still be set
    with pytest.raises(TheorySpecError, match="no data path"):
        prob.run_level(0.0, progress=False)


def test_unknown_scheme_is_refused():
    class Typo(ForwardProblem):
        scheme = "crank-nicolson"
    with pytest.raises(ValueError, match="scheme"):
        Typo()


def test_boundary_terms_by_name_act_on_that_boundary_only():
    # 2D Burgers' slip walls: a resistance beta (u . n)(w . n) on the walls,
    # not on the outlet -- integrated in the driver's quadrature.
    from ngsolve import BND, CoefficientFunction, InnerProduct, Integrate
    from mras.live import load_example
    module = load_example("burgers_2d")
    prob = module.PROBLEM(h=0.35, dt=0.05, tend=0.1)
    u = GridFunction(prob.Ufes)
    u.Set(CoefficientFunction((1.0, 1.0)))  # crosses walls and outlet alike
    lf = LinearForm(prob.Ufes)
    for term in prob.model_terms(u, prob.Ufes.TestFunction()):
        lf += term
    lf.Assemble()
    n = prob.normal
    want = Integrate(module.WALL_RESISTANCE * InnerProduct(u, n) ** 2, prob.mesh,
                     BND, definedon=prob.mesh.Boundaries("wall"),
                     order=prob.order)
    got = InnerProduct(lf.vec, u.vec)  # the convection of a constant is zero
    assert got == pytest.approx(want, rel=1e-10)


def test_a_measure_on_a_boundary_keeps_its_quadrature():
    # NGSolve drops the rules of ds(intrules=...) restricted afterwards;
    # measures(order, boundary) builds the restriction with them.
    from ngsolve import H1, x
    from ngsolve.meshes import MakeStructured2DMesh
    from mras.solvers import measures
    mesh = MakeStructured2DMesh(quads=False, nx=1, ny=1)
    v = H1(mesh, order=1).TestFunction()
    lf = LinearForm(H1(mesh, order=1))
    lf += x ** 10 * v * measures(1, "bottom")[1]  # one point: (1/2)^10
    lf.Assemble()
    assert sum(lf.vec) == pytest.approx(0.5 ** 10)
