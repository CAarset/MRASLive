"""Noisy data and the parameter law's safeguards (mras.robust)."""

import numpy as np
from ngsolve import GridFunction, InnerProduct, Integrate, sin, x, y

from mras.live import build
from mras.robust import LEAKAGE, Smoother, Watch
from mras.scheme import State

QUICK = dict(h=0.35, dt=0.05, k_q=1)


def test_effective_noise_is_zero_only_where_the_true_parameter_is_in_q_h():
    """The data -- the estimator's own discrete model at q^dagger -- and
    their residual in that model at P_h q^dagger: positive where q^dagger
    lies outside Q_h (an unfitted mesh), zero (to the forward model's
    tolerance) on a fitted one, where P_h q^dagger = q^dagger."""
    unfitted = build("fisher", h=0.2, dt=0.05, tend=0.5)
    fitted = build("fisher", h=0.2, dt=0.05, tend=0.5, fitted=True)
    for p in (unfitted, fitted):
        p.discrepancy = None
    assert unfitted.run_level(0.0, progress=False).meta["effective_noise"] > 1e-4
    assert fitted.run_level(0.0, progress=False).meta["effective_noise"] < 1e-8


def test_the_smoother_removes_about_what_the_noise_put_in():
    """alpha follows the discrepancy rule for the filtered data's noise
    level, ||R z - z|| = delta_f ||z||, chosen on the first datum that is
    not zero (a zero datum passes as it is), and the smoothed datum is
    closer to the clean one."""
    p = build("burgers_1d", tend=0.5, **QUICK)
    clean = GridFunction(p.Ufes)
    clean.Set(sin(3.14159 * x / 6) * sin(3.14159 * y / 1.2))
    noisy = p.noise.draw(clean, 0.05, 1)
    smoother = Smoother(p, 0.05)
    smoother(GridFunction(p.Ufes))  # data at rest: nothing to choose on
    assert smoother.alpha is None
    smoother.count = 0  # the filter restarts on the noisy datum
    smoothed = smoother(noisy)

    def norm(f):
        return Integrate(InnerProduct(f, f), p.mesh) ** 0.5
    target = smoother.filtered_level()
    change = norm(smoothed - noisy) / norm(noisy)
    assert abs(change - target) < 0.1 * target  # the bisection's 5 %
    assert norm(smoothed - clean) < norm(noisy - clean)


def test_projection_keeps_q_within_its_bounds():
    p = build("burgers_1d", tend=0.25, h=0.35, dt=0.05, k_q=0)
    p.q0 = -0.5  # outside q >= 0 from the start
    p.discrepancy = None
    result = p.run_level(0.0, progress=False)
    assert result.state.gfq.vec.FV().NumPy().min() >= 0.0


def test_leakage_is_off_by_default_and_acts_on_noisy_data_only():
    p = build("burgers_1d", tend=0.25, **QUICK)
    state = State(p, 0.05)
    p.init_run(state)
    p._safeguards(state, 0.05, True)
    assert state.leak == 0.0  # it moves the fixed point: not by default
    p.leakage = True
    for level in (0.0, 0.05):
        state = State(p, level)
        p.init_run(state)
        p._safeguards(state, level, True)
        assert state.leak == LEAKAGE * level


def test_the_watch_stops_at_tau_times_the_effective_level():
    watch = Watch()
    for _ in range(99):
        watch.add(1.0, 1.0, 1.0)
    assert not watch.ready and not watch.below(1.1)  # still warming up
    watch.add(1.0, 1.0, 1.0)
    assert watch.ready and watch.below(1.1)
    for _ in range(1000):  # the running mean forgets its start
        watch.add(4.0, 1.0, 1.0)  # a residual of twice the level
    assert not watch.below(1.1)
    np.testing.assert_allclose(watch.residual, 2.0, rtol=1e-6)


def _level(p, z, clean):
    def sq(f):
        return Integrate(InnerProduct(f, f), p.mesh)
    e = GridFunction(p.Ufes)
    e.vec.data = z.vec - clean.vec
    return e, (sq(e) / sq(clean)) ** 0.5


def test_matern_noise_has_the_papers_level_and_follows_the_torch_seed():
    """Scaled per datum to ||z - u||_L2 = delta ||u||_L2, zero on the
    Dirichlet dofs, smoother than white noise, and reproducible: a run's
    start seeds torch's generator, which ngsrng draws from, so the same
    seed gives the same draws -- to round-off for a scalar field: ngsrng
    draws a single sample inside NGSolve's TaskManager, whose parallel
    kernels sum in a varying order (batches, as on a vector space, repeat
    bit for bit)."""
    from ngsolve import grad
    from mras.noise import GaussianFEMNoise, MaternFEMNoise
    p = build("burgers_1d", tend=0.5, **QUICK)
    clean = GridFunction(p.Ufes)
    clean.Set(sin(3.14159 * x / 6) * sin(3.14159 * y / 1.2))
    noise = MaternFEMNoise(p.Ufes, seed=3)
    noise.start()
    z = noise.draw(clean, 0.05)
    e, level = _level(p, z, clean)
    np.testing.assert_allclose(level, 0.05, rtol=1e-10)
    fixed = ~np.array(p.Ufes.FreeDofs(), dtype=bool)
    assert np.abs(e.vec.FV().NumPy()[fixed]).max() == 0.0
    following = noise.draw(clean, 0.05).vec.FV().NumPy()
    assert not np.array_equal(following, z.vec.FV().NumPy())
    noise.start()
    np.testing.assert_allclose(noise.draw(clean, 0.05).vec.FV().NumPy(),
                               z.vec.FV().NumPy(), rtol=1e-12)
    np.testing.assert_allclose(noise.draw(clean, 0.05).vec.FV().NumPy(),
                               following, rtol=1e-12)
    white, _ = _level(p, GaussianFEMNoise(p.Ufes, seed=3).draw(clean, 0.05, 7),
                      clean)

    def roughness(f):
        return (Integrate(InnerProduct(grad(f), grad(f)), p.mesh)
                / Integrate(InnerProduct(f, f), p.mesh)) ** 0.5
    assert roughness(e) < 0.5 * roughness(white)


def test_matern_noise_on_a_velocity_space():
    """One field per component of a vector state, at the paper's level,
    both in one draw: the same seed repeats it bit for bit."""
    from mras.noise import MaternFEMNoise
    from ngsolve import CoefficientFunction
    p = build("burgers_2d", h=0.35, dt=0.05, tend=0.1)
    clean = GridFunction(p.Ufes)
    clean.Set(CoefficientFunction((sin(x), 1 + 0.1 * y)))
    noise = MaternFEMNoise(p.Ufes, seed=0)
    noise.start()
    z = noise.draw(clean, 0.02)
    e, level = _level(p, z, clean)
    np.testing.assert_allclose(level, 0.02, rtol=1e-10)
    fixed = ~np.array(p.Ufes.FreeDofs(), dtype=bool)
    assert np.abs(e.vec.FV().NumPy()[fixed]).max() == 0.0
    first, second = e.vec.FV().NumPy().reshape(2, -1)
    assert abs(first @ second) < 0.5 * np.linalg.norm(first) \
        * np.linalg.norm(second)  # independent components
    noise.start()
    np.testing.assert_array_equal(noise.draw(clean, 0.02).vec.FV().NumPy(),
                                  z.vec.FV().NumPy())


def test_the_time_filter_keeps_a_constant_datum():
    """zbar <- zbar + w (z - zbar) leaves a constant datum as it is (an
    NGSolve expression reading zbar while writing it once shrank the data
    by 1 - w each step)."""
    p = build("burgers_1d", tend=0.5, **QUICK)
    datum = GridFunction(p.Ufes)
    datum.Set(sin(3.14159 * x / 6) * sin(3.14159 * y / 1.2))
    smoother = Smoother(p, 0.05, tau=4 * p.dt, space=False)
    for _ in range(5):
        out = smoother(datum)
    np.testing.assert_allclose(out.vec.FV().NumPy(), datum.vec.FV().NumPy(),
                               rtol=1e-12, atol=1e-14)
