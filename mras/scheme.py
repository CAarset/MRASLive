"""The MRAS time loop: data feed, the (q, u) step of scheme (4), errors.

A problem declares its physics once -- the model operator f(q, u) as its
q-free part ``model`` and its q-dependent part ``coupling`` (with the
adjoint density ``coupling_adjoint``), the tracking scale of C, the source
g, and optionally Dirichlet data -- and everything else is built here from
those hooks: the forward (data-generating) model (:mod:`mras.forward`), the
MRAS step of the selected time scheme, and its linear solves.
"""

import functools
from collections.abc import Mapping
import time
import warnings

import numpy as np
from ngsolve import (BilinearForm, CoefficientFunction, FESpace, GridFunction,
                     InnerProduct, Integrate, LinearForm, Norm, Parameter,
                     Projector, IfPos, TaskManager, grad, specialcf)

from .floor import ModelFit
from .forward import BoundaryData, ForwardRun
from . import robust
from .noise import GaussianFEMNoise, MaternFEMNoise
from .solvers import Solve, factorize, mass, measures

SCHEMES = ("implicit", "semi-implicit", "explicit")


class State:
    """Mutable objects shared by the driver and a problem's hooks."""

    def __init__(self, problem, noise_level):
        self.problem = problem
        self.noise_level = noise_level
        self.mesh = problem.mesh
        self.Ufes, self.Qfes = problem.Ufes, problem.Qfes
        self.X = problem.X
        self.u, self.q = problem.u, problem.q
        self.v, self.s = problem.v, problem.s
        self.n = specialcf.normal(self.mesh.dim)
        self.dt = problem.dt
        self.t = Parameter(problem.tstart)                    # t^n
        self.t_next = Parameter(problem.tstart + problem.dt)  # t^{n+1}
        self.step = 0  # n: the steps taken; the step from t^n reads datum n+1
        self.gfu = GridFunction(problem.Ufes)
        self.gfq = GridFunction(problem.Qfes)
        self.z = GridFunction(problem.Ufes)
        self.z_old = GridFunction(problem.Ufes)
        self.zt = GridFunction(problem.Ufes)
        self.u_true = GridFunction(problem.Ufes)
        self.g = None  # the source g(t^{n+1}) of the step, set by data()
        self.e_old = GridFunction(problem.Ufes)   # u^n - z^n
        self.q_next = GridFunction(problem.Qfes)  # q^{n+1} while it is solved
        self.z_started = False
        self.has_truth = True  # data() clears it for purely observational runs
        self.forward = None  # the run's ForwardRun, built by the first datum
        self.joint_steps = False  # the Schur solve failed: factorise instead
        # Running histories, appended by the driver after every step; the
        # live view reads them while the run is in progress.
        self.error_q = None
        self.error_u = None
        self.misfit_u = []
        self.model_fit = None  # where the estimator plateaus (mras.floor)
        # Noisy data and the parameter law's safeguards (mras.robust).
        self.smoother = None   # R_sp(R_ti(.)) of noisy data
        self.reference = False  # effective noise level at P_h q^dagger
        self.watch = None      # residual against effective noise level
        self.q_start = None    # q_0, the leakage's anchor
        self.leak = 0.0        # the leakage rate lambda
        self.effective_noise = []
        self.residual = []

    def set_z(self, z):
        """Copy a datum into the persistent z and refresh zt = (z - z_old)/dt.

        The first datum of a run also initialises ``z_old``, so ``zt``
        starts at zero and every noise level is a fresh statistical start.
        """
        if not self.z_started:
            self.z_old.vec.data = z.vec
            self.z_started = True
        self.z.vec.data = z.vec
        self.zt.vec.data = (1 / self.dt) * (self.z.vec - self.z_old.vec)


def series_steps(tsteps):
    """Steps after which the solution is kept, besides the initial one:
    1, 5, 10, 25, 50, 100, ... below ``tsteps``, and the last."""
    marks, decade = [1], 1
    while decade * 5 < tsteps:
        marks += [m * decade for m in (5, 10, 25) if m * decade < tsteps]
        decade *= 10
    return frozenset(marks + [tsteps])


def _terms(value):
    """A hook's weak form(s) as a list: none, one expression, or several."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [term for term in value if term is not None]
    return [value]


# A run stops once error_q is at most TARGET_FLOOR * floor
# (MRASProblem.stop_target), or at tend.
TARGET_FLOOR = 1.25


class TheorySpecError(ValueError):
    """A problem class is missing specification the driver or the theory
    checklist cannot guess; the message names every missing piece."""


class MRASProblem:
    """Base class: the physics a problem declares, and the scheme built on it.

    Physics hooks
    -------------
    ``coupling(q, u, w)``          density of the q-dependent part of f:
                                   ``<f(q, u) - f(0, u), w> = int coupling dx``.
    ``coupling_adjoint(q, u, w)``  density of the adjoint of its q-derivative,
                                   ``<f_q(q, u) p, w> = int p coupling_adjoint
                                   dx``; linear in ``w``.
    ``model(u, w)``                density of the q-free part ``<f(0, u),
                                   w>`` over the domain, and
    ``model_boundary(u, w)``       its density over the boundary -- or, by
                                   boundary name, ``{"wall": density}``;
                                   both default to none.  A list of densities
                                   keeps linear and nonlinear terms apart,
                                   so the linear ones are assembled once.
    ``model_strong(u)``            f(0, u) itself, as a function (its strong
                                   form, e.g. ``-Trace(u.Operator("hesse"))``
                                   for -Laplace u), for the sigma-term: the
                                   parameter row tests the data's residual
                                   with Q_h's functions, and a discontinuous
                                   one has no weak derivative to take.
                                   Optional; without it the weak density is
                                   used elementwise, which a manufactured
                                   source matches, generated data do not.
    ``source(t)``                  g at time ``t``, a Parameter: build the
                                   source from NGSolve functions of it, or
                                   read ``t.Get()`` for data indexed by time.
    ``tracking_scale(q)``          c in the tracking operator C = c (-Laplace);
                                   the constant ``C_scale`` unless overridden.
    ``tracking_boundary(u, w)``    a boundary part of C (bilinear, symmetric,
                                   non-negative, q-free; by boundary name as
                                   for ``model_boundary``): C = c (-Laplace
                                   + B).  A linear q-free term of f can go
                                   here as well as there: it then acts on the
                                   state, not on the datum, and the data's
                                   noise does not reach the estimator through
                                   it, with the same error equation and fixed
                                   point and C's coercivity only improved.
    ``dirichlet(t)``               the Dirichlet data at time ``t``, by
                                   boundary: ``{name: value}`` for boundaries
                                   of the Dirichlet part of U_h, the others
                                   held at zero; None (the default) for
                                   homogeneous data.  Boundary data drive a
                                   problem as a source does: the state and
                                   the data model carry them, imposed on the
                                   constrained dofs (``forward.BoundaryData``).
    ``u0``, ``q_true``             initial state and true parameter.
    ``tend``, ``target``           the run's time limit, and the
                                   reconstruction error at which it stops
                                   before that: 1.25 times the floor
                                   unless set; 0 runs to ``tend``.
    ``q0``                         the initial estimate qhat_0 (a number,
                                   coefficient or grid function), set by
                                   whoever runs the problem; it overrides
                                   ``initial_guess()``, the example's own
                                   start (0 unless it says otherwise).

    The hooks are written once and used everywhere: with the trial function
    of the data space they are the forward model, with the datum the MRAS
    step, and the checklist (:mod:`mras.verify`) probes the same coupling
    pair.  ``examples/blank_problem.py`` is the smallest complete problem.

    Data
    ----
    By default the data are generated: ``u_t + f(q_true, u) = g`` stepped
    by :class:`~mras.forward.ForwardRun` in lockstep with the estimator, on
    its mesh and in its U_h.  Only the true parameter has a mesh of its own
    (``q_domains``): it is the physical quantity, the state, the datum and
    the estimate one consistent discretisation.  ``truth(t)`` instead supplies
    an analytic u^dagger -- with ``truth_rate(t)`` its time derivative,
    from which ``manufactured_source`` builds g = u^dagger_t +
    f(q_true, u^dagger) -- and ``observations(state)`` supplies measured
    data with no truth at all.  Synthetic data get Gaussian FEM noise at
    the requested noise level.

    Time scheme
    -----------
    ``scheme`` discretises the MRAS (4) in time:

    ``"implicit"`` (default)  backward Euler: q^{n+1}, u^{n+1} and z^{n+1}
                              in both rows, the tracking term
                              f_q(q^{n+1}, z^{n+1})^* (u^{n+1} - z^{n+1})
                              included.  q and u couple both ways; with
                              sigma = 1 the step is iterated to
                              ``newton_tol``.
    ``"semi-implicit"``       the paper's (13)-(14): the parameter row
                              explicit in u (u^n, z^n), f(q^{n+1}; q^n,
                              z^{n+1}) linearised about q^n.
    ``"explicit"``            the parameter row entirely at the old
                              iterate, then the state row with q^{n+1}
                              known: the scheme read as prediction and
                              correction.  For sigma = 0 it coincides with
                              ``"semi-implicit"``.

    Every scheme takes the norm-dependence of C at q^n, C(||q^n||), as the
    paper does ("significantly more feasible", p. 88): it is a scalar of the
    whole field, and implicit in it the step would couple every dof.  And
    every scheme takes the tracking factor f_q(q~, z)^* at q~ = q^n: the
    theory leaves q~ open, and at q^n the step needs no f_qq.

    ``forward_scheme`` steps the data model the same way: backward Euler by
    Newton's method, one Newton step per step (linearly implicit Euler), or
    forward Euler.

    ``split`` chooses how a step is solved.  ``True`` solves the rows apart
    with the constant matrices factorised once: the parameter row, then the
    state row, for the block-triangular semi-implicit and explicit steps;
    the Schur complement on the state, by CG to round-off, for the implicit
    step.  ``False`` assembles and factorises the joint system on
    U_h x Q_h every step, the reference.  ``"auto"`` (default) is ``True``
    unless a problem overrides ``forms`` with a joint form of its own: that
    form is checked on the first step and solved by rows when its parameter
    row does not see u^{n+1}.

    Other configuration
    -------------------
    ``q_domains``        the ``Domains`` the true parameter is defined on,
                         a mesh that follows its jumps (pictures paint it
                         there, exactly); the estimator's mesh need not.
    ``positive_state``,  guarantees, set by the problem's author, that the
    ``positive_source``  state and the source stay positive: the pictures
                         then use the fields' own ranges instead of ranges
                         symmetric about zero.
    ``decaying``         whether the fields relax over the run instead of
                         being sustained: the pictures of sustained fields
                         (the default) are pinned to their largest values.
    ``series_steps``     steps after which the solution is kept (1, 5, 10,
                         25, ...); the run takes ``tsteps`` steps to ``tend``.
    ``on_step``          ``on_step(state)``, called after ``init_run`` and
                         after every step; the live view's hook.
    ``noise_model``      the data noise of synthetic runs: "matern" (the
                         default; :class:`mras.noise.MaternFEMNoise`, of
                         correlation length ``noise_length`` -- 5 % of the
                         domain's diagonal unless set -- and smoothness
                         ``noise_nu``; drawn by ngsrng, seeded with
                         ``seed`` through torch's global generator at the
                         start of a run) or "white", the paper's discrete
                         Gaussian noise (:class:`mras.noise.GaussianFEMNoise`);
                         either scaled to the noise level in L2 per datum.
    ``smooth_data``      noisy data enter smoothed, R_sp(R_ti(z^delta))
                         (Kaltenbacher-Nguyen 2021, (28), whatever the
                         noise): a time filter of time constant
                         ``smooth_time`` (``smooth_scale`` delta^(2/3)
                         dt^(1/3) unless set, ``smooth_scale`` being
                         ``robust.FILTER_TIME`` unless a problem's data vary
                         more slowly)
                         and, with ``smooth_space`` (on), one heat step in
                         space, R_sp taking data bounded in Z to data close
                         in U (:class:`mras.robust.Smoother`); clean data
                         enter as they are.
    ``discrepancy``      tau of the discrepancy stop: the run stops once the
                         data's residual in the discrete model at the
                         estimate, ||D_t z + f(q, z) - g||_U*, is at tau
                         times the effective noise level (None, the
                         default: no stop; robust.DISCREPANCY = 1.1 is the
                         usual start) -- the same residual at P_h q^dagger, so
                         that the data's mismatch with the estimator's model
                         counts.
    ``projection``,      q projected onto ``q_bounds`` ((0, None): q >= 0)
    ``q_bounds``         after every step.
    ``leakage``          -lambda (q - q_0) in the parameter law, lambda the
                         noise level times ``robust.LEAKAGE``; off by
                         default: it moves the fixed point to lambda (q_0 -
                         q^dagger), which the MRAS theory does not allow.
    ``data_cache``       a :class:`~mras.forward.DataCache` the generated
                         noise-free data are read from and kept in, shared
                         by the runs of one example at several noise levels
                         (``scripts/run_examples.py``); a run resumed from a
                         checkpoint generates its data afresh.
    """

    name = "problem"
    tstart = 0.0
    saved_every = 100
    sigma = 1.0
    L = None
    u0 = None
    q0 = None
    target = None
    q_true = None
    q_domains = None
    on_step = None
    data_cache = None
    # Noisy data and the parameter law's safeguards (mras.robust)
    noise_model = "matern"
    noise_length = None
    noise_nu = 1.0
    smooth_data = True
    smooth_time = None
    smooth_scale = robust.FILTER_TIME
    smooth_space = True
    discrepancy = None  # robust.DISCREPANCY = 1.1 turns the stop on
    projection = True
    q_bounds = (0.0, None)
    leakage = False
    scheme = "implicit"
    forward_scheme = "implicit"
    split = "auto"
    newton_tol = 1e-10
    newton_maxit = 50
    manufactured_source = False
    # Display hints, user-set guarantees: the state (and so the datum) and the
    # source stay positive.  Pictures of fields of either sign are drawn
    # symmetrically about zero; a guaranteed positive one between its own
    # extremes (mras.live).
    positive_state = False
    positive_source = False
    # Whether the fields decay over the run (a reference relaxing to rest)
    # rather than being sustained by their forcing: pictures of sustained
    # fields keep their largest values as colour range (mras.live).
    decaying = False

    def __init_subclass__(cls, **kwargs):
        """Record the arguments of the outermost constructor in
        ``init_args``, so a checkpoint can build the problem again."""
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is None:
            return

        @functools.wraps(init)
        def recording(self, *args, **kw):
            if "init_args" not in self.__dict__:
                self.init_args = (args, dict(kw))
            init(self, *args, **kw)
        cls.__init__ = recording

    def __init__(self, domains, dt=1e-3, tend=5.0, k_u=3, seed=0):
        self.mesh, self.Ufes, self.Qfes = domains
        self.dt = dt
        self.tend = tend
        self.seed = int(seed)
        self.tsteps = int(round((tend - self.tstart) / dt))
        self.series_steps = series_steps(self.tsteps)
        self.X = FESpace([self.Ufes, self.Qfes])
        self.u, self.q = self.X.TrialFunction()
        self.v, self.s = self.X.TestFunction()
        self.normal = specialcf.normal(self.mesh.dim)
        # One quadrature for every form of the step, exact for the cubic
        # products of P3 states that the models here carry.
        self.order = 2 * max(self.Ufes.globalorder, self.Qfes.globalorder) + 2
        self.dx, self.ds = measures(self.order)
        self._noise = None  # built on first use (``noise``)
        self.boundary_data = BoundaryData(self, self.Ufes)  # dirichlet(t) on U_h
        self._snaps = None
        self._cache = None
        self._q_parts = None  # the true parameter's share of error_q
        self._u_scratch = None
        self._rows_apart = None  # the "auto" verdict on a custom forms()
        self._validate_theory_spec()

    # -- physics hooks ----------------------------------------------------
    C_scale = 1.0
    """Tracking operator C = C_scale * (-Laplace); None when C depends on
    the parameter (override ``tracking_scale``)."""

    coupling = None
    """``coupling(q, u, w)``: the q-dependent part of f as a weak density,
    ``<f(q, u) - f(0, u), w> = int coupling(q, u, w) dx`` -- the hook the
    checklist derives sigma, (A1) and (A2) from.  Required."""

    coupling_adjoint = None
    """``coupling_adjoint(q, u, w)``: ``<f_q(q, u) p, w> = int p *
    coupling_adjoint(q, u, w) dx``, transcribed independently of
    ``coupling`` (the checklist compares the two).  Required."""

    adjoint_deviation = None
    """One-line reason why the scheme's cross blocks are expected not to
    be adjoint; None means machine precision expected."""

    def model(self, u, w):
        """Density of the q-free part of f, ``<f(0, u), w>``, over the domain."""

    def model_boundary(self, u, w):
        """Density of the q-free part of f over the boundary (``ds``), or
        ``{boundary: density}`` for densities on named boundaries only."""

    model_strong = None
    """``model_strong(u)``: f(0, u) as a function, for the sigma-term's
    pairing with Q_h's functions (see the class docstring)."""

    def source(self, t):
        """The source g at time ``t`` (a Parameter), or None (the default:
        no source, as for a problem driven by its boundary data)."""

    def tracking_scale(self, q):
        """c in C(||q||) = c (-Laplace)."""
        return self.C_scale

    def tracking_boundary(self, u, w):
        """Density of the boundary part of the tracking operator C (see
        the class docstring), or ``{boundary: density}``."""

    def dirichlet(self, t):
        """The Dirichlet data at time ``t``: ``{boundary: value}``, each
        boundary a name (or regular expression) of the mesh's boundaries on
        the Dirichlet part of U_h, the rest of which is zero; None means
        homogeneous data."""

    def truth(self, t):
        """Analytic reference state u^dagger at time ``t``, or None."""

    def truth_rate(self, t):
        """Its time derivative, for ``manufactured_source``."""

    def observations(self, state):
        """Measured data for this step (coefficient or grid function)."""

    def initial_guess(self):
        """The example's starting value of the parameter estimate, used
        unless ``q0`` is set."""
        return 0.0

    def init_run(self, state):
        """Set gfu / gfq / u_true at tstart for one noise level."""
        start = self.u0 if self.u0 is not None else self.truth(state.t)
        if start is None:
            raise NotImplementedError(
                f"{self.name}: set u0, or give the analytic truth(t)")
        state.gfu.Set(start)
        state.u_true.Set(start)
        start = self.q0 if self.q0 is not None else self.initial_guess()
        state.gfq.vec.data = self.project_q(start).vec

    def model_terms(self, u, w, order=None, boundary=True):
        """``<f(0, u), w>`` as weak forms, in the quadrature of ``order``
        (the driver's unless given).

        ``boundary=False`` leaves out the boundary densities, for test
        functions of Q_h: an L2 function has no boundary values.  Boundary
        densities keyed by boundary name are integrated over those only.
        """
        order = order or self.order
        dx, ds = measures(order)
        terms = [term * dx for term in _terms(self.model(u, w))]
        if boundary:
            terms += self._on_boundary(self.model_boundary(u, w), order)
        return terms

    @staticmethod
    def _on_boundary(densities, order):
        """Boundary densities as weak forms in the quadrature of ``order``:
        those keyed by boundary name over that boundary only."""
        if not isinstance(densities, Mapping):
            densities = {None: densities}
        terms = []
        for region, density in densities.items():
            surface = measures(order, region)[1]
            terms += [term * surface for term in _terms(density)]
        return terms

    def f_terms(self, q, u, w, order=None, boundary=True):
        """``<f(q, u), w>`` as weak forms: the model's, then the coupling."""
        order = order or self.order
        return (self.model_terms(u, w, order, boundary)
                + [self.coupling(q, u, w) * measures(order)[0]])

    # -- specification ------------------------------------------------------
    def _validate_theory_spec(self):
        cls = type(self)
        missing = []
        if cls.coupling is None:
            missing.append(
                "coupling(q, u, w): the q-dependent part of f as a weak "
                "density -- e.g. f(q, u) = q*exp(u) gives "
                "\"return q * exp(u) * w\"")
        if cls.coupling_adjoint is None:
            missing.append(
                "coupling_adjoint(q, u, w): the density of f_q(q, u)^* w -- "
                "\"return exp(u) * w\" for the example above")
        if missing:
            raise TheorySpecError(
                f"{cls.__name__} is not runnable as specified; missing:\n  - "
                + "\n  - ".join(missing)
                + "\nSee examples/blank_problem.py for the smallest "
                "complete example.")
        for attribute in ("scheme", "forward_scheme"):
            if getattr(self, attribute) not in SCHEMES:
                raise ValueError(f"{attribute} is one of {SCHEMES}, not "
                                 f"{getattr(self, attribute)!r}")
        if (self.sigma and cls.model_strong is None
                and not self.manufactured_source
                and "l2" in self.Qfes.type):
            warnings.warn(
                f"{cls.__name__}: sigma = 1 on a discontinuous Q_h without "
                "model_strong(u): the sigma-term tests the model's weak form "
                "elementwise, which drops its derivatives' fluxes, so the "
                "data's residual does not vanish at the true parameter",
                stacklevel=3)

    def _check_data_path(self):
        cls = type(self)
        if (self.q_true is None and cls.truth is MRASProblem.truth
                and cls.observations is MRASProblem.observations
                and cls.data is MRASProblem.data):
            raise TheorySpecError(
                f"{cls.__name__} has no data path: set q_true (the forward "
                "model generates the data), give the analytic truth(t), or "
                "observations(state)")

    def _truth_known(self):
        """True when some data path supplies a reference state."""
        return (type(self).data is not MRASProblem.data
                or type(self).observations is MRASProblem.observations
                or type(self).truth is not MRASProblem.truth)

    def _constant_tracking(self):
        return type(self).tracking_scale is MRASProblem.tracking_scale

    # -- data ---------------------------------------------------------------
    def data(self, state):
        """Feed one step: the source g, the reference u_true and the datum z,
        all at t^{n+1} (the paper's step (13))."""
        state.t_next.Set(self.tstart + (state.step + 1) * self.dt)
        state.g = self.source(state.t_next)
        obs = self.observations(state)
        truth = self.truth(state.t_next)
        if obs is not None:
            state.has_truth = truth is not None
            if truth is not None:
                self._set_reference(state, truth)
            state.set_z(self._as_gf(obs))
            return
        if truth is not None:
            self._set_reference(state, truth)
        elif self.data_cache is not None:
            # The same noise-free datum another run stepped the forward
            # model to, read back (``data_cache``).
            state.u_true.vec.FV().NumPy()[:] = self.data_cache.datum(
                self, state.step + 1)
        else:
            # Datum n+1 is the forward state after n+1 steps.
            if state.forward is None:
                state.forward = ForwardRun(self)
            if state.forward.steps > state.step:
                raise RuntimeError("the forward run cannot step backwards")
            while state.forward.steps <= state.step:
                snap = state.forward.step()
            state.u_true.vec.data = snap.vec
        state.set_z(self._observed(state, state.step + 1))

    def _observed(self, state, counter):
        """The datum the MRAS sees: the reference with the run's noise
        drawn on it, smoothed when it is noisy (``smooth_data``)."""
        if not state.noise_level:
            z = GridFunction(self.Ufes)
            z.vec.data = state.u_true.vec
            return z
        z = self.noise.draw(state.u_true, state.noise_level, counter)
        return z if state.smoother is None else state.smoother(z)

    @property
    def noise(self):
        """The data noise (``noise_model``), built on first use."""
        if self._noise is None:
            if self.noise_model == "white":
                self._noise = GaussianFEMNoise(self.Ufes, seed=self.seed)
            elif self.noise_model == "matern":
                self._noise = MaternFEMNoise(self.Ufes, seed=self.seed,
                                             length=self.noise_length,
                                             nu=self.noise_nu)
            else:
                raise ValueError(f"noise_model is 'matern' or 'white', not "
                                 f"{self.noise_model!r}")
        return self._noise

    @noise.setter
    def noise(self, value):
        self._noise = value

    def _set_reference(self, state, value):
        if isinstance(value, GridFunction):
            state.u_true.vec.data = value.vec
        else:
            state.u_true.Set(value)
        if type(self).dirichlet is not MRASProblem.dirichlet:
            # As for the projected generated datum: the constrained dofs
            # carry the boundary data themselves, which the state carries
            # too, so the error u - z stays in the homogeneous space.
            state.u_true.vec.data = (Projector(self.Ufes.FreeDofs(), True)
                                     * state.u_true.vec
                                     + self.boundary_data.lift(
                                         state.t_next).vec)

    def _as_gf(self, value):
        if isinstance(value, GridFunction):
            return value
        gf = GridFunction(self.Ufes)
        gf.Set(value)
        return gf

    def _forward_data(self):
        """Every forward snapshot on U_h at once, for whole-run analyses.

        The identification loop never calls this: it steps its own
        ForwardRun in lockstep.  Theory checks that need the full datum
        trajectory (e.g. ess-inf witnesses) use this cached array.
        """
        if self._snaps is None and self.q_true is not None:
            with TaskManager():
                run = ForwardRun(self)
                snaps = np.empty((self.tsteps, self.Ufes.ndof),
                                 dtype=np.float32)
                for k in range(self.tsteps):
                    snaps[k] = run.step().vec.FV().NumPy()
            self._snaps = snaps
        return self._snaps

    def _source_terms(self, state, w, boundary=True, strong=False):
        """``<g^{n+1}, w>`` as weak forms (see model_terms for boundary;
        ``strong`` takes a manufactured source's model part in its strong
        form, as the sigma-term does the data's)."""
        if self.manufactured_source:
            rate = self.truth_rate(state.t_next)
            if strong:
                return [InnerProduct(rate + self.model_strong(state.u_true),
                                     w) * self.dx,
                        self.coupling(self.q_true, state.u_true, w)
                        * self.dx]
            return ([InnerProduct(rate, w) * self.dx]
                    + self.f_terms(self.q_true, state.u_true, w,
                                   boundary=boundary))
        if state.g is None:
            return []
        return [InnerProduct(CoefficientFunction(state.g), w) * self.dx]

    def _bracket(self, state, qk, s, linearised):
        """``<D_t z + f(qk, z) - g, s>``, the sigma-bracket of (14) at qk;
        ``linearised`` drops its part linear in qk, which the matrix carries
        as ``q f_q(qk, z)``."""
        z = state.z
        coupling = self.coupling(qk, z, s)
        if linearised:
            coupling = coupling - qk * self.coupling_adjoint(qk, z, s)
        strong = self.model_strong is not None
        model = ([InnerProduct(self.model_strong(z), s) * self.dx] if strong
                 else self.model_terms(z, s, boundary=False))
        return ([InnerProduct(state.zt, s) * self.dx]
                + model
                + [coupling * self.dx]
                + [(-1) * term
                   for term in self._source_terms(state, s, boundary=False,
                                                  strong=strong)])

    # -- the joint form -----------------------------------------------------
    def forms(self, state, qk=None):
        """The joint weak form of one step on U_h x Q_h, linearised at qk
        (default q^n) -- the reference solve, and what mras.adjoint probes."""
        qk = state.gfq if qk is None else qk
        u, q, v, s = self.u, self.q, self.v, self.s
        dt, sig, ca = self.dt, self.sigma, self.coupling_adjoint
        z, gfu, gfq = state.z, state.gfu, state.gfq
        implicit = self.scheme == "implicit"
        scale = self.tracking_scale(gfq)
        a = BilinearForm(self.X)
        res = LinearForm(self.X)
        # state row (13): f(q^{n+1}; qk, z) = f(qk, z) + f_q(qk, z)(q - qk)
        a += InnerProduct(u, v) * self.dx
        a += dt * scale * InnerProduct(grad(u), grad(v)) * self.dx
        for term in self._on_boundary(self.tracking_boundary(u, v), self.order):
            a += dt * scale * term
        a += dt * q * ca(qk, z, v) * self.dx
        res += InnerProduct(gfu, v) * self.dx
        res += dt * scale * InnerProduct(grad(z), grad(v)) * self.dx
        for term in self._on_boundary(self.tracking_boundary(z, v), self.order):
            res += dt * scale * term
        for term in self._source_terms(state, v):
            res += dt * term
        for term in self.model_terms(z, v):
            res += -dt * term
        if sig:
            res += -dt * (self.coupling(qk, z, v) - qk * ca(qk, z, v)) * self.dx
        # parameter row (14)
        a += q * s * self.dx
        res += gfq * s * self.dx
        if state.leak:  # -lambda (q - q_0), implicit
            a += dt * state.leak * q * s * self.dx
            res += dt * state.leak * state.q_start * s * self.dx
        if implicit:  # the tracking factor at q~ = q^n in every scheme
            a += -dt * ca(gfq, z, u) * s * self.dx
            res += -dt * ca(gfq, z, z) * s * self.dx
        else:
            state.e_old.vec.data = gfu.vec - state.z_old.vec
            res += dt * ca(gfq, state.z_old, state.e_old) * s * self.dx
        if sig:
            a += sig * dt * q * ca(qk, z, s) * self.dx
            for term in self._bracket(state, qk, s, linearised=True):
                res += -sig * dt * term
        return a, res

    # -- constant matrices ----------------------------------------------------
    def _prepare(self):
        """Assemble and factorise what is constant over a run, once."""
        if self._cache is None:
            u, v = self.Ufes.TnT()
            stiff = BilinearForm(self.Ufes)
            stiff += InnerProduct(grad(u), grad(v)) * self.dx
            stiff.Assemble()
            cache = self._cache = {"Mu": mass(self.Ufes, self.dx),
                                   "Au": stiff.mat,
                                   "Cu": stiff.mat,
                                   "Mq": mass(self.Qfes, self.dx)}
            walls = self._on_boundary(self.tracking_boundary(u, v), self.order)
            if walls:  # C = -Laplace + its boundary part
                tracking = BilinearForm(self.Ufes)
                tracking += InnerProduct(grad(u), grad(v)) * self.dx
                for term in walls:
                    tracking += term
                tracking.Assemble()
                cache["Cu"] = tracking.mat
            cache["Mq_inv"] = factorize(cache["Mq"], symmetric=True)
            # The state matrix M + dt c A: its values are rewritten in place
            # when c varies, and it is factorised once when c does not.
            base = BilinearForm(self.Ufes)
            base += InnerProduct(u, v) * self.dx
            base.Assemble()
            cache["Su"] = base.mat
            cache["scale"] = None
            if self._constant_tracking():
                self._state_operator(self.C_scale)
        return self._cache

    def _state_operator(self, scale):
        """M + dt C on U_h and its Cholesky factors, rebuilt only when c
        changes."""
        cache = self._cache
        if cache["scale"] != scale:
            cache["Su"].AsVector().data = (cache["Mu"].AsVector()
                                           + (self.dt * scale)
                                           * cache["Cu"].AsVector())
            cache["Su_inv"] = factorize(cache["Su"], self.Ufes.FreeDofs(),
                                        symmetric=True)
            cache["scale"] = scale
        return cache["Su"], cache["Su_inv"]

    def _mass_q(self, leak):
        """(1 + dt lambda) M_q and its factors: the parameter row's matrix
        for sigma = 0, with the leakage's implicit term."""
        cache = self._cache
        if not leak:
            return cache["Mq"], cache["Mq_inv"]
        if cache.get("leak") != leak:
            mat = cache["Mq"].CreateMatrix()
            mat.AsVector().data = (1 + self.dt * leak) * cache["Mq"].AsVector()
            cache["Mq_leak"] = (mat, factorize(mat, symmetric=True))
            cache["leak"] = leak
        return cache["Mq_leak"]

    def _parameter_operator(self, qk, z, leak=0.0):
        """The parameter-row matrix M_q (+ sigma dt f_q(qk, z) weighted mass,
        + dt lambda M_q for the leakage) and its factors."""
        if not self.sigma:
            return self._mass_q(leak)
        q, s = self.Qfes.TnT()
        form = BilinearForm(self.Qfes)
        form += (1 + self.dt * leak) * q * s * self.dx
        form += self.sigma * self.dt * q * self.coupling_adjoint(qk, z, s) * self.dx
        form.Assemble()
        return form.mat, factorize(form.mat)

    # -- the step -------------------------------------------------------------
    def _step(self, state):
        """Advance state.gfu, state.gfq from step n to n + 1."""
        custom = type(self).forms is not MRASProblem.forms
        rows = self.split
        if rows == "auto":
            rows = self._custom_rows(state) if custom else True
        if not rows:
            return self._step_joint(state)
        if custom:
            return self._step_custom_rows(state)
        if self.scheme == "implicit":
            return self._step_implicit(state)
        return self._step_rows(state)

    def _state_rhs(self, state, scale, f_q, u_old=None):
        """M u^n + dt C z + dt <g - f(0, z) - (q-part of f), v> on U_h
        (u^n the estimate's, or ``u_old``)."""
        cache = self._cache
        w = self.Ufes.TestFunction()
        form = LinearForm(self.Ufes)
        for term in self._source_terms(state, w):
            form += self.dt * term
        for term in self.model_terms(state.z, w) + f_q:
            form += -self.dt * term
        form.Assemble()
        u_old = state.gfu.vec if u_old is None else u_old
        return (cache["Mu"] * u_old
                + (self.dt * scale) * (cache["Cu"] * state.z.vec)
                + form.vec).Evaluate()

    def _equation_base(self, state):
        """<D_t z + f(0, z) - g, v> on U_h: the q-free part of the data's
        residual in the discrete model, for ``_equation_residual``."""
        w = self.Ufes.TestFunction()
        form = LinearForm(self.Ufes)
        form += InnerProduct(state.zt, w) * self.dx
        for term in self.model_terms(state.z, w):
            form += term
        for term in self._source_terms(state, w):
            form += -1 * term
        form.Assemble()
        return form.vec

    def _equation_residual(self, state, base, q):
        """The data's residual in the estimator's discrete model at q,
        D_t z + f(q, z) - g, and the parameter's term in it, f(q, z) -
        f(0, z), both as squared U* norms -- the model fit's integrand
        (mras.floor).  At P_h q^dagger it is the effective noise level: the
        data's noise, smoothing error and mismatch with the model at once,
        zero for data the discrete model generated at a parameter in Q_h;
        it is large at a poor q and falls toward that level as q improves,
        as the discrepancy principle needs."""
        w = self.Ufes.TestFunction()
        form = LinearForm(self.Ufes)
        form += self.coupling(q, state.z, w) * self.dx
        form.Assemble()
        inv = self._u_norm_inverse()
        total = base.CreateVector()
        total.data = base + form.vec
        return robust.dual2(inv, total), robust.dual2(inv, form.vec)

    def _u_norm_inverse(self):
        """The inverse Riesz map of the U-norm on U_h: C's energy norm
        (||grad v||, with a boundary part of C its own term as well) where
        U_h has a Dirichlet part, the H1 norm where it has none (as the
        checklist's)."""
        cache = self._prepare()
        if "Ku_inv" not in cache:
            free = self.Ufes.FreeDofs()
            gram = cache["Cu"].CreateMatrix()
            gram.AsVector().data = cache["Cu"].AsVector()
            if free.NumSet() == len(free):
                gram.AsVector().data += cache["Mu"].AsVector()
            cache["Ku"] = gram
            cache["Ku_inv"] = factorize(gram, free, symmetric=True)
        return cache["Ku_inv"]

    def _project(self, q):
        """q onto ``q_bounds``: clipped -- exactly on P0, where coefficients
        are values; elsewhere the L2 projection of the clipped field."""
        lo, hi = self.q_bounds
        if lo is None and hi is None:
            return
        values = q.vec.FV().NumPy()
        if self.Qfes.globalorder == 0:
            values[:] = robust.clip_values(values, self.q_bounds)
            return
        field = q
        if lo is not None:
            field = IfPos(field - lo, field, lo)
        if hi is not None:
            field = IfPos(field - hi, hi, field)
        q.vec.data = self.project_q(field).vec

    def _step_rows(self, state):
        """Semi-implicit / explicit step: the parameter row, then the state
        row with q^{n+1} known."""
        cache = self._prepare()
        q, s = self.Qfes.TnT()
        w = self.Ufes.TestFunction()
        dt, sig, ca = self.dt, self.sigma, self.coupling_adjoint
        z, gfu, gfq, qn = state.z, state.gfu, state.gfq, state.q_next
        semi = self.scheme == "semi-implicit"
        # (14), explicit in u: the tracking term at u^n, z^n
        state.e_old.vec.data = gfu.vec - state.z_old.vec
        form = LinearForm(self.Qfes)
        form += dt * ca(gfq, state.z_old, state.e_old) * s * self.dx
        if sig:
            for term in self._bracket(state, gfq, s, linearised=semi):
                form += -sig * dt * term
        form.Assemble()
        rhs = (cache["Mq"] * gfq.vec + form.vec).Evaluate()
        if state.leak:
            rhs += (dt * state.leak) * (cache["Mq"] * state.q_start.vec)
        _, inv = (self._parameter_operator(gfq, z, state.leak) if semi
                  else self._mass_q(state.leak))
        qn.vec.data = inv * rhs
        # (13) with q^{n+1} known: f(q^{n+1}; q^n, z) is a load
        if sig and semi:
            f_q = [(self.coupling(gfq, z, w)
                    + (qn - gfq) * ca(gfq, z, w)) * self.dx]
        else:
            f_q = [self.coupling(qn, z, w) * self.dx]
        scale = self.tracking_scale(gfq)
        rhs = self._state_rhs(state, scale, f_q)
        su, su_inv = self._state_operator(scale)
        boundary = self.boundary_data.lift(state.t_next).vec
        rhs -= su * boundary
        gfu.vec.data = boundary + su_inv * rhs
        gfq.vec.data = qn.vec

    def _step_implicit(self, state):
        """Backward Euler step through the Schur complement on the state.

        With M_q' the parameter-row matrix and K the coupling block
        (dt int q f_q(qk, z)^* v), the step reads
            M_q' q - K^T u = r_q,   K q + S_u u = r_u,
        and eliminating q leaves (S_u + K M_q'^-1 K^T) u = r_u - K M_q'^-1
        r_q, symmetric positive definite and solved by CG to round-off,
        preconditioned with the factors of S_u.  For sigma = 1 f is
        nonlinear in q: Newton's method, f linearised at the iterate qk with
        its declared derivative, to ``newton_tol``.  The tracking factor
        f_q(q~, z)^* stays at q~ = q^n (see ``forms``), so the iteration never
        needs f_qq -- which does not exist for q |q|^{2/3} at q = 0.  The
        coupling block of the state row then differs from that of the
        parameter row while qk moves, and the Schur system is solved by
        GMRES.  When the Krylov solve does not reach round-off -- a coupling
        so strong that K M_q'^-1 K^T swamps the preconditioner -- the step is
        the factorised joint solve instead, and so are the run's later steps.
        """
        if state.joint_steps:
            return self._step_joint(state)
        cache = self._prepare()
        q, s = self.Qfes.TnT()
        w = self.Ufes.TestFunction()
        dt, sig, ca = self.dt, self.sigma, self.coupling_adjoint
        z, gfu, gfq, qk = state.z, state.gfu, state.gfq, state.q_next
        free = self.Ufes.FreeDofs()
        boundary = self.boundary_data.lift(state.t_next).vec
        scale = self.tracking_scale(gfq)
        su, su_inv = self._state_operator(scale)
        x = gfu.vec.CreateVector()
        qk.vec.data = gfq.vec
        track = BilinearForm(trialspace=self.Qfes, testspace=self.Ufes)
        track += dt * q * ca(gfq, z, w) * self.dx
        track.Assemble()
        kt = track.mat
        for _ in range(self.newton_maxit if sig else 1):
            mq, mq_inv = self._parameter_operator(qk, z, state.leak)
            if sig:
                couple = BilinearForm(trialspace=self.Qfes,
                                      testspace=self.Ufes)
                couple += dt * q * ca(qk, z, w) * self.dx
                couple.Assemble()
                k = couple.mat
            else:
                k = kt
            form = LinearForm(self.Qfes)
            form += -dt * ca(gfq, z, z) * s * self.dx
            if sig:
                for term in self._bracket(state, qk, s, linearised=True):
                    form += -sig * dt * term
            form.Assemble()
            rq = (cache["Mq"] * gfq.vec + form.vec + kt.T * boundary).Evaluate()
            if state.leak:
                rq += (dt * state.leak) * (cache["Mq"] * state.q_start.vec)
            f_q = ([(self.coupling(qk, z, w) - qk * ca(qk, z, w)) * self.dx]
                   if sig else [])
            ru = self._state_rhs(state, scale, f_q)
            ru -= su * boundary
            solve = Solve(su_inv, free, symmetric=not sig)
            solve(su + k @ mq_inv @ kt.T, (ru - k * (mq_inv * rq)).Evaluate(),
                  x)
            if not solve.converged or not np.isfinite(Norm(x)):
                state.joint_steps = True
                return self._step_joint(state)
            q_new = (mq_inv * (rq + kt.T * x)).Evaluate()
            change = Norm((q_new - qk.vec).Evaluate())
            qk.vec.data = q_new
            if not sig or change <= self.newton_tol * max(Norm(q_new), 1e-300):
                break
        else:
            warnings.warn(f"{self.name}: implicit step not converged to "
                          f"{self.newton_tol:g} at t = {state.t.Get():g}",
                          stacklevel=2)
        gfu.vec.data = boundary + x
        gfq.vec.data = qk.vec

    def _step_joint(self, state):
        """The reference solve: the joint system on U_h x Q_h, factorised;
        iterated like the Schur solve when the implicit step is nonlinear."""
        if self.scheme == "explicit" and self.sigma:
            raise ValueError("the explicit scheme with sigma = 1 is solved "
                             "by rows only (split=True)")
        generic = type(self).forms is MRASProblem.forms
        iterate = generic and self.scheme == "implicit" and self.sigma
        qk = state.q_next
        qk.vec.data = state.gfq.vec
        sol = GridFunction(self.X)
        for _ in range(self.newton_maxit if iterate else 1):
            a, res = self.forms(state, qk) if generic else self.forms(state)
            a.Assemble()
            res.Assemble()
            sol.vec[:] = 0.0
            sol.components[0].vec.data = self.boundary_data.lift(
                state.t_next).vec
            r = (res.vec - a.mat * sol.vec).Evaluate()
            sol.vec.data += a.mat.Inverse(self.X.FreeDofs(),
                                          inverse="umfpack") * r
            change = Norm((sol.components[1].vec - qk.vec).Evaluate())
            qk.vec.data = sol.components[1].vec
            if change <= self.newton_tol * max(Norm(qk.vec), 1e-300):
                break
        state.gfu.vec.data = sol.components[0].vec
        state.gfq.vec.data = sol.components[1].vec

    def _custom_rows(self, state):
        """For a custom ``forms``: does its parameter row see u^{n+1}?
        Checked once, on the first step, on the assembled matrix."""
        if self._rows_apart is None:
            a, _ = self.forms(state)
            a.Assemble()
            nu = self.Ufes.ndof
            x = a.mat.CreateColVector()
            x.FV().NumPy()[:] = 0.0
            x.FV().NumPy()[:nu] = np.random.default_rng(0).standard_normal(nu)
            y = (a.mat * x).Evaluate().FV().NumPy()
            self._rows_apart = bool(np.linalg.norm(y[nu:])
                                    <= 1e-14 * max(np.linalg.norm(y), 1e-300))
        return self._rows_apart

    def _step_custom_rows(self, state):
        """A custom block-triangular joint form, solved by rows."""
        from ngsolve import BitArray
        a, res = self.forms(state)
        a.Assemble()
        res.Assemble()
        nu, n = self.Ufes.ndof, self.X.ndof
        rows_q, rows_u = BitArray(n), BitArray(n)
        rows_q.Clear()
        rows_u.Clear()
        free = self.X.FreeDofs()
        for i in range(n):
            if free[i]:
                (rows_u if i < nu else rows_q).Set(i)
        sol = GridFunction(self.X)
        sol.components[0].vec.data = self.boundary_data.lift(state.t_next).vec
        r = (res.vec - a.mat * sol.vec).Evaluate()
        sol.vec.data += a.mat.Inverse(rows_q, inverse="umfpack") * r
        r = (res.vec - a.mat * sol.vec).Evaluate()
        sol.vec.data += a.mat.Inverse(rows_u, inverse="umfpack") * r
        state.gfu.vec.data = sol.components[0].vec
        state.gfq.vec.data = sol.components[1].vec

    # -- the run --------------------------------------------------------------
    def run_level(self, noise_level, progress=True, resume=None):
        """One identification run at a noise level: ``noise_level`` is a
        fraction, 0.05 means noise level 5%, 0.0 is noise-free.

        ``resume`` is a restart point (:meth:`mras.checkpoint.Checkpoint.
        restart`) to continue from instead of the initial state; the run
        then goes on exactly as it would have without the interruption.
        """
        self._check_data_path()
        with TaskManager():
            return self._run_level(noise_level, progress, resume)

    def _restore(self, state, restart):
        """Put a restart point into a fresh state: estimate, datum,
        reference, histories and the forward model where it stood."""
        state.step = int(restart["step"])
        state.t.Set(self.tstart + state.step * self.dt)
        for name in ("gfu", "gfq", "z", "z_old", "u_true"):
            getattr(state, name).vec.FV().NumPy()[:] = restart[name]
        state.zt.vec.data = (1 / self.dt) * (state.z.vec - state.z_old.vec)
        state.z_started = True
        state.has_truth = bool(restart["has_truth"])
        if "forward_steps" in restart:
            state.forward = ForwardRun(self)
            state.forward.restore(restart)
        if "noise_rng" in restart:
            self.noise.set_rng_state(restart["noise_rng"])
        if state.q_start is not None and "q_start" in restart:
            state.q_start.vec.FV().NumPy()[:] = restart["q_start"]
        if state.smoother is not None and "z_bar" in restart:
            state.smoother.restore(restart, restart)
        if "watch_count" in restart:
            state.watch.restore(restart)
        for name in ("effective_noise", "residual"):
            if restart.get(name) is not None:
                getattr(state, name)[:] = list(restart[name])

    def _safeguards(self, state, noise_level, truth_known):
        """A run's noisy-data machinery (:mod:`mras.robust`): the smoother of
        noisy data, the reference parameter P_h q^dagger (whose residual is
        the effective noise level), the residual watch, the leakage."""
        if noise_level > 0 and self.smooth_data:
            state.smoother = robust.Smoother(self, noise_level,
                                             self.smooth_time,
                                             self.smooth_space)
        if self.q_true is not None and truth_known:
            self._q_ref = self._true_parts()["reference"]
            state.reference = True
        state.watch = robust.Watch()
        if self.leakage and noise_level > 0:
            state.leak = robust.LEAKAGE * noise_level
            state.q_start = GridFunction(self.Qfes)
            state.q_start.vec.data = state.gfq.vec

    def _run_level(self, noise_level, progress, resume=None):
        from .results import LevelResult
        started = time.perf_counter()
        state = State(self, noise_level)
        self.init_run(state)
        # The initial state takes the Dirichlet data exactly, as the
        # forward model's does.
        keep = Projector(self.Ufes.FreeDofs(), True)
        boundary = self.boundary_data.lift(state.t).vec
        for gf in (state.gfu, state.u_true):
            gf.vec.data = keep * gf.vec + boundary
        truth_known = self._truth_known()
        self._safeguards(state, noise_level, truth_known)
        if resume is None and noise_level > 0:
            self.noise.start()
        if resume is not None:
            self._restore(state, resume)
        elif truth_known:
            # z^0, the datum at t^0: step 1 then reads f_q(z^0)^*(u^0 - z^0)
            # in the q-row and D_t z^1 = (z^1 - z^0)/dt, as the paper's (14)
            # does.  (Observation-only runs have no datum at t^0, so their
            # first datum primes z_old instead -- see State.set_z.)
            state.set_z(self._observed(state, 0))
        # The tracked quantity is the parameter reconstruction error, and
        # the paper's tables report errors relative to the true field
        # (Section 8): error_q = ||qhat - q||_L2 / ||q||_L2.
        error_q = None
        if truth_known and self.q_true is not None:
            if self._true_parts()["norm"] > 0.0:
                error_q = [self.q_error(state.gfq)]
        error_u = [self.u_distance(state.gfu, state.u_true)] \
            if truth_known else None
        if resume is not None:  # the histories so far, not the restart's
            for name, history in (("error_q", error_q), ("error_u", error_u)):
                if history is not None:
                    history[:] = list(resume[name])
            state.misfit_u[:] = list(resume["misfit_u"])
        misfit_u = state.misfit_u
        state.error_q, state.error_u = error_q, error_u
        if self.on_step is not None:
            self.on_step(state)
        saved_u = [state.gfu.vec.FV().NumPy().copy()]
        saved_q = [state.gfq.vec.FV().NumPy().copy()]
        series_u = [state.gfu.vec.FV().NumPy().copy()]
        series_q = [state.gfq.vec.FV().NumPy().copy()]
        floor = self.q_floor() if error_q is not None else None
        target = self.stop_target(floor) if error_q is not None else 0.0
        stopped = "tend"
        fit = state.model_fit = (ModelFit(self) if error_q is not None
                                 else None)

        while state.step < self.tsteps:
            self.data(state)
            misfit_u.append(self.u_distance(state.gfu, state.z))
            if progress:
                headline = (error_q[-1] if error_q is not None
                            else error_u[-1] if error_u is not None
                            else misfit_u[-1])
                print(f"\r{self.name} noise_level={noise_level}: "
                      f"t={state.t.Get():.3f}"
                      f"/{self.tend}  err={headline:.3e}", end="", flush=True)
            self._step(state)
            if self.projection:
                self._project(state.gfq)
            base = self._equation_base(state)
            residual2, signal2 = self._equation_residual(state, base,
                                                         state.gfq)
            if state.reference:
                effective2, signal2 = self._equation_residual(
                    state, base, self._q_ref)
            else:  # the declared level stands in
                effective2 = noise_level ** 2 * signal2
            state.watch.add(residual2, effective2, signal2)
            # per step, relative to the parameter's running share of it
            size = max(state.watch.data2, 1e-300)
            state.residual.append((residual2 / size) ** 0.5)
            state.effective_noise.append((effective2 / size) ** 0.5)
            if error_u is not None and state.has_truth:
                error_u.append(self.u_distance(state.gfu, state.u_true))
            if error_q is not None and state.has_truth:
                error_q.append(self.q_error(state.gfq))
            if fit is not None:
                fit.offer(state)  # while zt is still (z^{n+1} - z^n) / dt
            state.z_old.vec.data = state.z.vec
            state.step += 1
            state.t.Set(self.tstart + state.step * self.dt)
            if state.step % self.saved_every == 0:
                saved_q.append(state.gfq.vec.FV().NumPy().copy())
                saved_u.append(state.gfu.vec.FV().NumPy().copy())
            if state.step in self.series_steps:
                series_u.append(state.gfu.vec.FV().NumPy().copy())
                series_q.append(state.gfq.vec.FV().NumPy().copy())
            plateau = False
            if fit is not None and fit.due(state.step):
                # Below the model fit and no longer falling: as far as these
                # data and this mesh allow -- for clean data only, as the
                # model fit of noisy data fits their noise.
                fit.update(state.step, state.t.Get(), error_q[-1])
                plateau = (self.target is None and fit.stalled
                           and not noise_level
                           and error_q[-1] <= fit.value)
            # The discrepancy stop: the residual at the effective noise level
            # the data allow.
            discrepant = bool(self.discrepancy) and state.watch.below(
                self.discrepancy)
            if self.on_step is not None:
                self.on_step(state)
            if error_q is not None and error_q[-1] <= target:
                stopped = "target"
                break
            if plateau:
                stopped = "plateau"
                break
            if discrepant:
                stopped = "discrepancy"
                break

        if fit is not None and fit.kept and not (
                fit.values and fit.values[-1][0] == state.t.Get()):
            fit.update(state.step, state.t.Get(), error_q[-1])  # the whole run

        if progress:
            print()
        return LevelResult(
            problem=self.name, noise_level=noise_level,
            error_u=np.array(error_u) if error_u is not None else None,
            error_q=np.array(error_q) if error_q is not None else None,
            misfit_u=np.array(misfit_u),
            effective_noise=np.array(state.effective_noise),
            residual=np.array(state.residual),
            saved_gfu=np.array(saved_u), saved_gfq=np.array(saved_q),
            series_gfu=np.array(series_u), series_gfq=np.array(series_q),
            state=state,
            meta={"dt": self.dt, "tend": self.tend, "scheme": self.scheme,
                  "dofs_u": len(state.gfu.vec.FV().NumPy()),
                  "dofs_q": len(state.gfq.vec.FV().NumPy()),
                  "steps": len(misfit_u),
                  "q_floor": floor, "target": target or None,
                  "model_fit": None if fit is None else fit.value,
                  "stopped": stopped, "t_end": state.t.Get(),
                  "noise_level": noise_level,
                  "effective_noise": (state.watch.effective
                                      if state.watch.count else None),
                  "residual": (state.watch.residual
                               if state.watch.count else None),
                  "smooth_alpha": (None if state.smoother is None
                                   else state.smoother.alpha),
                  "seconds": round(time.perf_counter() - started, 1)})

    def stop_target(self, floor=None):
        """The reconstruction error at which a run stops before ``tend``:
        ``target`` if set (0: none, the run goes to ``tend``), else
        TARGET_FLOOR times the floor.  The floor itself is
        no target: no estimate in Q_h gets below it, so the error only
        approaches it.  The rule reads the true parameter, which only
        benchmarks have; a run without it goes to ``tend``."""
        if self.target is not None:
            return float(self.target)
        floor = self.q_floor() if floor is None else floor
        return TARGET_FLOOR * (floor or 0.0)

    def q_floor(self):
        """Best relative error any estimate in Q_h can reach, or None.

        ``||P q - q|| / ||q||`` with ``P`` the L2 projection onto Q_h:
        ``error_q`` can never go below it, and since ``qhat - P q`` is
        orthogonal to ``P q - q``, ``error_q**2 - q_floor()**2`` is exactly
        the part of the error the estimator can still remove.
        """
        if self.q_true is None:
            return None
        parts = self._true_parts()
        if parts["norm"] == 0.0:
            return None
        return parts["floor"] / parts["norm"]

    # -- error norms ----------------------------------------------------------
    def _true_parts(self):
        """The true parameter's share of ``error_q``, computed once.

        P q, the L2 projection of q onto Q_h (``project_q``); the mass
        matrix of Q_h in project_q's quadrature; and in that quadrature
        ||q|| and ||q - P q||.  P q being the projection in that very
        quadrature, qhat - P q is orthogonal to P q - q for every qhat in
        Q_h, and Pythagoras splits the error exactly,

            ||qhat - q||^2 = ||qhat - P q||^2 + ||P q - q||^2,

        its first term a mass-matrix norm on Q_h.  q itself, which may live
        on a mesh of its own, is evaluated here only, not at every step.
        """
        if self._q_parts is None:
            order = self.order + 4  # project_q's quadrature
            q = CoefficientFunction(self.q_true)
            reference = self.project_q(self.q_true)
            gap = reference - q
            self._q_parts = {
                "reference": reference,
                "mass": mass(self.Qfes, measures(order)[0]),
                "norm": max(float(Integrate(q * q, self.mesh, order=order)),
                            0.0) ** 0.5,
                "floor": max(float(Integrate(gap * gap, self.mesh,
                                             order=order)), 0.0) ** 0.5,
                "scratch": reference.vec.CreateVector(),
                "image": reference.vec.CreateVector()}
        return self._q_parts

    def q_error(self, q):
        """||q - q_true|| / ||q_true|| for ``q`` a grid function on Q_h, by
        the exact split of :meth:`_true_parts`."""
        parts = self._true_parts()
        d, md = parts["scratch"], parts["image"]
        d.data = q.vec - parts["reference"].vec
        md.data = parts["mass"] * d
        gap2 = max(InnerProduct(d, md), 0.0)
        return (gap2 + parts["floor"] ** 2) ** 0.5 / parts["norm"]

    def u_distance(self, a, b):
        """||a - b||_L2 for grid functions ``a``, ``b`` on U_h: through the
        mass matrix of U_h in the driver's quadrature, which integrates the
        square of a U_h function exactly on straight elements."""
        self._prepare()
        if self._u_scratch is None:
            self._u_scratch = (a.vec.CreateVector(), a.vec.CreateVector())
        d, md = self._u_scratch
        d.data = a.vec - b.vec
        md.data = self._cache["Mu"] * d
        return max(InnerProduct(d, md), 0.0) ** 0.5

    def project_q(self, value):
        """The L2 projection of ``value`` onto Q_h, as a grid function.

        Assembled: the element mass matrices of Q_h on the (curved)
        geometry, against the load of ``value`` in a quadrature four orders
        above the driver's.  ``GridFunction.Set`` on an L2 space inverts
        the mass matrix of the straight element, so on curved elements it
        is not the projection: it puts a constant 0.5 % off on the Fisher
        mesh.
        """
        dx = measures(self.order + 4)[0]
        s = self.Qfes.TestFunction()
        load = LinearForm(self.Qfes)
        load += InnerProduct(CoefficientFunction(value), s) * dx
        load.Assemble()
        gf = GridFunction(self.Qfes)
        gf.vec.data = factorize(mass(self.Qfes, dx), symmetric=True) * load.vec
        return gf

    def run(self, noise_levels, progress=True):
        """One run per noise level; runs at different noise levels are
        statistically independent."""
        return [self.run_level(p, progress) for p in noise_levels]
