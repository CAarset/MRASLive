# Implementation notes

How the code realises the paper's theory, which free modelling choices were
made concretely, and which behaviour is a known theoretical limit rather than
a bug. The reference implementations for the published figures exist
separately; this codebase targets the theory in the paper text with a clean,
extensible design, so its numbers are its own.

## Noise model — `mras/noise.py`

- Observations are `z = u_dagger + noise_level * e / ||e||_M * ||u_dagger||_M`,
  where `e ~ N(0, M^-1)` is an L2-Gaussian FEM field: the whitening
  `e = M^(-1/2) eta`, `eta ~ N(0, I)` on the degrees of freedom, gives the
  continuum-consistent covariance `M^-1` (nodal or mass-lumped whitening
  would distort the spatial spectrum).
- `M^(-1/2) eta` is applied matrix-function-free: the mass matrix is
  diagonally scaled, `M~ = D^(-1/2) M D^(-1/2)` (condensation ~228 at
  h = 0.1 on P3, from ~1.3e5 unscaled), and `M~^(-1/2) (D^(1/2) eta)` is
  approximated by a Clenshaw-evaluated Chebyshev polynomial of degree
  doubling until `max |1 - p(x) sqrt(x)| <= 5e-4` on `[0.7 lambda_min,
  1.2 lambda_max]`, whose endpoints come from a Lanczos run of length 150.
  Mass lumping is not used because row-sum lumping of a P3 mass matrix is
  not positive.
- The draw is renormalised by the exact energy `sqrt(e . M e)`, so
  `||z - u_dagger||_L2 = noise_level * ||u_dagger||_L2` holds exactly;
  `noise_level = 0` (noise-free) returns the field unchanged.
- Every draw is seeded: the RNG is keyed by `(seed, noise_level, counter)`, so a
  run is fully reproducible and each time step is statistically independent.

## Initial states, sources, references

- All four benchmarks start from the paper's deterministic fields, and the
  estimator starts in the exact initial state: error histories begin at
  t = 0 with `error_u[0] = 0`, which isolates estimator convergence from
  initial-condition transients (a test enforces this for all four).
- Darcy (`mras/problems/darcy.py`): initial state
  `sin(pi x) sin(pi y)`; the source is a fresh i.i.d. standard-normal
  128x128 pixel field per time step (the paper's source model), rendered
  onto the data mesh as piecewise-constant voxel coefficients. Truth runs
  on the fine data mesh with the same per-step sources.
- Fisher-KPP (`mras/problems/fisher.py`): deterministic Gaussian bump
  `u0 = 3 exp(-r^2 / 0.4)`; forcing `g(x, t) = u0(x) cos(2 t)`; ring-shaped
  true parameter 1 inside r = 0.5 and 0.25 outside r = 0.9.
- Potential and Allen-Cahn: references
  `u_dagger(x, t) = u_wot(x) (DECAY_T - t) / DECAY_T + OFFSET` with
  (DECAY_T, OFFSET) = (6, 6) and (10, 1) respectively; `tend = 5` for both.
  The time derivative `ut = -u_wot / DECAY_T` is analytic, and because the
  profile is linear in t the backward-Euler difference of exact samples
  equals `ut` identically.
- Tracking coefficient: the potential and Allen-Cahn q-rows use
  coefficient 1, the exact value of the linearisation factor
  `1 + (5/3) |q_tilde|^(2/3)` at the paper's linearisation point
  `q_tilde = 0` (Examples 5.2, 5.3).

## Discretisation and driver — `mras/scheme.py`

- Joint FEM solve on `X = FESpace([Ufes, Qfes])`, `solvers.BVP` with a
  local (block-Jacobi) preconditioner; semi-implicit Euler. All
  linearisation factors are taken at the new time level, except the two the
  paper makes explicit: the tracking coefficient `f'_q(q_tilde, z^n)` and
  the stabilisation bound `C(.)`, which stay at the previous level.
- Potential and Allen-Cahn solve for the increments `(du, dq)` and add them
  (the paper's schemes (35), (38)/(41) evaluate the parameter response
  through trial increments); Darcy and Fisher solve directly for `(u, q)`.
- NGSolve drops BilinearForm terms without a trial function silently, so
  every term linear in an unknown enters as an implicit increment plus an
  explicit previous-level half in the residual.
- Each noise level is a fresh start: `run()` builds a new `State` per noise level,
  `init_run` supplies the t = 0 reference and the initial estimate, and
  `State.set_z` primes `z_old` with the noise level's first datum, so the
  difference term starts at zero on every noise level.
- Dirichlet sets follow the reference fields: Fisher carries "wall"
  conditions on U_h, Darcy's Q_h space is free, and potential/Allen-Cahn
  use free spaces because their references do not vanish on the boundary.
- With the P0 L2 parameter space, `Grad(s) = 0` and `s * ds` boundary pairs
  evaluate to zero, so the s-row boundary-flux terms are omitted.
- P3 U_h / P0 Q_h on the 2D unit geometries of `mras/domains.py`; data
  meshes, sampling strides and figure instants are per-problem
  (`data_h`, `data_order`, `series_counters`).
- Error metrics: `error_q` is the relative reconstruction error
  `||qhat - q||_L2 / ||q||_L2` (the paper's benchmark tables report
  relative errors, Sec. 8); `error_u` and the data misfit stay absolute.

## Burgers pipe — `examples/burgers_pipe.py` (no paper benchmark)

The Burgers identification is an application of the abstract scheme (4),
not a published benchmark; the audit trail of its theory choices:

Rigorous (the paper's assumptions are verified or exact):

- `f(q, u) = -div(q grad u) + u u_x` is linear in `q`, so the
  regularisation term is dropped (`sigma = 0`, the paper's rule for
  `f` linear in `q`) and (A1) holds *exactly* with `L = 0` — the
  tracking term is the exact adjoint, not a bound.
- Adjoint: `f'_q(q, u)[s] = -div(s grad u)`, so the q-row of scheme (4)
  is `D_t q = grad z . grad (u - z)` — Darcy's estimator equation
  (Prop. 4.1) verbatim, since the advection `u u_x` carries no `q`.
- Remark 2.1 placement: the state row evaluates the nonlinearity on the
  datum — stabilisation `C(||q||)(u - z)` and explicit data-side
  advection `-(z z_x, v)` — which is what cancels `z z_x` from the
  continuous error system.
- (A3): `C = -Laplace` with wall-homogeneous Dirichlet; the pipe sits in
  `(0,6) x (0,1.2)`, so Poincare gives `lambda_1 >= pi^2 (1/36 + 1/1.44)`
  `~ 7.13`, and with `L = 0` any `M <= lambda_1` satisfies (A3) (`M = 1`
  as for Darcy).  The explicit q-flow gain is
  `dt ||grad z||_inf^2 / (2 lambda_1)`; measured `||grad z||_inf ~ 12`
  makes it `0.21` per unit `dt`, so `dt = 0.02` runs with margin.

Ad-hoc modelling (no theory attached, none claimed):

- The truth generator is backward Euler with lagged advection on the
  data mesh; it produces observations only and enters no assumption.
- Wake parameter `q_true`: logistic decay off the three hole centres
  (`Q_WAKE = 0.15` -> `Q_FREE = 1.0`, scale `Q_SMOOTH = 0.0875`); the
  smooth transition is a modelling choice (abrupt disks leave most of
  the L2 error in the kink layer: measured wake-region error drops from
  2.09 to 0.76 when the wake is smoothed, at unchanged global error).
- Zero initial estimate `qhat = 0`; Gaussian-bump initial state.

Open — where the code cannot force conformance:

- (A2) coercivity of the coupled linearisation is unverified, exactly as
  the paper leaves it for its own Darcy benchmark (Remark 4.2); the
  justification there is persistent excitation (PE) of the source.
- PE evidence (noise level 0% (noise-free), `h = 0.35`, `dt = 0.02`, relative `error_q`):

  | source / horizon / wake | `error_q` | min over run |
  |---|---|---|
  | single frequency, abrupt wake, `T = 5` | 0.805 | 0.805 |
  | same, `T = 15` | 0.788 | 0.788 |
  | single frequency, smooth wake, `T = 5` | 0.785 | 0.785 |
  | same, `T = 15` | 0.765 | 0.765 |
  | single frequency, smooth wake, `dt = 0.005`, `T = 5` | 0.830 | 0.830 |
  | distinct frequencies (in phase), smooth wake, `T = 5` | 0.776 | 0.776 |
  | same, `T = 10` | 0.767 | 0.767 |
  | drifting heater centres (spatial PE surrogate), `T = 10` | 0.773 | 0.773 |

  while `||u - z||` stays near 0.02: the state tracks without
  identifying `q`.  The plateau is insensitive to step size (finer is
  worse: the update is rate-limited, not discretisation-limited),
  horizon (longer helps only marginally), source frequency richness and
  even spatial sweeping of the heaters — the signature of an
  identifiability (PE/(A2)) gap in the `q -> z` map for this source
  geometry, not of a discretisation or implementation defect.  For
  comparison, the paper's own Darcy benchmark — with per-step i.i.d.
  random pixel sources, i.e. maximal PE — recovers `q` only to relative
  errors 0.33–0.60 (Table 6).
- The shipped demo takes the best-behaving variants (smooth wake, distinct
  frequencies on top of the original phase stagger, `T = 10`; measured
  noise-free relative error 0.777 — the phase-staggered variant behaves like
  the in-phase row above, within the plateau band).  The deeper question —
  whether to accept the heater grid as PE surrogate, switch to per-step
  randomised spatial sources (the Darcy benchmark's own recipe), or change
  the model so that (A2) becomes verifiable — is open and recorded here.
- Framework verdict (`mras/verify.py`, reproducible with
  `scripts/checklist.py`): adjoint pairing machine-precise, (A1) exact
  (`sigma = 0`, `L = 0`), (A2) FAIL — the kernel probe measures a
  sign-indefinite pairing, the gradient-coupling situation of the
  paper's Remark 4.2, (A3) UNVERIFIABLE (no `C_scale` claim to check
  once (A2) fails), persistent excitation OK, reconstruction descending,
  seeded.  The failure is therefore localised to the one assumption the
  paper itself does not verify for its gradient-coupled benchmarks: the
  implementation passes every probe that does not presuppose (A2), and
  the error plateau sits in the same band as the paper's own Darcy
  numbers.  This is what "no (A2), no convergence-rate guarantee" looks
  like in practice; the Arrhenius pair is the same code on the same
  checklist with the hooks filled in, and recovers.

## Arrhenius reaction pair — `mras/problems/arrhenius.py` (sixth/seventh examples)

`Dt u - Laplace(u) + q(x) r(u) = g` with `r(u) = 1 + u` (linear member,
the "linearised" twin) and `r(u) = exp(u)` (nonlinear member) on one and
the same experimental design: three phase-staggered Gaussian heaters as
`g`, a two-bump smooth `q_true`, zero initial state and estimate.
Zeroth-order (pointwise) coupling in `q`, so this is the example class
where the paper's theory is fully verifiable:

- Affine in `q`: `sigma = 0` exactly, `L = 0` exactly, and the tracking
  kernel `f'_q(q, z)[s] = s r(z)` equals the state-side coupling — the
  adjoint pairing test measures it at machine precision (~1e-15).
- (A2): `f(q, z) - f(p, z) = (q - p) r(z)`, so `C_coe` is any a.e. lower
  bound of `r(z)` over the datum.  `c_coe()` computes it by bisection on
  the *weak* essential lower bound (`essinf_datum`, the integrated
  negative part `(r0 - u)^+`), because nodal minima of the P3 datum undershoot the
  true bound (Gibbs ringing, cf. the noise-model section).  Linear
  member: witness `1 + lo`; exponential member: `exp(lo)`.
- (A3): `C = -Laplace` with `C_scale = 1`, `lambda_1 = 0.12500000000069`
  on the shipped `4 pi` box (CG inverse iteration, mass-normalised;
  analytic `2 (pi / L)^2`; the `2 pi` box measured exactly `0.5` the same
  way); with `L = 0` any `M <= lambda_1` is admissible (the pair declares
  `N_tilde = 1`, the strict inequality `lambda_1 > L^2 / (2 C_coe)` holds
  with margin `0.125 > 0`).
- Measured recovery (noise level 0% (noise-free), `h = 0.3, dt = 0.02, data_h = 0.15`):
  relative `q`-error 0.37 (t = 4), 0.13 (t = 8), 0.081 (t = 28) for both
  members, tail decay ~1%/time-unit.  Both members therefore sit inside
  the "good" band (< 10%) at long horizon; "excellent" (< 5%) is not
  reached inside the tested horizon and we do not claim it.
- The plateau is rate-limited, not a floor: `dt` refinement leaves it
  unchanged, `h` refinement of `Q_h` moves the P0 representation floor of
  `q_true` from 0.85% (h = 0.3) to 0.00% (h = 0.15) while the error
  stays at its plateau, and the long-horizon curve keeps decaying at
  ~1%/time-unit.  The decay rate of the zero-order flow scales like
  `r^2 / lambda_1(C)` - confirmed by the two design levers: datum
  amplitude (`r`, error at t = 10: 0.23 / 0.17 / 0.12 for amplitude 1 /
  1.5 / 3) and domain size (`lambda_1 ~ L^-2`).  The shipped
  configuration bakes the `4 pi` box and the triple heater amplitude
  into the module defaults (`SIZE = 4 pi`, `PREHEAT = 0.9`, heaters
  3.6 / 3.0 / 2.4, widened Gaussians, `h = 0.3, dt = 0.02, tend = 10,
  data_h = 0.15`).

## Automated theory checklist — `mras/verify.py`

Every example — past or future — through the same fixed questions:
adjoint pairing, (A1)/sigma, (A2) monotonicity, (A3) coercivity margin,
persistent excitation of the datum, reconstruction descent, seeded
reproducibility.  Statuses are `ok` / `fail` / `deviation` (documented,
via `adjoint_deviation`) / `unverifiable` (no probe exists — flagged,
never silently skipped).  The probes and their honest caveats:

- Theory hooks are plain attributes defaulting to `None` on `Problem`
  (`coupling_kernel`, `monotone_pair`, `C_scale`, `adjoint_deviation`,
  `c_coe`); the checks dispatch on callability, so a problem that never
  declares a hook is reported unverifiable rather than crashing.
- The (A2) pair probe localises test pairs on <= 3 cells: full-domain
  random pairs only measure the datum-weighted *mean* of the
  monotonicity density and average out sign-changing data.  Related
  lesson: Allen-Cahn's nodal datum values ring below zero at sine
  extrema (63% negative!) while its cell means stay >= 1 - |u_wot| > 0;
  `tests/test_verify.py` pins both facts so the verdict cannot silently
  flip through an interpolation artifact.
- For affine problems `c_coe()` (analytic witness) outranks the discrete
  probe: weakly interpolating a P0 direction into the P3 datum
  under-reports the witness (0.166 vs the true ~1).
- Gradient couplings (Darcy, Fisher, Burgers) are exactly degenerate on
  P0 (`grad q = 0` a.e.); the probe's H1-relaxed lift is sign-indefinite
  and reported `fail` with the Remark 4.2 note — the paper itself does
  not claim (A2) there, and (A3) is then `unverifiable`, not `fail`.
- `lambda_1(C)` is computed by CG inverse iteration
  (`solvers.CG(a.mat, rhs, pre=..., sol=...)`, mass-normalised each
  step); plain `BVP` inner solves under-resolve the low modes and
  over-report `lambda_1` by ~30x on the coarse box.

`scripts/checklist.py` runs the registry and writes
`results/checklist.md`; `tests/test_verify.py` pins the documented
verdict per example.

## Known theoretical limits (expected behaviour)

- Potential problem, constant mode: the `C(||c||)` H1 stabilisation acts
  through `-Laplacian` and is blind to constants (the paper's Remark 6.2),
  so with the offset reference `u_wot (...) + 6` the state error settles at
  a floor set by that mode rather than decaying to zero.
- The stabilisation bound uses the worst-case ratio `Z_MAX / Z_MIN` of the
  admissible state range instead of instantaneous norms; this over-stabilises
  and slows the observed decay but never destabilises.

## Data — `mras/data/pdebench.py`

- `2D_DarcyFlow_beta100.0_Train.hdf5`: despite the filename convention
  (beta = 100), the stored fields have 10:1 contrast. The synthetic
  cross-check set in `scripts/darcy_dataset.py` brackets this with
  NGSolve-built fields of matched contrast and centred inclusions.

## Environment behaviour (load-bearing)

- ngsolve 6.2.2608: `GridFunction.Set(coefficient)` on an H1 space
  *projects* (Gibbs overshoot at kinks — pointwise assertions on H1 are
  invalid; the tests use L2 comparisons).
- Dirichlet conditions are applied at solve time via `FreeDofs`, so a
  `Set` may write constrained dofs without violating the BC after a solve.
- `FreeDofs()` iterates as boolean flags (`np.fromiter(..., dtype=bool)`).
- `to_gridfunc` maps pixel arrays through `VoxelCoefficient`; analytic
  coefficient functions must go through `GridFunction.Set` directly.
- `ImplicitStepper` needs `a.Assemble()` before construction (its matrix
  is merged in the constructor).
- ngsolve 6.2.2608: `BilinearForm` has no `matvec` — use `a.mat.matvec`;
  `solvers.CG(mat, rhs, pre=, sol=, tol=, maxsteps=)` is positional
  `(mat, rhs)`; `vec.data = <numpy>` demands float64 (the seeded noise
  snapshots are float32 — cast before assignment); there is no
  `CoM`/cell-center accessor on `Mesh` (localised P0 probes must work
  topologically, on cells, not on centres).

## Verification anchors

- `pytest tests/` (84 tests) covers the fixed-point bookkeeping (t = 0
  error seeding for all problems, seeded reproducibility, fresh-start
  independence), the noise whitening (covariance moments, exact relative
  noise level), degree-of-freedom anchors, the field helpers, the Burgers
  conformance form, the live helper (hook throttling, relative-error
  seeding, the fixed-framerate movie stitch), the adjoint pairing test
  (machine precision for the exact-coupling examples, documented ratios
  for the paper-literal linearisations), the per-example theory
  checklist verdicts, and the Arrhenius pair (recovery, (A2) witnesses,
  linearisation limit).
- `scripts/reproduce.py --micro` and `scripts/darcy_reproduce.py --micro`
  are seconds-long wiring checks for every problem; full campaigns are the
  operator-run handoff scripts.
