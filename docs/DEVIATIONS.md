# Implementation notes

How the code realises the paper's theory, which free modelling choices were
made concretely, and which behaviour is a known theoretical limit rather than
a bug. The reference implementations for the published figures exist
separately; this codebase targets the theory in the paper text with a clean,
extensible design, so its numbers are its own.

## Noise model — `mras/noise.py`

- Observations are `z = u_dagger + level * e / ||e||_M * ||u_dagger||_M`,
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
  `||z - u_dagger||_L2 = level * ||u_dagger||_L2` holds exactly; level 0
  returns the field unchanged.
- Every draw is seeded: the RNG is keyed by `(seed, level, counter)`, so a
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
- Each noise level is a fresh start: `run()` builds a new `State` per level,
  `init_level` supplies the t = 0 reference and the initial estimate, and
  `State.set_z` primes `z_old` with the level's first datum, so the data
  difference term starts at zero on every level.
- Dirichlet sets follow the reference fields: Fisher carries "wall"
  conditions on U_h, Darcy's Q_h space is free, and potential/Allen-Cahn
  use free spaces because their references do not vanish on the boundary.
- With the P0 L2 parameter space, `Grad(s) = 0` and `s * ds` boundary pairs
  evaluate to zero, so the s-row boundary-flux terms are omitted.
- P3 U_h / P0 Q_h on the 2D unit geometries of `mras/domains.py`; data
  meshes, sampling strides and figure instants are per-problem
  (`data_h`, `data_order`, `series_counters`).

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

## Verification anchors

- `pytest tests/` (34 tests) covers the fixed-point bookkeeping (t = 0
  error seeding for all four problems, seeded reproducibility, fresh-start
  independence), the noise whitening (covariance moments, exact relative
  level), degree-of-freedom anchors, and the field helpers.
- `scripts/reproduce.py --micro` and `scripts/darcy_reproduce.py --micro`
  are seconds-long wiring checks for every problem; full campaigns are the
  operator-run handoff scripts.
