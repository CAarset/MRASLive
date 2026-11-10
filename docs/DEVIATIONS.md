# Implementation notes

How the code realises the paper's theory, which free modelling choices were
made concretely, and which behaviour is a known theoretical limit rather than
a bug. The reference implementations for the published figures exist
separately; this codebase targets the theory in the paper text with a clean,
extensible design, so its numbers are its own.

## Data — one discretisation, the true parameter apart (2026-11-04)

- The data are the estimator's own discrete model -- its mesh, its `U_h`,
  the forward scheme -- stepped at the true parameter (`ForwardRun`).
  Only `q_true` has a mesh of its own (`q_domains`, following its jumps)
  or none (an analytic field): the user's rule, since the true parameter
  is a physical quantity and everything else -- state, datum, estimate,
  errors -- is one internally consistent computational tool.  The
  driver's `data_domains` (a finer data mesh, `data_h`/`data_order`) is
  gone; Fisher's `data_h` survives as `q_h`, the width of the ring's
  own mesh.
- Until then the data came from a finer mesh (`h/2`, or `data_h` with a
  higher order), projected onto `U_h`.  That followed the AMMC paper,
  which generates its Darcy data at `h = 0.03, k = 4` against `0.04, 3`
  and its Fisher data at `0.09, 4` against `0.1, 3`, "to avoid inverse
  crime"; here the code departs from the paper.
- The theory's grounds.  KN21 (1)-(3) analyses data `z = u*` that solve
  the model at `q*`, and the error system (19)-(21) closes because the
  coupling terms cancel there; discretely that holds for data from the
  estimator's own discrete model.  With `q_true` outside `Q_h` the rest is
  `f(q_true - P_h q_true, z)`, exactly the effective noise level of
  `mras/robust.py`.  Data from another discretisation instead perturb the
  model: a perturbation present at all times, outside the noise class
  `L^p(0, inf; Z)` of KN21 (27), whose noise theory is for `sigma = 1`
  only; KN21's outlook names perturbations of the model as future work.
  The inverse-crime concern is met where the theory puts it: `q_true` is
  not in `Q_h` (unfitted estimator meshes), and noise enters explicitly,
  by (27)-(28).
- With `fitted=True` (and Darcy's fitted grid) `q_true` lies in `Q_h`:
  the data are then the estimator's exact discrete model, the effective
  noise level is zero, and a run checks the theory's exact-data
  statement itself, not its robustness.
- Earlier evidence: 1D Burgers' own-mesh data removed a bias the
  fine-mesh data left (the curved-mesh study below), and 2D Burgers at
  `S = 5` semi-converged on fine-mesh data (the static-inflow table).
  Results before 2026-11-04 are not comparable with later ones.

## Error norms (2026-11-04)

- `error_q` splits exactly, `||qhat - q||^2 = ||qhat - P q||^2 +
  ||P q - q||^2` (Pythagoras: `P q`, `project_q`, is the L2 projection in
  the very quadrature the norms use, four orders above the driver's), so
  each step takes a mass-matrix norm on Q_h, and the true parameter, on
  its own mesh, is evaluated once a run (`MRASProblem._true_parts`, which
  `q_floor`, the effective noise's reference and the model fit share).
  `error_u` and the misfit go through the mass matrix of U_h in the
  driver's quadrature.  Both agree with direct integration in the same
  quadrature to 1e-15.  They replace `Integrate` at NGSolve's default
  order 5, which under-resolved a true parameter jumping inside elements:
  `error_q` moves by 6e-4 (2D Burgers) and 3e-3 (Fisher) relative, the
  misfit by 5e-12.  At the 2D production size the norms went from 0.30 s
  to 0.008 s a step (a 10-step noisy smoke 4.0 s -> 2.7 s a step, under a
  heavier load the second time).

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

- The estimator starts in the exact initial state: error histories begin
  at t = 0 with `error_u[0] = 0`, which isolates estimator convergence
  from initial-condition transients.  The initial state takes the
  Dirichlet data exactly (interpolation alone leaves it off by the
  projection error on the boundary), for the estimator and the forward
  model alike.
- Darcy (`examples/darcy.py`): initial state `sin(pi x) sin(pi y)`; the
  source is i.i.d. standard-normal 128x128 pixel fields drawn at the
  keyframes `t = k * 0.02` from a stream keyed by (seed, k) and linear in
  time between them (2026-10-29), so it is a function of time, the same
  for every step size; the paper's fresh field per step is the case
  `dt = 0.02`, which it reproduces exactly.  (A field per step made the
  forcing depend on `dt`: its effect on the state shrinks like `sqrt(dt)`,
  so a finer step meant weaker excitation.)  The true permeability is the
  PDEBench sample as PDEBench defines it, constant on each of its
  128 x 128 cells (a voxel coefficient, exact at every point, painted on
  the cell grid itself).  The estimator's mesh is unstructured and as fine
  as the cells (`h = 1/128`), the data on the same mesh (on it refined
  once until 2026-11-04); it does not follow the cells (`fitted=True`
  puts the estimator on the cell grid),
  so q is estimated in P0 across elements its jumps cut through: best fit
  7.9 % off at the defaults (P1 5.1 %; 5.6 % and 3.6 % at `h = 1/256`),
  the error along the jumps.  Before 2026-10-29 it was interpolated
  bilinearly and ramped to zero over a 5 % strip along the boundary (the
  paper's code), so it lay in no `Q_h` (P1 floor 12 % at `h = 0.08`) and
  the diffusion vanished at the wall; before 2026-10-28 the errors were
  also measured against a different projection than the data used.  The
  PDEBench sample is binary, 0.1 and 1.0 (a 10:1 contrast, though the file
  is named beta100: beta is PDEBench's forcing parameter, not the
  contrast).
- Fisher-KPP (`examples/fisher.py`): rest start, a ring-shaped true
  parameter, 1 between r = 0.3 and r = 1.05 and 0.25 elsewhere (widened
  2026-10-29 from 0.5 - 0.9, the paper's).  The ring is defined on a mesh
  of its own that follows both circles, exactly; the estimator's mesh and
  the data's do not, so q is estimated in P0 across elements the circles
  cut through: best fit 8.1 % off at the defaults (`h = 0.04`), the error
  along the circles.  The
  forcing (2026-10-29, not the paper's) is a train of waves running out
  from the centre, `6 sin(2 pi r / 0.8 - 2 t)` (a centre wandering on a
  circle of radius 0.15 until the user's simplification, 2026-10-29); the
  amplitude keeps the datum below 1 (peak 0.88), past which the logistic
  term blows the state up.  The paper's Gaussian pulse at the centre,
  `3 exp(-r^2 / 0.4) cos(2 t)` from `u0 = 3 exp(-r^2 / 0.4)`, is 4e-4 of
  its peak in the corners, which it therefore never excites: on a coarse
  check (`h = 0.15`, `dt = 0.02`, `t = 8`) it ends at 25.8 % with the
  corners at 116 %, the waves at 0.51 %, corners 0.8 %, core 0.6 %.
  `scripts/reproduce.py --problem fisher` therefore no longer reproduces
  the paper's Fisher figures.
- Piecewise constant parameters (Fisher, Allen-Cahn, 2D Burgers, Darcy;
  2026-10-29) live on a mesh of their own that follows their jumps
  (`problem.q_domains`), where they are exact -- a P0 field set by
  subdomain (`mras.fields.subdomain_field`), or Darcy's voxel
  coefficient -- and the pictures paint them there.  The estimator's
  mesh and the data's do not follow the jumps: a mesh that does hands the
  estimator q's discontinuities (and the data model them, an inverse
  crime), and the best fit of q in P0 would be exact.  They evaluate q
  across their elements by point search, so the best fit is off along
  the jumps, and the error concentrates there.  `fitted=True` puts the
  estimator (q-hat and, as one step solves for both on one mesh, u) on a
  fitted mesh; the data stay unfitted.  On the fitted mesh, the vertices
  lie on the interfaces, where a test of the coordinates lands on either
  side by round-off: the subdomains, not the coordinates, say which piece
  an element is in.
- Potential and Allen-Cahn: analytic references
  `u_dagger(x, t) = u_wot(x) (DECAY_T - t) / DECAY_T + OFFSET` with
  (DECAY_T, OFFSET) = (6, 6) and (10, 1) respectively, and manufactured
  sources `g = u_dagger_t + f(q_dagger, u_dagger)` built by the driver
  from the same `f` the estimator uses.  Both models are the paper's
  (Props. 6.1, 7.1): `f(c, z) = -Laplace z + c z + c |c|^{2/3} z` and
  `-Laplace z + c z^3 + c |c|^{2/3} z`.  Before 2026-10-28 the checklist's
  `coupling` hook of the potential carried a 3/5 factor the forms did not
  (`q + 0.6 q |q|^{2/3}`), the forms used the Picard multiplier
  `(1 + |q^n|^{2/3})` and the tracking factor at `q~ = 0`, so the cross
  blocks were not adjoint (measured ratios 0.65 and 0.96); now they are,
  to machine precision.  Potential uses P1 for q (q_dagger = pi - r is
  continuous: best fit 0.03 %, P0 2 %); the Allen-Cahn q_dagger, piecewise
  constant on three pieces, lives on a mesh of its own that follows their
  interfaces, and the estimator's does not: best fit in P0 7.1 % off at
  the defaults (`h = 0.12`; P1 4.6 %), the error along the interfaces.
- Potential and Allen-Cahn boundary conditions (2026-10-29): the state is
  prescribed on the whole boundary, `u = h` as in the paper's (26) and
  (38), with `h` the reference state's boundary values (`dirichlet(t)`;
  the analytic datum takes the same values on the constrained dofs, so
  `u - z` vanishes there exactly).  Before, both ran with natural
  conditions on all of `H1` (homogeneous Dirichlet data would have
  precluded the offset reference), where `C = c (-Laplace)` vanishes on
  the constants: (A3) failed, and the state error settled at a floor set
  by the untracked constant mode.  With `u - z` in `H1_0`, `-Laplace_Tr`
  of the paper's Prop. 6.1 acts as `-Laplace`, and `C` is coercive.
- The potential's `u_wot` is the speed of a steady vorticity field,
  solved by damped Picard iteration (2026-10-29) until the residual on the
  free dofs has dropped by 1e-10, in 35 iterations and about 3 s at the
  defaults.  The paper's code took 15 iterations against a residual that
  counted the Dirichlet rows, which never fall, so it always stopped at 15
  -- 3e-5 short of the fixed point -- and refactorised two sparse matrices
  each time: 26 s of setup.  The Laplacian is now factorised once and the
  vorticity corrections are solved by GMRES preconditioned with it.  The
  forcing of that solve is a seeded normal value per dof, so `u_wot`, and
  with it the reference state, changes with the mesh.

## Discretisation and driver — `mras/scheme.py`, `mras/forward.py`

- One physics, everywhere.  A problem declares `f(q, u)` once, as the
  q-free `model` and the q-dependent `coupling` with its adjoint density
  `coupling_adjoint`; the driver builds the forward model, the MRAS step of
  every scheme, and the checklist's probes from those hooks.  Before
  2026-10-28 every example hand-wrote its joint step forms and its forward
  forms separately, and nothing tied either to the hooks the checklist
  verified.
- One quadrature.  Every form the driver builds uses one explicit rule of
  order `2 max(k_u, k_q) + 2` (`mras.solvers.measures`), exact for the
  cubic P3 products of the models here.  NGSolve otherwise picks the rule
  from the space a form lives on, so the same integrand was integrated
  differently on Q_h (degree 2 for P1) and on U_h x Q_h (degree 6) -- a 5 %
  difference in one parameter step between solving by rows and jointly.
- Time schemes (`scheme`): backward Euler on (4) (`"implicit"`, the
  default), the paper's (13)-(14) (`"semi-implicit"`), and the sequential
  prediction-correction reading (`"explicit"`).  All three take the
  norm-dependence of C at q^n, as the paper does ("significantly more
  feasible", p. 88), and the tracking factor `f_q(q~, z)^*` at `q~ = q^n`:
  the theory leaves q~ open, and at q^n the step never needs `f_qq`, which
  does not exist for `q |q|^{2/3}` at q = 0 (a Newton step through it was
  singular on the zero initial estimate).  sigma = 1 makes the implicit
  step nonlinear; it is iterated by Newton's method on f with the declared
  derivative, to `newton_tol = 1e-10`.  The explicit scheme is only
  conditionally stable: with the potential's stiff reaction (z ~ 100) it
  diverges at dt = 0.05.
- Solves (`split`).  The joint system is never formed by default.  In the
  semi-implicit and explicit schemes the parameter row does not see
  u^{n+1}: q^{n+1} is solved for first (an L2 mass matrix, block
  diagonal), then u^{n+1} with `S_u = M + dt c A` factorised once per run
  (once per change of c).  In the implicit scheme the q-u coupling runs
  both ways and the step goes through the Schur complement on the state,
  `(S_u + K M_q^-1 K^T) u = ...`, by CG to round-off preconditioned with
  the factors of `S_u`.  `split=False` assembles and factorises the joint
  system every step, the reference: the two agree to round-off
  (`tests/test_generic.py`).  A problem that overrides `forms` with a
  joint form of its own is checked on its first step and solved by rows
  when its parameter row does not see u^{n+1}.  A step whose parameter row
  depends on u^{n+1} -- the implicit one here, or any implicit treatment of
  the tracking term (Crank-Nicolson, higher-order implicit Runge-Kutta),
  or a parameter in a constraint or a boundary condition -- is not block
  triangular.
- Constant matrices are assembled once per run: the masses of U_h and Q_h,
  the stiffness of C, the state matrix and its factors.  A step assembles
  only what depends on the datum: the load vectors, the coupling block of
  the implicit step, the sigma = 1 parameter matrix.
- Threads.  Each run executes under NGSolve's `TaskManager` (assembly,
  interpolation and integration were single-threaded before 2026-10-28).
  Threaded assembly sums in an order that varies between runs, so
  same-seed runs agree to round-off (2e-15 relative; 6e-10 for the
  potential, whose C ~ 1e6 amplifies it), not bitwise; the reproducibility
  check compares at 1e-8, far below a different noise draw.
- Forward model (`mras.forward.ForwardRun`): `u_t + f(q_true, u) = g` by
  backward Euler (`forward_scheme = "implicit"`, Newton's method to
  `newton_tol`), one Newton step (`"semi-implicit"`, linearly implicit
  Euler) or forward Euler (`"explicit"`).  Terms of f found linear in u by
  probing are assembled once with the mass matrix; a linear model (Darcy,
  the template) is one direct solve per step with
  factors computed once, and a nonlinear one solves its Jacobian by GMRES
  to round-off, preconditioned with the factors of the step matrix
  linearised at the initial state (renewed when the iteration count
  grows).  On the Burgers data mesh of the time (27k dofs) that is 10 iterations and
  67 ms against 550-750 ms for a fresh LU, agreeing to 3.6e-15.  Before
  2026-10-28 Burgers lagged its advection (Picard) instead.
- Forward data are generated in lockstep: each run owns a `ForwardRun`
  that takes one step whenever the estimator asks for the next datum, so
  nothing is precomputed or stored.  `MRASProblem._forward_data()` still
  returns the whole trajectory for theory checks that need it (Arrhenius
  ess-inf witnesses).  Darcy and Fisher precomputed their whole truth
  trajectories before 2026-10-28.
- Dirichlet data (`dirichlet(t)`, `mras.forward.BoundaryData`): declared
  by boundary, `{name: value}` for boundaries on the Dirichlet part of
  `U_h` (the others held at zero), and imposed as NGSolve imposes boundary
  values -- `Set(value, BND, definedon=mesh.Boundaries(name))`, the L2
  projection onto the traces, on that boundary's dofs only (2026-10-29;
  before, one coefficient for the whole Dirichlet part, so 2D Burgers
  picked its inlet out by a test of the height).  Naming a boundary that
  is not on the Dirichlet part, or none, is an error.  The new state takes
  the boundary values at t^{n+1} on the constrained dofs and the step
  solves for the free ones, in the forward model and the estimator
  alike.  The datum, the estimator's own discrete model, takes the same
  boundary values on its constrained dofs, so the MRAS error u - z stays
  in the homogeneous space.  (Until 2026-11-04 the datum was projected
  from a finer data mesh, and its constrained dofs were reset to the
  boundary data: the projection smeared steep boundary layers, 2 % at the
  inlet of an inflow starting into fluid at rest.)
- Time levels of the data (the paper's step (13)): step n -> n+1 reads the
  datum `z^{n+1}` and the source `g^{n+1} = g(t^{n+1})`, and the forward
  model that generated `z^{n+1}` was driven by the same `g^{n+1}`.  With
  `g(t^n)` instead (before 2026-10-27) exact data are no fixed point of the
  step: the residual `dt g_t` biased the Burgers state by
  `||u - u_dagger||` ~ 0.02.  `tests/test_generic.py` pins the fixed point
  for every scheme and solve.
- Each noise level is a fresh start: `run()` builds a new `State` per
  noise level, `init_run` supplies the t = 0 reference and the initial
  estimate, and the driver primes `z` and `z_old` with the datum `z^0` at
  t = 0 (the reference plus its own seeded noise draw), so step 1 reads
  `f_q(z^0)^*(u^0 - z^0)` and `D_t z^1 = (z^1 - z^0)/dt` as in the paper's
  (14).  Observation-only problems have no datum at t = 0; their first
  datum primes `z_old`.
- `MRASProblem.q_floor()` is the relative L2-projection error of `q_true` in
  `Q_h`: no estimate can go below it, and because `qhat - P q` is
  orthogonal to `P q - q`, `sqrt(error_q^2 - q_floor^2)` is exactly the
  part the estimator can still remove.  It is recorded in `meta["q_floor"]`
  of every run with a known parameter.
- Q_h test functions have no trace: the sigma-bracket of the parameter row
  leaves out boundary densities (`model_boundary`).  On P0 the gradient of
  a test function vanishes too, so the bracket's Laplacian drops out
  entirely; on P1 (potential) it is the elementwise gradient.  Both cancel
  exactly on noise-free data.
- The implicit step's Schur solve falls back to the factorised joint
  solve when CG does not reach round-off, and the run takes the joint
  solve from then on (2026-10-29).  Before, `Solve` fell back only when
  handed a direct factorisation, which the Schur solve was not, so an
  unconverged CG iterate was taken as the step without a word.  That
  happens once the coupling swamps the preconditioner -- `K M_q^-1 K^T`
  against `M + dt c A`, with `|grad z|^2` in the thousands -- and the step
  is then not the energy-stable one: in the 2D Burgers channel at
  `S U = 100` the discrete energy `(|u - z|^2 + |q - P q|^2) / 2` fell
  until t = 0.9 and then doubled every three steps (to 22.7 at t = 2 on
  `h = 0.25`); with the fallback it falls monotonically (0.15 at t = 2).
- Projections onto Q_h (`MRASProblem.project_q`: the floor, the initial
  estimate, the painted q) are assembled -- the element mass matrices on
  the curved geometry against a load four quadrature orders above the
  driver's (2026-10-29).  `GridFunction.Set` on an L2 space inverts the
  straight element's mass matrix, so on curved elements it is not the
  projection: it put the constants of the Fisher ring 0.5 % off (a
  floor of 0.19 % for a field that lies in P0), and overstated the
  Burgers floors (P1 at `h = 0.05`: 0.074 % for 0.069 %).
- A run stops once its reconstruction error has come down to its target
  (2026-10-29), at the latest at `tend` -- the time limit, 100 for most
  examples (the potential and Allen-Cahn: their decay times, 6 and 10).
  The target is 1.25 times the floor unless set
  (`problem.target`, `run_live(target=...)`, `--target`; 0 runs to
  `tend`): the floor itself is no target, as no estimate in `Q_h` gets
  below it.  Until 2026-11-04 it was max(1 %, 1.25 times the floor),
  which stopped the noise-free 1D Burgers run at 1 % on its way down to
  its 0.069 % floor.  The rule reads the true parameter, which only benchmarks
  have; with noise a discrepancy principle is to replace it (not
  implemented).  Runs record how they ended (`meta["stopped"]`,
  `"target"` or `"tend"`, and `meta["t_end"]`); the checklist audits at a
  fixed horizon (target 0).  Step `n` is at `t = tstart + n dt`, never
  accumulated.  The paper's code looped `while t < tend - 2 dt` with
  `t += dt`, which stopped one or two steps short, depending on
  round-off.
- Two lines mark how far the estimate can come (2026-10-29).  The L2
  floor (dotted, "best fit in Q_h") is the closest any estimate in `Q_h`
  gets to q, a bound.  The model fit (dashed, "discretization error",
  `mras.floor`) estimates where the estimator plateaus: the q_h that best
  explains the data in the estimator's own discrete model, weighted as its
  error equation weights the residual (by the inverse tracking operator),
  refitted at steps 50, 100, 200, ... and at the end, each time from the
  last fit (a Gauss-Newton iteration for the nonlinear examples).  It sits
  above the L2 floor: on an unfitted mesh the data fix a gradient-weighted
  average of q in the elements a jump cuts, not its area average.  It is
  an estimate, not a bound: on Fisher (`h = 0.1`, to t = 4) the L2 floor
  is 13.3 %, the model fit 17.7 %, the estimator's plateau 16.5 %.  (Weighted
  by the mass matrix's inverse diagonal instead, the fit came out at
  25.8 %, far above the plateau: that weighting treats fine-scale residuals
  like smooth ones.)  A run whose error is at or below the model fit and
  fell by less than 2 % since the last refit stops (`"plateau"`).
- The trajectory a checkpoint keeps: every step, until the budget is
  full; then every other stored step is dropped and the stride doubles, so
  a run of any length keeps evenly spaced steps up to the budget.  Movies
  of a finished run know its end, and their time axis ends there; the
  live view's ends at the time shown, and its colour ranges' time
  constants grow with the time so far.
- Snapshot steps (`series_steps`) are 1, 5, 10, 25, 50, ... below the
  step count, plus the last step, kept after the initial state; scripts
  that need the paper's figure instants set them.
- Error metrics: `error_q` is the relative reconstruction error
  `||qhat - q||_L2 / ||q||_L2` (the paper's benchmark tables report
  relative errors, Sec. 8); `error_u` and the data misfit stay absolute.

## Scalar Burgers in the pipe — `examples/burgers_1d.py` (no paper benchmark)

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
  `dt ||grad z||_inf^2 / (2 lambda_1)`; the measured runs stayed stable
  with margin up to `dt = 0.2` (the step sweep below), where accuracy,
  not stability, is the limit.

Ad-hoc modelling (no theory attached, none claimed):

- The truth generator is the estimator's discretisation stepped at
  `q_true` (backward Euler, Newton; on a finer data mesh until
  2026-11-04); it produces observations only and enters no assumption.
- Wake parameter `q_true`: two smooth low-diffusion wake bands straddling
  the hole row (`y = 0.4`/`0.8`, mildly modulated along `x`).  The formula
  is written with `Q_MIN, Q_MAX = 0.12, 0.55`, but `wake / (1 + wake)`
  never exceeds about 1/2, so the field actually spans about 0.33 to 0.55.
  Its best P0 approximation (`q_floor()`) has relative error 5.6% at
  `h = 0.35`, ~3% at `h = 0.15` and ~2.3% at `h = 0.1`; in discontinuous
  P1 0.52% and 0.30%, in P2 0.07% and 0.03% (measured with
  `GridFunction.Set`, slightly above the exact projection on the curved
  elements: P1 0.26 % at `h = 0.1`).  The smooth bands
  are a modelling choice (abrupt disks leave
  most of the L2 error in the kink layer: measured wake-region error
  drops from 2.09 to 0.76 when the wake is smoothed, at unchanged global
  error).
- Zero initial estimate `qhat = 0`; state starts at rest (the beams drive
  it).
- Geometry: the pipe meshes are curved to the state order
  (`mras.domains.burgers_pipe`).  Straight-sided meshes of widths `h` and
  `h/2` cut different polygons out of the pipe at the holes, so the
  fine-mesh data violated the estimator's Dirichlet condition near every
  hole; the parameter flow absorbed that residual as a bias that capped
  `error_q` near 0.07 and made it *rise* once the excitation was strong
  (evidence below).
- Source (2026-10-28, tuned by the user): three beams of light, one
  emitted from each hole -- a wedge of 15 +- 5 degrees, constant along its
  direction out to the walls or the pipe ends, flat across it with a smooth
  fall-off at its edges (`exp(-(angle/half)^4)`), pulsing as `|sin|`
  (positive light; `signed=True` heats and cools), breathing in width and
  turning slowly about its hole, alternate beams in opposite senses.  Light
  does not pass the holes: the shadow of hole j as seen from hole k is
  `(D dy)^2 < (R L)^2, D dx > 0, D dx < L^2` (all holes on the centre line,
  `D` their distance, `L` the distance from hole k), three IfPos tests and
  no root.  Periods `P_n = 24 * 2**frac(n PHI)` with the user's
  `PHI = (1 + sqrt 5) / 3`: 25 - 39 time units, at least 1.4 apart,
  pairwise incommensurate; the turns at twice that; all stretched by
  `period_scale`.  `burgers_1d_dark` is the same pipe with the left
  beam off.  Since 2026-11-07 (the user's): holes of radius 0.27 (from 0.18),
  a larger shadow for the dark pipe, and light never below a quarter of its
  peak (`PULSE_FLOOR`; the pulse is `1/4 + 3/4 |sin|`).  Since 2026-11-09
  (the user's): the middle beam turns with period 50 (`TURNS`), two full
  turns by t = 100, where twice its drawn period, 66.6, gave 1.50; its start
  (120 degrees), sense, pulse and width are as before, and so are the other
  beams (1.77 and 1.28 turns by t = 100).  Production at t = 100 (h = 0.05,
  dt = 0.025), against 2026-11-08's: `burgers_1d` 0.39 % / 6.55 % / 26.1 %
  at noise 0 / 1 % / 5 % (0.43 / 6.24 / 24.8 %); `burgers_1d_dark` 25.0 % /
  26.1 % / 35.5 % (28.9 / 29.9 / 38.6 %), whose post-hoc PE now passes
  (mu >= 0.163, it failed).  Noise-free by thirds (left / middle / right):
  0.34 / 0.07 / 0.55 % and, dark, 44.6 / 0.19 / 0.56 %.  Every run is at
  its least error at t = 100.  Before (2026-10-27/28): six Gaussian heaters on a 3x2 grid
  sweeping Lissajous paths, then three heaters circling the holes (which
  left the pipe ends unexcited and q there unidentified, error ~12 %);
  earlier still the same six heaters fixed in place.  For the long-time
  average of the PE Gramian only distinct frequencies matter; its rank and
  conditioning are set by the spatial response patterns the forcing
  excites and their amplitude (the identification rate scales like
  `|grad z|^2`).
Forcing and step size, 2026-10-28 (defaults `h = 0.1`, P1, implicit
scheme, noise-free; `error_q` at `t = 50 / 100 / 200 / 300 / 400`):

| forcing, `dt` | error_q | below 1 % from |
|---|---|---|
| six heaters, fast (pulses 1.4 - 2.7), 0.05 | 0.124 / 0.068 / 0.028 / 0.0145 / 0.0090 | t = 376 |
| six heaters, fast, 0.025 | 0.122 / 0.066 / 0.027 / 0.0140 / 0.0088 | t = 366 |
| six heaters, slow (periods 21.6 - 50), 0.05 | 0.127 / 0.072 / 0.028 / 0.0132 / 0.0078 | t = 359 |
| six heaters, slow, 0.025 | 0.127 / 0.072 / 0.028 / 0.0131 / 0.0078 | t = 359 |
| three heaters circling the holes, 0.05 | 0.296 / 0.223 / 0.126 / 0.123 / 0.118 | -- |

Halving the step does not move the reconstruction, and neither does the
forcing's speed: what limits it is the excitation's spatial reach.  The
three heaters circling the holes reach x = 0.9 ... 5.1 only; the pipe ends
beyond the outer orbits, between them and the absorbing inlet and outlet,
are never heated, the datum has almost no gradient there, and q is not
identified there -- the error plateaus near 12 %, all of it at the two
ends, while the middle of the pipe is identified as well as before.  The
identification also comes in bursts, at strong pulses.  At a noise level of
10 % the three heaters do not move the estimate past its first 5 % on the
coarse test mesh (h = 0.35, t = 4), where the six did.
- Defaults (2026-10-28, set by the user) `h = 0.05, dt = 0.025,
  tend = 75, k_q = 1`: `Q_h` is discontinuous P1, the lowest order whose
  floor is under 1% (0.069 % at `h = 0.05`, 0.26 % at `h = 0.1`; P0 1.1 %
  and 2.3 %), so a reconstruction within 1% is possible in principle.

Burgers PE evidence, 2026-10-27 (noise-free, `h = 0.15`, `dt = 0.05`,
`k_u = 3`, `error_q` at `t = 10 / 25 / 50 / 100 / 150 / 200`, floor 3.1%):

| configuration | error_q |
|---|---|
| fixed heaters (previous source) | 0.587 / 0.501 / 0.426 / 0.346 / 0.298 / 0.264 |
| fixed heaters, amplitude x2 | 0.438 / 0.345 / 0.259 / 0.179 / 0.147 / 0.133 |
| sweeping heaters, straight meshes | 0.294 / 0.149 / 0.083 / 0.061 / 0.069 / 0.073 |
| sweeping heaters, amplitude x2, straight meshes | 0.130 / 0.075 / 0.074 / 0.075 / 0.081 / 0.086 |
| sweeping, straight, `dt = 0.025` | 0.290 / 0.147 / 0.082 / 0.061 / 0.069 / 0.074 |
| sweeping, straight, advection `z^n d_x z^{n+1}` | 0.294 / 0.149 / 0.083 / 0.061 / 0.068 / 0.073 |
| sweeping, data on the estimator's own mesh | 0.303 / 0.159 / 0.088 / 0.041 / 0.036 / 0.035 |
| sweeping, own mesh, `q_true` replaced by its P0 projection | 0.301 / 0.155 / 0.082 / 0.025 / 0.0099 / 0.0038 |
| **sweeping heaters, curved meshes (shipped)** | **0.305 / 0.161 / 0.090 / 0.042 / 0.036 / 0.035** |

Reading: sweeping heaters identify about 6x faster than the fixed grid.
With straight meshes the error then rises again to a plateau that is
neither temporal (`dt` and the advection time level change nothing) nor
excitation (stronger forcing raises it); data on the estimator's own mesh
remove it, and the inverse-crime run (truth already in `Q_h`) converges
exponentially, to 0.4% at `t = 200`.  Curving both meshes gives the
own-mesh curve with fine-mesh data: `error_q` 3.5% at `t = 200`, of which
3.1% is the P0 floor and 1.5% (`sqrt(e^2 - floor^2)`) is left to the
estimator.  Cost: 180 ms per step (estimator plus lockstep forward run on
the curved `h/2` mesh).

Beams, 2026-10-28 (h = 0.1, P1, implicit scheme, noise-free; relative
`error_q` at the end, and over the left / middle / right third of the pipe,
x < 1.5 / 1.5 - 4.5 / > 4.5):

| run | `tend` | `dt` | steps | periods x | error_q | left / middle / right |
|---|---|---|---|---|---|---|
| three beams, light (`|sin|`) | 75 | 0.025 | 2999 | 1 | 0.0267 | 0.034 / 0.017 / 0.033 |
| three beams, heating and cooling | 75 | 0.025 | 2999 | 1 | 0.0262 | 0.034 / 0.016 / 0.033 |
| same sweeps in half the time | 37.5 | 0.025 | 1499 | 0.5 | 0.0219 | 0.029 / 0.015 / 0.025 |
| same, as many steps as the first | 37.5 | 0.0125 | 2999 | 0.5 | 0.0219 | 0.029 / 0.015 / 0.025 |
| same sweeps in twice the time | 150 | 0.025 | 5998 | 2 | 0.0334 | 0.042 / 0.021 / 0.043 |
| same, as many steps as the first | 150 | 0.05 | 2999 | 2 | 0.0333 | 0.042 / 0.021 / 0.043 |
| left beam off (`burgers_1d_dark`) | 75 | 0.025 | 2999 | 1 | 0.0618 | **0.123** / 0.020 / 0.011 |

At the defaults (`h = 0.05`, `dt = 0.025`, `tend = 75`; P1 floor 0.069 %)
the beams reach 1.13 % (left / middle / right 1.6 / 0.5 / 1.4 %), passing
0.66 / 0.15 / 0.043 at a quarter / half / three quarters of the run and
still falling about fourfold per quarter at its end; the dark-left pipe
reaches 6.3 % (12.7 / 1.5 / 1.5 %).  Each run takes about 45 minutes on 16
threads.

Reading: the sign of the light does not matter, and the number of steps
does not either (at fixed `tend`, halving `dt` changes the fourth digit).
With the sweeps scaled to the horizon, so that the same area is lit in the
same order, the reconstruction is not set by the total time: twice the time
ends worse, at 3.3 % against 2.7 %, and half the time better, at 2.2 %.  The
longer runs are ahead half-way (0.115 against 0.149 and 0.189) and then
stall over their last quarter.  Excitation acts where the light falls: the
excitation dose `D = int |grad z|^2 dt` per element (the local density of the
identification Gramian) ranks the local error with Spearman -0.96 half-way
through the three-beam run (`h = 0.1`).  By its end every region has had
enough light and the relation is weak (at the defaults, Spearman -0.23;
median local error 0.5 % in the least-lit decile of elements, 0.1 - 0.35 %
in the others).  With the left beam off, the left end is barely lit and
stays unrecovered: the least-lit decile keeps a median error of 15 %, the
next 2.6 %, the rest 0.1 - 1.2 % (Spearman -0.65;
`archive/burgers_pipe/excitation_vs_error.png`, at the defaults).

Order of `Q_h`, 2026-10-27 (shipped source, curved meshes, noise-free,
`dt = 0.05`; `error_q` at `t = 10 / 25 / 50 / 100 / 150 / 200 / 250`):

| configuration (floor) | error_q |
|---|---|
| `h = 0.1`, P1 -- defaults (0.30%) | 0.320 / 0.188 / 0.124 / 0.068 / 0.043 / 0.028 / **0.020** |
| `h = 0.1`, P0 (2.29%) | 0.311 / 0.171 / 0.102 / 0.045 / 0.030 / 0.027 / 0.026 |
| `h = 0.15`, P1 (0.52%) | 0.318 / 0.183 / 0.115 / 0.055 / 0.032 / 0.020 / **0.013** |

Reading: P0 at `h = 0.1` has stalled at its floor plus 1.1%.  P1 removes
the floor but triples the parameter unknowns, which the same forcing
identifies more slowly, so the P1 curve trails P0 until `t ~ 200` and passes
it after.  Both P1 runs are still falling by about 30% per 50 time units
at `t = 250`: below 1% needs about `t = 350` at `h = 0.1` and `t = 300` at
`h = 0.15` (extrapolated, not run).  The coarser P1 mesh is ahead at every
time, with half the unknowns and half the cost per step.

Open — where the code cannot force conformance:

- (A2) coercivity of the coupled linearisation is unverified, exactly as
  the paper leaves it for its own Darcy benchmark (Remark 4.2); the
  justification there is persistent excitation (PE) of the source.
- PE evidence, earlier campaign (Gaussian-bump start-up, logistic wake,
  horizons 5–15, `dt = 0.005`–`0.02`): every single-knob variant — single
  frequency, richer frequencies, finer `dt`, drifting heater centres —
  left `error_q` in the 0.77–0.83 band while `||u - z||` stayed near
  0.02.  The diagnosis held (the state tracks without identifying `q`)
  and fed the redesign above.
- Re-measured after the `g^{n+1}` and `z^0` fixes, before the sweeping
  source and curved meshes (2026-10-27; fixed heaters, noise-free,
  `h = 0.35`, `dt = 0.05`, one run to `T = 800`):
  `error_q` 0.234 / 0.150 / 0.110 / 0.090 at `t = 200 / 400 / 600 / 800`
  against 0.221 / 0.138 / 0.109 / 0.096 with the misaligned source below,
  with the state now tracking at `||u - u_dagger||` = 0.0010 instead of
  0.018.  The misaligned source is faster early, because its spurious
  residual acts as extra excitation, but levels off on its bias; the
  aligned run overtakes by `t` ~ 600 and is still descending at 800.
  With the floor of 5.6% at this mesh, the removable part at `T = 800`
  is still `sqrt(0.090^2 - 0.056^2)` ~ 7%: the "discretisation-limited
  floor near 0.09" read off the table below was partly the P0 floor,
  partly the source-time bias, and partly not yet converged.
- PE evidence, fixed-heater design, measured with the source at `t^n` (before
  2026-10-27; rest start, quasi-periodic heaters; noise-free, `h = 0.35`,
  `k_u = 3`, `seed = 0`; `error_q` at cutoff, tail slope over the last
  20% of steps):

  | horizon (`dt = 0.05`) | `error_q` | tail slope / step |
  |---|---|---|
  | `T = 10` | 0.572 | -3.9e-4 |
  | `T = 25` | 0.479 | -1.7e-4 |
  | `T = 50` | 0.394 | -1.6e-4 |
  | `T = 100` | 0.308 | -7.0e-5 |
  | `T = 200` | 0.221 | -3.5e-5 |
  | `T = 400` | 0.138 | -1.5e-5 |
  | `T = 800` | 0.096 | -2.2e-6 |

  every run is monotone and still descending at its cutoff; only at
  `T = 800` does the descent level off, near 0.09, with a small wiggle
  at the very end — the discretisation-limited floor of this
  configuration.  Short runs overstate the eventual error and the
  horizon is the dominant lever; no plateau appears at any shorter
  horizon.  Against the previous integer-harmonic source (same heaters,
  same phases, exact common period) the aperiodic source is 4.8% better
  at `T = 50` (0.394 vs 0.414) and 7.8% at `T = 100` (0.308 vs 0.331) —
  a real but secondary effect, widening with horizon.
- Step size saturates for the parameter (aperiodic, `T = 50`,
  noise-free): `dt = 0.2/0.1/0.05/0.02/0.01` gives `error_q`
  0.459/0.402/0.394/0.392/0.392 while the state error keeps improving
  (0.072 -> 0.004): small steps pay for the state up to a point and buy
  the parameter nothing below `dt` about 0.05 — half the forcing period
  of the fastest heater.
- Noise floors the parameter, not the state (`T = 400`): `error_q`
  0.138 noise-free, 0.260 at noise level 1% (tail still creeping down),
  0.532 at noise level 5% with a flat to rising tail — the floor grows
  roughly linearly with the noise (about a 10x amplification, the
  signature of weak identifiability of `q` through the gradient
  coupling), while the state tracks at ~0.02 in every case.  Without
  (A2) there is no robustness claim for `qhat`, and this is the
  concrete shape of that gap.
- The noise measurements above predate the sweeping source, the curved
  meshes and the `g^{n+1}` / `z^0` fixes, and need re-measuring.  Per-step
  randomised spatial sources (the Darcy benchmark's recipe) were
  deliberately not used: the forcing stays deterministic and smooth, and
  sweeping heaters supply the excitation.
- Framework verdict (`mras/verify.py`, reproducible with
  `scripts/checklist.py`): adjoint pairing machine-precise, (A1) exact
  (`sigma = 0`, `L = 0`), (A2) FAIL — the kernel probe measures a
  sign-indefinite pairing, the gradient-coupling situation of the
  paper's Remark 4.2, (A3) OK (`L = 0`, so it is the coercivity of
  `C = -Laplace` alone, `M = 1`, whatever (A2) does), persistent
  excitation FAIL over the audit horizon `t <= 1` (the PE Gramian is
  singular to the accuracy of the solve: the beams have not yet lit the
  top corner of the inlet, where its eigenvector sits), reconstruction
  descending, seeded.  The failure is therefore localised to the one assumption the
  paper itself does not verify for its gradient-coupled benchmarks: the
  implementation passes every probe that does not presuppose (A2), and
  noise-free reconstruction descends below the band of the paper's own
  Darcy benchmark (0.33-0.60, Table 6) given enough horizon.  This is
  what "no (A2), no convergence-rate guarantee" looks
  like in practice; the Arrhenius pair is the same code on the same
  checklist with the hooks filled in, and recovers.

## Arrhenius reaction pair — `examples/arrhenius.py` (sixth/seventh examples)

Since 2026-10-29 the defaults are `h = 0.25` with `Q_h` discontinuous P1
(best fit of the smooth rate field 0.057 %; P0 at `h = 0.3` was 2.4 %),
and the template `examples/blank_problem.py` runs at `h = 0.5` with P1
(0.22 %; before, P0 at `h = 1.25`: 12 %).

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
  on the shipped `4 pi` box (inverse iteration, mass-normalised;
  analytic `2 (pi / L)^2`; the `2 pi` box measured exactly `0.5` the same
  way), so `||grad .||` is a norm on `U_h`, and in it `<Cv, v> =
  ||v||_U^2`: with `L = 0`, `M~ = M = N~ = 1`.
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
adjoint pairing, coupling pre-verification, (A1) remainder, (A2)
monotonicity, (A3) coercivity, persistent excitation of the data, the
convergence statement of the declared sigma, reconstruction descent,
seeded reproducibility.  Statuses are `ok` /
`fail` / `estimated` (measured, warned, never promoted) / `deviation`
(documented, via `adjoint_deviation`) / `unverifiable` (no probe exists —
flagged, never silently skipped).

What a program can settle decides the verdicts (2026-10-29; before, a
positive sampled minimum confirmed (A2), (A3) needed (A2), and the
persistent excitation was a proxy, the relative variation of the datum
in time):

- Discrete identities are confirmed to round-off: the adjoint pairing of
  the assembled step, `coupling_adjoint` against the numerical
  q-derivative of `coupling`, affinity in `q` (hence `L = 0`).
- `sigma` is the author's choice (2026-10-29; before, the checklist
  failed `sigma = 1` for `f` affine in `q` as "forced" to 0).  The energy
  identity of the error equations says what each choice gives, and the
  convergence row reports it: `sigma = 1`, the decay estimate from
  (A1)–(A3) (failed where the (A2) pairing is indefinite: the energy then
  grows from `eps_q(0) = p`, `eps_u(0) = 0`, `<f_q p, p> < 0`); `sigma =
  0` with `f` affine in `q`, convergence through the persistent
  excitation only — (A2) plays no part, and a reaction coupling, compact
  from `H` to `U*`, has no uniform rate at all (the proceedings'
  Remark 1) — with a note on the rate `sigma = 1` would give where (A2)
  is confirmed; `sigma = 0` with `f` nonlinear, no estimate.
- Constants that are eigenvalues are computed exactly on `Q_h`, `U_h`
  at the sampled data: the (A2) constant where `f` is affine in `q`
  (smallest eigenvalue of the pairing matrix), the coercivity of `C`, the
  persistent-excitation constant.  Dense on `Q_h` up to
  `DENSE_MAX_DOFS = 5000`, which sets the Arrhenius audit at `h = 0.5`.
- Infima and suprema over a nonlinear dependence on `q` are only
  sampled, and a sample bounds them from one side: a negative (A2)
  pairing refutes, a positive minimum (and the (A1) remainder maximum)
  is `estimated`.  Confirmation takes an analytic bound the author
  declares — `c_coe()`, `L` — which the checklist evaluates on the data
  and tests against the pairing on `Q_h`: a declared `C_coe` above it
  (by more than 1 %) fails.  Potential and Allen–Cahn declare `C_coe =
  ess inf u^dagger` and `(ess inf u^dagger)^3` (the monotone `q |q|^{2/3}`
  drops out, paper (30)).
- The U-norm is `||grad v||` where `U_h` has a Dirichlet part (a norm by
  Poincare-Friedrichs, the paper's choice) and the full H1 norm where it
  has none; (A1) measures the remainder in its dual (before:
  `mass + dt * stiffness`), (A3) the coercivity of `C` in it, the
  persistent excitation the sensitivity in its dual.
- (A3) with `L = 0` is the coercivity of `C` alone (`M~ = M`), whatever
  (A2) does: Darcy, Fisher and both Burgers examples pass with `M = 1`,
  as the paper argues for Darcy.  Without a Dirichlet part `C = c
  (-Laplace)` vanishes on the constants and (A3) fails for any `c` (so
  did potential and Allen–Cahn before they prescribed the boundary
  state); for their norm-dependent `C(||q||)` it is coercive for every
  `c > 0`, and the bound `c(r) >= L(r)^2 / (2 C_coe) + M` stays
  unverifiable without a declared `L(r)`.
- Persistent excitation: the window mean of `||f_q(q~, z) p||_U*^2`
  against `gamma ||p||_H^2`, on two windows of half the horizon (8 data
  instants); `gamma_h` is the smallest eigenvalue of the windowed Gramian
  `B^T K^-1 B` relative to the mass of `Q_h`, and its eigenvector the
  direction the data see least, reported by where it lives.  Zero to the
  accuracy of the solve (`1e-10` of the largest) fails.  For `sigma = 0`
  and `C = c K`, eliminating the fast state error leaves `D_t eps_q =
  -(B^T C^-1 B) eps_q`, so the slowest direction e-folds in about
  `c / gamma_h`: about 900 time units for Darcy, 60 for Fisher, 1.5e5 for
  the Arrhenius pair, whose weakest directions are mesh-scale (a
  pointwise coupling sees `q` through `(-Laplace)^-1`).  The verdict
  holds for the audit horizon: both Burgers examples fail there because
  their data up to `t = 1` have not reached a part of the domain.  So the
  check is repeated post hoc over the horizon a run took, which is known
  only once it has stopped: `scripts/run_examples.py` does it after each
  example's runs (`check_pe_after`: the audit discretization, CHECK or
  else QUICK, data sampled at 50 instants up to the longest `t_end`; the
  `pe` column), and `scripts/checklist.py --after results/summary.json`
  for finished runs.
- The (A2) pair probe localises test pairs on <= 3 cells: full-domain
  random pairs only measure the datum-weighted *mean* of the
  monotonicity density and average out sign-changing data.
- Gradient couplings (Darcy, Fisher) are exactly degenerate on P0
  (`grad q = 0` a.e.); the probe's H1-relaxed lift is sign-indefinite
  and reported `fail` with the Remark 4.2 note.  On Burgers' P1 `Q_h`
  the elementwise pairing exists and its exact smallest eigenvalue is
  negative (`fail`, same Remark).

`scripts/checklist.py` runs the registry and writes
`results/checklist.md`; `tests/test_verify.py` pins the documented
verdict per example.

## Noisy data and the parameter law's safeguards — `mras/robust.py`

(2026-11-03.)  Before, noisy data entered the MRAS raw — discrete white
noise on `U_h`, drawn afresh every step — and runs went on to their time
limit, which the theory (Kaltenbacher–Nguyen 2021, §2.2; the AMMC paper's
§2.4) does not cover: it inserts smoothed data, and its error bound grows
with the noise's `L^p(0, t)` norm, so persistent noise needs a stop.

- Noise model (`noise_model`, 2026-11-03): `"matern"` by default — a
  Gaussian field with Matern covariance in space (ngsrng, smoothness
  `nu = 1`, correlation length 5 % of the domain's diagonal, Robin
  boundary treatment), one field per velocity component, zero on the
  constrained dofs, drawn afresh at every step and scaled per datum to
  `||z^delta - u||_L2 = delta ||u||_L2` as the paper scales its noise;
  `"white"` is the paper's discrete Gaussian noise.  The MRAS theory
  assumes nothing of the noise but its size; smoother noise only shrinks
  the smoothing error.  ngsrng draws the field directly (one
  `Gaussian.sample_into` per datum, every component in it), from torch's
  global generator, which a run seeds with its `seed` at its start; the
  restart point carries the generator's state, so a resumed run draws on
  as the uninterrupted one.  The noise levels of one example thus share a
  realisation, scaled.  Same-seed draws repeat bit for bit on a vector
  space; a scalar field repeats to round-off only (~3e-16 relative):
  ngsrng draws a single sample inside NGSolve's TaskManager, whose
  parallel kernels sum in a varying order.  (Until 2026-11-03 the
  standard normals came from the keyed stream of the white noise, ngsrng
  only colouring them, `Gaussian.apply`.)  At 1 % on the Burgers 1D mesh
  the Matern noise is 4.8 % of the state's gradient norm, the white noise
  42 %.
- Smoothing (`smooth_data`, noisy data only): `R_sp(R_ti(z^delta))`, a
  causal exponential filter of time constant `smooth_time` (default
  `5 delta^(2/3) dt^(1/3)`: the balance of its lag, `O(tau D_t z)`, against
  the noise it leaves, `O(delta (dt/tau)^(1/2))`, fitted on the Burgers
  pipe, where Matern noise was best filtered at about 2 steps at 1 % and 5
  at 5 %: 0.188 and 0.311 at t = 30, against 0.192 and 0.615 unfiltered,
  0.186 noise-free) and (`smooth_space`, on) one implicit heat step
  `(M + alpha A)^-1 M` with the boundary data kept, whatever the noise:
  `R_sp` is the theory's map from data bounded in `Z` to data close in `U`
  (KN21 (28)), which asks nothing of the noise's law.  (On 2026-11-03 it
  was for white noise only at first, after a smoke where it changed
  nothing for Matern noise: an empirical choice the theory does not
  make.)  `alpha` by the smoother's discrepancy rule (`||R z - z||_H` the
  noise level of what it smooths), chosen once, on the first datum that
  is not zero, and kept.  The smoothed data replace `z` everywhere, the
  sigma-term's `D_t z` included.  The filter was written first as the
  NGSolve expression `zbar.data += w (z - zbar)`, which reads `zbar` while
  writing it and shrank the data by `1 - w` every step (zero for one-step
  filters); every smoothing smoke before the fix (2026-11-03) is void.  On data with steep fronts (the Burgers
  beams) the heat step makes the data inconsistent with the model — the
  model applied to `z + alpha Laplace z` — and raises the effective noise
  level (below): the theory's smoothing error is in the U-norm, the rule
  controls it in H.
- Effective noise level: the data's residual in the estimator's discrete
  model, `||D_t z + f(q, z) - g||_U*` (the model fit's integrand), at
  `q = P_h q^dagger` — the data's noise, their smoothing error and their
  mismatch with the estimator's discrete model at once — relative to the
  parameter's term in it, `||f(q, z) - f(0, z)||_U*`; reported per step
  (`effective_noise`) and per run (`meta["effective_noise"]`, the `eff`
  column of `scripts/run_examples.py`).  First (same day) it was the
  tracking residual `||u_ref - z||_H` of a reference state held at
  `P_h q^dagger`, which failed: the state tracks the data, so `u - z`
  starts at zero and stays below the reference's for any q, and the
  discrepancy stop fired the moment its running means were ready (both
  Burgers 1D and Arrhenius at 5 % noise stopped at t = 5); smoothing
  inflated it too (to 182 % for Arrhenius), the heat step's data being
  inconsistent with the model.  Noise-free data have one: Fisher
  at `h = 0.2` 4.6 % (the ring Q_h cannot represent), 0.45 % fitted, and
  3e-12 fitted with the data from the estimator's own model
  (`data_domains = None`) — zero exactly only then.  Without a true
  parameter the declared noise level stands in.
- Discrepancy stop (`discrepancy`; off by default, tau = 1.1 the usual
  start): the running mean (50 steps)
  of the residual at the estimate at tau times that of the effective
  level, after two time constants; the model-fit plateau stop now applies to clean data
  only (the model fit of noisy data fits the noise).  A priori stopping
  from the bound (12) is not implemented.
- Projection (`projection`, on, `q_bounds = (0, None)`): clipped
  coefficients on P0, the L2 projection of the clipped field otherwise.
  Onto a convex set holding `q^dagger` the projection is non-expansive,
  so the MRAS energy estimate survives it.  Leakage (`leakage`, off):
  `-lambda (q - q_0)` implicit in the parameter row of every scheme,
  `lambda = LEAKAGE` times the noise level; it moves the fixed point by
  `lambda (q_0 - q^dagger)`, which the MRAS theory does not allow.  A dead
  zone (the parameter frozen while the residual was at the effective
  level) was tried and dropped the same day: no support in the MRAS
  papers, and in the smokes it bursts after each frozen phase (Arrhenius
  at 5 % noise: 0.54 against 0.30 without it).  The principle (the
  user's, 2026-11-03): tools that directly support the MRAS theory only.
- All persist in the checkpoint's restart point (reference state, filter,
  smoother's `alpha`, running means), so a resumed run goes on exactly.

## Known theoretical limits (expected behaviour)

- The stabilisation bound uses the worst-case ratio `Z_MAX / Z_MIN` of the
  admissible state range instead of instantaneous norms; this over-stabilises
  and slows the observed decay but never destabilises.

## Data — `mras/data/pdebench.py`

- `2D_DarcyFlow_beta100.0_Train.hdf5` (1.3 GB, h5py): `load_nu_grid`
  reads one 128 x 128 sample of `/nu` lazily, so a run holds one field,
  not the 10000.  Despite the filename (beta = 100 is PDEBench's forcing
  parameter), the fields are binary, 0.1 and 1.0 -- a 10:1 contrast.  The
  synthetic cross-check set in `scripts/darcy_dataset.py` brackets this
  with NGSolve-built fields of matched contrast and centred inclusions;
  its comparison found the axis convention of the pixel grid (raw rows
  against rolled) moving the final error by 5 points (0.41 against 0.46).
- What the set holds: per sample a permeability map `/nu` and the one
  steady solution `/tensor` of `-div(nu grad u) = beta`; the
  `t-coordinate` (10 values to 2.25) is a placeholder, there is no time
  series.  The examples read the maps only and generate their own
  time-dependent data.
- How the field enters (`mras.fields.pixel_field(..., linear=False)`,
  2026-10-29): constant on each cell, as PDEBench defines it, and painted
  on the cell grid (`mras.domains.pixel_square`), where it lies in P0;
  the estimator's mesh does not follow the cells.  Before, it was
  interpolated bilinearly between cell centres
  (continuous ramps one pixel wide) and ramped linearly to zero over a
  5 % strip along the boundary, a choice of the paper's code that made the
  model degenerate (no diffusion) at the boundary and left the field in no
  `Q_h` (P0 floor 21 %, P1 12 % at `h = 0.08`).
- Issues found on 2026-10-28: the errors used to be measured against the
  P0 projection of the field on the estimator mesh while the data were
  generated with its P0 projection on the data mesh (two different
  "truths"); the field is now one coefficient for both.

## Two-dimensional Burgers — `examples/burgers_2d.py` (no paper benchmark)

- Vector Burgers, `u_t - div(q grad u) + S (u . grad) u = 0` (convection
  strength `S = 1`, `S = 5` until 2026-11-04; inflow speed 10, see below), up a vertical
  channel (`mras.domains.burgers_channel`, 2026-10-29): inlet at the
  bottom, outlet at the top, walls bulging inward at staggered heights and
  three irregular barriers in the lanes between the bulges.  The walls are
  slip walls (a stiff resistance to flow through them, none along them);
  the barriers are no-slip.  The true viscosity (the user's design) is
  0.5, and 1.0 in a five-pointed star left of the lowest barrier, a heart
  right of the middle one and a diamond left of the top one, each at its
  barrier's height (`mras.domains.CHANNEL_INCLUSIONS`).  The star and the
  diamond share one x, their lanes' mean centre rounded to 0.1 (1.0); the
  heart sits mirrored about the channel's middle (2.2), so the positions
  follow the channel's width (3.2, the user's).  All three fill bounding
  squares of side 1.05 (`CHANNEL_SHAPE_SIDE`); a shape that would touch a
  wall, a barrier or another shape is an error.  The estimate starts at
  3/4, between the background and the shapes (`Q_START`).  They are exact
  on a mesh of their own that
  follows them; the estimator's mesh and the data's do not, so the best
  fit in P0 is off along the outlines: 5.6 % at the defaults (`h = 0.05`;
  112k velocity dofs; until 2026-11-04 also 444k on a data mesh).  (Earlier the same day: a smaller
  star and heart between the barriers, at width 1.6 (P0 floor 5.5 % at
  `h = 0.05`), and before that low wakes above the barriers; the strength
  table below was measured with those.)
  Until then the example ran in the scalar example's perforated pipe,
  where the flow
  died out before the outlet: Burgers flow has no pressure to push it, so
  no-slip walls and obstacles drain its momentum.  No source -- the flow
  is driven by a time-dependent inflow (non-homogeneous Dirichlet data on
  the inlet, imposed on the state and on the data model at every step;
  no-slip on walls and barriers; do-nothing outflow, the natural
  condition `q grad u . n = 0` of the weak form).  The theory
  carries over verbatim: f is linear in q (sigma = 0, L = 0), the
  parameter law is `q_t = grad z : grad(u - z)`, C is the vector
  Laplacian with Poincare through the Dirichlet part, (A2) is the
  Remark 4.2 situation.  The datum and the state carry the same inflow,
  so the error is homogeneous (the datum's constrained dofs are the
  boundary data, see the driver notes).
- Excitation comes from the inflow alone (2026-11-04, the user's design):
  straight up the channel, tapering to half its centre at the (slip)
  walls, its amplitude rising from rest by a smoothstep over `RAMP = 5`
  and then held, so that the flow settles to a steady state; no transverse
  or skewing motion (a slow sway came back on 2026-11-05, below, and the
  ramp went to `RAMP = 20` on 2026-11-06).  With
  steady data Baumeister's PE asks that
  `h -> div(h grad z_inf)` be injective on Q_h: q is identified where the
  steady flow shears, which the taper provides on the centreline.
  Until then the inflow added a mode skewing it toward either wall and a
  transverse component, modulated at mutually incommensurate frequencies
  (stretched by the user's `PERIOD_SCALE`).  The noise-free production run
  (`h = 0.05`, `S = 5`, to t = 100) fell from 0.44 to 0.098 at t = 6, rose
  to 0.32 at t = 55 and swung between 0.17 and 0.26 after, with an
  effective noise level of 6.5 - 8.3 %; at 1 % noise it diverged, to 540 %.
- Static inflow, convection strength and taper (2026-11-04; noise-free,
  `h = 0.1`, `dt = 0.05`, `U_MEAN = 10`, to t = 30; best fit in Q_h 7.6 %
  at this `h`; "unsheared": share of the area where the steady
  `|grad z|^2` is under 1 % of its mean):

  | `S` | wall taper | `error_q` at t = 5 / 10 / 30 (minimum) | effective noise, steady | model fit | unsheared |
  |---|---|---|---|---|---|
  | 5 | 0.5 | 0.178 / 0.236 / 0.275 (0.160 at t = 4) | 8.1 % | 26 % | 12 % |
  | **1 (shipped)** | **0.5** | 0.159 / 0.122 / 0.113 (at t = 30, still falling) | 2.6 % | 9.6 % | 4 % |
  | 1 | 1 (plug) | 0.174 / 0.167 / 0.170 (0.167 at t = 7.6) | 2.8 % | 10.4 % | 25 % |

  Every flow is steady by t = 8 (`|dz/dt| / |z|` ~ 1e-10).  These ran on
  data from the finer data mesh, as the code then had them: at `S = 5`
  the estimate semi-converges even on the steady flow, so the old run's
  rise was the data's mismatch with the estimator's model, which the
  oscillation only moved about.  The layers are about `q / (S U)` wide,
  0.01 at `S = 5` and 0.05 at `S = 1`, and the mismatch fell with them.
  The plug inflow leaves a quarter of the channel unsheared, and its
  estimate stalls.
- The same on one mesh (2026-11-04, the data the estimator's own model;
  `h = 0.15`, to t = 20; best fit 9.5 % at this `h`):

  | `S` | `error_q` at t = 5 / 10 / 20 | effective noise, steady | unsheared |
  |---|---|---|---|
  | **1 (shipped)** | 0.168 / 0.134 / 0.127, still falling | 2.9 % | 4 % |
  | 5 | 0.134 / 0.123 / 0.118, still falling | 2.3 % | 12 % |

  The semi-convergence at `S = 5` is gone: it was the data mesh's.  What
  is left of the effective noise is `q_true - P_h q_true`.  `S = 1`
  stays for the user's design -- a flow spread evenly through the
  channel, with layers the mesh resolves -- where `S = 5` runs on as a
  jet with a less sheared core; `S = 5` would also do.
- A slow sway and a stronger inflow (2026-11-05, the user's call).  The
  static production run (`h = 0.05`, `U_MEAN = 10`) crept from 0.104 at
  t = 20 to 0.0935 at t = 100 against its 5.6 % floor, and too little of
  the flow reached the outlet.  The inflow now sways slowly from side to
  side: a transverse part `SWAY sin(2 pi t / SWAY_PERIOD) (1 - xi^2)
  U_MEAN`, peak `SWAY = 0.25` of the centre speed mid-channel and zero at
  the slip walls, period 50 against a settling time under 10, so the flow
  passes through near-steady states.  `U_MEAN` went 10 -> 20 (`S U = 20`).
  Noise-free, `h = 0.15` (best fit 9.7 %), `dt = 0.05`, to t = 50;
  "outflow" is the datum's flux through the outlet over the inlet's:

  | inflow | `error_q` at t = 10 / 20 / 30 / 40 / 50 | minimum | outflow | effective noise |
  |---|---|---|---|---|
  | static, `U = 10` | 0.1349 / 0.1281 / 0.1253 / 0.1238 / 0.1228 | 0.1228 (t = 50) | 45 % | 2.9 % |
  | sway, `U = 10` | 0.1324 / 0.1251 / 0.1222 / 0.1248 / 0.1199 | 0.1199 (t = 50) | 41 - 46 % | 2.7 - 3.3 % |
  | sway, `U = 15` | 0.1212 / 0.1176 / 0.1215 / 0.1249 / 0.1204 | 0.1176 (t = 20) | 48 - 54 % | 2.6 - 3.2 % |
  | **sway, `U = 20` (shipped)** | 0.1178 / 0.1150 / 0.1215 / 0.1243 / 0.1209 | 0.1150 (t = 20) | 52 - 58 % | 2.5 - 3.1 % |
  | sway, `U = 10`, `S = 2` | 0.1226 at t = 10, then stopped | | 56 % | 2.9 % |

  The flow's shape depends on `S U` alone (`U = 20` and `S = 2` give one
  outflow to three digits), but the parameter law scales with the datum
  squared, so the faster inflow also moves the estimate faster: hence `U`,
  not `S`.  The sway ends every run below the static one by t = 50, and
  moves the error with it: as the flow leans left across the star and the
  diamond (t = 25 - 50), the effective noise -- the part of `q_true`
  outside `Q_h`, which the flow sees through its shear -- rises to 3.1 -
  3.3 % and the error peaks near t = 40 at about 0.125.  The best fit in
  `Q_h` that the data allow moves with the flow; the static flow
  approaches its single one monotonically.
- Noisy data (2026-11-06).  The production runs of 2026-11-05 (`U = 20`,
  `RAMP = 5`) gave 0.071 noise-free (floor 0.056), but 3.19 at 1 % noise and
  4.24 at 5 %: `qhat` ran to 0 and to 25 and the state came apart by t = 5
  (`error_u` 22 against a clean 0.02).  1D had taken the same noise well.
  There were three causes, each measured.

  *The stiff walls read the noise.*  f's wall term `beta (z . n)(w . n)` sat
  on the datum, with `beta = 100` and no restoring force on the state: a
  load of the noise's normal trace.  Each term's U* norm on the data's noise
  `e` at steady flow, relative to the parameter's own term (`h = 0.05`,
  from the checkpoints):

  | noise | wall term | `C e` | `q div grad e` | convection | `D_t` |
  |---|---|---|---|---|---|
  | 1 % | **1.6 - 2.7** | 0.22 | 0.12 | 0.08 | 0.01 |
  | 5 % | **2.5 - 4.0** | 0.37 | 0.20 | 0.19 | 0.03 |

  That one term was the effective noise of 2.0 and 3.1, and its sublinear
  growth with the level.  Fix: the term joins C as well (`tracking_boundary`,
  C + B in place of C).  The estimator is `u_t + (C + B)(u - z) + f(q, z) =
  g` with B in f, so B z cancels and B acts on u alone; the error equation
  gains the dissipation `beta ||eps . n||^2` and keeps its fixed point.
  Noise-free the change is 0.1150 -> 0.1156 at t = 20 (`h = 0.15`).  At 1 %
  on the production mesh, t = 2: `qhat` peaks at 1.5 for 3.8, error 0.47 for
  0.62.  The U-norm of the residuals is now C's energy norm.

  *The filter lagged the inlet.*  The filter ran over every dof, the
  constrained ones too, so the datum lagged the inflow the state took
  exactly: `u - z` left the homogeneous space during the ramp.  Fix: the
  constrained dofs stay out of the filter, `u = z = h` on the boundary as
  in the papers (AMMC (11), Section 6).  That leaves a jump to the lagging
  interior, one element wide, which pins about 1.5 % of the cells at 0 (at
  the inlet, during the ramp).  Tried and reverted on 2026-11-07, as
  departures from (11), which keeps g itself and has `u = z = h`: the state
  taking the filtered boundary values, and a source filtered as the data
  are.  The first removed the pinned cells but not the error (`h = 0.1`,
  `RAMP = 20`, exact against filtered boundary: 1 %, tau = 0.86, t = 20:
  0.189, 0.222; 5 %, tau = 0.75, t = 10: 0.437, 0.366; 5 %, tau = 2.5,
  t = 15: 0.375, 0.370).  The second helped 1D (`h = 0.1`, t = 40; 5 %:
  0.343 -> 0.207, 1 %: 0.142 -> 0.132) and rested on an argument of ours in
  neither paper: the filter commutes with `D_t` and a linear f, so filtered
  data solve the model with filtered inputs.  With the source exact, a
  time-dependent g leaves `-tau D_t g` in the data's residual.  Also since
  2026-11-03: the papers' `R_sp` is pointwise in time and `R_ti` enters only
  the sigma-term, so at sigma = 0 they have no time filter; ours feeds every
  term.  Proposition 2.3's proof uses only the bound (28), with the lag
  counted in `delta_sp`, and it is for sigma = 1 alone.

  *The filter was too short, the ramp too fast.*  What the first two leave
  is a bias and a wander.  Noise in the regressor attenuates: `qhat` sits
  low in the mean by about 0.45 of its rms error in every noisy 1D run
  (-0.0074 at 1 %, -0.030 at 5 %, mean q 0.15), as errors-in-variables
  regressions do.  1D has excitation everywhere to hold it; most of this
  channel is near plug flow and has none.  Medians of `qhat` on the 0.5
  background at `h = 0.1`, 1 %: 0.457 at the old filter, 0.500 at the long
  one.  The residual is now the convection's commutator with the filter,
  which scales as `(tau / RAMP)^2`, hence the slow ramp.  At `h = 0.1`,
  `RAMP = 20` (`error_q`: lowest, and at the end of the run):

  | noise | `tau` | lowest (at t) | at t = 30 | at t = 40 |
  |---|---|---|---|---|
  | 1 % | 0.086 (`RAMP = 5`) | 0.32 (stalled) | | |
  | 1 % | 0.86 | 0.197 (24) | 0.260 | |
  | 1 % | **2.6** | **0.152 (24.5)** | 0.177 | 0.213 |
  | 1 % | 5 | 0.236 by t = 30 | 0.236 | |
  | 5 % | 0.25 (`RAMP = 5`) | 0.60 (t = 5, rising) | | |
  | 5 % | 2.5 | 0.319 (9.5) | 0.419 | 0.404 |
  | 5 % | **5** | **0.311 (24.5)** | 0.356 | 0.342 |

  All three shapes stand out at both levels; what is left runs along the
  shear layers, where `|grad z|` makes `q_t` largest.  The error rises again
  past t = 25 as the sway carries those layers on: the theory's term growing
  with time for noise that never stops, which is what a discrepancy stop is
  for.  (The table is at `U = 20` with the filtered boundary and source
  later reverted; see below for `U = 40`.)  Shipped: `smooth_scale = 100` (tau = 1.7 at 1 %, between
  the two tried there, and 5 at 5 %).  The production mesh follows the
  coarse runs (`h = 0.05`, `RAMP = 20`, tau = 2.6 and 5, to t = 30): 1 %:
  0.207, lowest 0.190; 5 %: 0.403, lowest 0.370; against 3.19 and 4.24
  before.  That is above `h = 0.1` (0.152, 0.311): the richer the space, the
  less each coefficient averages, so with noise a finer Q_h is noisier.  P1
  shows the same (1 %, `h = 0.05`, t = 15: 0.253 for P0's 0.216).

  What did not work, so that nobody repeats it.  The tracked datum from an
  independent stream of draws: no effect, as the attenuation does not come
  from there.  The parameter law's regressor from an independent stream
  (instrumental variables): the block system loses its symmetry and the
  energy identity its cancellation, and it diverged, q to 92 at 5 %.  The
  filter on z minus the harmonic extension of the boundary data: more
  overshoot in the ramp, 2.45 at 1 % and 5.85 at 5 %.
- More flow at the outlet (2026-11-07, the user's call: slip barriers,
  `U_MEAN = 40`, no source).  Forward model, `h = 0.1` unless said, flux
  through the outlet over the inlet's:

  | lever | outflow |
  |---|---|
  | no-slip barriers, `U = 20 / 30 / 40 / 60` (t = 12.5; 7.5 for 30, 40) | 55 / 60 / 62 / 63 %, saturating |
  | slip barriers, `U = 20` | 71 % |
  | slip barriers, `h = 0.05`, `U = 20 / 30 / 40` (t = 15.6) | 67 / 74 / 79 % |
  | a uniform body force `G = 20 / 40 / 80` (rejected) | 89 / 114 / 153 % |

  `U` runs out with no-slip barriers because Burgers flow has no pressure:
  nothing refills a barrier's shadow, or turns flow along a bulging wall,
  but viscosity.  The body force would have stood in for the missing
  pressure drop, but the flow stays driven by its inflow alone.  The sway
  went 20 % slower on the same day (period 50 -> 62.5), and `RAMP` to 40:
  the filter's lag and its commutator with the convection go with
  `D_t u_in`, so doubling U doubled them (at `h = 0.1`, t = 15, `RAMP` 20
  -> 40: 1 % 0.35 -> 0.25, 5 % 0.60 -> 0.43).

  Accuracy at `U = 40`, forward model at `RAMP = 20`: the peak velocity is
  the inflow's, with an undershoot of 0.05 (no overshoot past Burgers'
  maximum principle); `dt / 2` changes the flow by 2e-5 at t = 20; `h / 2`
  at half strength (t = 10) by 2.5e-4 in L2, 1.3 % in the gradient; and
  `h = 0.1` against `h = 0.05` at full strength (t = 20) differ by 8.6e-4,
  6.1 % in the gradient, outflow 82.4 and 82.3 %.  The `h / 2` run at full
  strength was stopped under load before it reached full strength.  At the
  production settings (`RAMP = 40`; `scripts/flow_check.py`, t = 20 / 40 /
  50): `dt / 2` changes the flow by at most 6.4e-6 in L2 and 1.3e-5 in the
  gradient; `h = 0.1` and `h = 0.05` differ by at most 7.3e-4 and 6.0 %; the
  vertical velocity stays in [0.07, the inflow's peak]; the horizontal one
  reaches 19 against the inflow's 9.5 -- the slip surfaces turn the flow, and
  couple the components, so no per-component maximum principle holds there;
  the outflow is 70 / 87 / 86 % of the inflow.
- Inflow strength (history, with the modulated inflow and `S = 5`;
  2026-10-29; noise-free, `dt = 0.05`, to `t = 10`;
  reach = first time the datum at the outlet's centre exceeds 10 % of the
  inflow speed).  With the star and heart, the user's `h = 0.05` and
  `PERIOD_SCALE = 10` (U = 10, 20, 50 on the fitted estimator mesh,
  13 and 16 on the unfitted one):

  | `U_MEAN` (`S = 5`) | `S U` | farthest 10 % by t = 10 | `error_q` at t = 2.5 / 5 / 7.5 / 10 |
  |---|---|---|---|
  | **10 (then shipped)** | 50 | y = 3.1 (t = 1.9) | 0.58 / 0.54 / 0.53 / 0.51 |
  | 13 | 65 | y = 3.1 (t = 1.2); outlet 6 % | 0.44 / 0.67 / 0.88 / 0.71 |
  | 16 | 80 | y = 4.4 (t = 9.8) | 0.51 / 1.32 / 1.61 / 1.52 |
  | 20 | 100 | y = 4.4 (t = 9.9) | 1.02 / 2.29 / 2.76 / 3.01 |
  | 50 | 250 | outlet (t = 0.6) | 17 / 12 / 14 / 28 |

  Earlier the same day, on the wake viscosity with `PERIOD_SCALE = 3` and
  `h = 0.12`: `S U = 10 - 20` died by y = 1.9; `S U = 50` reached the
  outlet by t = 3.95 (the modulation's peak) at 0.20 error by t = 10; 100
  ended at 1.9 and 250 at 7.5.

  Reach depends on `S U` alone -- `u -> S u` maps the strengths onto one
  another exactly -- and in a no-slip channel without pressure it needs a
  Reynolds number `S U W / q` of a few hundred.  At that strength Burgers
  flow steepens into fronts about `2 q / (S du)` wide (0.01 at `S U = 100`),
  far below any affordable mesh: the Galerkin datum oscillates past
  Burgers' maximum principle, the data mesh and the estimator's resolve
  the fronts differently, and the datum misses the estimator's discrete
  model by a residual the parameter flow integrates.  Refining does not
  move the bound (at `S U = 100`: 1.9 at `h = 0.12`, 3.0 at `h = 0.05`):
  the residual is the estimator's own discretisation error at the fronts.
  With data generated on the estimator's own mesh -- no residual but the
  `q_true - P q_true` part, and an inverse crime -- the discrete energy
  falls monotonically even at `S U = 100`.  The estimator also prefers a
  small datum, since its parameter flow scales with the datum's square
  (`U = 25, S = 2`, `S U = 50`, ended at 410 % where `U = 10, S = 5`
  ended at 21 %, before the Schur fallback of the driver notes).  Hence
  the strong convection on a slow inflow then shipped: stable, and reaching
  the middle of the channel within ten time units, not the outlet (not
  measured past t = 10).  In the old perforated pipe, holes
  across the centre line drained the flow, and even `S U = 250` hardly
  reached the outlet.

## Planned: Navier-Stokes in the pipe (not implemented)

- Velocity-pressure state (Taylor-Hood), inflow Dirichlet data (the hook
  exists), no-slip walls and cylinders, do-nothing outflow; a viscosity
  field written as `-div(2 q eps(u))`, whose parameter law is
  `q_t = 2 eps(z) : eps(u - z)`.  Flow past the three cylinders at Re ~ 100
  sheds vortices, a rich and naturally unsteady excitation.
- Open: the natural setting is the divergence-free space, where testing
  the error equation with the divergence-free error removes both
  pressures, the datum's unobserved one included.  The datum, though, is
  divergence-free only approximately -- projected from the data mesh, and
  noisy -- so u - z leaves that space.  Whether to project the datum, or
  carry the consistency error, is to be decided before implementing.
- The driver needs an "observed component" (errors, misfit, noise and
  the parameter row act on the velocity only); the estimator's matrix is
  then a constant Stokes block, factorised once.

## Environment behaviour (load-bearing)

- ngsolve 6.2.2608: `GridFunction.Set(coefficient)` on an H1 space
  *projects* (Gibbs overshoot at kinks — pointwise assertions on H1 are
  invalid; the tests use L2 comparisons).  It is a local L2 projection
  averaged at shared dofs, even for P1: the Burgers beams (non-negative,
  peak 30) came out at -3 ... -4 in P3 and at -4.6 in P1.  `Set(...,
  dual=True)` interpolates through the dual basis -- point values at the
  nodes for P1, which keeps a non-negative source non-negative, but not
  for P3 (-1.7).  The live view and the movies therefore paint the source
  as the coefficient itself (`mras.live._Forcing`), as the solver uses it:
  the forcing panel shows exactly 0 where there is no light.
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

- `pytest tests/` covers the fixed-point bookkeeping (t = 0 error seeding,
  seeded reproducibility, fresh-start independence), the exact-data fixed
  point of every scheme and solve, rows against joint solves, the forward
  model (linear terms assembled once, the three forward schemes, Dirichlet
  data), the noise whitening (covariance moments, exact relative noise
  level), degree-of-freedom anchors, the field helpers, the Burgers
  conformance form, the live helper (hook throttling, relative-error
  seeding, the movie stitch), the adjoint pairing (machine precision for
  every example in both the implicit and the semi-implicit scheme), the
  per-example theory checklist verdicts, and the Arrhenius pair
  (recovery, (A2) witnesses, linearisation limit).
- `scripts/reproduce.py --micro` and `scripts/darcy_reproduce.py --micro`
  are seconds-long wiring checks for every problem; full campaigns are the
  operator-run handoff scripts.
