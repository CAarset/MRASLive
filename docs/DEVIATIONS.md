# Deviation report

Everything the new code does differently from the paper text or from a
naively literal reading of the original notebooks, and everything that was
left behind on purpose. Items marked **[ported quirk]** reproduce the
original runs (which generated the published figures) rather than the paper
equations; changing them would change the numbers.

Status: all four problems and both demos are micro-verified with the new
code (suite: 30 passed). Full-size reproduction of Figs 4–12 and
Tables 6–8 is operator-run via the handoff scripts in `scripts/`; any entry
that fails its table check will be flagged there, not hidden here.

## Noise model

- **[ported quirk]** Noise is uniform on [−1, 1] drawn on a coarse
  `peaks x peaks` voxel grid and resampled, where the paper says Gaussian
  128x128. Peaks per problem: Darcy 20, Fisher 12, potential 20, AC 20.
- Noise is normalised so that `||noise||_L2 = level * ||field||_L2`
  exactly, after the voxel resample; the paper only states the percentage
  informally.
- **[ported quirk]** Reseeding differs per problem: the potential problem
  re-seeds before every draw (so its noise field is frozen-identical
  across time, as in the original notebook), while Darcy, Fisher and AC
  draw unseeded fresh noise each step.
- The Fisher noise box spans the whole mesh; its dead boundary regex
  (`"top|bottom|left|right"` against `"wall"` labels) means no Dirichlet
  condition ever applies to the Fisher state space. **[ported quirk]**

## Darcy (`mras/problems/darcy.py`)

- Data come from PDEBench `2D_DarcyFlow_beta100.0_Train.hdf5`. Despite the
  name, that file is the 10:1 contrast set (beta=100 is the *filename*
  convention); the paper's 100:1 contrast therefore shows up as 10:1 here.
  The NGSolve-built cross-check dataset (`scripts/darcy_dataset.py`) exists
  to bracket this; it validates, it does not replace.
- **[ported quirk]** The initial state `u0` is the (unseeded) first random
  grid of the generator, not a deterministic field.
- **[ported quirk]** The `a-dagger` auxiliary field is built with
  `to_gridfunc0`, i.e. L2-P0, which bands the coefficient relative to the
  paper's presentation.
- **[ported quirk]** Rows of the PDEBench sample are rolled by −12 before
  slicing, centring the sample-0 zone the original run used.
- This is the only problem that seeds its error histories at t=0 (the
  original notebook computed the initial errors; the others start empty).

## Fisher–KPP (`mras/problems/fisher.py`)

- **[ported quirk]** `u0` is a random draw; the paper's figure caption
  suggests the deterministic `3*exp(...)` initialisation.
- **[ported quirk]** The reaction feedback `g` divides by 0.4 **and** lags
  one step (`z_old`), where the paper writes an instantaneous `z`.
- **[ported quirk]** `z_old` is initialised to the constant 1, and the
  residual uses `(g − z + z^2)` (mixed old/current terms).
- Geometry: rectangle (−1.25, ±1.25); combined with the dead boundary
  regex above the estimator solves a fully natural (no-Dirichlet) problem.

## Nonlinear potential (`mras/problems/potential.py`)

- **[ported quirk]** The contrast law uses the notebook's `C(gfq)`
  formulas rather than the paper's `C_{c_n}` notation; same functions,
  different labels.
- **[ported quirk]** `L` is effectively hardcoded through the
  `(1 + 0*(5/3)*absq23)` idiom (the zero multiplier keeps the notebook
  shape while freezing the dependence).
- **[ported quirk]** `z_min` late-binding: offset 0.1, init +1, ratio
  1000, exactly as evaluated in the original notebook closure.
- **[ported quirk]** The u-row of the joint system carries no trial-`q`
  coupling (AC does); the parameter update therefore acts through the
  q-row only.
- `tend = 6` by default while the paper states T = 5 (the original run
  over-ran by one time unit).
- The disk mesh (radius π, `bc="circle"`) carries Dirichlet dofs only
  through that label; the legacy `definedon=circle` no-ops were dropped.

## Allen–Cahn (`mras/problems/allen_cahn.py`)

- **[ported quirk]** Same `C(gfq)` formulas as the potential problem.
- **[ported quirk]** Offset 0.2 in the shift of the observed field.
- **[ported quirk]** `sigma` multiplies the noise scale at a different
  placement than the paper's formula (it ends up inside the normalisation,
  not outside).
- `tend = 10` by default while the paper states T = 5.
- The u-row does carry the trial-`q` coupling here.
- The s-row boundary-flux (`ds`) terms of the original notebook were
  dropped: with the P0 test space they evaluate to zero on this mesh and
  segfaulted on assembly in ngsolve 6.2.2608 (PAPER_TARGETS item 8).

## Cross-cutting

- **[ported quirk]** `error_q` is appended before `gfq` is updated, so the
  q-error curve lags the state error by one step — matching the original
  notebooks' append order.
- **[ported quirk]** Solve style per problem: Darcy and Fisher solve
  directly into `gf`; potential and AC solve into `aux` and add
  (`gf.vec.data += aux.vec.data`, incremental mode).
- Times sampled for the field figures follow the paper's figure instants,
  converted to step indices (`series_counters`).

## Environment behaviour (not deviations, but load-bearing)

- ngsolve 6.2.2608: `GridFunction.Set(coefficient)` on an H1 space
  *projects* (Gibbs overshoot at kinks — pointwise assertions on H1 are
  invalid; the tests use L2 comparisons).
- Dirichlet conditions are applied at solve time via `FreeDofs`, so a
  `Set` may write constrained dofs without violating the BC after a solve.
- `ImplicitStepper` needs `a.Assemble()` before construction (its matrix
  is merged in the constructor).

## Not ported (deliberate)

- The Helmholtz notebook's sample-21 nu zone-shift and per-step random
  numpy sources: `examples/helmholtz_in_parameter.py` uses a
  deterministic decaying analytic source instead.
- **[ported quirk, for the record]** In the decaying-Darcy demo the
  reaction source is the *constant final* random grid of the generator
  (`grids[-1]`) at every step — an original-notebook quirk that the demo
  preserves and documents inline.

## Verification anchors

Micro runs (h 0.1–0.2, T 0.01, 5% noise, ngsolve 6.2.2608):
darcy 0.045 / 0.397, fisher 0.080 / 1.410, potential 0.577 / 6.763,
AC 0.579 / 6.831 (err_u / err_q). Helmholtz demo `--quick`:
0.0043 / 0.0010 against `q_exact = 0`.
Table 6 anchor for the full-size Darcy campaign (h = 0.04, 0% noise,
T = 5): err_u ≈ 0.02, err_q ≈ 0.24, acceptance |Δ| ≤ 0.02.
