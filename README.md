# MRAS — data assimilation via model reference adaptation

Reference implementation of the numerical scheme in

> B. Kaltenbach, C. Aarset, T. T. N. Nguyen, *Data assimilation via model
> reference adaptation for linear and nonlinear dynamical systems*,
> Applied Mathematics for Modern Challenges **8** (2026) 80–112,
> [doi:10.3934/ammc.2026005](https://doi.org/10.3934/ammc.2026005).

The estimator evolves the state `u` and the unknown parameter `q` jointly on
a product finite-element space, driven only by noisy scalar-field data `z` of
`u`.  Benchmarks: the four from the paper (Darcy flow on PDEBench data,
Fisher–KPP, a nonlinear-in-state potential problem, Allen–Cahn); a time-
dependent Burgers flow in a perforated pipe as the template for your own
problem; and an Arrhenius reaction pair — `Dt u − Δu + q(x) r(u) = g` with
`r(u) = 1 + u` and `r(u) = e^u` as linearised/nonlinear twins of one
experiment — built so every assumption of the paper's convergence theory is
numerically verifiable (see the checklist below).

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[pdebench,notebook,dev]"  # ngsolve, numpy, matplotlib,
                                                     # pydantic, h5py, anywidget, pytest
```

The Darcy benchmark reads one PDEBench file:

```bash
mkdir -p data
curl -L -o data/2D_DarcyFlow_beta100.0_Train.hdf5 \
  https://darus.uni-stuttgart.de/api/access/datafile/133221
# expected md5: 7c09f3b1bb097737d3fd52ffb0c6f1c8
```

## Quickstart

```python
from mras.problems.darcy import DarcyProblem
from mras.plotting import error_figure

prob = DarcyProblem(h=0.04, dt=1e-3, tend=5.0)      # mesh + orders + data
results = prob.run(noise_levels=[0.0, 0.05])        # noisy data -> MRAS loop
for r in results:
    print(r.noise_level, r.final_error_u, r.final_error_q)
error_figure(results, prob.dt, fname="errors.png")
```

`noise_level` is always a fraction: `0.0` is noise-free, `0.05` is noise
level 5%.

## Live notebook

`notebooks/mras_live.ipynb` runs one noise level and stitches the frames
into a fixed-framerate mp4 with static colorbars —

```
datum z  |  data error u - z  |  ground truth u at final time
estimate qhat  |  error qhat - q  |  true parameter q
```

Each colorbar spans the global extremes of its category (both state
panels share one range, the parameter panels share theirs, both error
panels are symmetric about zero), and the panels carry no interactive
chrome, so colours are comparable across every frame.  Pass `live=True`
to stream the same six panels interactively in `ngsolve.webgui` canvases
instead (those stream from a CDN, so the browser needs internet).  The
notebook only picks an example (`burgers`, `darcy`, `fisher`,
`potential`, `allen_cahn`, `arrhenius`, `arrhenius_linear`) and a noise
level; `mras/live.py` holds all
the plumbing (demo sizing, frame recording, static ranges, the ffmpeg
stitch).  Run it with `jupyter lab notebooks/mras_live.ipynb`.

## Your own reconstruction problem, step by step

**Start by copying `examples/blank_problem.py`.**  It is a complete,
self-contained problem — recovery of a reaction coefficient `q(x)` in
`Dt u - Δu + q u = g` — with four clearly marked TODO blocks (mesh,
source, `q_true`, joint weak form) and the theory hooks already wired:
run it, and it identifies `q` at noise level 0% (noise-free) and noise
level 5%, prints the full checklist, and writes its figures.  Rename the
class, edit the blocks, keep the structure; the constructor refuses any
problem that is *not* runnable and names every missing piece (see below).
`examples/burgers_pipe.py` (pipe geometry, heater-grid source) and
`mras/problems/arrhenius.py` (the fully verifiable reference) show the
same contract under more geometry and more nonlinearity.

A problem is a `mras.Problem` subclass on `(U_h, Q_h) = P3 x P0` over any
NGSolve mesh.  The constructor validates the contract and raises
`TheorySpecError` listing exactly what is missing — the weak form, a data
hook, or the coupling declaration — so nothing fails halfway into a
campaign.

**1. Domain.**  Reuse a builder (`mras.domains.darcy(maxh=h, order_u=k_u,
order_q=0)` for a rectangle, `fisher`, `potential`) or return your own
`Domains(mesh, Ufes, Qfes)`: geometry (see the CSG2d pipe in
`examples/burgers_pipe.py`), Dirichlet boundaries, element orders.

**2. Where do the data come from?**  Exactly one of three hooks, resolved in
this order:

* `observations(state)` — you have measurements, as a `GridFunction` that
  already contains its own noise; run at `noise_level=0.0` (noise-free).
* `truth(state)` — synthetic data: observations = truth + seeded FEM Gaussian
  noise at the requested `noise_level`; `error_u`, `error_q` become available.
* `forward_forms(step)` — no data at all: the driver steps *your forward weak
  form* (backward Euler) on a finer mesh (`data_domains`) to generate the
  observations, with `source(t)`, `u0` and `q_true` as inputs.

`q_true` and `initial_guess()` (zero by default) set the parameter field and
the estimator's start — the initial error *is* an experimental choice.
`symmetric = False` routes the nonsymmetric joint solve through GMRES.

**3. The MRAS step — the only mandatory hook.**  `forms(state)` returns the
joint `(BilinearForm, LinearForm)` of one step: mass rows `u*v`, `q*s`, the
diffusion of your choice (the `C` slot), the `sigma`-bracket
`Dt z + f(q,z) - g`, and the aligned tracking term from `state.z`/`z_old`.
Then run:

```python
p = MyProblem(h=0.2, dt=0.05, tend=1.5, seed=0)
res = p.run_level(0.0)               # LevelResult: error_q, error_u, series
p.run(noise_levels=[0.0, 0.05])      # campaign; then error_figure(results, p.dt)
.venv/bin/python examples/blank_problem.py       # the template, end to end
```

**4. Two things are verdicts, not choices.**  `sigma` is forced by the
structure of `f`: 0 iff `f` is affine in `q` — the checklist derives the
family from your declarations and *fails* a mismatch.  `adjoint_deviation`
may be set only to document a deliberate deviation from the paper-literal
row, with a reason; that turns a FAIL into a documented DEVIATION.

**5. Theory hooks — each declaration unlocks one rung of the checklist**
(`run_checks(p, res.state, res.error_q)` on any instance, or one registry
line in `mras/verify.py` plus `check_example("my_problem")`):

| declare | unlocks automatically | without it |
|---|---|---|
| `coupling(q, z, s)` — the q-dependent part of f as a weak functional (return the integrand; the framework appends `*dx`; z-only terms of f cancel in every checklist difference and stay out) | (A1) affine/nonlinear family + sigma consistency; (A2) measured monotonicity ladder | the constructor refuses the problem — or `theory_exempt = "<reason>"` waives the ladder honestly |
| `coupling_adjoint(q, z, s)` — independently transcribed density of `f_q(q~(z), .)^*` | coupling pre-verification: the declared pairing is checked against the numerical q-derivative of `coupling` before any solving | pre-verification unverifiable (warned at construction) |
| `L` — Lipschitz remainder of `f` in q | (A1) OK analytically | nonlinear f: remainder ESTIMATED numerically, loudly warned, never fed into (A3) |
| `c_coe()` — analytic witness (ess-inf or range minimum of the multiplier over the datum range; helpers `data_range()`, `essinf_datum()` exist per family) | (A2) OK *with provenance*, feeding the (A3) margin | probe-only measured value |
| `C_scale` (`C = C_scale*(-Delta)`) — or norm-dependent `C` inside `forms` | (A3) margin `lambda_1(C) > L^2/(2 C_coe)` with the slack reported for `M` | (A3) unverifiable |

What you get with *no* declarations at all, never silently: adjoint pairing
(probed on the **assembled step operator**, needs no hooks), persistent
excitation, reconstruction descent, and — via `check_example` — seeded
reproducibility.  The ladder `OK > DEVIATION > ESTIMATED > UNVERIFIABLE >
FAIL` is the contract pinned by `tests/test_verify.py`, which includes a
*new* example built by pure inheritance — rate law `r(w) = 1 + w + w^2`
with its witness minimised over the datum range
(`QuadraticRateProblem`) — a nonlinear-in-`q` variant whose remainder is
*estimated*, not declared (`SquaredQProblem`), and a bare variant
asserting the honest degradation.  Copy that pattern for your problem.

An ESTIMATED verdict means the checklist measured the constant from your
coupling hook (radius-3 sampled remainder bounds, K-dual norm) and warns:
an estimate must never silently feed the (A3) margin.  Declare `L` to
confirm it.

## Verification checklist

Every example — past or future — runs through the same fixed questions
(`mras/verify.py`): adjoint pairing (`<Ax, y> = <x, A*y>` to numerical
precision), coupling pre-verification (declared `coupling_adjoint` vs the
numerical q-derivative of `coupling`), the (A1) Lipschitz remainder and
the `sigma` it forces, the (A2) monotonicity witness `C_coe`, the (A3)
coercivity margin `lambda_1(C) > L^2/(2 C_coe)`, persistent excitation of
the datum, reconstruction descent, seeded reproducibility.  Verdicts are
`ok`, `fail` (the honest negative), `estimated` (measured, warned, never
promoted), `deviation` (documented), or `unverifiable` (flagged, never
silently skipped):

```bash
.venv/bin/python scripts/checklist.py       # writes results/checklist.md
.venv/bin/python scripts/demo_arrhenius.py  # q-reconstruction figure, both pair members
```

`tests/test_verify.py` pins the documented verdict per example, so a
silent flip of any rung fails the suite.

## Run configurations (YAML)

Runs are described by `mras.config.RunConfig` (pydantic-validated) and load
from YAML: one mapping, a sequence, a `{defaults, runs}` merge, or multiple
documents.  `configs/burgers.yaml` is the reference:

```yaml
defaults: {dt: 0.01, tend: 5.0, k_u: 3, noise_levels: [0.0, 0.05, 0.2],
           times: [2.0, 5.0]}
runs:
  - {h: 0.15}
  - {h: 0.075}
```

```bash
.venv/bin/python examples/burgers_pipe.py --config configs/burgers.yaml
```

## Running the studies (Tables 6–8, Figs 4–12)

Each campaign runs one noise level per process and resumes where a previous
run stopped (finished noise levels are skipped).  Figures and the markdown table
are assembled from the saved runs afterwards.

| paper item | campaign | figures |
|---|---|---|
| Fig 4, Table 6 (Darcy) | `scripts/darcy_reproduce.py --out results/darcy` | `scripts/make_figures.py --problem darcy --in results/darcy` |
| Table 7, Figs 5–8 (Fisher, potential) | `scripts/reproduce.py --problem fisher` / `--problem potential` | `scripts/make_figures.py --problem fisher` etc. |
| Table 8, Figs 9–11 (Allen–Cahn) | `scripts/reproduce.py --problem allen_cahn` | `scripts/make_figures.py --problem allen_cahn` |
| Fig 12 / Table 8 (q0 sensitivity) | vary the initial estimate in `init_run`, then `scripts/reproduce.py --problem fisher` / `--problem allen_cahn` | as above |
| Darcy data cross-check | `scripts/darcy_dataset.py --quick` | writes `comparison.json` |

Full-size runs are hours-long per noise level on 32 cores; every script accepts
`--micro` for a seconds-long wiring check and `--h/--dt/--tend/
--noise-levels` overrides.  Keep `OMP_NUM_THREADS` small if other work is
running.

## Repository layout

```
mras/               package: scheme (MRAS loop + data resolution), config
                    (YAML/pydantic), domains, fields, noise, stepper, live
                    (webgui panels), verify (theory checklist),
                    problems/{darcy,fisher,potential,allen_cahn,arrhenius},
                    data/pdebench, results, plotting
notebooks/        live six-panel notebook (Burgers preset)
examples/           runnable demos (blank template to copy, Burgers pipe,
                    Helmholtz, decaying Darcy, quickstart)
configs/            YAML run configurations
scripts/            reproduction campaigns, figure assembly, the theory
                    checklist, the Arrhenius reconstruction demo
tests/              pytest suite (domains, fields, noise, scheme, live,
                    config, plotting, adjoint, arrhenius, verify)
docs/               cleanup plan and cleanup flags, paper-target mapping,
                    deviation report, assumption-by-assumption status
```

## Tests

```bash
.venv/bin/pytest -q
```

## Notes on fidelity

The package implements the paper's schemes on deterministic, analytic
benchmark fields with L2-Gaussian FEM observation noise; the modelling
choices, theoretical limits and environment caveats are documented in
[docs/DEVIATIONS.md](docs/DEVIATIONS.md).
