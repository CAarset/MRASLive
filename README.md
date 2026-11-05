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
.venv/bin/pip install -e ".[notebook,dev]"  # ngsolve, numpy, matplotlib,
                                            # pydantic, h5py, anywidget, pytest,
                                            # ngsplot and ngsrng from GitHub
```

[ngsplot](https://github.com/CAarset/ngsplot) (the FEM renderer behind every
figure) and [ngsrng](https://github.com/CAarset/ngsrng) (random fields; it
brings torch and scipy) are installed from GitHub, unpinned: the latest commit
of each default branch.  While the repositories are private, `git` needs
GitHub credentials (`gh auth login` sets them up).  To update both:
`.venv/bin/pip install -U --force-reinstall --no-deps "ngsplot @
git+https://github.com/CAarset/ngsplot.git" "ngsrng @
git+https://github.com/CAarset/ngsrng.git"`; to work on one locally,
`.venv/bin/pip install -e ../ngsplot`.

Movies need `ffmpeg` on the `PATH` (any build with libx264).

The Darcy benchmark reads one PDEBench file:

```bash
mkdir -p data
curl -L -o data/2D_DarcyFlow_beta100.0_Train.hdf5 \
  https://darus.uni-stuttgart.de/api/access/datafile/133221
# expected md5: 7c09f3b1bb097737d3fd52ffb0c6f1c8
```

## Quickstart

```python
from mras.live import build
from mras.plotting import error_figure

prob = build("darcy", h=0.04, dt=1e-3, tend=5.0)  # example DEFAULTS, overridden
results = prob.run(noise_levels=[0.0, 0.05])      # noisy data -> MRAS loop
for r in results:
    print(r.noise_level, r.final_error_u, r.final_error_q)
error_figure(results, prob.dt, fname="errors.png")
```

`noise_level` is always a fraction: `0.0` is noise-free, `0.05` is noise
level 5%.

Or end to end from the shell, saving figures and error histories under
`results/cli/<name>/` as it goes — the runner owns all argument handling, the
examples themselves have none:

```bash
.venv/bin/python -m mras.live darcy --h 0.04 --dt 1e-3 --tend 5
```

## Live notebook, movies and checkpoints

`notebooks/mras_live.ipynb` runs one noise level with a seven-panel view
rendered by [ngsplot](https://github.com/CAarset/ngsplot) — the same FEM
renderer NGSolve's own viewer uses — that follows the solve step for step:

```
               |  forcing g                      |
state u        |  data error u - z               |  datum z
estimate qhat  |  reconstruction error qhat - q  |  true parameter q
```

Below the panels a strip plots `||qhat - q|| / ||q||` against time; it
spans the curve, from its lowest value so far to its largest (1 for an
estimate starting at zero), so the plot moves down with it.  A dotted
line comes into view as the curve approaches it: the best fit in `Q_h`
(`MRASProblem.q_floor()`), below which no estimate goes.  The run also
stops on a plateau (`"plateau"`): at or below the model fit
(`mras.floor`), an estimate of where the estimator settles with these
data on this mesh, and no longer falling.  The title reads `t / tend`.  Data are generated in lockstep with
the estimator — the forward model takes one step whenever the MRAS asks
for the next datum — so the view opens at t = 0 with no up-front forward
run, and painting runs on a background thread beside the solve.
`q` and `qhat` share one fixed range, `q`'s extremes padded by 5 % of its
span at each end: `q` stays clear of the colour bar's ends, an estimate
that over- or undershoots saturates at them, and a converged `qhat` looks
like `q` colour for colour; a piecewise constant `q` is painted on its own
mesh, which follows its jumps (`q_domains`).
Every other panel follows its field with
moving exponential averages over a 250th of the horizon -- of its
extremes, or for the two errors of twice their root mean square, so the
bulk of an error spans the colour bands and its spikes saturate red and
blue.  `u` and `z` share one range.  The forcing, the state and the datum
are painted symmetrically about zero, like the errors, unless the problem
guarantees them positive (`positive_state`, `positive_source`); a vector
state is painted as its length.  For a problem whose fields are sustained
rather than decaying (`decaying = False`, the default) their ranges are
pinned to the largest values so far, and follow a field down gently only
after nothing has come within 90 % of them for 5 % of the horizon.

Every run is written to an **HDF5 checkpoint** (`mras.checkpoint`): the
settings it ran with (problem, constructor arguments, scheme, versions,
git revision), the error histories of every step, the trajectory — the
coefficient vectors of `u`, `q`, `z` from the initial state on, every step
that fits in `budget` megabytes (256 by default, 1024 in
`scripts/run_examples.py`; `budget=None` keeps every step) — and a
double-precision restart point, rewritten every twentieth of
the run, from which `mras.checkpoint.resume(path, tend=...)` continues the
run exactly.  `load(path).field("q", k)` gives any stored step back as a
grid function.

The **movie** is rendered from the checkpoint once the run is done
(`mras.movie.render_movie`): a frame for every stored step, uniform in
model time, up to 1200 frames (thinned evenly beyond that), lasting 4 to 10
seconds -- long runs play ten seconds at up to 120 frames a second, short
ones four (`fps` and `every` override the pace) -- painted in parallel by
worker processes and streamed into ffmpeg as H.264 MP4.  In the notebook
the movie is embedded in the output, so it survives re-opening.

The notebook only picks an example and a noise level; `mras/live.py`
holds the plumbing.  From the command line, `scripts/run_examples.py` runs
every example (or those named) the same way:

```bash
# every run stops at its target error -- 1.25 x the floor,
# --target to change it, 0 to run to the limit -- or its time limit --tend
.venv/bin/python scripts/run_examples.py                      # all, DEFAULTS
.venv/bin/python scripts/run_examples.py burgers_1d fisher  # these two
.venv/bin/python scripts/run_examples.py --resume results/burgers_1d/burgers_1d_noise0.h5 --tend 400
.venv/bin/python scripts/run_examples.py --movie-from results/burgers_1d/burgers_1d_noise0.h5
.venv/bin/python scripts/run_examples.py burgers_1d --noise-levels 0 0.05 0.1  # three levels
.venv/bin/python scripts/speed.py burgers_2d --steps 10    # where a step's time goes
```

An example's noise levels run one after another on the same noise-free
data: the first run generates them, the next read them back, so the
forward model (most of the time for Burgers 2D) is stepped once.

## Your own reconstruction problem, step by step

**Start by copying `examples/blank_problem.py`.**  It is a complete,
self-contained problem — recovery of a reaction coefficient `q(x)` in
`Dt u - Δu + q u = g` — with four clearly marked TODO blocks (mesh,
`q_true`, source, model): run
`.venv/bin/python -m mras.live blank_reaction --quick`, and it identifies
`q` at noise level 0% (noise-free) and noise level 5%, prints the full
checklist, and writes its figures.  `examples/burgers_1d.py` (curved
pipe, beams of light turning about its holes), `examples/burgers_2d.py` (vector flow driven by
its inflow) and `examples/arrhenius.py` (the fully verifiable reference)
show the same contract under more geometry and more nonlinearity.

**A problem declares its physics, nothing else.**  The model operator
`f(q, u)` of `Dt u + f(q, u) = g` is written once, as densities (the
driver attaches one quadrature to every form):

| hook | what it is |
|---|---|
| `model(u, w)` (and `model_boundary(u, w)` for `ds` terms) | the q-free part `<f(0, u), w>`; a list keeps linear and nonlinear terms apart |
| `coupling(q, u, w)` | the q-dependent part `<f(q, u) - f(0, u), w>` — required |
| `coupling_adjoint(q, u, w)` | the density of `f_q(q, u)^* w` — required |
| `source(t)` | `g` at time `t` (a `Parameter`); none by default |
| `tracking_scale(q)` / `C_scale` | `C = c (-Δ)`, constant by default (`C_scale = 1`) |
| `dirichlet(t)` | Dirichlet data by boundary, `{"inlet": value}`: imposed on those boundaries of the Dirichlet part of `U_h`, the rest held at zero; homogeneous by default.  A problem may be driven by boundary data alone (2D Burgers) |
| `u0`, `q_true`, `initial_guess()` | initial state, true parameter, the example's start of the estimate (0 unless it says otherwise) |
| `q0` | the initial estimate `qhat_0` (number, coefficient or grid function) set by whoever runs the problem, overriding `initial_guess()`: `run_live(..., q0=...)`, `scripts/run_examples.py --q0 0.3` |
| `q_domains` | the mesh a piecewise constant `q_true` is defined on, following its jumps -- the one quantity with a mesh of its own: the data are the estimator's own discrete model at `q_true`; the estimator's mesh need not follow the jumps, and does not by default (`fitted=False` in the examples) |
| `positive_state`, `positive_source` | guarantees the state and source stay positive (pictures then use their own ranges, not ranges symmetric about zero) |
| `decaying` | the fields relax over the run rather than being sustained (default `False`: pictures of the forcing, state and datum keep their largest values as colour range) |

From these the driver builds everything: the forward model that generates
the data (`u_t + f(q_true, u) = g` on `data_domains`, stepped in lockstep
with the estimator), the MRAS step, and its solves; the checklist probes
the same coupling pair.  The constructor raises `TheorySpecError` naming
whatever is missing.

**Where the data come from.**  By default they are generated from the
physics.  `truth(t)` instead supplies an analytic `u†` — with
`truth_rate(t)` and `manufactured_source = True` the source follows as
`g = u†_t + f(q_true, u†)` — and `observations(state)` supplies measured
data with no truth at all (run at `noise_level=0.0`).

**The time scheme** (`scheme`, and `forward_scheme` for the data model):

| scheme | MRAS step |
|---|---|
| `"implicit"` (default) | backward Euler on (4); q and u couple both ways, sigma = 1 iterated by Newton's method |
| `"semi-implicit"` | the paper's (13)–(14): the q-row explicit in u, f linearised about qⁿ |
| `"explicit"` | the q-row at the old iterate, then the u-row with qⁿ⁺¹ known (= semi-implicit for sigma = 0) |

Every scheme takes `C(‖qⁿ‖)` and the tracking factor at `qⁿ`.  `split`
picks the solve: rows apart with the constant matrices factorised once
(`True`; the default `"auto"`), or the joint system factorised every step
(`False`, the reference — the two agree to round-off).

```python
p = MyProblem(h=0.2, dt=0.05, tend=1.5, seed=0)
res = p.run_level(0.0)               # LevelResult: error_q, error_u, series
p.run(noise_levels=[0.0, 0.05])      # campaign; then error_figure(results, p.dt)
```

Once the file carries a `PROBLEM = MyProblem` marker (and a `DEFAULTS`
dict), the runner does everything else —
`.venv/bin/python -m mras.live my_problem --h 0.2 --dt 0.05`; the example
file itself never grows argparse code.

**`sigma` is a choice, and the checklist says what it gives.**  The
energy identity of the error equations decides: with `sigma = 1` the
decay estimate follows from (A1)–(A3); with `sigma = 0` and `f` affine in
`q`, (A2) plays no part and the parameter converges only through the
persistent excitation of the data (the paper's choice for affine `f`);
with `sigma = 0` and `f` nonlinear in `q` there is no estimate.  The
checklist's convergence row reports which statement the declared `sigma`
has, and whether its hypotheses are confirmed.

**Theory hooks — each declaration unlocks one rung of the checklist**
(`run_checks(p, res.state, res.error_q)` on any instance, or
`check_example("my_problem")` for a registered one):

| declare | unlocks automatically | without it |
|---|---|---|
| `coupling`, `coupling_adjoint` (required) | (A1) affine/nonlinear family; the convergence statement of the declared sigma; (A2) on `Q_h`: exact where `f` is affine in q, sampled patch pairs where not; the pairing pre-verified against the numerical q-derivative of `coupling` before any solving; the persistent-excitation Gramian | — |
| `L` — Lipschitz remainder of `f` in q | (A1) OK analytically | nonlinear f: remainder ESTIMATED numerically, loudly warned, never fed into (A3) |
| `c_coe()` — analytic bound (ess-inf or range minimum of the multiplier over the datum range; helpers `data_range()`, `essinf_datum()` exist per family) | (A2) OK *with provenance*, tested against the pairing on `Q_h` (a bound above it fails), feeding the (A3) margin | affine f: the exact constant on `Q_h`; nonlinear f: ESTIMATED (a sampled minimum bounds `C_coe` from above only) |
| constant `C_scale` | (A3) in the U-norm: with `L = 0` the coercivity of `C` alone, `M = C_scale`; otherwise `C_scale > L^2/(2 C_coe)` with the slack reported for `M` | (A3) unverifiable (norm-dependent `C`) |

The adjoint pairing (probed on the assembled step operator), persistent
excitation, reconstruction descent and — via `check_example` — seeded
reproducibility need no declarations.  A `C = c (-Laplace)` on a `U_h`
without Dirichlet part fails (A3) whatever is declared: it vanishes on
the constants.  The ladder `OK > DEVIATION >
ESTIMATED > UNVERIFIABLE > FAIL` is the contract pinned by
`tests/test_verify.py`, which includes a *new* example built by pure
inheritance — rate law `r(w) = 1 + w + w^2` with its witness minimised
over the datum range (`QuadraticRateProblem`) — and a nonlinear-in-`q`
variant whose remainder is *estimated*, not declared (`SquaredQProblem`).

## Verification checklist

Every example — past or future — runs through the same fixed questions
(`mras/verify.py`): adjoint pairing (`<Ax, y> = <x, A*y>` to numerical
precision), coupling pre-verification (declared `coupling_adjoint` vs the
numerical q-derivative of `coupling`), the (A1) Lipschitz remainder and
the `sigma` it forces, the (A2) monotonicity constant `C_coe`, the (A3)
coercivity of `C` in the U-norm, the persistent excitation of the data
(the smallest eigenvalue of the windowed sensitivity Gramian on `Q_h`, and
where its eigenvector lives), reconstruction descent, seeded
reproducibility.  Verdicts are
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
documents.  The YAML is the control switch, never a restatement: fields it
does not mention fall through to the example's `DEFAULTS` (on top of the
constructor signature); `--quick` swaps in the audited `QUICK` sizes; CLI
flags sit on top of everything.  `configs/burgers.yaml` is the reference:

```yaml
defaults: {dt: 0.01, tend: 5.0, k_u: 3, noise_levels: [0.0, 0.05, 0.2],
           times: [2.0, 5.0]}
runs:
  - {h: 0.15}
  - {h: 0.075}
```

```bash
.venv/bin/python -m mras.live burgers_1d --config configs/burgers.yaml
```

## Running the studies (Tables 6–8, Figs 4–12)

Each campaign runs one noise level per process, writes into
`results/paper/<problem>/`, and resumes where a previous run stopped
(finished noise levels are skipped).  Figures and the markdown table
are assembled from the saved runs afterwards.

| paper item | campaign | figures |
|---|---|---|
| Fig 4, Table 6 (Darcy) | `scripts/darcy_reproduce.py` (into `results/paper/darcy`) | `scripts/make_figures.py --problem darcy` |
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
mras/               package: scheme (MRASProblem: the physics hooks, the
                    time schemes, the MRAS loop), forward (the data model
                    built from the same hooks), solvers (factorised
                    constants, Krylov to round-off), config (YAML/pydantic),
                    domains, fields, noise, stepper, live (example catalog,
                    run runner, live ngsplot panels), verify (theory
                    checklist), adjoint, data/pdebench, results, plotting
notebooks/        live seven-panel notebook (Burgers preset)
examples/         every problem in one folder: one file per example
                  (darcy, fisher, burgers_1d, burgers_1d_dark, burgers_2d, the arrhenius pair,
                  potential, allen_cahn), each
                  declaring PROBLEM/PROBLEMS plus its DEFAULTS/QUICK/CHECK
                  sizes, plus blank_problem.py (template to copy)
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
