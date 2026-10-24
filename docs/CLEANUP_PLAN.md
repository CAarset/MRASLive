# MRAS cleanup — plan & progress log (overnight campaign)

Objective: total from-scratch cleanup of the MRAS code into one
Git-standardized, self-contained repo that (1) correctly implements the
theory of the paper (AMMC 2026, DOI 10.3934/ammc.2026005) — the published
figures stay reproducible from the separate GRO upload, this codebase
targets the theory, (2) exposes a simple interface —
user provides NGSolve mesh & weak form and MRAS "just happens", (3) is
streamlined/clean/optimized, (4) carries honest comments describing current
behaviour only. No backwards compatibility, no shims.

## Author decisions (2026-10-20/21)
- GRO folder is canonical for paper numerics (normalised relative-L2 noise).
- 2026-10-21 afternoon, supersedes the quirk-porting policy: theory
  correctness and extensibility beat numeric reproduction. The ported-quirk
  layer is removed in favour of canonical choices: Gaussian FEM noise
  (M^-1/2 whitening), every draw seeded, deterministic paper initial states
  and sources, aligned time-level indexing. Numerics need not match the
  paper; they must be logically and theoretically consistent.
- Interface: user supplies mesh, FE spaces, f(q,u) and q-directional
  derivative dq_f as NGSolve-expression hooks; driver assembles the joint
  semi-implicit scheme, time loop, noise, errors, saving.
- Deliverable: pip-installable `mras` package + example scripts + figure/
  table scripts + thin driver notebooks (WebGUI) per benchmark.
- Reproduction: seeded RNGs (deviation from paper noted), campaign runs
  main figures first, then Tables 6–8 in checkpointed parallel batches.
- Git: linear history on `main`, clean up in place (no fresh tree).
- Extras kept: Helmholtz dropped, **decaying-Darcy kept** as 5th demo.
- Environment: NO global pip installs. Project venv at `.venv` (created with
  stdlib `python3 -m venv` — `uv` is not installed on this host and would
  need a global install; deviation from the usual workflow, same effect).
  All runs via `.venv/bin/python` / `.venv/bin/pytest`, `OMP_NUM_THREADS`
  pinned per run, heavy runs under `systemd-run --user --scope -p MemoryMax=…`.

## Target layout
```
mras/                  package
  __init__.py          public API re-exports
  domains.py           square/rect/disk mesh builders (was domain_*.py ×4)
  fields.py            VoxelCoefficient↔GridFunction mapping, ICs
  noise.py             single relative-L2 noise model, seeded
  problem.py           MRASProblem: f(q,u), dq_f, data, spaces, BCs
  scheme.py            joint semi-implicit Euler driver (incremental form)
  run.py               run(problem, cfg) → Result (errors, series, timing)
  plotting.py          publication plots (cmunrm font, colorbars, panels)
  data/pdebench.py     checksummed downloader for the PDEBench hdf5
  problems/            darcy.py, fisher.py, potential.py, allen_cahn.py,
                       darcy_decaying.py  (data recipes + weak-form hooks)
examples/              one thin script per benchmark + figures + tables
notebooks/             WebGUI drivers (import the package; no logic)
tests/                 pytest suite
docs/                  this file, PAPER_TARGETS.md, DEVIATIONS.md
data/                  gitignored; downloaded artefacts
results/               gitignored; campaign outputs (npy/png/csv)
pyproject.toml, README.md, LICENSE (MIT, authors as in paper), .gitignore
```
Legacy `GRO/`, `MRAS/` folders deleted at the end (preserved in history via
checkpoint commit 4749311).

## Phases
- [x] P0 environment: .venv + deps, ngsolve smoke solve, HDF5 readable.
- [x] P0 git: .gitignore (hdf5/npy/vol/PDF excluded), checkpoint commit.
- [x] P1 paper targets → docs/PAPER_TARGETS.md.
- [x] P2 package skeleton + unit tests (noise normalisation, field mapping,
      seeding determinism, disk/square mesh dof counts vs Tables 2–5,
      known-solution steady check for the stepper).
- [x] P3 core scheme + Darcy end-to-end (micro-verified; the Table 6 row
      h=0.04/0% check (u 0.02, q 0.24 at T=5) runs full-size via
      `scripts/darcy_reproduce.py`, which the operator runs).
- [x] P4 Fisher–KPP (Fig 5/6, ring a†, nonlinear implicit data solver;
      campaign via `scripts/reproduce.py --problem fisher`).
- [x] P5 Nonlinear potential (disk, vorticity-built u₀, Figs 7/8;
      campaign via `scripts/reproduce.py --problem potential`).
- [x] P6 Allen–Cahn (Figs 9/10/11; campaign via
      `scripts/reproduce.py --problem allen_cahn`).
- [x] P7 Decaying-Darcy demo (`examples/darcy_decaying.py`) and Helmholtz
      interface demo (`examples/helmholtz_in_parameter.py`).
- [ ] P8 Campaign execution: Tables 6/7/8 + Fig 12 full-size, resumable
      handoff scripts, then `scripts/make_figures.py` panels and the
      paper-vs-new table diff (operator-run on this machine).
- [x] P9 README, notebook, LICENSE, pyproject, legacy folders deleted,
      green suite, deviation report (docs/DEVIATIONS.md).

## Reproduction acceptance
- Main figures: qualitatively matching panels regenerated from new code.
- Tables 6–8: treated as order-of-magnitude context (docs/PAPER_TARGETS.md);
  new-code results must be logically consistent — noise-free runs decay, noisy
  runs plateau at the noise scale, Table-8 sensitivity trends reproduced.
- Any systematic mismatch → flagged in the final report, not hidden.

## Progress log
- 2026-10-21 02:5x CEST — Setup complete: venv (.venv, ngsolve 6.2.2608,
  numpy/matplotlib/h5py/pandas/tqdm/pytest), smoke Poisson solve OK,
  PDEBench hdf5 readable (nu 10000×128×128), checkpoint commit 4749311,
  PAPER_TARGETS.md written. Repo: no remote; linear history on main.
- 2026-10-21 05:2x CEST — P2 complete: `mras/` package (domains, fields,
  noise, stepper, scheme, results, 4 problems, PDEBench loader) installed
  editable in .venv; pytest suite 27/27 green (dof anchors, boundary
  behaviour, noise normalisation/determinism, stepper residual, PDEBench
  grid, mini-run determinism + level chaining). Notable API findings on
  6.2.2608: `GridFunction.Set` on H1 now projects rather than interpolates
  (it also writes constrained dofs; Dirichlet is applied at solve time —
  cross-build mini-runs agree, so behaviour stays faithful), coordinate
  call is `gf(x, y)`, `BitArray.NumFree()` gone, `Grad(coord)` needs the
  space dimension.
- (next entries appended below)
- 2026-10-21 08:0x CEST — P3–P6 complete: all four problems implemented in
  `mras/problems/` and micro-verified (h=0.1–0.2, T=0.01, 5% noise):
  darcy 0.045/0.397, fisher 0.080/1.410, potential 0.577/6.763,
  AC 0.579/6.831 — stable, no blow-ups. Legacy noise conventions (uniform
  peaks noise, per-problem re-seeding, solve style, error-append order)
  ported deliberately; see docs/DEVIATIONS.md.
- 2026-10-21 09:3x CEST — Incident: a detached full-Darcy run OOM-killed
  the operator's Strata session (the machine holds ~50 GiB of Strata
  shared memory). Decision: the agent runs only light/micro jobs;
  full-size campaigns are operator-run via the resumable handoff scripts
  (`scripts/darcy_reproduce.py`, `scripts/reproduce.py`).
- 2026-10-21 12:4x CEST — Final assembly: `mras/plotting.py` (headless
  field/error figures, out-of-domain pixels masked), examples (quickstart
  runner for all four problems, Helmholtz interface demo, decaying-Darcy
  demo), `scripts/reproduce.py` + `scripts/make_figures.py` (npz -> panels
  + table.md), quickstart notebook, README, MIT LICENSE, legacy GRO/ and
  MRAS/ folders deleted (notebooks preserved in git history). Suite:
  30 passed.
- 2026-10-21 16:1x CEST — Theory-consistency pass (three commits):
  (1) `noise.py` Gaussian FEM noise — Chebyshev-filtered M^-1/2 whitening
  with diagonal mass scaling, Lanczos spectral bounds, exact relative-level
  renormalisation, per-draw seeding (seed, level, counter).
  (2) Canonical driver — fresh start per level, t=0 seeding of both error
  histories, z_old priming, Dirichlet canonicalisation, single seed knob on
  the base Problem; trial-free BilinearForm terms (NGSolve drops them
  silently) rewritten as implicit-increment + explicit-half pairs.
  (3) Canonical setups — deterministic paper fields everywhere (Darcy
  sin-pixel initial state and per-step N(0,1) sources, Fisher bump,
  potential/AC decay+offset references), tracking coefficient 1 at the
  paper's q~ = 0, analytic reference time derivatives, aligned
  counter/u_true/source indexing; dropped real_root and PDEBench row
  rolling. Suite: 34 passed. Docs rewritten (DEVIATIONS.md as
  implementation notes, PAPER_TARGETS.md as table reference). Micro probes
  confirm fixed-point consistency (clean exact-q runs stay at the
  discretisation floor) and noise-ordered error plateaus.
- 2026-10-24 — Examples unification (supersedes the Target layout above):
  `mras/problems/` dissolved into `examples/` — every problem lives in
  `examples/` directly, declaring its `PROBLEM`/`PROBLEMS` class marker
  plus `DEFAULTS` (construction), `QUICK` (cheap check) and `CHECK`
  (checklist) sizes. All argument handling sits in `mras/live.py`, which
  also owns the loader, catalog and layered runner
  (constructor signature < DEFAULTS < --quick < YAML defaults < YAML runs
  < CLI), so example files stay untouched once a problem is well-formed
  and the YAML is the control switch. All field rendering goes through
  `ngsplot.Plot` (editable install of the local checkout required, see
  README); the notebook streams six live ngsplot panels during the solve
  and delivers an animated HTML movie at the end.
