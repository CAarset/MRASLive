# MRAS cleanup — plan & progress log (overnight campaign)

Objective: total from-scratch cleanup of the MRAS code into one
Git-standardized, self-contained repo that (1) reproduces the paper
(AMMC 2026, DOI 10.3934/ammc.2026005), (2) exposes a simple interface —
user provides NGSolve mesh & weak form and MRAS "just happens", (3) is
streamlined/clean/optimized, (4) carries honest comments describing current
behaviour only. No backwards compatibility, no shims.

## Author decisions (2026-10-20/21)
- GRO folder is canonical for paper numerics (normalised relative-L2 noise).
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
docs/                  this file, PAPER_TARGETS.md
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
- [ ] P2 package skeleton + unit tests (noise normalisation, field mapping,
      seeding determinism, disk/square mesh dof counts vs Tables 2–5,
      known-solution steady check for the stepper).
- [ ] P3 core scheme + Darcy end-to-end (reproduce Fig 4 + Table 6 row
      h=0.04/0%: u 0.02, q 0.24 at T=5).
- [ ] P4 Fisher–KPP (Fig 5/6, ring a†, nonlinear implicit data solver).
- [ ] P5 Nonlinear potential (disk, vorticity-built u₀, Figs 7/8).
- [ ] P6 Allen–Cahn (Figs 9/10/11, 4-noise-row figure).
- [ ] P7 Decaying-Darcy demo (no paper targets, just a clean run).
- [ ] P8 Campaign: Tables 6/7/8 (≈136 runs) + Fig 12, parallel batches,
      resumable via results/campaign_state.json, paper-vs-new CSV diff.
- [ ] P9 README, notebooks, LICENSE, pyproject, delete legacy folders,
      final green suite, final commit, report with flagged deviations.

## Reproduction acceptance
- Main figures: qualitatively matching panels regenerated from new code.
- Tables 6–8: same setup re-run with new code; agreement target |Δ| ≤ 0.02
  per entry; entries marked "–" must blow up or be flagged.
- Any systematic mismatch → flagged in the final report, not hidden.

## Progress log
- 2026-10-21 02:5x CEST — Setup complete: venv (.venv, ngsolve 6.2.2608,
  numpy/matplotlib/h5py/pandas/tqdm/pytest), smoke Poisson solve OK,
  PDEBench hdf5 readable (nu 10000×128×128), checkpoint commit 4749311,
  PAPER_TARGETS.md written. Repo: no remote; linear history on main.
- (next entries appended below)
