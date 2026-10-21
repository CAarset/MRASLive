# MRAS — data assimilation via model reference adaptation

Reference implementation of the numerical scheme in

> B. Kaltenbach, C. Aarset, T. T. N. Nguyen, *Data assimilation via model
> reference adaptation for linear and nonlinear dynamical systems*,
> Applied Mathematics for Modern Challenges **8** (2026) 80–112,
> [doi:10.3934/ammc.2026005](https://doi.org/10.3934/ammc.2026005).

The estimator evolves the state `u` and the unknown parameter `q` jointly on
a product finite-element space, driven only by noisy scalar-field data `z` of
`u`.  Four benchmarks are included: a Darcy flow driven by PDEBench data, a
Fisher–KPP equation, a nonlinear-in-state potential problem, and an
Allen–Cahn equation.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[pdebench,dev]"     # ngsolve, numpy, matplotlib, h5py, pytest
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
results = prob.run(levels=[0.0, 0.05])              # noisy data -> MRAS loop
for r in results:
    print(r.noise_level, r.final_error_u, r.final_error_q)
error_figure(results, prob.dt, fname="errors.png")
```

## Bring your own mesh and weak form

A problem is a `mras.Problem` subclass on `(U_h, Q_h) = P3 x P0` over any
NGSolve mesh.  Two hooks are required — the data feed and the joint
bilinear/linear forms of one time step:

```python
from mras.scheme import Problem

class MyProblem(Problem):
    name = "my_problem"

    def data(self, state):                    # set the datum and the reference
        state.u_true.Set(exact_at(state.counter))
        state.set_z(noisy_copy_of(state.u_true))

    def forms(self, state):                   # one MRAS step as a joint BVP
        a, res = BilinearForm(self.X), LinearForm(self.X)
        a += self.u * self.v * dx + self.q * self.s * dx
        # ... model terms and the q-gradient term from the paper ...
        return a, res
```

`examples/helmholtz_in_parameter.py` is a complete, runnable version of this
pattern (nonlinear-in-parameter Helmholtz); `examples/darcy_decaying.py`
shows the lower-level `ImplicitStepper` API.

## Running the studies (Tables 6–8, Figs 4–12)

Each campaign runs one noise level per process and resumes where a previous
run stopped (finished levels are skipped).  Figures and the markdown table
are assembled from the saved runs afterwards.

| paper item | campaign | figures |
|---|---|---|
| Fig 4, Table 6 (Darcy) | `scripts/darcy_reproduce.py --out results/darcy` | `scripts/make_figures.py --problem darcy --in results/darcy` |
| Table 7, Figs 5–8 (Fisher, potential) | `scripts/reproduce.py --problem fisher` / `--problem potential` | `scripts/make_figures.py --problem fisher` etc. |
| Table 8, Figs 9–11 (Allen–Cahn) | `scripts/reproduce.py --problem allen_cahn` | `scripts/make_figures.py --problem allen_cahn` |
| Fig 12 / Table 8 (q0 sensitivity) | vary the initial estimate in `init_level`, then `scripts/reproduce.py --problem fisher` / `--problem allen_cahn` | as above |
| Darcy data cross-check | `scripts/darcy_dataset.py --quick` | writes `comparison.json` |

Full-size runs are hours-long per level on 32 cores; every script accepts
`--micro` for a seconds-long wiring check and `--h/--dt/--tend/--levels`
overrides.  Keep `OMP_NUM_THREADS` small if other work is running.

## Repository layout

```
mras/               package: scheme (MRAS loop), domains, fields, noise,
                    stepper, problems/{darcy,fisher,potential,allen_cahn},
                    data/pdebench, results, plotting
examples/           runnable demos (quickstart, Helmholtz, decaying Darcy)
scripts/            reproduction campaigns and figure assembly
tests/              pytest suite (domains, fields, noise, scheme, plotting)
docs/               cleanup plan, paper-target mapping, deviation report
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
