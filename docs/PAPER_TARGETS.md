# Paper reference and campaign targets

Source of truth: `MRAS_Paper.pdf` (AMMC 8 (2026) 80–112,
DOI 10.3934/ammc.2026005, "Data assimilation via model reference adaptation
for linear and nonlinear dynamical systems"). The tables below are
transcribed from the published Tables 1–8; the code implements the paper's
theory with deterministic analytic fields and seeded L2-Gaussian FEM noise
(`docs/DEVIATIONS.md`), so its numbers are its own. Use the transcribed
values as order-of-magnitude context for the operator-run campaigns, not as
bit-level targets.

## Benchmarks (Table 1)

| # | Name | PDE | Type | Unknown |
|---|------|-----|------|---------|
| 1 | Darcy flow | Dt u − ∇·(a∇u) = g | linear in a | a |
| 2 | Fisher–KPP | Dt u − ∇·(a∇u) + u − u² = g | nonlinear in u | a |
| 3 | Nonlinear potential | Dt u − Δu + cu + c\|c\|^{2/3}u = g | nonlinear in c | c |
| 4 | Modified Allen–Cahn | Dt u − Δu + cu³ + c\|c\|^{2/3}u = g | nonlinear in u, c | c |

Shared discretisation: state space U_h ⊂ H¹ P³, parameter space Q_h ⊂ L²
piecewise constant, joint monolithic semi-implicit Euler, Δt = 1e-3 unless a
study varies it, T per example (Darcy 5, Fisher 10, potential 5, AC 5).

## Paper setup facts (Tables 2–5)

* **Darcy**: Ω = (0,1)²; a† from PDEBench `2D_DarcyFlow_beta100.0_Train.hdf5`
  sample 0; source i.i.d. N(0,1) 128² pixel fields, fresh per step; noise
  levels 0–50% (Tables 6–7); main h = 0.04, data mesh h = 0.03 (k = 4).
* **Fisher–KPP**: Ω = (−1.25,1.25)²; u0 = 3exp(−(x₁²+x₂²)/0.4);
  g = 3exp(−(x₁²+x₂²)/0.4)cos(2t); a† = ring 1 on 0.5² < x₁²+x₂² < 0.9²,
  else 0.25; a₀ = 1 on [−1.15,1.15]²; main h = 0.1; noise 0/3%.
* **Potential**: Ω = B_π(0); u† = u₀(6−t)/6 + 6; c† = π − |x|; c₀ = 0;
  main h = 0.1; noise 0/5%.
* **Allen–Cahn**: Ω = (−2,2)×(−1,1); u† = sin(π(x₁−2)/4)sin(π(x₂−1)/2)
  ·(10−t)/10 + 1; c† piecewise 4/2/1; c₀ = 0; main h = 0.1;
  noise 0/5/10/20%.

## Table 6 — space × noise (final-time relative L² errors, u / q)

Noise levels 0, 5, 10, 20, 50%.  "–" = algorithm did not converge (blow-up).

Darcy (T=5):

| h | 0% | 5% | 10% | 20% | 50% |
|-------|------------|------------|------------|------------|------------|
| 0.04  | 0.02 / 0.24 | 0.07 / 1.08 | 0.17 / 1.39 | 0.33 / 1.48 | 0.77 / 1.76 |
| 0.1   | 0.02 / 0.23 | 0.05 / 0.89 | 0.14 / 1.36 | 0.30 / 1.51 | 0.64 / 1.78 |
| 0.2   | 0.02 / 0.34 | 0.03 / 0.55 | 0.12 / 1.25 | 0.29 / 1.79 | 0.64 / 2.54 |

Fisher–KPP (T=10):

| h | 0% | 5% | 10% | 20% | 50% |
|-------|------------|------------|------------|------------|------------|
| 0.1   | 0.05 / 0.52 | 0.07 / 0.66 | 0.12 / 0.79 | 0.21 / 0.87 | 0.58 / 0.92 |
| 0.25  | 0.04 / 0.55 | 0.09 / 0.72 | 0.14 / 0.82 | 0.24 / 0.89 | 0.83 / 0.92 |
| 0.5   | 0.07 / 0.66 | 0.08 / 0.62 | 0.13 / 0.72 | 0.24 / 0.85 | 0.82 / 0.93 |

Nonlinear potential (T=5):

| h | 0% | 5% | 10% | 20% | 50% |
|-------|------------|------------|------------|------------|------------|
| 0.2   | 0.08 / 0.04 | 0.02 / 0.02 | 0.03 / 0.04 | 0.06 / 0.07 | 0.16 / 0.28 |
| 0.4   | 0.09 / 0.05 | 0.09 / 0.10 | 0.18 / 0.18 | 0.36 / 0.35 | 0.91 / 1.06 |
| 0.8   | 0.10 / 0.09 | 0.06 / 0.10 | 0.12 / 0.13 | 0.24 / 0.20 | 0.59 / 0.55 |

Modified Allen–Cahn (T=5):

| h | 0% | 5% | 10% | 20% | 50% |
|-------|------------|------------|------------|------------|------------|
| 0.1   | 0.39 / 0.28 | 0.42 / 0.29 | 0.44 / 0.29 | 0.50 / 0.33 | – |
| 0.25  | 0.41 / 0.30 | 0.42 / 0.30 | 0.44 / 0.30 | 0.51 / 0.32 | 0.62 / 0.68 |
| 0.5   | 0.41 / 0.31 | 0.42 / 0.31 | 0.44 / 0.31 | 0.50 / 0.32 | 0.74 / 0.39 |

## Table 7 — time step × noise (at the main h of each example)

Darcy (h=0.04):

| dt | 0% | 5% | 10% | 20% | 50% |
|-------|------------|------------|------------|------------|------------|
| 0.001 | 0.02 / 0.24 | 0.07 / 1.08 | 0.17 / 1.39 | 0.33 / 1.48 | 0.77 / 1.76 |
| 0.01  | 0.02 / 0.25 | 0.06 / 0.90 | 0.11 / 1.05 | 0.23 / 1.09 | 0.70 / 1.32 |
| 0.1   | diverges at every noise level | | | | |

Fisher–KPP (h=0.1):

| dt | 0% | 5% | 10% | 20% | 50% |
|-------|------------|------------|------------|------------|------------|
| 0.001 | 0.05 / 0.52 | 0.07 / 0.66 | 0.12 / 0.79 | 0.21 / 0.87 | 0.58 / 0.92 |
| 0.01  | 0.05 / 0.53 | 0.08 / 0.69 | 0.13 / 0.80 | 0.23 / 0.88 | – |
| 0.1   | 0.05 / 0.52 | 0.08 / 0.69 | 0.13 / 0.80 | 0.22 / 0.88 | 0.82 / 0.92 |

Nonlinear potential (h=0.2):

| dt | 0% | 5% | 10% | 20% | 50% |
|-------|------------|------------|------------|------------|------------|
| 0.001 | 0.08 / 0.04 | 0.02 / 0.02 | 0.03 / 0.04 | 0.06 / 0.07 | 0.16 / 0.28 |
| 0.01  | 0.08 / 0.04 | 0.02 / 0.03 | 0.03 / 0.04 | 0.07 / 0.07 | 0.17 / 0.29 |
| 0.1   | 0.08 / 0.04 | 0.02 / 0.02 | 0.03 / 0.04 | 0.06 / 0.07 | 0.16 / 0.28 |

Modified Allen–Cahn (h=0.1):

| dt | 0% | 5% | 10% | 20% | 50% |
|-------|------------|------------|------------|------------|------------|
| 0.001 | 0.39 / 0.28 | 0.42 / 0.29 | 0.44 / 0.29 | 0.50 / 0.33 | – |
| 0.01  | 0.39 / 0.28 | 0.42 / 0.29 | 0.44 / 0.29 | 0.50 / 0.33 | – |
| 0.1   | 0.35 / 0.28 | 0.39 / 0.29 | 0.41 / 0.29 | 0.46 / 0.33 | – |

(dt=0.001 rows reproduce the corresponding Table 6 main-h rows.)

## Table 8 — initial-guess sensitivity (u / q at 0% and 10% noise)

Fisher–KPP (h=0.1), constant a₀:

| noise | a₀ = 1 | a₀ = 2 | a₀ = 3 | a₀ = 4 |
|-------|------------|------------|------------|------------|
| 0%    | 0.05 / 0.56 | 0.17 / 1.23 | 0.29 / 2.02 | 0.41 / 2.83 |
| 10%   | 0.14 / 0.83 | 0.17 / 1.17 | 0.23 / 1.65 | 0.30 / 2.18 |

Modified Allen–Cahn (h=0.1), constant c₀:

| noise | c₀ = 0 | c₀ = 5 | c₀ = 10 | c₀ = 15 |
|-------|------------|------------|------------|------------|
| 0%    | 0.08 / 0.12 | 0.27 / 0.25 | 0.47 / 0.42 | 0.68 / 0.59 |
| 10%   | 0.23 / 0.17 | 0.28 / 0.24 | 0.38 / 0.34 | 0.49 / 0.45 |

## Figures to regenerate

- Fig 1: the four ground-truth parameter fields (2×2 panel).
- Fig 4: Darcy, u-row and a-row over t ∈ {0.001, 0.05, 0.1, 0.5, 1.5, 5} +
  truth panels, MRAS (top) vs exact (bottom), clean data.
- Fig 5: Fisher, evolution panels for clean + 3% noise; Fig 6: error field
  a(T)−a† for both.
- Fig 7: potential evolution, clean + 5%; Fig 8: error fields c(T)−c†.
- Fig 9: AC parameter evolution rows for 0/5/10/20%; Fig 10: error fields;
- Fig 11: L²-error-in-time curves for c and u at the four noise levels.
- Fig 12: sensitivity to initial q₀: Fisher a₀=4, AC c₀=15 — panels
  initial / reconstructed / truth.

Figure time instants: {0.001, 0.05, 0.1, 0.5, 1.5, 5} for
Darcy/Fisher/potential and {0.001, 0.05, 0.075, 0.2, 0.5, 1.5, 5} for AC.

## Darcy dataset facts (PDEBench file, author report input)

Empirical facts about `2D_DarcyFlow_beta100.0_Train.hdf5`, sample 0:
- `/nu[0]` takes exactly two values **{0.1, 1.0}** — a 10:1 contrast, *not*
  100:1. The "beta100" filename implies the paper intended a∈[1,100]; the
  stored field is 10:1.
- `/nu[0]` max-zone (the a=1.0 inclusion) occupies rows[43,123]×cols[23,101]
  — an off-centre, irregular blob (28.2% of cells).
- `/tensor[0,0]` (the state field) is never used for the ground truth; the
  truth is recomputed by implicit time-stepping with a fresh random source.
  Only the coefficient field is taken from PDEBench.
- `scripts/darcy_dataset.py` builds NGSolve-only fields of matched contrast
  with centred inclusions and brackets the PDEBench field in a side-by-side
  run; the comparison is a validation flag on the dataset choice, not a
  replacement.

## Paper-vs-original-code inconsistencies (flag to authors in final report)

1. Paper (Sec. 5.2) says Fisher noise is Gaussian; the original GRO code
   draws uniform ± on a 128² grid (normalised to the stated relative level).
2. GRO `functions_darcy.py` has two noise functions (`add_noise_draw` used
   only when `counter==4`, `add_noise` otherwise); the MRAS-folder notebook
   keeps the `counter==4` quirk but its functions file dropped
   `add_noise_draw`, so that pair cannot run. The GRO pair is canonical.
3. Paper Sec 4.2 says the Darcy source is i.i.d. N(0,1); the original code
   uses uniform draws. (This codebase follows the paper: N(0,1), seeded.)
4. Potential: the paper's u† decay formula is garbled in the PDF text layer;
   the effective notebook formula is `u₀(6−(t+dt))/6` with an offset, and
   the noise/offset globals were late-bound (z_min 1→0.1).
5. Table 4 states the source term with u†|c†|^{2/3}c† ordering vs Eq (26)
   c|c|^{2/3}u — same thing, notation only.
6. The Darcy notebook contains a stray `p` line after `q_exact.Set`
   (NameError if executed); cell-level execution order is not guaranteed.
7. Original noise was drawn from a preallocated unseeded list of
   `np.random.rand(128,128)` arrays, so the published figures are not
   bit-reproducible. (This codebase seeds every draw.)
8. The potential/Allen-Cahn notebooks pair each noisy `dx`
   integration-by-parts term with a boundary flux on `ds`. The pairs on the
   state (H1) test-function row assemble correctly; the pairs on the
   parameter row integrate an H1 flux against the L2 test function `s` on
   the boundary. NGSolve builds up to 6.2.2304 silently assemble those to
   zero (verified against a form whose true value is 2.0), and builds from
   6.2.2403 crash the assembler. Since the paper runs completed, the terms
   contributed nothing to the published numbers; this codebase omits the
   parameter-row `ds` fluxes. Cross-build agreement verified on mini-runs
   (6.2.2202 vs 6.2.2608 agrees to ~1e-5).
