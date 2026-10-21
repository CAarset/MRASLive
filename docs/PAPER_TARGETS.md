# Paper reproduction targets

Source of truth: `MRAS_Paper.pdf` (AMMC 2026, DOI 10.3934/ammc.2026005,
"Data assimilation via model reference adaptation for linear and nonlinear
dynamical systems"). Text extract kept at `/tmp/paper.txt` during setup; the
numbers below are transcribed from Tables 6–8 and the setup Tables 2–5.

Canonical code for paper numerics: the **GRO** folder (public upload), as
decided by the authors. The MRAS folder is the private superset with
exploratory variants.

## Benchmarks (Table 1)

| # | Name | PDE | Type | Unknown | Canonical notebook |
|---|------|-----|------|---------|--------------------|
| 1 | Darcy flow | Dt u − ∇·(a∇u) = g | linear | a | GRO/adaptive_darcy_joint gitlab.ipynb |
| 2 | Fisher–KPP | Dt u − ∇·(a∇u) + u − u² = g | nonlinear in u | a | GRO/adaptive nonlinear in state joint.ipynb |
| 3 | Nonlinear potential | Dt u − Δu + cu + c\|c\|^{2/3}u = g | nonlinear in c | c | GRO/adaptive nonlinear c problem joint_new.ipynb |
| 4 | Modified Allen–Cahn | Dt u − Δu + cu³ + c\|c\|^{2/3}u = g | nonlinear in u, c | c | GRO/adaptive modified allen cahn_gro.ipynb |
| 5 | (Decaying Darcy — not in paper, kept as extra demo) | like Darcy, decaying data | linear | a | MRAS/adaptive_darcy_decaying.ipynb |

Shared discretization: state space U_h ⊂ H¹₀ P³ (k=3, Dirichlet), parameter
space Q_h ⊂ L² piecewise constant (k=0), joint monolithic semi-implicit Euler,
Δt = 0.001 unless a study varies it, T per example.

## Per-benchmark data recipes

Canonical = GRO notebooks, verified line-by-line 2026-10-21. Deviations of
code vs paper text are inline-tagged "≠paper".

### 1. Darcy (T=5, dt=1e-3, h_max=0.04, k_u=3; fine data mesh h=0.03, k=4)
- a†: PDEBench sample 0 `/nu` (values {0.1, 1.0}), np.roll rows by 12,
  VoxelCoefficient((0,0),(1,1), linear=True) onto Q_h times zero-band ramps
  `IfPos(x-.05,1,20x)…` (to_gridfunc_0).
- g(·,t): fresh `np.random.rand(128,128)` uniform [0,1] grid per step (5001
  draws, unseeded), VoxelCoefficient onto V/Ufes. Data loop uses gⁿ =
  np_random[counter-1] (lagged); MRAS loop feeds np_random[counter] — one-step
  source mismatch vs the true driving term (reproduce as-is; ≠paper).
- u†: implicit Euler on fine mesh, u†(0) = random grid np_random[0]; fully
  implicit diffusion (M+dtK)⁻¹, source lagged; L²-interpolate to coarse U_h.
- Joint form: a: uv + dt∇u∇v + dt q∇z∇v + qs; res: gfu·v + dt g·v + dt∇z∇v +
  gfq·s + dt∇z_old(∇gfu−∇z_old)s. Diffusion in the reference model is 1, not
  q (model reference adaptation). Solve: CG + Jacobi on joint X.
- a₀: indicator of B₀.₄₂((0.5,0.5)); u₀ = gfu init = to_gridfunc(np_random[0])
  (≠paper's u₀=u†(0)); z_old init 0, updated at loop TOP.
- Counter starts 1 (z = add_noise(u_exact[1],…)); loop `while t < tend-2dt`;
  errors seeded with initial error, then appended after each solve
  (error_q before gfq.Set, error_u after gfu.Set).
- Noise: 2U(0,1)−1 on 20² grid, VoxelCoefficient (0,0)-(1,1), normalised to
  exact δ‖u†_h‖ (≠paper: Gaussian 128²). Noise levels [0, 0.05].

### 2. Fisher–KPP (T=10, dt=1e-3, h_max=0.1, k_u=3; fine data mesh h=0.09, k=4; domain [−1.25,1.25]², NO Dirichlet on U_h — regex mismatch)
- a†: ring: 1 if 0.5² < x²+y² < 0.9², else 0.25.
- g(x,t) = 3 exp(−(x²+y²)/0.4) cos(2t) (both coords /0.4, ≠paper 0.41/0.42).
- u†: implicit Euler on fine V, u†(0) = to_gridfunc(np_random[0]) random
  (≠paper: 3exp Gaussian); a reassembled per step with lagged reaction
  −gf·u·v (semi-implicit logistic), source coef_f = g(t) at current t.
- a₀: 1 on [−1.15,1.15]², 0 elsewhere ✓; z_old init = 1 (const), updated at
  loop BOTTOM; gfq init a₀; gfu init u_exact[0]; counter starts 1;
  loop `while t < tend-2dt`; noise peaks 12 on unit box (≠domain box —
  edge-extrapolated over [−1.25,1.25]²).
- Joint form: a: uv + dt∇u∇v + dt q∇z∇v + qs; res: gfu·v + dt(g−z+z²)v +
  dt∇z∇v + gfq·s + dt∇z_old(∇gfu−∇z_old)s. Noise [0, 0.03].

### 3. Nonlinear potential (T=6!, dt=1e-3 (6000 steps), h_max=0.1, k_u=3, σ=1; domain disk B_π, U_h dirichlet="circle", Q_h L2 free)
- u†(·,t) = u₀·(6−(t+dt))/6 + 0.1 evaluated per step (≠paper: T=5 and offset
  1; z_min was 1 at set_u definition but late-bound global becomes 0.1 —
  reproduce effective behaviour, flag).
- u₀ = 25·|velocity| from seeded (rng seed 2) nonlinear vorticity/stream
  solve on the disk, clamped `IfPos(0.5-(pi-r), pi-r, 0.5)*speed` (→0 at ∂B).
- c†(x) = π − √(x₁²+x₂²) (cone); c₀ = 0 (gfq.Set(0)).
- Joint scheme (σ=1): a: qs + σdt(|gfq|^{2/3}qz + qz)s; uv + dt·C(gfq)∇u∇v +
  dt gfq z v + dt|gfq|^{2/3}gfq z v (note: trial q absent from u-eq — f linear
  in q); res: σdt[u_t + |c†|^{2/3}c†u† + c†z]s − σdt[zt + |gfq|^{2/3}gfq z +
  gfq z]s + dt·z_old(gfu−z_old)s (+ Δ-noise-cancellation terms iff δ>0);
  v-side: dt[|c†|^{2/3}c†u† + z_t + c†u†]v + dtC(∇z−∇gfu)∇v −
  dt(|gfq|^{2/3}+1)gfq z v. ABSOLUTE (not incremental) solve for gf.
- C(gfq) = 2 + (1000·‖gfq‖^{2/3} + ‖c†‖^{2/3} + ‖c†‖_{L³}^{2/3})² (z_max/z_min
  = 100/0.1 late-bound globals; ≠paper formula, flag).
- Noise: add_noise re-seeds np.random.seed(0) EVERY call → frozen identical
  pattern (zt = 0 after step 1); peaks 20 on box (−3,3)². Levels [0, 0.05].
- z_old init 0 (loop top has NO z_old←z; it is at bottom). u-dof check:
  U_h(dirichlet) 34279 free? (paper Table 3: 34279/7552 at h=0.1).

### 4. Modified Allen–Cahn (T=10!, dt=1e-3, h_max=0.1, k_u=3, σ=1; domain [−2,2]×[−1,1], U_h full Dirichlet, Q_h free)
- c†: 4 if x+y≥1; 2 if x+y<1 ∧ x−2y>−0.4; else 1 (v_bg=1, v_inc1=4, v_inc2=2;
  v_side=3 unused). ✓paper
- u† = sin(0.25π(x−2))·sin(0.5π(y−1))·(10−(t+dt))/10 + 0.2 (≠paper offset 1,
  T=5; z_min=0.2/z_max=1). u₀ = u†(0), c₀ = 0, gfq init 0.
- Joint scheme: incremental (solve aux, gf += aux). a: qs + σdt q z³ s +
  σdt(|gfq|^{2/3}qz + qz)s; uv + dt·C(gfq)∇u∇v + dt q z³ v +
  dt|gfq|^{2/3} q z v; res: σdt[u_t + |c†|^{2/3}c†u† + c†z]s
  − σdt[zt + |gfq|^{2/3}gfq z + gfq z³]s + dt·z_old(gfu−z_old)s
  (+Δ-terms iff δ>0); v-side: dt[|c†|^{2/3}c†u† + z_t + c†u†]v
  − dt(|gfq|^{2/3}gfq z + gfq z³)v + dtC(∇z−∇gfu)∇v (+Δ-terms iff δ>0).
- C(gfq) = 1 + 2·(z_max²/z_min)·(‖gfq‖^{2/3}+‖c†‖^{2/3}+‖c₀=c† L³‖^{2/3})²
  = 1 + 10(…)². Loop `while t < tend-dt`; counter 0-based; series snapshots
  at counter ∈ {1,5,25,50,75,100,200,500,1500,2500,5000,tsteps−2}.
- Noise: add_noise returns (z, noise); peaks 20 on box (−2,−1)-(2,1);
  z_old init 0, updated at top-of-form (after draw). Levels [0, 0.1, 0.2,
  0.05] (iteration order!).

## Table 6 — space × noise (final-time relative L² errors; u / q)

Darcy (T=5):
- h=0.04: u 0.02/0.07/0.17/0.33/0.77 ; q 0.24/1.08/1.39/1.48/1.76 (0,5,10,20,50%)
- h=0.1:  u 0.02/0.05/0.14/0.30/0.64 ; q 0.23/0.89/1.36/1.51/1.78
- h=0.2:  u 0.02/0.03/0.12/0.29/0.64 ; q 0.34/0.55/1.25/1.79/2.54

Fisher–KPP (T=10):
- h=0.1:  u 0.05/0.07/0.12/0.21/0.58 ; q 0.52/0.66/0.79/0.87/0.92
- h=0.25: u 0.04/0.09/0.14/0.24/0.83 ; q 0.55/0.72/0.82/0.89/0.92
- h=0.5:  u 0.07/0.08/0.13/0.24/0.82 ; q 0.66/0.62/0.72/0.85/0.93

Nonlinear potential (T=5):
- h=0.2:  u 0.08/0.02/0.03/0.06/0.16 ; q 0.04/0.02/0.04/0.07/0.28
- h=0.4:  u 0.09/0.09/0.18/0.36/0.91 ; q 0.05/0.10/0.18/0.35/1.06
- h=0.8:  u 0.10/0.06/0.12/0.24/0.59 ; q 0.09/0.10/0.13/0.20/0.55

Allen–Cahn (T=5):
- h=0.1:  u 0.39/0.42/0.44/0.50/– ; q 0.28/0.29/0.29/0.33/–
- h=0.25: u 0.41/0.42/0.44/0.51/0.62 ; q 0.30/0.30/0.30/0.32/0.68
- h=0.5:  u 0.41/0.42/0.44/0.50/0.74 ; q 0.31/0.31/0.31/0.32/0.39
(– = algorithm did not converge / blow-up)

## Table 7 — time step × noise (at main h per example)

Darcy (h=0.04): dt=0.001 = Table 6 h=0.04 row; dt=0.01: u
0.02/0.06/0.11/0.23/0.70, q 0.25/0.90/1.05/1.09/1.32; dt=0.1: diverges all.
Fisher (h=0.1): dt=0.01: u 0.05/0.08/0.13/0.23/–, q 0.53/0.69/0.80/0.88/–;
dt=0.1: u 0.05/0.08/0.13/0.22/0.82, q 0.52/0.69/0.80/0.88/0.92.
Potential (h=0.1? paper main h=0.1; Table 6 rows are 0.2/0.4/0.8 — for
Table 7 assume h=0.2): dt=0.01: u 0.08/0.02/0.03/0.07/0.17, q
0.04/0.03/0.04/0.07/0.29; dt=0.1: u 0.08/0.02/0.03/0.06/0.16, q
0.04/0.02/0.04/0.07/0.28.
Allen–Cahn (h=0.1): dt=0.01: u 0.39/0.42/0.44/0.50/–, q 0.28/0.29/0.29/0.33/–;
dt=0.1: u 0.35/0.39/0.41/0.46/–, q 0.28/0.29/0.29/0.33/–.

## Table 8 — initial-guess sensitivity (clean and 10% noise)

Fisher (h=0.1): a₀ ∈ {1,2,3,4} const:
- 0%: u 0.05/0.17/0.29/0.41 ; q 0.56/1.23/2.02/2.83
- 10%: u 0.14/0.17/0.23/0.30 ; q 0.83/1.17/1.65/2.18
Allen–Cahn (h=0.1): c₀ ∈ {0,5,10,15} const:
- 0%: u 0.08/0.27/0.47/0.68 ; q 0.12/0.25/0.42/0.59
- 10%: u 0.23/0.28/0.38/0.49 ; q 0.17/0.24/0.34/0.45

## Figures to regenerate

- Fig 1: the four ground-truth parameter fields (2×2 panel).
- Fig 4: Darcy, u-row and a-row over t ∈ {0.001?,0.05?,0.1?,0.5?,1.5,5} + truth
  panels, MRAS (top) vs exact (bottom), clean data.
- Fig 5: Fisher, evolution panels for clean + 3% noise; Fig 6: error field
  a(T)−a† for both.
- Fig 7: potential evolution, clean + 5%; Fig 8: error fields c(T)−c†.
- Fig 9: AC parameter evolution rows for 0/5/10/20%; Fig 10: error fields;
  Fig 11: L²-error-in-time curves for c and u at the four noise levels.
- Fig 12: sensitivity to initial q₀: Fisher a₀=4, AC c₀=15 — panels
  initial / reconstructed / truth.

Time instants shown in figures (read from notebook outputs):
{0.001, 0.05, 0.1, 0.5, 1.5, 5} for Darcy/Fisher-like and
{0.001, 0.05, 0.075, 0.2, 0.5, 1.5, 5} for AC (Figure 9 header).

## Darcy dataset critique (author request 2026-10-21)

Empirical facts about `2D_DarcyFlow_beta100.0_Train.hdf5`, sample 0 (read via
h5py):
- `/nu[0]` takes exactly two values **{0.1, 1.0}** — a 10:1 contrast, *not*
  100:1. The "beta100" filename implies the paper intended a∈[1,100]
  (100:1); the stored field is 10:1. If the paper's a† is meant to be this
  field as-is, the actual coefficient contrast is 10:1.
- `/nu[0]` max-zone (the a=1.0 inclusion) occupies rows[43,123]×cols[23,101]
  — an off-centre, irregular blob (28.2% of cells). The `roll(12)` in the
  notebook merely nudges it to rows[31,111] to look more centred; it is not a
  true recentering.
- `/tensor[0,0]` (the state field, range 0.17–45.25) is **never used** for the
  ground truth. The notebook recomputes u† by implicit time-stepping with
  q_exact=nu and a fresh random source g. So only the coefficient nu is taken
  from PDEBench; the "dataset" is really just an a† field generator.
- q_exact is set with `to_gridfunc_0` (zero band at boundary) so a† vanishes
  near ∂Ω — a Dirichlet artefact that inflates boundary parameter error.

Action (quick, non-blocking): build a **second, NGSolve-only Darcy dataset** —
random binary coefficient a† = thresholded smoothed-Gaussian-field with a
clean centred inclusion, both at 10:1 and 100:1 contrast — run the same MRAS
driver, and compare reconstruction error vs the PDEBench field. Outcome is a
validation note or a flag for the authors, not a replacement of the paper set.

## Known paper↔code inconsistencies (flag to authors in final report)

1. Paper (Sec. 5.2) says Fisher noise is Gaussian; GRO code draws uniform
   ± on a 128² grid (normalised to the stated relative level).
2. GRO functions_darcy.py has two noise functions (add_noise_draw used only
   when `counter==4`, add_noise otherwise); the MRAS notebook keeps the
   counter==4 quirk but its functions file dropped add_noise_draw → the MRAS
   pair cannot even run. GRO pair is consistent; canonical = GRO.
3. Paper Sec 4.2 says Darcy source is i.i.d. N(0,1); code uses uniform.
4. Potential: paper's u† decay formula is garbled in the PDF text layer;
   take the notebook `set_u(t)` formula as truth (verify).
5. Table 4 states source term with u†|c†|^{2/3}c† ordering vs Eq (26)
   c|c|^{2/3}u — same thing, notation only.
6. Darcy notebook contains a stray `p` line after `q_exact.Set` (NameError if
   executed); cell-level execution order of notebooks is not guaranteed.
7. Noise is drawn per time step at counter+1 from a preallocated list of
   `np.random.rand(128,128)` arrays — unseeded → paper figures are not
   bit-reproducible; the campaign adds explicit seeding.
8. Potential (5.2) / Allen-Cahn (5.3) notebooks pair each noisy `dx`
   integration-by-parts term with a boundary flux on `ds`.  The pairs on the
   *state* (H1) test-function row assemble correctly and are kept.  The two
   pairs on the *parameter* row integrate the H1 flux against the L2 test
   function `s` on the boundary: NGSolve builds up to 6.2.2304 accept those
   forms but silently assemble them to zero (verified with 6.2.2202/2304
   wheels; standalone and product-space probes both give 0.0 for a form whose
   true value is 2.0), and builds from 6.2.2403 on crash the assembler
   outright.  The paper runs completed without error, so those terms
   contributed nothing to the published numbers; `mras` therefore omits the
   parameter-row `ds` fluxes and keeps their `dx` halves.  Verified: the same
   `mras` code on the 6.2.2202 and 6.2.2608 builds of the noisy mini-runs
   agrees to 1.2e-5 (potential: err_u 0.576936 vs 0.576948, err_q 6.812 vs
   6.763; AC: err_u 0.568 vs 0.579, err_q 6.825 vs 6.831 at h=0.2, tend=0.01).
