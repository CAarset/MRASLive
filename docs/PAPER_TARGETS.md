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

### 1. Darcy (T=5, h_max=0.04; fine data mesh h=0.03, k=4)
- a†: PDEBench `2D_DarcyFlow_beta100.0_Train.hdf5`, sample 0, `/nu` entry,
  row-rolled by 12 to centre the 1-zone, mapped to Q_h via VoxelCoefficient
  with a zero band strip (to_gridfunc_0).
- g(·,t): i.i.d. N(0,1) values on a 128×128 grid → VoxelCoefficient, redrawn
  every time step (paper says N(0,1); the notebook draws uniform — see
  inconsistencies below).
- u†: implicit Euler on fine mesh (h=0.03, k=4); data z = L²-interpolation of
  u† to the coarse U_h, then noise added.
- a₀: indicator of B₀.₄₂((0.5,0.5)); u₀ = u†(0).
- Noise: relative L², ‖z−u†‖ = δ‖u†‖, per-step draw on 128² grid
  (GRO model: 2·U(0,1)−1 normalised to exact relative level).

### 2. Fisher–KPP (T=10, h_max=0.1; fine data mesh h=0.09, k=4)
- a†: ring: 1 if 0.52 < x²+y² < 0.9², else 0.25.
- g(x,t) = 3 exp(−(x²/0.41 + y²/0.42)) cos(2t).
- u₀ = 3 exp(−(x²/0.41 + y²/0.42)); u† simulated by implicit Euler
  (Newton/nonlinear solve since f is nonlinear in u).
- a₀: 1 on [−1.15,1.15]², 0 elsewhere (paper); paper scenarios δ ∈ {0, 3%}.
- Noise: paper says discrete Gaussian scaled to δ‖u†‖; code (GRO) uses
  uniform ± draw normalised to δ‖u†‖. Flag for authors.

### 3. Nonlinear potential (T=5, h_max=0.1, domain B_π(0) disk)
- u†(·,t) = u₀ · (6−t)/12 … i.e. linear decay from predetermined positive
  u₀ (see notebook `create_state`, vorticity/stream-function construction);
  text: u†(·,t) := u₀ (6−t)/6+6/6 — verify against notebook formula
  (the extracted PDF text is garbled here).
- c†(x) = π − √(x₁²+x₂²) (cone); g = Dt u† − Δu† + c†u† + u†|c†|^{2/3}c†.
- c̃ = c₀ = 0; boundary h > 0 (nonhomogeneous; handled via incremental form).
- Noise scenarios: 0%, 5% (main figure); Tables 6–7 use 0–50%.

### 4. Modified Allen–Cahn (T=5, h_max=0.1, domain [−2,2]×[−1,1])
- c†: three-part material: 1 if x₁+x₂<1 ∧ x₁−2x₂<−0.4; 4 if x₁+x₂≥1;
  2 if x₁+x₂<1 ∧ x₁−2x₂≥−0.4.
- u† = sin(π/4(x₁−2)) sin(π/2+(x₂−1))·(10−t)/10 + 1;
  g = Dt u† − Δu† + c(u†)³ + u†|c|^{2/3}c.
- u₀ = u†(0), c₀ = 0; noise 0,5,10,20% (Figure 9/11); Tables 6–7 0–50%.

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
