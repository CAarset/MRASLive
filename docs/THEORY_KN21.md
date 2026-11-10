# Theory source II: Kaltenbacher–Nguyen 2021, the MRAS convergence theory

B. Kaltenbacher, T. T. N. Nguyen, *A model reference adaptive system approach
for nonlinear online parameter identification*, Inverse Problems **37**(5)
(2021) 055006; arXiv:2012.09908v2 (<https://arxiv.org/abs/2012.09908v2>).
Local copy: `docs/references/KaltenbacherNguyen21_arXiv2012.09908v2.pdf`
(not in git; fetch it again from the arXiv link). Numbers below refer to the
arXiv v2 version.

This is reference [25] of the AMMC paper (`MRAS_Paper.pdf`, ledger in
`docs/ASSUMPTIONS.md`): the AMMC paper's Assumption 2.2 and Propositions 2.3
are adapted from it, with the assumptions renumbered. The checklist
(`mras/verify.py`) and the proceedings' theory section rest on both.

## Numbering

| KN21 (Assumption 1.1) | AMMC (Assumption 2.2) | content | checklist row |
|---|---|---|---|
| (A1) | — (stated in the text) | the exact state exists uniquely; `z = u*` | not checked (well-posedness; KN21 Prop. 3.1 gives sufficient conditions) |
| (A2) | (A1) | Lipschitz remainder of the linearization, `L(‖q‖_H)` | "(A1) Lipschitz remainder" |
| (A3) | (A2) | coercivity `⟨f(q,z) − f(q*,z), q − q*⟩ ≥ C_coe ‖q − q*‖²_H`, **required only if `L ≢ 0`, i.e. `σ = 1`** | "(A2) monotonicity" |
| (A4) | (A3) | the tracking operator: `⟨Cv,v⟩ ≥ (L²/(2C_coe) + M)‖v‖²_U`, `⟨Cv,w⟩ ≤ Ñ‖v‖_U‖w‖_U` | "(A3) coercivity" |
| Remark 1.6 | Remark 4.2 ("persistence of excitation") | persistent excitation, after Baumeister et al. 1997 | "persistent excitation" (Gramian form, see below) |

## Setting (§1)

* Triples `Q ↪ U ↪ Z`, `U ↪ H ↪ U*`, `Q ↪ H ↪ Q*` (the last two Gelfand
  triples); full observations `z = G u* = u*`, `G = Id : U → Z`.
* MRAS (7)–(9): `D_t q + σ(D_t z + f(q,z) − g) − f_q'(q̃,z)*(u − z) = 0`,
  `D_t u + f(q,z) + C(‖q‖_H)(u − z) = g`.
* **σ** (p. 4): zero if the remainder `f(q,z) − f(q*,z) − f_q'(q̃,z)(q − q*)`
  vanishes (e.g. `f` linear in `q`), one otherwise.
* Solution spaces (10)–(11): `u ∈ L²(U) ∩ H¹(U*) ∩ L^∞(H)`,
  `q ∈ L²(H) ∩ H¹(Q*) ∩ L^∞(H)`.

## Scope of the convergence theory

* **Remark 1.3.** With `f` linear in `q`, `L ≡ 0` and coercivity (A3) "is
  not needed"; the linear case is the one of Baumeister et al. (1997), and the
  paper "only consider[s] the nonlinear case `L ≢ 0`, `σ = 1`". So
  Proposition 2.1 (exponential decay with rate `min{C_coe, 2M C_{U→H}}`) is
  proved for `σ = 1`. With `σ = 0` the parameter converges by persistent
  excitation, and `C_coe` plays no part — the content of the proceedings'
  Remark 1, and of the checklist's convergence row.
* **Remark 1.4.** If coercivity holds in a stronger norm of `q − q*`, the
  Lipschitz bound (A2) may be stated in that norm too, which admits stronger
  nonlinearity; `L` itself must stay a function of `‖q‖_H` (the norm the
  decay estimate bounds uniformly in time).
* **Remark 1.5.** Local Lipschitz continuity of `f` in `q` on balls, with
  `f(·, z)` Gâteaux differentiable, gives (A2).
* **Remark 1.6 — persistent excitation.** For `f` linear in `q`, `u` is
  *uniformly persistently excited* iff
  `∃ℓ, µ > 0 ∀h ∈ Q∖{0} ∀t₀ ≥ 0 ∃t₁, t₂ ∈ [t₀, t₀+ℓ] ∃v ∈ U∖{0}:
  ∫_{t₁}^{t₂} ⟨f(h, u(t)), v⟩ ds ≥ µ ‖h‖_Q ‖v‖_U`.
  Coercivity implies this with the weaker norms `‖h‖_H ‖v‖_H` (take `v = h`,
  `µ = ℓ C_coe`). The checklist's persistent-excitation check uses the
  Gramian form instead — the window mean of `‖f_q'(q̃,z(s)) p‖²_{U*}` against
  `γ ‖p‖²_H`, the classical finite-dimensional condition, which gives the
  decay rate of the parameter error directly (`mras/verify.py`); it is not the
  same statement as the one above (an integral of `f(h, u)` over a
  subinterval against `‖h‖_Q`, where cancellation over time matters).
* **Remark 1.7.** Nonautonomous `f(t, q, u)`: Carathéodory conditions, (A2)
  and (A3) uniformly for a.e. `t`.

## Exact data (§2.1)

Error system (19)–(21) with `r = u − u*`, `e = q − q*`, tested by `(r, e)`:

```
½ d/dt (‖r‖²_H + ‖e‖²_H) + ⟨f(q,z) − f(q*,z), e⟩ + ⟨f(q,z) − f(q*,z) − f_q'(q̃,z)e, r⟩ + ⟨C r, r⟩ = 0
```

(the coupling terms cancel). (A3) bounds the second term below, (A2) and
Young the third, (A4) absorbs `L²/(2C_coe)`: (22)
`½ E' + (C_coe/2)‖e‖²_H + M‖r‖²_U ≤ 0`, hence the energy decreases and
(23) `E(t) ≤ exp(−min{C_coe, 2M C_{U→H}} t) E(0)`, as printed, with
`C_{U→H}` "the norm of the continuous embedding". From (22) and
`‖r‖_H ≤ C_{U→H}‖r‖_U` the rate is `min{C_coe, 2M / C_{U→H}²}`; the printed
one follows from it only where `C_{U→H} ≤ 1` (docs/DEVIATIONS.md,
2026-11-10; the checklist reports `min{C_coe, 2M ρ_h}`). Proposition 2.1:
regularity (i), the integrated bound (ii), the exponential rate (iii). The
`σ`-term is what produces `C_coe ‖e‖²`: without it (σ = 0) the identity
dissipates `r` only.

## Noisy data (§2.2)

* Noise (27): `‖z^δ − z‖_{L^p(0,∞; Z)} ≤ δ` — a bound over infinite time, so
  noise of fixed level at all times is outside this class.
* The data must be smoothed (28): `R_sp : Z → U` pointwise in time and
  `R_ti : L^p(Z) → W^{1,p}(H)` with
  `‖R_sp(z^δ(t)) − z(t)‖_U ≤ δ̃_sp(t)`, `‖D_t(R_ti(z^δ) − z)‖_{L^p(H)} ≤ δ̃_ti`,
  `p ≥ 2`; the smoothed data replace `z` in every term of the MRAS (the
  σ-term takes `D_t R_ti(z^δ)`, the rest `R_sp(z^δ)`).
* Again `σ = 1` only. Further assumptions (Assumption 2.2):
  **(B1)** `f_q'(q̃, ·)` Lipschitz in the state, `‖f_q'(q̃,v) − f_q'(q̃,w)‖_{L(H,U*)} ≤ L̃₀‖v − w‖_U`;
  **(B2)** `f(q, ·)` Lipschitz in the state into `U*` and into `H`
  (constants `L̃₁`, `L̃₂`); and the tracking condition is strengthened, (29):
  `⟨C v, v⟩ ≥ ((L + L̃₀ ‖δ̃_sp‖_{L^∞(0,t)})²/(2C_coe) + M) ‖v‖²_U` — noisy
  data need a stronger `C`.
* Proposition 2.3: for any `ω < min{C_coe, 2M C_{U→H}}`,
  `E(t) ≤ e^{−ωt} E(0) + C (‖δ̃_sp‖²_{L^p(0,t)} + (δ̃_ti)²)`.
  The noise term grows with `t` unless `δ̃_sp` is integrable in time: for
  persistent noise this bounds the error only up to a growing term, i.e.
  semi-convergence, which is why noisy runs need a stopping rule.

## Well-posedness of the MRAS (§2.3)

Pseudomonotonicity of the MRAS operator in `(u, q)` (using (A3) for `q`),
Galerkin approximation bounded by Proposition 2.1, and uniqueness from (22).
Again in the `σ = 1` setting.

## Examples (§3) — the conditions the paper verifies

* **Existence (Prop. 3.1)** for `D_t u − ∇·(a∇u) + cu + φ(u)ψ(a,c) = g`:
  `a ∈ L^∞ ∩ W^{1,3}`, `a > 0`, `c ∈ L²` close to some `c̃ ∈ L^∞`,
  `g ∈ W^{1,∞,2}`, `u₀ ∈ H²`, `ψ` bounded, `φψ` monotone,
  `|φ(u)| ≤ C(1 + |u|³)`; then `u ∈ L^∞((0,T) × Ω)`.
* **§3.1, a potential** `D_t u − Δu + qu + φ(u)ψ(q) = g`: positivity
  `z ≥ c > 0` from a maximum principle with **inhomogeneous Dirichlet data**
  `z = h ≥ c` (and `u₀ ≥ c`, `g ≥ q*c + M_z M_q*`); homogenized with an
  extension `h̄`, `U = H¹₀`, **`Q = H¹`**, `H = L²`. Coercivity with
  `C_coe = c` from positivity and monotonicity of `φ(z)ψ`; (A2) with
  `L_{q̃,z}(‖q‖)` for `1 ≤ β ≤ 5/3`; `C = (L + 1)(−Δ)`. Remark 3.2:
  `φψ = z³|q|^{2/3}q`. Remark 3.5: keep `L` small (q̃, q near q*) to avoid a
  stiff system.
* **§3.2, a diffusion coefficient**: coercivity needs the model **perturbed
  by `u Δq`**, `f(q,u) = −∇·(q∇u) + uΔq + φψ`, a negative state `z ≤ −c < 0`
  and `Q = H¹₀ ∩ H²`; then `⟨f(q,z) − f(q*,z), q − q*⟩ = −∫ z |∇(q − q*)|² ≥
  c ‖q − q*‖²_{H¹₀}` — coercivity in the stronger norm (Remark 1.4). Without
  the perturbation (Darcy, Fisher, Burgers here) the pairing is not coercive,
  as the AMMC paper's Remark 4.2 recalls.

## What this changes for the repository

* `σ = 0` examples (Darcy, Fisher, both Burgers, Arrhenius, blank): the
  coercivity row is reported, but it is not a hypothesis of their
  convergence (Remark 1.3); their convergence is the persistent-excitation
  statement. `σ = 1` restores the coercivity route for affine `f`
  (`arrhenius_sigma1`).
* Noise: the theory's route is smoothing (28) plus a stronger tracking
  operator (29), and it bounds the error only up to a term growing with `t`
  for persistent noise — the basis for data smoothing and a stopping rule.
* Potential and Allen–Cahn: the prescribed boundary state `u = h` is the
  paper's route to positivity of the data, hence to coercivity.
