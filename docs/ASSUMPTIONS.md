# Assumption ledger: Assumption 2.2 across all examples

Theory source: Kaltenbach–Aarset–Nguyen, *Data assimilation via model reference
adaptation for linear and nonlinear dynamical systems*, AMMC 8 (2026) 80–112
(DOI 10.3934/ammc.2026005; arXiv:2602.10920v2). Page references are to the
typeset PDF (`MRAS_Paper.pdf`). This ledger pairs the paper's standing
assumptions with (i) the analytical verification per example and (ii) what the
repository automatically checks today (`mras/verify.py`, `mras/adjoint.py`),
plus the pipeline design for the constants we do not yet derive.

---

## 1. The assumptions, verbatim (Assumption 2.2, p. 5–6)

Standing setting: `U ⊆ H ≅ H* ⊆ U*` Gelfand triple; `q` the unknown parameter
(pathwise in `Q ∩ L^∞(0,T;H)`), `z` the datum (given), `u` the MRAS state,
`q†` the true parameter, `q̃` any point where `q ↦ f(q, z)` is
Gateaux-differentiable (existence is part of the theory; `q̃` need not be
computable). The model `f : Q × U → U*` and its partial `q`-derivative act as
`f_q(q̃, z) ∈ L(Q, U*)` with Banach adjoint `f_q(q̃, z)* ∈ L(U, Q*)`.

* **(A1) differentiability / Lipschitz remainder** — for a.e. `t` and all `q`:

  ```
  ‖f(q,z) − f(q†,z) − f_q(q̃,z)(q − q†)‖_{U*} ≤ L(‖q‖_H) ‖q − q†‖_H      (5)
  ```

  with `L : [0,∞) → [0,∞)` monotonically increasing. Note the quantifier on
  `q`: `L` may grow with the ball the iterate lives in — it is a
  *local-on-balls* Lipschitz modulus of the first-order Taylor remainder of
  `q ↦ f(q, z)` at the (unknown!) point `q̃`, uniform over `q†` with the
  difference measured from `q` alone.

* **(A2) monotonicity (coercivity in the parameter)** — for all `q ∈ Q`:

  ```
  ⟨f(q,z) − f(q†,z), q − q†⟩_{Q*,Q} ≥ C_coe ‖q − q†‖²_H,   C_coe > 0.
  ```

  No `q̃` appears: this is plain monotonicity of `q ↦ f(q, z)` against the
  truth, and it is the assumption that carries identifiability. It is the
  assumption that *fails* for the gradient-coupling examples (Remark 4.2).

* **(A3) state-operator bounds** — for all `v, w ∈ U`:

  ```
  ⟨C(‖q‖_H) v, v⟩_{U*,U} ≥ ( L(‖q‖_H)² / (2 C_coe) + M ) ‖v‖²_U =: M̃(‖q‖_H) ‖v‖²_U   (6)
  ⟨C(‖q‖_H) v, w⟩_{U*,U} ≤ Ñ(‖q‖_H) ‖v‖_U ‖w‖_U                                        (7)
  ```

  with `M > 0` free. `C` is *design freedom of the algorithm* (Remark 6.2:
  the choice is not unique). The coupling is explicit: (A3) must absorb the
  (A1) constant through `L²/(2 C_coe)`; `M` buys the stability margin.

* **Proposition 2.3** — under (A1)–(A3) the joint error decays exponentially,
  rate `C′ = min{ C_coe, 2 M̃ / C_{U→H} }` (plus persistent-excitation terms):
  (A2) and (A3) *both* enter the convergence rate, (A1) only through M̃.

* **σ ∈ {0,1}** (paper eq. (4)): `σ = 0` iff `q ↦ f(q,z)` is linear, `σ = 1`
  otherwise; `σ` multiplies the `D_t z + f(q,z) − g` drag term on the parameter
  equation's left. Not an assumption — a construction switch forced by the
  linearity of `f` in `q`.

Three structural observations used throughout:

1. **(A1) sees only the q-nonlinear remainder.** Split
   `f(q,z) = f₀(z) + A(z)q + R(q,z)` with `A(z)q` the affine-in-q skeleton and
   `R` the genuinely nonlinear remainder. Then the (5)-remainder equals
   `R(q,z) − R(q†,z) − R_q(q̃,z)(q−q†)` — the skeleton contributes *exactly
   zero*, even though `A(z)` itself is unbounded as a map `H → U*` (e.g.
   Darcy's `h ↦ −∇·(h∇z)` has no `L²`-bound). Hence **L ≡ 0 for every example
   affine in q** (Darcy, Fisher, Burgers, Arrhenius, constant-q Helmholtz),
   with nothing to prove beyond the affineness, and `L` is a non-trivial
   increasing function of `‖q‖_H` exactly when `R ≢ 0` (potential, AC).
2. **(A3) is by construction, not by discovery.** Every operator the paper
   uses (`−Δ` on `H¹₀` with `‖v‖_U := ‖∇v‖₂`; `(L²/2C_coe + M)·(−Δ_Tr)` on `H¹`
   with `‖v‖²_U := ‖∇v‖² + ‖Tr v‖²_{L²(∂Ω)}`) is a positive scalar function of
   `‖q‖` times an operator whose form *is* the squared U-norm. Then
   `⟨C(‖q‖)v,v⟩ = c(‖q‖)·‖v‖²_U`, i.e. `M̃ = Ñ = c(‖q‖)` holds with equality
   for every `v` and every `‖q‖` — the paper *defines* `C` so that (6)–(7) are
   tautological. Verification reduces to checking the code assembles that
   exact form.
3. **(A2) is the only assumption that can genuinely fail** — and the paper
   says so (Remark 4.2): for the gradient-coupled examples (Darcy-type
   `⟨ε, −∇·(ε∇z)⟩ ~ −½∫ε²Δz`) coercivity would require perturbing the model by
   `+u∇a`, or the alternative *persistent excitation* route; the paper keeps
   the unperturbed model and lets numerics speak. Our audits quantify exactly
   this gap.

---

## 2. Master table

Legend. `⟨·,·⟩` without subscript = `L²(Ω)`; `z̲ := ess inf z`, `z̄ := ess sup
|z|` over the data box; `C_{H¹→L⁶}` the Sobolev embedding constant of the
paper; `Δ_Tr` the operator `⟨−Δ_Tr v, w⟩ := ∫∇v·∇w + ∫∂Ω Tr v Tr w`.
Adjointness column: independent numerical check of the *analytical pair*
`⟨f_q(q,z)p, s⟩ = ⟨p, f_q(q,z)* s⟩` over ~100 random draws of `(q,z,p,s)`
**before and independent of any MRAS run** (design in §5; the two listed
figures are what exists today — §5 explains why they must be separated).
"machine" = ratio < 1e-14.

| # | example | f(q, z) | f_q(q, z) p | f_q(q, z)* s | adjointness (100 draws) | σ | (A1) L(‖q‖_H) | (A2) C_coe | (A3) C(‖q‖_H); M̃, Ñ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Darcy** (§4.1, U=H¹₀ ‖∇·‖₂, H=L²) | `−∇·(q∇z)` | `−∇·(p∇z)` | `∇z·∇s` | analytic identity; scheme op: machine (8.6e-16) | 0 | **0** (affine; remainder ≡ 0) | *not claimed by paper* (Rem. 4.2); probe: **FAIL −2.36** | `−Δ` (const); `M̃ = M`, `Ñ = 1` |
| 2 | **Fisher–KPP** (§5.1, same spaces) | `−∇·(q∇z) + z − z²` | `−∇·(p∇z)` | `∇z·∇s` | analytic identity; scheme op: machine (3.1e-15) | 0 | **0** (z-nonlinearity drops out of the q-remainder; L = 0 "remains", §5.1) | *not claimed*; probe: **FAIL −0.422** | `−Δ` (const); `M̃ = M`, `Ñ = 1` |
| 3 | **Burgers pipe** (repo; U=H¹₀, H=L²) | `−∇·(q∇z) + z ∂_x z` | `−∇·(p∇z)` | `∇z·∇s` | scheme op: machine (1.8e-15) | 0 | **0** (affine) | probe: **FAIL −1.26** (Remark 4.2 situation) | `−Δ` (const); `M̃ = M`, `Ñ = 1` |
| 4 | **Nonlinear potential** (§6.1, U=H¹ ‖∇·‖²+‖Tr·‖², H=L²) | `−Δz + q z + q\|q\|^{2/3} z` | `p z (1 + (5/3)\|q\|^{2/3})` | `s z (1 + (5/3)\|q\|^{2/3})` | analytic pair self-adjoint (multiplication); scheme op: DEVIATION 0.65 — see §4.3 | 1 | `(5/3) z̄ C_{H¹→L⁶} (‖q‖+‖q†‖+‖q̃‖)^{2/3}` (Prop 6.1, derived) | `z̲ = ess inf u† > 0` (positivity (27)); probe OK ≥ 23.1 | `(L²/(2 z̲) + M)(−Δ_Tr)`; `M̃ = Ñ = L²/(2 z̲) + M` by construction |
| 5 | **Modified Allen–Cahn** (§7.1, same spaces) | `−Δz + q z³ + q\|q\|^{2/3} z` | `p (z³ + (5/3)\|q\|^{2/3} z)` | `s (z³ + (5/3)\|q\|^{2/3} z)` | self-adjoint multiplication; scheme op: DEVIATION 0.96 — see §4.3 | 1 | same as #4 (Prop 7.1: "Lipschitz continuity wrt. parameter remains unchanged") | `z̲ = ess inf u†`; probe OK ≥ 2.42 | same as #4 |
| 6 | **Arrhenius exp** (repo; U=H¹₀ 4π-periodic box, H=L²) | `−Δz + q e^{z}` | `p e^{z}` | `s e^{z}` | machine (2.8e-15) | 0 | **0** exactly | `e^{z̲}` (analytic witness `c_coe()`; measured ≥ 0.748) | `−Δ` (const); `L = 0 ⇒ M̃ = M`; measured λ₁(C) = 0.125 margin |
| 7 | **Arrhenius linear** (repo) | `−Δz + q (1+z)` | `p (1+z)` | `s (1+z)` | machine (6.8e-15) | 0 | **0** exactly | `1 + z̲` (witness ≥ 0.706) | `−Δ` (const); as #6 |
| 8 | **Helmholtz, q = k² ∈ ℝ** (rework, §6) | `−Δz + q·r(z)` (sign matched to datum; `r(z) = −z` for `u_t = Δu + q u + g`, q = k²) | `p·r(z)` | `s·r(z)` (identified with ℝ via ∫) | scalar pairing `p s ∫r(z)dx` both sides: machine by construction | 0 | **0** (affine) | `(1/\|Ω\|) ∫ r(z) dx` — **volume average**, sign-matched (§6.3: weaker than ess-inf) | `c·(−Δ)` or `c·I_U`; `L = 0 ⇒ M̃ = M` designable |

Row notes.
*1–3:* the `z−z²` and `z ∂_x z` nonlinearities never touch (5)/(A2): both
cancel in `f(q,z) − f(q†,z)` (they are `q`-free). This is the paper's Remark
2.1 ("the constructed MRAS is affine linear in u although the model is
nonlinear in u") and the reason Fisher "shares the Darcy situation".
*4–5:* the derivative of `q|q|^{2/3}` is `(5/3)|q̃|^{2/3}` (Gateaux point
`q̃` from the mean-value step; the paper keeps `q̃` symbolic in `L`).
*6–7:* the multiplier `r(z)` (exp/linear/any bounded law) only enters the
adjoint pair symmetrically — the pair is a *self-adjoint multiplication* by
`r(z)`, so the adjoint is the same kernel, and the 100-draw check validates
the implementation, not a transcription guess.
*8:* the current repo "helmholtz" example is *not* this example (its truth
model contains no `q`: `q_true = 0`, quadratic self-step; all checks vacuous).
The rework is designed in §6.

---

## 3. What the paper proves, example by example (quotations)

* **Darcy, Prop. 4.1 (p. 10–11).** "as equation (16) is linear in the
  parameter a, the MRAS (4) has σ = 0"; `f_a(ã,z)h = −∇·(h∇z)`,
  `⟨f_a(ã,z)h, u−z⟩ = ∫ h ∇z·∇(u−z)`; "the Lipschitz constant in Assumption
  (A1) is L_{ã,z} = 0"; `C(‖a‖_H)v := −Δv ⇒ ⟨Cv,v⟩ = ‖v‖²_U`, `M = 1`,
  `Ñ = 1` with `‖u‖_U := ‖∇u‖_{L²}`. **The proposition states only "(A1), (A3)
  hold" — (A2) is not claimed**; Remark 4.2: coercivity "will require to
  perturb the underlying equation by adding u∇a … However, numerical results
  empirically prove good convergence even without this modification given
  small initial data error", with persistent excitation as the alternative.
* **Fisher, Prop. 5.1 (p. 13).** "As equation (21) is linear in a, the
  Lipschitz constant in Assumption (A1) remains L_{ã,z} = 0. Similarly as
  for the Darcy equation, Assumption (A3) holds with … C(‖a‖_H) := −Δ" —
  again (A1)+(A3) only, silent on (A2), same Remark-4.2 situation.
* **Potential, Prop. 6.1 (p. 18–19).** The paper's fully rigorous template.
  σ = 1; `f_c(c̃,z)h = h z + (5/3)|c̃|^{2/3} h z`, hence
  `f_c(c̃,z)*(u−z) = z (1 + (5/3)|c̃|^{2/3})(u−z)` (self-adjoint
  multiplication). **(A2):** via monotonicity of `ψ(c) = c|c|^{2/3}`,
  `⟨f(c,z) − f(c†,z), c−c†⟩ ≥ z̲‖c−c†‖²_{L²}` with `z̲ = ess inf u†` from the
  positivity assumption (27) on the true state. **(A1):** the worked
  estimate — mean value + Hölder with exponents `(2, 3, 6)` + Sobolev
  embedding + `(a+b)^{2/3} ≤ a^{2/3} + b^{2/3}`:
  `L_{c̃,z}(‖c‖) = (5/3) z̄ C_{H¹→L⁶} (‖c‖_{L²} + ‖c†‖_{L²} + ‖c̃‖_{L²})^{2/3}`,
  manifestly increasing. **(A3):** `C(‖c‖)v := (L²/(2 z̲) + M)(−Δ_Tr v)`,
  `M̃ = Ñ = L²/(2 z̲) + M`. Remark 6.2: the choice of `C` is not unique.
* **Allen–Cahn, Prop. 7.1 (p. 23).** "the analysis … largely agrees with
  Proposition 6.1 … one need only change cz to cz³ in f(c,z) whenever it
  appears, as this is a nonlinearity in the state. Also by this reason,
  Lipschitz continuity wrt. parameter remains unchanged, resulting in the
  same … operator C(‖c‖_H)."
  *Transcription note:* (39)/(41) print the tracking multiplier as
  `z(1 + (5/3)|c̃|^{2/3})(u−z)`, but `∂_c(c z³) = z³` forces
  `z³ + (5/3)|c̃|^{2/3} z` (the table's row 5 is the corrected form). The repo
  (examples/allen_cahn) follows the printed factor `z_old(u−z)` with
  `c̃ = 0`, which is one cause of its measured scheme-level DEVIATION (§4.3).
  The repo's own monotonicity witness (`phi(q) = q z³ + 0.6 q|q|^{2/3} z`,
  `z³`-correct) is unaffected — its measured (A2) ≥ 2.42 stands.

---

## 4. Adjointness: three distinct statements, three distinct checks

The user-facing column "adjointness" conflates three claims; the pipeline
keeps them separate so a failure localizes.

### 4.1 Analytical pair self-adjointness (theory, per row)
`⟨f_q(q,z)p, s⟩ = ⟨p, f_q(q,z)* s⟩` *as mathematics*, i.e. that the third
column really is the Banach adjoint of the second. For every row this is a
one-line computation: Darcy-type rows use the definition of the distributional
divergence (`⟨−∇·(p∇z), s⟩ = ∫p∇z·∇s = ⟨p, ∇z·∇s⟩_{H, H*}`); potential/AC/
Arrhenius rows are multiplication operators with `L²`-symmetric kernels.

### 4.2 Implementation pre-verification (the 100-draw check — §5)
Both pair members are *implemented code paths* in a new example. The paper's
own verification style ("explicit model reference adaptive system
construction, verified assumptions") suggests the numerical analogue: draw
~100 independent quadruples `(q, z, p, s)` — `q` uniform on a plausible
parameter ball, `z` on or near the datum manifold produced by the example's
own data operator, `p, s` random coefficient fields — and verify
`|⟨f_q p, s⟩ − ⟨p, f_q* s⟩| / (|⟨f_q p, s⟩| + |⟨p, f_q* s⟩|) < 1e-10` per draw.
This runs **at checklist time, before any solving** (the constructor only
warns once when `coupling_adjoint` is missing), from declared hooks — it
validates the transcription `f_q ↔ f_q*` and nothing else. All eight rows
are expected machine precision: for
Darcy-type it compares the two code paths `∫p∇z·∇s` (left, U*-action paired
with s) and `∫p(∇z·∇s)` (right); the identity is real at the level of *the
assembler's quadrature of two written expressions*, which is exactly what a
new example can get wrong (e.g. writing the adjoint of `−∇·(p∇z)` as
`−∇·(z∇s)`).

### 4.3 Scheme-operator self-adjointness (post-run, exists today)
The transpose identity of the *full step operator* `N = Kᵀ` between the
parameter and state rows (`mras/adjoint.py`), evaluated at a run's frozen
linearisation point. Its ratios are the "machine / DEVIATION" figures in the
table: machine precision for all σ=0 rows (2.8e-15 … 8.6e-16), but 0.65
(potential) and 0.96 (AC) for the paper-literal σ=1 discretizations — the
tracking multiplier `z(u−z)` (with `(5/3)|c̃|^{2/3} ≡ 0` at `c̃ = 0`) is not
the discrete adjoint of the σ-bracket linearised at `q̃ = qⁿ`. This is a
*discretisation-scheme* property (the aligned implicit q-row was
tested-and-rejected on stability grounds); it says nothing about §4.1/§4.2.
Conflating the three is what made the old verdict table confusing.

---

## 5. The L / C pipeline: derive, or estimate on permission, never silently

*Status (implementation ledger in §7): R0, R2, the §4.2 pre-verification
and K1 are implemented in `mras/verify.py`; R1 (closed-form template),
R3 (prompt) and the K0 certificate remain designed-but-unbuilt.*

### 5.1 Ladder for L(‖q‖_H) — rung by rung, most rigorous first

* **R0 — structural zero (automatic).** The example declares
  `coupling_kernel` (affine-in-q form `A(z)q` + q-free terms) and `sigma = 0`.
  By observation 1 of §1, the (5)-remainder is *identically zero*: `L ≡ 0` is
  closed-form, exact, and the right constant is 0 — not "a small
  number". Automatic witness (runs without user input): a 100-draw
  *quadratic-remainder probe*
  `‖f(q,z) − f(q†,z) − f_q(q̃,z)(q−q†)‖_{U*} ≤ 1e-9 · ‖q−q†‖_H`
  evaluated from the declared forms (finite-difference `f_q` or declared pair
  hook). Probe failure with σ=0 declared ⇒ FAIL, because (5) is an identity
  then — R0 can *fail*, which makes it a real check, not a declaration.
  Covers rows 1, 2, 3, 6, 7, 8.

* **R1 — closed-form modulus, derived by a template (semi-automatic).** When
  the remainder is Nemytskii-separable, `R(q,z)(x) = ψ(q(x)) · m(z(x))`, with
  scalar `ψ ∈ C¹(ℝ)` and multiplier `m`, the paper's Prop 6.1 proof *is a
  recipe*, not a miracle:
  1. mean value + Hölder: remainder in `L¹ ≤ ‖ψ'‖-modulus_{L³}(q, q†, q̃) · ‖m(z)‖_{L^∞}`-type triple;
  2. Hölder `(2, 3, 6)` + `U ↪ L⁶` embedding moves the test-function norm
     into `‖·‖_U`, producing the factor `C_{U→L⁶}`;
  3. subadditivity `(a+b)^θ ≤ a^θ + b^θ` (θ ≤ 1) converts the pointwise
     modulus into `C · (‖q‖ + ‖q†‖ + ‖q̃‖)^θ`.
  The example declares `(ψ, m, θ, exponents)`; the framework emits
  `L(R) = κ (R + ‖q†‖_{L²} + ‖cst‖)^θ` and *proves monotonicity
  programmatically* (closed-form `L' ≥ 0` or structural: positive combination
  of increasing functions). The remaining gap is the *certified constant* κ —
  e.g. the embedding constant `C_{H¹→L⁶}(Ω)`. The paper leaves it symbolic;
  a pipeline has three admissible answers (§5.3): the user declares a value,
  the framework estimates it (power iteration maximising
  `‖v‖₆ / ‖v‖_U` on the discrete space — an evidence-based lower bound,
  flagged as an estimate), or the run refuses to proceed. For rows 4–5 this
  rung reproduces the paper's `L` up to κ.

* **R2 — numerical modulus (automatic, on permission).** No declared
  structure ⇒ estimate the definition (5) directly. Protocol:
  - grid of radii `R ∈ {R₀, 2R₀, …, R_max}` (`R₀` from the initial guess's
    H-norm, `R_max` from the data box's plausible parameter range);
  - per `R`, draw ~100 `(q, q†, q̃, z)` in the `H`-ball of radius `R` (q) and
    the datum manifold (z); evaluate
    `ρ = ‖f(q,z) − f(q†,z) − f_q(q̃,z)(q−q†)‖_{U*} / ‖q−q†‖_H`
    with the dual norm solved exactly: `‖F‖_{U*}² = Fᵀ K_U⁻¹ F`
    (`K_U` = U-inner-product matrix, CG solve — cost is a few linear solves
    per draw);
  - ascent step from the best draws (coordinate perturbations, keep max);
  - fit an increasing closed form `L̂(R) = a R^θ + b` to the sampled maxima
    (log-log least squares for the exponent);
  - report status **ESTIMATED** (new status) with the full protocol record:
    seed, draws, radii, best values, fitted `(a, θ, b)`.
  Honesty rule: a sampled supremum is a *lower bound* of the true modulus;
  ESTIMATED output may never be presented as a verification of (A1), only as
  evidence plus a working constant. (This is the same evidence-vs-proof line
  that separates UNVERIFIABLE from OK today.)

* **R3 — NEEDS_INPUT (fail-closed prompt).** If neither R0 nor R1 applies and
  estimation was not permitted, `run_checks` emits a new status
  **NEEDS_INPUT** — not UNVERIFIABLE, which currently reads "cannot say" —
  with an actionable prompt:

  ```
  (A1) needs one of:
    [declare]  problem.lipschitz = lambda R: ...        # closed form, e.g. paper Prop 6.1 form
    [derive]   problem.remainder = (psi, m, theta, p, q) # Nemytskii template → framework derives L(‖q‖)
    [estimate] problem.allow_estimated_lipschitz = True  # R2 protocol, budget: draws, radii
  ```

  The framework then *executes* the chosen branch and re-runs. The doctrine
  is absolute: no silent default constant enters the (A3) chain, because
  `M̃ = L²/(2 C_coe) + M` propagates `L` into the state row's diffusion, i.e.
  a guessed `L` is an unstated change of the algorithm.

### 5.2 Ladder for C(‖q‖_H) — mostly free thanks to design freedom

Because `C` is the algorithm's own choice, the right question is not "is the
given C coercive?" but "does the code assemble a C for which (6)–(7) hold by
construction?"

* **K0 — tautological construction certificate (automatic).** If the state
  row assembles `c(‖q‖) · B` where `B` is *the Riesz map of the declared
  U-norm* (`∫∇u∇v` for `H¹₀`-with-gradient-norm; `∫∇u∇v + ∫∂Ω Tr u Tr v` for
  the paper's `H¹`-norm on rows 4–5), then
  `M̃(R) = Ñ(R) = c(R)` exactly — verify by a form-equality probe (one
  symmetric positive test vector + boundary-term presence check + symmetry
  `B = Bᵀ` at quadrature level) instead of today's λ₁ probe, which
  mechanically reports UNVERIFIABLE for norm-dependent `C` (the
  potential/AC "C is norm-dependent" line — an artifact of the probe, not a
  gap in the theory). Covers rows 1–7 as implemented (`C_scale = 1`,
  `C = c(‖q‖)·(−Δ)`), and the Helmholtz rework.
* **K1 — eigenvalue bound.** If `B` is only *a* fixed SPD form, `M̃(R) =
  c(R)·λ_min(B)` and `Ñ(R) = c(R)·λ_max(B)` — machinery exists (`_lambda1`,
  CG inverse iteration; Arrhenius λ₁ = 0.125). This is the rung that produced
  the "margin 0.125 available for M" figures.
* **K2 — constructive default.** Given any admissible pair `(L̂, Ĉ_coe)` from
  §5.1/§5.4, *define* `C(R) := (L̂(R)²/(2 Ĉ_coe) + M)·B` (the paper's own
  ansatz, Remark 6.2 "not unique") — then (A3) holds by K0 for every rung
  combination, including estimated inputs (with the honesty flag inherited).
  The pipeline therefore never *needs* a (A3)-verification; it needs the
  (A1)+(A2) inputs and the construction certificate.

### 5.3 Ladder for the auxiliary constants (κ, C_{U→L⁶}, M)
Three options only, chosen by the same prompt as §5.1-R3: declare (a number
with provenance), estimate (protocol + ESTIMATED flag), or refuse (run does
not proceed on a silent constant). `M` is free (`M > 0` arbitrary) — default
`M = 1` with the margin reported (rows 1–3, 6–7 show the "available margin"
wording).

### 5.4 Ladder for C_coe (A2) — unchanged, plus the remark context
Analytic witness hook `c_coe()` (Arrhenius `ess inf r(z)` over the data box —
rows 6–7 today, volume-average variant for row 8, §6.3; potential/AC `z̲ =
ess inf u†`); else localized probe pairs (existing); else UNVERIFIABLE.
**Interpretation upgrade:** a FAIL on Darcy-type rows is *expected from the
paper* (Remark 4.2) and should render as
`FAIL (expected, Remark 4.2): (A2) unclaimed by the paper; measured
worst-case pairing −2.36 < 0 → convergence, if any, runs on persistent
excitation, not coercivity` — with the persistent-excitation check as the
accompanying (conditional) justification, rather than an unexplained red row.

---

## 6. The Helmholtz rework: recovering q = k² as a scalar

### 6.1 What is wrong today
`examples/helmholtz_in_parameter.py` identifies nothing: its truth model is
`u_t = L Δu + g` with no parameter (`q_true = 0`), the q-update degenerates to
the quadratic self-step, and the audit correctly reports everything vacuous
(adjointness ratio 1.0 with "parameter absent from the truth model").

### 6.2 The framework piece: Q = ℝ via NGSolve's scalar FE space
Probed in this build (ngsolve 6.2.2608): `NumberSpace(mesh)` exists
(top-level `ngsolve`), has `ndof = 1`, embeds in a product space
`FESpace([H1(...), NumberSpace(mesh)])`, and **every coupling term assembles
separately**: `q*q_*dx`, `q*u_*dx`, `u*q_*dx`, `grad(u)*grad(u_)*dx` — the
joint space behaves as a mixed space with one scalar unknown, the scalar dof
is set/get-able (`gf.components[1].vec[0]`) and supports CF algebra
(`sqrt(q*q)**(2/3)` etc.). (Assembly of chained integral expressions
`a += x*dx - y*dx + z*dx` hit an operator-algebra quirk — terms must be added
one `+=` per integral; harmless.) The one convention to pin at
implementation time: whether `q*q_*dx` yields `|Ω| q q̂` or `q q̂` — settle
with a one-line assertion; the scheme must use the mass matrix it gets, and
`H = ℝ` is then `|·|` up to that fixed factor. So the framework work is: let
`Problem` accept a `Qfes` with `ndof = 1` (no mesh-based assumption on q) and
give `error_q` an `H`-norm for the scalar case (the scalar absolute value —
relative error `|q − q†|/|q†|` on `k²`, well-defined since `k² > 0`).

### 6.3 Theory, scalars first (why excellent recovery is expected)
With `Q = H = ℝ` (constants embedded in `L²(Ω)`), `f(q,z) = −Δz + q·r(z)`:

* σ = 0, `f_q(q,z)p = p·r(z)` (constant multiplier), `f_q(q,z)* s = s · r(z)`
  as `U → ℝ` (i.e. `s ↦ s ∫ r(z) dx` after identifying `ℝ* ≅ ℝ`). Pairing
  check row of the table #8: both sides equal `p s ∫r(z)dx` — machine
  precision for any `(q, z, p, s)` draws; the adjoint is independent of `q`
  altogether.
* (A1): affine ⇒ L ≡ 0 (R0, with the identity probe).
* **(A2) with the scalar miracle:** `q − q†` factors out of the *function*:
  `⟨f(q,z) − f(q†,z), q − q†⟩ = (q − q†)² ∫_Ω r(z) dx`. Hence
  `C_coe = (1/|Ω|) ∫ r(z) dx` — a **volume average**, not an essential
  infimum. Whenever `r > 0` (Arrhenius-type `r = e^z`, or `r = z` with
  one-signed datum) monotonicity needs only the datum's *mean* to keep the
  sign convention; the Dirichlet-datum ess-inf = 0 obstruction that kills the
  field-parameter version evaporates. Sign convention of `+q·r(z)` vs the
  datum must match the mean's sign — the example will *assert*
  `C_coe > 0` at construction.
* (A3): `L = 0` ⇒ any `C = c·B` works with `M̃ = c` (K0); the state row is a
  linear heat solve plus a bounded zero-order coupling.
* Expectation "excellent q recovery": this is finite-dimensional identification
  of one scalar driven by the scalar ODE `D_t q = (lift of) ∫ r(z)(u−z)`.
  Once `(q − q†)`-excitation `∫ r(z) (u−z) dt` is persistent (a scalar
  version of the existing persistent-excitation check), the descent is
  geometric with rate governed by `C′ = min{C_coe, 2M̃/C_{U→H}}` — and with a
  well-excited scalar there is no representation floor (no P0 field to
  resolve): relative error on `k²` should fall to the noise/discretization
  floor, i.e. *excellent* (< 5%) and plausibly below the < 1% true target.
  Deliverable when implemented: `examples/helmholtz_q2.py` + table row +
  recovery figure with the standard seeded protocol.

---

## 7. Status of the machinery vs the pipeline

| rung / piece | paper | repo today | gap |
|---|---|---|---|
| σ switch | forced by q-linearity | `sigma` attribute; `check_a1_sigma` verifies kernel/pair consistency | — |
| L = 0 (R0) | implicit | `_probe_affine` chord-gap probe (8 draws, < 1e-9 ⇒ affine ⇒ `L = 0` exact); declared `L = 0` upgraded to analytic OK | — |
| L closed form (R1) | rows 4–5 done by hand | manual | Nemytskii template + monotonicity-by-construction + κ provenance |
| L estimated (R2) | — | `_estimate_lipschitz`: radius × draw grid, K-dual norm ⇒ ESTIMATED + construction-time-independent `UserWarning`; the estimate never feeds (A3) | — |
| prompt (R3) | — | UNVERIFIABLE (reads "cannot say") | new NEEDS_INPUT status + three-branch prompt |
| (A2) witness | rows 4–5 `z̲`; 6–7 `ess inf r(z̲)` | `c_coe()` analytic hook + localised patch probe with U_h-view relaxation; FAIL wording carries the Remark-4.2 context | volume-average hook (row 8, waits on the Helmholtz rework) |
| (A3) certificate (K0) | tautological C | K1 done: `_lambda1` CG-preconditioned inverse iteration (local Riesz) on `C = C_scale*(-Delta)`; norm-dependent C ⇒ honest UNVERIFIABLE | form-equality certificate (K0); constructive default (K2) |
| adjointness pre-verification | "verified assumptions" style | `check_pairing`: 100-draw declared-pair precheck against the numerical q-derivative of `coupling`, before any solving; construction warns once when `coupling_adjoint` is missing | — |
| scalar parameter space | — | mesh-based `Qfes` assumed | `NumberSpace(mesh)` probed working; scheme + error_q adaptation |

ESTIMATED (measured, extrapolated) is live in the ladder, strictly ordered
`OK (proved/witnessed) > DEVIATION (measured, explained) >
ESTIMATED (measured, extrapolated) > UNVERIFIABLE (silent inputs insufficient)
> FAIL (contradicted)`; NEEDS_INPUT remains proposed with R3 (three-branch
prompt), its prerequisite.
