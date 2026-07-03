# Subspace Dislocation — Operational Proposal

*Combining and selecting between the `ukraine_shock` and `iran_shock` factor models by working with the factor **subspace** (a point on the Grassmannian) rather than with individual factors.*

> Status: design proposal. No code written yet. Numbers in §5 are reproduced by `scripts/subspace_dislocation_demo.py` (to be added); they were computed and verified in numpy during drafting.

---

## 1. Objective

ShockArb scores a stock by decomposing its return into a macro-factor-explained part and a residual; a large positive residual is the mean-reversion candidate. With two regimes live at once (`ukraine_shock`, `iran_shock`), every ticker carries two such decompositions and the open question is how to *combine* them into one robust signal and how to *select* the regime that is the right lens for a given name.

This document proposes doing both through the geometry of the two factor **subspaces**, with no training data, and gives an exact `k = 3` worked example. The proposal rests on a result already proved in `factor_lab`: the subspace spanned by the estimated factors is recovered far more accurately than the individual factors are (Corollary 4, the Grassmannian distance carries only the irreducible floor; the in-subspace rotation cancels).

Term definitions appear at first use below.

---

## 2. Why the subspace, not the factors

Let a regime's frozen model supply an estimated factor basis `H ∈ ℝ^{d×k}` with orthonormal columns, where `d` is the dimension of the return space the decomposition lives in and `k` is the factor count (`k = 3` for both live regimes). Write `Π_H = H Hᵀ` for the orthogonal projection onto `col(H)`, the **factor subspace**.

ShockArb's signal is built entirely from `Π_H`:

- explained part of a return `r`: `Π_H r`;
- dislocation (residual): `r − Π_H r`;
- fit: `r²_H = |Π_H r|² / |r|²`.

The projection `Π_H` is **invariant** to any rotation or sign flip of the basis within the subspace: for any orthogonal `R ∈ O(k)`, replacing `H` by `HR` gives `(HR)(HR)ᵀ = H R Rᵀ Hᵀ = H Hᵀ = Π_H`. The dislocation, the residual, and `r²` are therefore unchanged by the eigenvector rotation that occurs when a model is refit on a new window. The signal is a coordinate-free (Grassmannian) object; the individual `h_j` are merely one representation of it, and not the representation ShockArb needs.

This matters quantitatively because the calibration window is short (`n ≈ 35` trading days). At that `n` the individual factor directions are substantially rotated away from their population targets, but the subspace they span is comparatively well determined. The `factor_lab` decomposition makes the split precise: the per-factor (frame) error is

    floor_j + rotation_j  =  δ²/(n λ̂_j + δ²)  +  (n λ̂_j)/(n λ̂_j + δ²) · sin²∠(ŵ_j, w_j),

whereas the subspace error keeps only the floors,

    d²_Gr(col(H), 𝓑)  =  Σ_j  δ²/(n λ̂_j + δ²).

The rotation term — which is non-zero whenever the factors are correlated (`G^∞_B ≠ I_k`) — cancels in the Grassmannian metric. Building the cross-regime signal from `Π_H` rather than from the `h_j` is thus not a stabilisation trick; it is using the only object the short window actually pins down.

---

## 3. The three operations

Let the two regimes supply orthonormal bases `H_u` (ukraine) and `H_i` (iran), with projections `Π_u`, `Π_i`. All three operations below are functions of these projections alone.

### 3.1 Combine — residual orthogonal to the union

Define the **union subspace** `S_u + S_i = col([H_u  H_i])` and its projection `Π_∪`. The **regime-robust dislocation** is the residual orthogonal to the union:

    robust_resid  =  r − Π_∪ r,      robust_r²  =  |Π_∪ r|² / |r|².

A dislocation survives only if *neither* macro lens explains it. Because `S_u + S_i` contains each individual subspace, `|Π_∪ r| ≥ max(|Π_u r|, |Π_i r|)`, so `robust_resid` is never larger than either single-regime residual. This is the continuous, geometric form of the existing "act only where both regimes agree" rule: agreement is replaced by a single conservative residual, and the binary tier match is replaced by a magnitude.

### 3.2 Select — attribute the disagreement to the symmetric-difference directions

The gap `r²_u − r²_i` is carried entirely by the directions one subspace contains and the other does not — the **symmetric difference** `(S_u ⊖ S_i)`, spanned by the ukraine-only and iran-only directions. Concretely, decompose the return space into

- **shared** directions `S_u ∩ S_i` (the broad-market factor both regimes carry),
- **ukraine-only** directions (in `S_u`, orthogonal to `S_i`),
- **iran-only** directions (in `S_i`, orthogonal to `S_u`),
- **neither** (orthogonal to the union).

The shared block explains the same energy under both regimes and cancels from `r²_u − r²_i`. What remains is

    r²_u − r²_i  =  ( |proj of r onto ukraine-only|² − |proj of r onto iran-only|² ) / |r|².

The selector reads off *which* block drives the disagreement and asks whether that block is economically named. The iran-only directions should be the energy / Strait-of-Hormuz factor that `ukraine_shock` lacks. If a stock looks dislocated under ukraine but its apparent dislocation lies in the iran-only block, the move is macro (energy), not idiosyncratic, and `iran_shock` is the correct lens. This is a closed-form attribution — no classifier, no labels — and, being projection-based, it is invariant across regimes and across refits.

The shared/specific split is obtained from the **principal angles** between `S_u` and `S_i`: the cosines are the singular values of `H_uᵀ H_i`. A singular value near 1 is a shared direction (angle ≈ 0); a singular value near 0 is a regime-specific direction (angle ≈ 90°). The count of near-1 singular values is the dimension of the (numerical) intersection.

### 3.3 Bias-correct — apply the James–Stein projection to r²

The raw fit `r²_H` is dispersion-biased *downward*: `|Π_H z|² < |Π_B z|²` (Corollary 3), where `Π_B` projects onto the true population subspace. ShockArb therefore **overstates** dislocation, and does so worst exactly at its operating point: the `r² > 0.50` gate sits in the weak-factor regime where the bias `Σ_j (1 − ψ_j²) c_j²` is largest. Here `ψ_j ∈ (0,1)` is the signal-to-noise cosine of factor `j`, `ψ̂_j = √max(0, 1 − δ̂² p / s_j²)` with `s_j` the `j`-th singular value of the calibration return matrix.

The James–Stein projection restores the norm:

    Π̂_B^{JS} = H D_ψ^{-1} Hᵀ,      D_ψ = diag(ψ̂_1, …, ψ̂_k),

so the corrected explained energy is `Σ_j (a_j / ψ̂_j)²` where `a = Hᵀ r` are the factor coordinates. Each coordinate is inflated by `1/ψ̂_j`, most strongly for the weakest factors. The corrected dislocation `|r|² − |Π̂_B^{JS} r|²` is the quantity to gate on; it is smaller than the raw residual, and re-ranking by it will push marginal `r² ≈ 0.5` names below the gate.

The residual floor `δ²/(n λ̂_j + δ²)` is the part the correction cannot remove — the per-regime noise floor on the smallest dislocation that can be trusted. The dual-regime construction reduces it on the *shared* block: the broad-market direction is estimated from two independent windows (2022 and 2026), so its floor can be pooled (effective `n` roughly doubled), while each regime-specific direction keeps its own floor. This is the statistical reason — beyond conviction — that the union-orthogonal signal is better determined than either single-regime residual.

---

## 4. Where it sits in the architecture

A new pure module, mirroring `engine.py`'s zero-I/O discipline:

```
shockarb/dislocation_geom.py     — pure linear algebra. Takes H_u, H_i, r (and optional ψ) in; returns scores out. No file or network I/O.
pipeline.py                      — loads the two frozen model.json files, extracts each factor basis, calls dislocation_geom, writes the augmented score columns.
report_compare.py                — gains the new columns as additional comparison fields.
```

Proposed API (`dislocation_geom.py`):

```python
def factor_basis(model) -> np.ndarray:            # d×k orthonormal H from a frozen model's stored factor directions
def projector(H) -> np.ndarray:                   # H Hᵀ
def principal_angles(H_u, H_i) -> np.ndarray:     # angles (radians); cosines = svals(H_uᵀ H_i)
def subspace_blocks(H_u, H_i, tol=1e-6) -> dict:  # orthonormal bases for shared / u_only / i_only / neither
def dislocation(r, H, psi=None) -> Dislocation:   # explained, residual, r²; JSE-corrected if psi given
def robust_dislocation(r, H_u, H_i) -> Robust:    # residual ⊥ (S_u+S_i); attribution of r²_u − r²_i to the blocks
def jse_cosines(singular_values, delta2, p) -> np.ndarray   # ψ̂_j
```

New columns on `live_alpha_*.csv`: `r2_robust` (union-orthogonal fit), `robust_resid` (regime-robust dislocation), `regime_pick` (which lens the attribution favours), `r2_u_jse` / `r2_i_jse` (bias-corrected fits).

**Implementation note (to confirm before coding):** the exact field that stores the factor basis in `model.json` must be read from the live serializer in `engine.py` / `pipeline.py` rather than assumed; `factor_basis()` should re-orthonormalise whatever is stored (`H, _ = np.linalg.qr(stored)`) so the projector identity holds regardless of how the basis was persisted.

---

## 5. Worked example (k = 3)

A deliberately transparent case in a `d = 6` return space, chosen so every quantity can be read by eye. Let `e₁,…,e₆` be an orthonormal basis with the interpretation:

| direction | meaning |
|---|---|
| `e₁` | broad market (shared by both regimes) |
| `e₂, e₃` | ukraine-only factors (tech/software risk-off) |
| `e₄, e₅` | iran-only factors (energy / Strait of Hormuz) |
| `e₆` | explained by neither regime |

The two `k = 3` regime subspaces are

    S_u = span{e₁, e₂, e₃},     S_i = span{e₁, e₄, e₅}.

Consider one stock whose return vector, in these coordinates, is

    r = (1.00, 0.20, 0.10, 0.70, 0.30, 0.15),     |r|² = 1.6525.

The components say: a large common market move (1.00), a small genuine tech-specific component (0.20, 0.10), a large energy/Strait component (0.70, 0.30), and a small idiosyncratic part explained by neither (0.15).

### 5.1 Single-regime decompositions

Projecting onto each subspace (squared norms add coordinate-wise because the construction is orthogonal):

| regime | explained `|Π r|²` | `r²` | residual `|r|² − |Π r|²` |
|---|---|---|---|
| ukraine | `1.00 + 0.04 + 0.01 = 1.05` | **0.635** | **0.6025** |
| iran | `1.00 + 0.49 + 0.09 = 1.58` | **0.956** | **0.0725** |

Read alone, the ukraine model flags a strong dislocation (`r² = 0.64`, large residual `0.60`); the iran model sees the same stock as almost fully explained (`r² = 0.96`). `Δr² = r²_u − r²_i = −0.321`.

### 5.2 Select — where does the disagreement live?

Principal angles between `S_u` and `S_i`: the cosines (singular values of `H_uᵀ H_i`) are `(1, 0, 0)`, i.e. angles `(0°, 90°, 90°)`. Exactly one shared direction (`e₁`); the other two pairs are orthogonal. So the disagreement must live in the symmetric-difference block. Indeed:

- ukraine-only energy `|⟨r, e₂,e₃⟩|² = 0.04 + 0.01 = 0.050` (3.0% of `|r|²`),
- iran-only energy `|⟨r, e₄,e₅⟩|² = 0.49 + 0.09 = 0.580` (35.1% of `|r|²`),

and `(0.050 − 0.580)/1.6525 = −0.321 = Δr²` exactly. The disagreement is driven by the iran-only (energy) block. The apparent ukraine dislocation is a macro energy move, not idiosyncratic mispricing — **the selector picks `iran_shock`** as the correct lens for this name.

### 5.3 Combine — the regime-robust dislocation

The union `S_u + S_i = span{e₁,…,e₅}` has dimension 5; its orthogonal complement is `span{e₆}`. Projecting:

    |Π_∪ r|² = 1.6525 − 0.0225 = 1.630,    robust_r² = 0.986,
    robust_resid² = 0.0225  (= the e₆ component, 0.15²).

Once *both* lenses are allowed, the trustworthy dislocation collapses to the genuine idiosyncratic part (`0.15`). A single-regime ukraine run would have acted on a residual of `0.60`; the dual-regime geometry shows 96% of that is macro and discards it. This is "act only where both agree," now a number rather than a tier vote.

### 5.4 Bias-correct — James–Stein on the ukraine fit

Suppose the ukraine model's three factors have signal-to-noise cosines `ψ = (0.97, 0.80, 0.55)` — strong, medium, weak. The factor coordinates are `a = H_uᵀ r = (1.00, 0.20, 0.10)`. The corrected explained energy is `Σ (a_j/ψ_j)²`:

| factor `j` | raw `a_j²` | corrected `(a_j/ψ_j)²` |
|---|---|---|
| 1 (strong) | 1.0000 | 1.0628 |
| 2 (medium) | 0.0400 | 0.0625 |
| 3 (weak) | 0.0100 | 0.0331 |
| **total** | **1.0500** | **1.1584** |

So the true ukraine explanation is `1.158`, not the raw `1.05`: corrected `r²_u = 0.701` (raw `0.635`), and the corrected dislocation is `1.6525 − 1.158 = 0.494`, versus the raw `0.6025`. The raw residual **overstates the dislocation by 0.108** — about 18% — and the overstatement is concentrated in the weak third factor, whose coordinate was inflated more than threefold (`1/0.55² ≈ 3.3×`). At a ticker sitting just above the `r² = 0.50` gate, a correction of this size is enough to move it across.

### 5.5 What the example demonstrates

One stock, four conclusions, all from projection operators: (i) the two regimes disagree sharply (`Δr² = −0.32`); (ii) the disagreement is fully attributable to the iran-only energy block, so iran is the lens; (iii) the regime-robust dislocation is tiny (`0.022`), so the name is *not* a genuine mispricing despite ukraine's large single-regime residual; (iv) even the ukraine residual itself is overstated by ~18% once the dispersion bias is corrected. The rotation-invariance claim of §2 was checked numerically: replacing `H_u` by `H_u R` for a random `R ∈ O(3)` changes the projector by `< 4×10⁻¹⁷`.

---

## 6. Validation plan

1. **Rotation-invariance unit test.** Assert `‖Π(H) − Π(HR)‖ < 1e-12` for random `R ∈ O(k)` and random sign flips. This is the formal guarantee that the signal is basis-free; it must hold exactly.
2. **Principal-angle diagnostic.** Compute the angles between the two live subspaces from the saved models. This quantifies how often selection even matters: if the subspaces nearly coincide, `Δr²` is structurally small and combine is trivial; large angles mean the energy block genuinely separates the regimes.
3. **Bias-correction effect at the gate.** Re-rank the live universe by JSE-corrected `r²` and report which names cross the `r² = 0.50` boundary in each direction. Confirm the movers are weak-factor names, as the theory predicts.
4. **Leave-one-regime-out backtest.** Using `run_backtest.py` / `score_history.py`, label realised mean-reversion P&L and test whether `robust_resid` (combine) and the attribution-based `regime_pick` (select) beat (a) each single-regime residual and (b) the current tier-agreement rule. Validation must be leave-one-regime-out, never random k-fold, because tickers within a shock are cross-sectionally correlated.

---

## 7. Caveats

- **`d` and the choice of return space.** The example uses an abstract `d = 6`. In the live system `d` is fixed by how `engine.py` represents returns for the projection; the module must take whatever that space is and the two bases expressed in it. Both regimes must share the same ambient space for the union and principal angles to be meaningful — confirm this in `pipeline.py` before wiring.
- **Oblique factors are where the Grassmannian advantage is largest.** The worked example is orthogonal for transparency, so the frame-vs-subspace gap is invisible in its numbers. The real models have correlated factors (`G^∞_B ≠ I_k`); that is exactly the regime in which the rotation term is large and the subspace formulation earns its keep (Corollary 4).
- **JSE is an asymptotic norm restoration.** Corrected `r²` can in principle exceed 1 under heavy over-correction; cap and monitor. Estimating `δ̂²` and `ψ̂_j` from a 35-day window is itself noisy — treat the correction as a re-ranking aid, not a precise point estimate.
- **Fixed-`F` determinism.** In this model `p → ∞` with `n, k` and `F` fixed, so `‖M_n − M‖` is a deterministic (unobserved) constant, not a random `O_P` quantity. The principal-angle and floor diagnostics are descriptive of the realised window, not sampling statements over hypothetical draws.

---

*End of proposal.*
