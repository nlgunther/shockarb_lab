# Opus Knowledge Transfer — Multi-Regime Signal Combination

*Captures the conceptual work of the 2026-06-18 Opus session on combining the `ukraine_shock` and `iran_shock` factor models. Records decisions and reasoning not derivable from the code. The full design lives in `docs/SUBSPACE_DISLOCATION.md`; this file is the briefing that explains why that design was chosen and what is still open.*

> Last updated: 2026-06-18 | Author: Opus session | Companion: `docs/SUBSPACE_DISLOCATION.md`, `factor_lab/docs/KT.md`

---

## 1. The problem addressed

Two regimes now score the same tickers (`ukraine_shock`, `iran_shock`), each producing its own `r²` and `confidence_delta`. The question was which ML approach best **combines** the two signals or **selects** the right regime per ticker — specifically whether XGBoost on metadata, Bayesian model averaging, or Kalman blending was appropriate. This resolves the KT.md note "ML/XGBoost regime-selection analysis pending (Opus-level task)."

The conclusion is that the right tool is not on that list. The problem is **geometric**, not a supervised-learning problem, and it is solved by working with the factor **subspace** (a point on the Grassmannian) rather than the individual factors. The decisive reason is a result already proved in `factor_lab`.

---

## 2. Decision and rationale

**Decision: combine and select via the two factor subspaces, using projection operators only.** Design and an exact `k = 3` worked example are in `docs/SUBSPACE_DISLOCATION.md`.

The reasoning chain, in order:

1. **The binding constraint is labels, not algorithm.** Seven regimes × ~35-day windows, and within a single shock all tickers move together, so the effective sample size is a handful of independent events, not the row count. Any supervised method overfits, and validation must be leave-one-regime-out, never random k-fold.

2. **ShockArb's signal is already a projection-operator quantity.** Dislocation is `r − Π_H r`, fit is `r²_H = |Π_H r|²/|r|²`, where `Π_H = H Hᵀ` projects onto the factor subspace `col(H)`. `Π_H` is invariant to rotation or sign flip of the basis within the subspace (`(HR)(HR)ᵀ = H Hᵀ` for `R ∈ O(k)`). So the signal is a coordinate-free Grassmannian object; the eigenvector rotation/flip that occurs on refit never touches it. This dissolves the basis-anchoring problem that an earlier turn worried about.

3. **The subspace is the only object the short window pins down.** From `factor_lab`, the per-factor (frame) error carries both an irreducible floor and an in-subspace rotation term, but the Grassmannian distance keeps only the floors — the rotation cancels (Corollary 4). At `n ≈ 35` the individual factors are badly rotated (see `factor_lab/rotation_check.py`: substantial `sin²∠` even at `n = 50`), while the subspace they span is well determined.

4. **This reframes combine-vs-select with no training data.** Combine = residual orthogonal to the union `S_u + S_i` (the continuous form of "act only where both agree"). Select = attribute `r²_u − r²_i` to the symmetric-difference directions, which are economically named (iran-only ≈ energy / Strait of Hormuz); the attribution says which lens fits. Both are closed-form and basis-invariant.

5. **One correction is owed to `r²` itself.** Raw `r²` is dispersion-biased downward (`|Π_H z|² < |Π_B z|²`, Corollary 3), worst at the `r² > 0.50` gate where factors are weak. Apply the James–Stein projection `Π̂_B^{JS} = H D_ψ⁻¹ Hᵀ` before gating; it inflates weak-factor coordinates most and shrinks the overstated dislocation.

---

## 3. Alternatives considered and why they were demoted

| Approach | Verdict | Why |
|---|---|---|
| **XGBoost on metadata** | Demoted to optional thin selector | Only ever intended for model/regime *selection*, not signal generation (Ken clarified). Even so, with this sample size it will mostly rediscover the sector rule already in KT.md and overfit the rest. If used, it must emit a *weight* (collapsing into the stacking framework), use depth-2 / monotone constraints / few features, and be validated leave-one-regime-out against the agreement-rule baseline. The geometry in §2 does the selection in closed form, so XGBoost is no longer on the critical path. |
| **Bayesian model averaging** | Superseded for *this* problem | The principled framing for combine-vs-select, and the closest of the three to right. But textbook BMA assumes one model is true (M-closed); neither regime is. Pseudo-BMA / stacking weights would have been the fallback. The subspace formulation subsumes it: the union projection *is* the combination, with no weights to estimate. |
| **Kalman filter (two-stream blend)** | Rejected | The latent-mispricing state is definable (OU/AR1, matching the reversion thesis) but its key parameter — the mean-reversion rate — is the very thing being traded and cannot be estimated from 35 post-shock days; it would have to be pooled, at which point it reduces to static precision-weighting plus a smoother. Not worth the machinery. |
| **Kalman filter (time-varying loadings)** | Held, not needed yet | Tracking `β_t` as a random walk is a genuine upgrade for window staleness, but it tracks the *frame*, which needs basis anchoring. The subspace formulation makes anchoring unnecessary; if temporal tracking is wanted later, do it as subspace tracking (a curve on the Grassmannian / principal-angle smoothing), not `β_t` tracking. |

The general lesson: the eigenvector instability that motivated the ML machinery is an artifact of choosing a frame representation. Once the signal is written as a projection, the instability disappears and most of the ML question dissolves.

---

## 4. The bridge to factor_lab theory

Three results from `factor_lab/docs/KT.md` are load-bearing here, and this is the non-obvious intellectual content of the session:

- **Corollary 4 (Grassmannian distance).** `d²_Gr(col(H), 𝓑) = Σ_j δ²/(n λ̂_j + δ²)` — only the floors; the in-subspace rotation cancels. Justifies building the signal from `Π_H`, not the `h_j`.
- **Corollary 3 (dispersion bias).** `|Π_B z|² − |Π_H z|² → Σ_j (1 − ψ_j²) c_j² > 0`. ShockArb overstates dislocation; the overstatement is largest at the weak-factor gate.
- **James–Stein correction.** `Π̂_B^{JS} = H D_ψ⁻¹ Hᵀ` with `ψ̂_j = √max(0, 1 − δ̂² p / s_j²)` restores the norm; weaker factors get larger inflation.

A further consequence worth keeping: the residual floor is the per-regime noise floor on the smallest trustworthy dislocation, and the dual-regime construction reduces it on the *shared* (broad-market) direction because that direction is estimated from two independent windows (2022 and 2026). This is the statistical — not merely conviction-based — reason the union-orthogonal signal beats either single-regime residual.

---

## 5. Design invariants (must hold in any implementation)

- **Rotation invariance is the contract.** `‖Π(H) − Π(HR)‖ ≈ 0` for any `R ∈ O(k)` and any sign flip. A unit test asserting this is the formal guarantee that the signal is basis-free. (Verified numerically this session at `< 4×10⁻¹⁷`.)
- **Pure module.** `shockarb/dislocation_geom.py` takes `H_u, H_i, r` (and optional `ψ`) in and returns scores out, with zero I/O — mirroring `engine.py`. `pipeline.py` loads the JSONs and feeds it.
- **Re-orthonormalise on load.** Whatever the basis field in `model.json` turns out to be, apply `H, _ = np.linalg.qr(stored)` so the projector identity holds regardless of serialization.

---

## 6. Open items and risks (verify before / during coding)

1. **Shared ambient space — the one assumption that silently corrupts everything.** The union `S_u + S_i` and the principal angles are only defined if both regimes decompose returns in the *same* space. `ukraine_shock` and `iran_shock` have different ETF baskets (10 vs 19 ETFs per KT.md). Confirm in `pipeline.py` what space `engine.py` projects in; if the bases do not share dimensions/coordinates, a reconciliation step is a real design decision, not coding — stop and escalate rather than forcing it.
2. **`model.json` basis field unknown.** Read the live serializer in `engine.py` / `pipeline.py` to find what is actually stored (loadings vs `Vt` vs components); do not assume.
3. **JSE estimation is noisy on 35 days.** Treat `Π̂_B^{JS}` re-ranking as an aid, not a point estimate; cap corrected `r²` (can exceed 1 under over-correction); monitor.
4. **Validation.** Leave-one-regime-out only, against both single-regime residuals and the current tier-agreement rule, using `run_backtest.py` / `score_history.py`.

---

## 7. Hand-off state

Design and worked example complete (`docs/SUBSPACE_DISLOCATION.md`). Implementation is **delegated to Sonnet**, scoped in stages:

- **Stage 1 (next):** build `dislocation_geom.py` with `factor_basis`, `projector`, `principal_angles`, and the rotation-invariance unit test. Run the principal-angle diagnostic on the two saved models. Sonnet must stop and flag if the two bases do not share an ambient space (item 6.1).
- **Opus checkpoint:** review the diff and the angle numbers before committing to scoring columns. The angles may change the plan — if the subspaces are nearly coincident, selection barely matters and the design simplifies to combine + bias-correct only.
- **Stage 2 (after checkpoint):** `robust_dislocation`, JSE columns, wiring into `pipeline.py` and `report_compare.py`, then the leave-one-regime-out backtest.

Files created this session: `docs/SUBSPACE_DISLOCATION.md`, `docs/Opus_KT.md`. No code written yet. No change to `engine.py` or any frozen model.

---

*End of Opus KT.*
