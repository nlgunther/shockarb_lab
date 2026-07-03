# Channel Knowledge Transfer — Information & Coding Theory for Regime Detection

*A conceptual briefing on treating regime detection as a decoding problem, and on the information-theoretic structure already latent in the ShockArb / factor_lab machinery. Reflection captured 2026-06-18 (Opus session). Companion to `docs/SUBSPACE_DISLOCATION.md` and `docs/Opus_KT.md`.*

> This is a framing and direction document, not a spec. It records why an information-theoretic lens is the right one for the multi-regime problem, what it unifies, and the one experiment that follows most directly. Term definitions appear at first use.

---

## 1. The lineage, stated precisely

James Simons came to finance from cryptanalysis at the Institute for Defense Analyses, not from algebraic coding theory in the textbook sense. The concrete bridge from that world to Renaissance is **Leonard Baum**, his IDA colleague and co-author of the Baum–Welch algorithm — the expectation–maximisation procedure for fitting a **hidden Markov model** (HMM: a Markov chain of unobserved states emitting noisy observations). Cryptanalysis is the problem of inferring a hidden sequence (plaintext, key) from an encoded, noisy stream. A market regime is a hidden state; the return series is the channel output. Regime detection is therefore not analogous to decoding — under the HMM formalism it *is* decoding, and the Viterbi algorithm (from convolutional-code theory) recovers the maximum-likelihood regime path through the state trellis.

The transferable idea is a posture: treat price as a received signal to be demodulated, not a number to be forecast.

---

## 2. The central isomorphism

A regime is a hidden state; observed returns are that state passed through a noisy channel. Two consequences for this codebase:

- **ShockArb is interference cancellation.** The macro factors are a strong, known carrier; the idiosyncratic dislocation is the weak modulation of interest. Projecting returns onto the orthogonal complement of the factor subspace (`engine.py`) is exactly successive interference cancellation in a multiuser detector: subtract the dominant known signal to expose the buried one. The residual's quality is therefore governed by how cleanly the carrier subspace is estimated — which is the dispersion-bias problem (see §3).

- **Regime selection is nearest-codeword decoding.** The 7-regime registry is a codebook. Choosing `ukraine_shock` vs `iran_shock` for a given cross-section is a minimum-distortion assignment to a codebook entry; the principal-angle geometry of `SUBSPACE_DISLOCATION.md` is the distortion metric on that codebook.

---

## 3. The unification with the factor_lab theory

The information-theoretic reading is not decoration; it is the same mathematics. Three identifications:

**The irreducible floor is a channel bound.** The floor `δ²/(n λ̂_j + δ²) = 1/(1 + SNR_j)` (`SNR_j = n λ̂_j/δ²`) is exactly the normalised minimum mean-square error (MMSE) of estimating a signal of variance `λ̂_j` in Gaussian noise — the Bayesian/Wiener MMSE — and the complement of the Gaussian channel capacity `½ log(1 + SNR)`. The floor is not merely an estimation artifact; it is the information limit of the channel.

**Corollary 4 is a sufficiency statement.** That the Grassmannian distance carries only the floor while the per-factor estimate also pays the in-subspace rotation term means: the **subspace is the sufficient statistic** for the factor signal, and the individual eigenvectors are a redundant, lossy re-encoding whose rotation is wasted Fisher information. Subspace estimation achieves the channel-limited bound; frame estimation spends information decoding a basis orientation that carries no signal. This is the information-theoretic content of the design decision in `SUBSPACE_DISLOCATION.md` to build the signal from `Π_H = H Hᵀ` rather than from the `h_j`.

**The factor model is a source code.** `k` factors are a rate-`k` code for the cross-section; PCA is the Karhunen–Loève transform, the rate–distortion-optimal linear code for a Gaussian source. The dispersion bias is then a decoding error within that code, and the James–Stein correction `Π̂_B^{JS} = H D_ψ⁻¹ Hᵀ` is its corrector.

---

## 4. What this makes actionable

Three concrete imports for the multi-regime problem, in increasing order of value.

**4.1 Mutual information generalises `r²`.** `r²` measures only linear, second-moment dependence between a stock and the macro subspace. The mutual information `I(return ; subspace projection)` measures all dependence. A regime that carries information about forward returns nonlinearly will show elevated MI with flat `r²` — a model-free detector of a live regime to which the linear lens is blind. Estimable with a k-NN or binned MI estimator on the existing score history; treat as a diagnostic, not a gate, given small samples.

**4.2 Regime switching is quickest-change detection.** "Has the world shifted from ukraine-type to iran-type?" is a sequential hypothesis test, not a daily re-classification. CUSUM and the Shiryaev–Roberts procedure minimise detection delay for a fixed false-alarm rate by accumulating the log-likelihood ratio (LLR) of returns under the two regime models. This replaces the ad-hoc "is the conflict active?" judgment with a principled stopping rule. Wald's SPRT is the fixed-hypothesis ancestor; the change-detection versions are the right form here because the switch time is unknown.

**4.3 Detectability lives in the symmetric-difference directions.** The speed at which the two regimes can be told apart is governed by the Kullback–Leibler divergence between the two regime hypotheses (expected detection delay ≈ `log(1/false-alarm) / KL`). For Gaussian factor models that divergence is essentially a function of the principal angles already computed: the two near-shared directions (angles 15°, 26°; cosines 0.97, 0.90) are common-mode and carry almost no regime-discriminating information, like a DC offset on the channel; the regime-specific direction (angle 57°, cosine 0.55) carries nearly all of it. **Run any regime detector on returns projected onto the symmetric-difference block; ignore the shared directions.** This is the same conclusion the subspace geometry reached, arrived at from the channel side — a useful cross-check that the two framings agree.

---

## 5. The counterweight (where the analogy misleads)

The cryptanalytic analogy must not be over-read, and the following caveats are load-bearing.

- **Markets are non-stationary and reflexive; ciphers are not.** A key does not change because you began decoding it; a tradable regime rotates the moment it is decoded and acted on. Classical information theory assumes stationarity or *known* change dynamics. This is where edges decay, and no transform fixes it.
- **The data-processing inequality is the governor.** No transformation manufactures information the returns do not contain. A ~35-day window holds a fixed, small amount of Fisher information about the regime. The Grassmannian/JSE work is valuable because it stops *discarding* information (the rotation term) and names the floor that cannot be beaten — not because it creates signal past that floor.
- **The horizon mismatch with Renaissance is real.** Medallion's edge lived at very short horizons, where the channel is closest to stationary and a law of large numbers over millions of "transmissions" does the work. ShockArb is event-driven and multi-day, with a *handful* of independent shock events in its entire history. A full hidden-Markov rebuild would be fitting a multi-state chain on data that cannot identify it. The tools that transfer cleanly to this regime are the ones that respect small, non-stationary samples — sequential change detection (§4.2) and MI diagnostics (§4.1) — not a wholesale HMM.
- **The inherited edge is discipline, not a theorem.** The durable IDA legacy is the posture: treat everything as decoding under noise, demand out-of-sample proof (leave-one-regime-out here), and operate where stationarity holds best. Applied at the horizon ShockArb actually trades.

---

## 6. The experiment that follows most directly

A **two-regime CUSUM on the log-likelihood ratio**, with returns projected onto the symmetric-difference direction (the 57° block). It turns the static "which lens" attribution of `SUBSPACE_DISLOCATION.md` into a *dated* regime-switch signal with a controllable false-alarm rate, and it is built almost entirely from machinery already on disk: the two frozen models supply the competing hypotheses; `dislocation_geom.py` supplies the projection; `score_history.py` supplies the return stream. Scope it as a diagnostic first (plot the CUSUM statistic against known event dates), validate the detection delay against the principal-angle KL estimate, and only then consider letting it drive regime selection.

---

## 7. Pointers

- `docs/SUBSPACE_DISLOCATION.md` — the subspace combine/select design; supplies the projection and the principal angles referenced in §4.3.
- `docs/Opus_KT.md` — why the geometric approach was chosen over BMA / Kalman / XGBoost; design invariants and open items.
- `factor_lab/docs/KT.md` — Corollaries 3 and 4, the floor, and the James–Stein correction (§3 here reinterprets them).
- Genealogy and methods: Baum–Welch (HMM EM), Viterbi (trellis decoding), Wald SPRT, Page CUSUM, Shiryaev–Roberts (quickest change detection), Karhunen–Loève / rate–distortion (PCA as optimal linear code).

---

*End of channel KT.*
