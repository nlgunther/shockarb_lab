# KT — ShockArb Regime Effectiveness & Signal Validation

> Last updated: 2026-05-30 | Trigger: manual | Staleness: Fresh

---

## 1. Project Overview

ShockArb is a quantitative trading system that scores stocks against a PCA-based macro factor
model calibrated on a historical geopolitical shock period (a "regime"). The core thesis: stocks
sold off by macro factors when fundamentals are intact will mean-revert.

This sub-project adds three capabilities to the existing system:

1. **`--save-recent` flag** — accumulates daily score output to a rolling parquet archive,
   enabling all subsequent analysis. Zero new data downloads required.

2. **Regime Health / SNR** — measures whether the active regime's factor structure still
   explains current return variance, and which regime (among all registered) is currently
   most explanatory. Runs as a daily printout appended to `shockarb score` output.

3. **Picks vs. Random validation** — honest alpha test: do ShockArb's ranked picks beat
   random selection from the same universe? Uses Spearman rank correlation as the
   primary statistic. No new data required beyond the `--save-recent` archive.

A fourth capability (Kalman adaptive blending across regimes) is explicitly deferred.

**Current status:** Active Development

---

## 2. Goals & Constraints

**Goals:**
- Accumulate daily score history with zero marginal data download cost
- Detect when the active regime's factor structure is degrading (SNR < threshold)
- Identify which registered regime is currently most explanatory
- Answer the foundational question: does ShockArb's ranking add alpha over random?
- Keep all new code consistent with existing ShockArb architecture and style

**Constraints:**
- Windows, Anaconda Prompt, conda environment "manager"
- Project root: `C:\Users\nlgun\personal\nlgcode\shockarb_lab`
- Must not break existing 281 passing tests
- No new data downloads for SNR or validation — use only what scoring already fetches
- Rolling archive retention: configurable, default 90 days; auto-purge on each run
- Score archive storage budget: trivial (~1KB/day × 90 days = <100KB)
- datamgr is a sibling package; zero imports from shockarb into datamgr

**Non-goals:**
- Kalman adaptive blending (deferred — build SNR first, validate it, then layer)
- Live forward-testing infrastructure (this is retrospective/diagnostic only)
- New data providers or ticker universe changes
- Portfolio construction changes (portfolio_sizer.py untouched)

---

## 3. Prototypes & Examples

### Score archive row schema

Each day appends one row per scored stock to `data/recent_scores/YYYY-MM-DD.parquet`:

```
date        | ticker | actual  | expected | delta  | r2   | conf_delta | regime          | model_file
2026-05-29  | TXN    | -0.0325 | -0.0012  | 0.0313 | 0.65 | 0.0202     | ukraine_shock   | ukraine_shock_us_20260528_143030.json
2026-05-29  | CPRT   | -0.0153 | +0.0003  | 0.0156 | 0.81 | 0.0127     | ukraine_shock   | ukraine_shock_us_20260528_143030.json
```

### Regime SNR definition

```
out_of_sample_r2(t) = 1 - Var(delta over window) / Var(actual over window)
SNR = out_of_sample_r2 / (1 - out_of_sample_r2)
```

Interpretation:
- SNR > 2.0 (R² > 0.67): regime is highly explanatory ✅
- SNR 1.0–2.0 (R² 0.50–0.67): moderate, regime drifting ⚠️
- SNR < 1.0 (R² < 0.50): regime explains less variance than noise ❌

### Regime competition output (appended to daily score)

```
REGIME HEALTH — 2026-05-30 (22-day window)
────────────────────────────────────────────────
  ukraine_shock         R²=0.41  SNR=0.70  ⚠️  DEGRADED
  gulf_war_recovery     R²=0.58  SNR=1.38  ✅  BEST FIT
  liberation_day        R²=0.33  SNR=0.49  ❌  POOR
→ Recommendation: gulf_war_recovery is currently most explanatory
```

### Spearman rank correlation test output

```
ALPHA VALIDATION — ukraine_shock (30-day window, 22 trading days)
────────────────────────────────────────────────────────────────────
  Spearman(Conf.Δ rank → next-day return rank):  0.34
  p-value:                                        0.021
  Bootstrap median (random):                      0.02
  Bootstrap 95th percentile (random):             0.18
  Result: ✅ SIGNIFICANT — model ranking is informative (p < 0.05)
```

---

## 4. Architecture & Key Files

### Existing files touched

```
shockarb/cli.py          — add --save-recent flag to score subcommand;
                           call save_score_row() and purge_stale_scores()
                           after each score run

shockarb/score_history.py — new module (see below); all archive logic lives here
```

### New file

```
shockarb/score_history.py — ScoreArchive class:
                             save_row(), load_window(), purge_stale(),
                             compute_snr(), regime_competition(),
                             spearman_alpha_test()
```

### New CLI subcommand

```
shockarb regime-health   — reads archive, prints SNR table + alpha test
                           requires --days N (default 30)
```

### Data layout

```
data/
  recent_scores/
    2026-05-29.parquet   — one file per trading day, one row per stock
    2026-05-30.parquet
    ...                  — auto-purged after RECENT_SCORES_RETENTION_DAYS
```

### Key design choices

**Why one file per day (not one appended file)?**
Purging by age is a simple `unlink()` on old files. A single appended parquet
would require re-reading and re-writing to drop old rows. Daily files also make
it trivial to inspect or delete a specific day.

**Why Spearman rank correlation (not Sharpe or IR)?**
Rank correlation is distribution-free, controls for universe-level beta automatically,
and directly answers "is the ordering informative" rather than "did the universe move."
A model that ranks stocks correctly but whose universe happened to fall is still
adding alpha. Sharpe would penalize it incorrectly.

**Why not download new data for regime competition?**
The ETF return vectors needed to project onto each regime's factor loadings are
already in memory during `score`. Projecting them onto additional factor matrices
is pure arithmetic — no network calls, no datamgr requests.

**Why defer Kalman blending?**
Kalman weights are only as good as the R² estimates driving them. R² over 22 days
is noisy. Building regime competition first validates that the R² estimates are
sensible before letting them control a blending process. Kalman on noisy inputs
can chase its own tail.

---

## 5. Recent Decisions & Rationale

**2026-05-30 — Defer Kalman blending**
Build SNR + regime competition first. Validate R² estimates are stable and
sensible before letting them drive adaptive weights. Kalman is the next layer,
not the first.

**2026-05-30 — Spearman rank correlation as primary alpha statistic**
Controls for universe beta. Distribution-free. Directly tests whether the model's
*ordering* is informative, not whether the universe happened to go up. Agreed in
design conversation.

**2026-05-30 — One parquet file per trading day in recent_scores/**
Simplifies purge logic (delete files older than cutoff). Avoids read-rewrite
cycle of a single appended file. Inspection/debugging is trivial.

**2026-05-30 — Regime competition uses no new data downloads**
ETF returns already fetched during score. Projection onto additional factor
matrices is arithmetic only. Zero marginal cost.

**2026-05-30 — --save-recent is the keystone**
All three deliverables depend on the archive. Build and verify it first before
building SNR or alpha test. Clock starts ticking on data accumulation from
first use.

---

## 6. Open Questions & Blockers

1. **Minimum window for reliable SNR** — 22 trading days (~1 month) is the proposed
   default. This gives weak but useful signal. Is 30 days a better default?
   Owner: Ken. Added: 2026-05-30.

2. **SNR degradation threshold** — proposed: SNR < 1.0 triggers ❌, SNR 1.0–2.0
   triggers ⚠️. Should the warning threshold be configurable in config.py?
   Owner: Ken. Added: 2026-05-30.

3. **Regime competition requires all regimes to have been built** — if gulf_war_recovery
   model file doesn't exist, competition silently skips it or errors. Define behavior.
   Proposed: skip missing models with a note in output.
   Owner: Cowork. Added: 2026-05-30.

4. **Alpha test minimum N** — 20 trading days gives weak signal; 60 gives reasonable
   confidence. Should the test refuse to run below a minimum? Proposed: warn but run,
   note low-N caveat in output. Owner: Cowork. Added: 2026-05-30.

---

## 7. Next Steps (Ordered)

### Step 1 — `--save-recent` flag and ScoreArchive ⭐ Build First

**Files:** `shockarb/score_history.py` (new), `shockarb/cli.py` (minor addition)

**What to build:**
- `ScoreArchive` class with:
  - `save_row(date, scores_df, regime_name, model_file)` — appends rows to daily parquet
  - `load_window(days=30)` — reads last N calendar days of parquet files, returns DataFrame
  - `purge_stale(retention_days=90)` — deletes files older than cutoff
- Add `--save-recent` flag to `score` subcommand in `cli.py`
- Call `archive.save_row()` and `archive.purge_stale()` after each score run when flag is set
- Named constant `RECENT_SCORES_RETENTION_DAYS = 90` at top of `score_history.py`

**Verification tests (Anaconda Prompt):**
```cmd
REM Run score with flag
python -m shockarb score --save-recent

REM Verify file was created
dir data\recent_scores\

REM Verify row count matches scored tickers
python -c "import pandas as pd; from pathlib import Path; f=sorted(Path('data/recent_scores').glob('*.parquet'))[-1]; df=pd.read_parquet(f); print(len(df), 'rows,', list(df.columns))"
REM EXPECTED: ~66 rows (stock count), columns include date/ticker/actual/expected/delta/r2/conf_delta/regime/model_file

REM Verify purge logic (manual test - set retention to 0 days, check files deleted)
python -c "from shockarb.score_history import ScoreArchive; a=ScoreArchive('data'); a.purge_stale(retention_days=0); print('purge ok')"
```

**Expected outcome:** ✅ High confidence. This is pure I/O with no statistical logic.
No surprises expected. The only edge case is the first run (no existing archive directory) — 
`ScoreArchive` should create `data/recent_scores/` if it doesn't exist.

---

### Step 2 — Regime SNR and Regime Competition

**Files:** `shockarb/score_history.py` (additions), `shockarb/cli.py` (print block)

**What to build:**
- `ScoreArchive.compute_snr(regime_name, days=30)` — computes out-of-sample R² and SNR
  from archive window. Returns dict with `{r2, snr, n_days, n_stocks, status}`.
- `ScoreArchive.regime_competition(days=30)` — calls `compute_snr` for each registered
  regime that has a model file present. Returns ranked list.
- Append regime health block to `score` output when `--save-recent` is active and
  archive has ≥ `MIN_WINDOW_DAYS` (= 5) days of data. Below minimum: print
  "Regime health: accumulating data (N/MIN_WINDOW_DAYS days)."

**Verification tests:**
```cmd
REM After running score --save-recent for several days, run:
python -m shockarb score --save-recent

REM Should show REGIME HEALTH block at bottom of output
REM Verify SNR computation directly
python -c "from shockarb.score_history import ScoreArchive; a=ScoreArchive('data'); print(a.compute_snr('ukraine_shock', days=30))"
REM EXPECTED: dict with r2 (float 0-1), snr (positive float), n_days (int), status (str)

REM Verify regime competition
python -c "from shockarb.score_history import ScoreArchive; a=ScoreArchive('data'); [print(r) for r in a.regime_competition(days=30)]"
REM EXPECTED: list of dicts, one per registered regime with a model file
```

**Expected outcome:** ⚠️ Medium confidence. The SNR formula is simple arithmetic but
the regime competition requires loading each regime's factor loadings from model files
and projecting the recent ETF return vectors. The model file loading path needs to handle
missing files gracefully. The output will be less meaningful until 10+ days of archive
data have accumulated — expect SNR values to be unstable with only a few days of data.
This is expected and not a bug.

---

### Step 3 — `regime-health` CLI Subcommand

**Files:** `shockarb/cli.py`

**What to build:**
- New subcommand `regime-health` with `--days N` option (default 30)
- Reads archive via `ScoreArchive.regime_competition()`
- Prints formatted SNR table + best-fit recommendation
- Exits with code 1 if archive is empty (prints helpful message)

**Verification tests:**
```cmd
python -m shockarb regime-health
REM EXPECTED: formatted SNR table, one row per regime with model file

python -m shockarb regime-health --days 10
REM EXPECTED: same but computed over 10-day window (may show low-N caveat)

python -m shockarb regime-health --days 999
REM EXPECTED: uses all available data, prints actual window size used
```

**Expected outcome:** ✅ High confidence once Step 2 is working. This is plumbing only —
formatting and routing. No statistical logic here.

---

### Step 4 — Spearman Alpha Test

**Files:** `shockarb/score_history.py` (addition), `shockarb/cli.py` (optional flag)

**What to build:**
- `ScoreArchive.spearman_alpha_test(regime_name, days=30, bootstrap_n=1000)`:
  - Load window of saved scores
  - For each day: rank stocks by `conf_delta` (model) and by `next_day_actual` (realized)
  - Compute Spearman correlation between the two rankings, per day
  - Average across days → mean Spearman ρ
  - Bootstrap: repeat with randomly shuffled conf_delta rankings M=1000 times
  - Report: ρ, p-value (fraction of bootstrap samples > ρ), bootstrap median, 95th pct
- Append alpha test output to `regime-health` command
- Requires `next_day_actual` column in archive — this means save_row() must also save
  *today's* realized returns for stocks scored *yesterday*. Implementation note: on each
  score run, look up yesterday's archive file and backfill the `next_day_actual` column.

**Implementation note — backfill pattern:**
```python
# In save_row(), after saving today's scores:
# 1. Load yesterday's parquet (if it exists)
# 2. For each ticker in yesterday's file, set next_day_actual = today's actual return
# 3. Re-write yesterday's parquet with the backfilled column
# This is the only write-twice pattern; it's unavoidable given we need t+1 returns
```

**Verification tests:**
```cmd
REM After 10+ days of --save-recent runs:
python -m shockarb regime-health

REM Should now include ALPHA VALIDATION section at bottom
REM Verify directly:
python -c "from shockarb.score_history import ScoreArchive; a=ScoreArchive('data'); print(a.spearman_alpha_test('ukraine_shock', days=30))"
REM EXPECTED: dict with rho (float), p_value (float 0-1), bootstrap_median, bootstrap_95th, n_days, significant (bool)

REM Edge case: insufficient data
python -c "from shockarb.score_history import ScoreArchive; a=ScoreArchive('data'); print(a.spearman_alpha_test('ukraine_shock', days=3))"
REM EXPECTED: raises InsufficientDataError or returns dict with significant=None and a warning message
```

**Expected outcome:** ⚠️ Medium confidence on implementation, ⚠️ Low confidence on
statistical results until 30+ days of data accumulate. The backfill pattern (writing
t+1 returns into yesterday's file) is the trickiest implementation detail — test it
carefully. The Spearman computation itself is straightforward (scipy.stats.spearmanr).
The bootstrap is simple: shuffle conf_delta ranks, re-compute correlation, repeat.

**Important caveat on early results:** With only 5–10 days of data, the Spearman test
will be extremely noisy and p-values will be unreliable. This is not a bug — it's
the nature of the test with small N. Print a clear low-N caveat when days < 20.

---

### Step 5 — Tests

**Files:** `tests/test_score_history.py` (new)

**What to build:**
Tests covering:
- `save_row()` creates file with correct schema
- `load_window()` returns correct date range
- `purge_stale()` deletes old files and keeps recent ones
- `compute_snr()` returns correct values for known input (use synthetic data)
- `spearman_alpha_test()` returns ρ=1.0 for perfectly-ranked synthetic data
- `spearman_alpha_test()` returns ρ≈0 for random synthetic data
- Missing model file in regime competition is handled gracefully
- Archive directory auto-created if missing
- `InsufficientDataError` (or equivalent) raised/handled when window too small

**Verification:**
```cmd
pytest tests\test_score_history.py -v
REM EXPECTED: all new tests pass

pytest tests\ -v
REM EXPECTED: 281 + N_new tests pass, 0 failed
```

**Expected outcome:** ✅ High confidence if Steps 1–4 are implemented cleanly.
The statistical tests (SNR, Spearman) are easiest to test with synthetic data where
the answer is known. Use a fixture that generates a DataFrame with perfect rankings
and verify ρ=1.0, then one with random rankings and verify ρ≈0.

---

## 8. Deferred: Kalman Adaptive Blending

Not in scope for this project. Build after:
1. Regime competition is live and producing stable SNR estimates (need 30+ days)
2. Spearman alpha test confirms the model is adding ranking alpha
3. SNR estimates from at least two regimes are available for comparison

The Kalman filter would use regime competition R² values as observation inputs and
maintain a posterior weight distribution over regimes. The blended signal would be
`Σ(weight_i × conf_delta_i)` across all regimes. Risk: Kalman on noisy R² estimates
(small N, high variance) can produce unstable weights. Validate the inputs before
building the filter.

---

## 9. Stock Inclusion Checklist (Reference)

Automatable checks that should eventually gate universe membership:

| Priority | Check | Implementation |
|----------|-------|----------------|
| 1 | R² floor ≥ 0.30 | Already computed; add hard cutoff in score output |
| 2 | Market cap ≥ $5B | yfinance info call, zero marginal cost |
| 3 | Avg daily volume sufficient | yfinance info call |
| 4 | Gross margin trend — 3-year slope not negative | yfinance financials, 3-year history |
| 5 | Net debt/EBITDA < 3x | yfinance balance sheet |
| 6 | Forward EPS direction positive | Already in fundamental scanner |
| 7 | Dividend coverage ratio sustainable | Already have both numbers |

Non-automatable (human judgment required):
- Secular category decline (Campbell's soup problem)
- Strategic pivot with opaque financials (Campbell's Rao's problem)
- Binary event risk (FDA, Supreme Court, M&A)

---

## 10. Last Updated

2026-05-30 | manual | Fresh | Initial creation for Cowork handoff.
Covers --save-recent infrastructure, regime SNR/competition, Spearman alpha test,
deferred Kalman blending, and stock inclusion checklist.
