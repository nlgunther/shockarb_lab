# ShockArb Cheatsheet

Quick reference for day-to-day use. Full details in [API.md](./API.md).

---

## CLI — Daily Workflow

```bash
# One-time: set your active regime (saved as a sticky default)
python -m shockarb set-regime ukraine_shock

# 1. Build (run once per event, or daily for live use)
python -m shockarb build

# 2. Score today's tape (saves to data/live_alpha_us.csv by default)
python -m shockarb score

# Score and archive to rolling history (run daily to accumulate data)
python -m shockarb score --save-recent

# 3. Score a historical date
python -m shockarb score --date 2022-03-01

# Save to a custom path
python -m shockarb score --out results.csv

# Suppress CSV output
python -m shockarb score --no-out

# Show model quality (add -v for full factor tables)
python -m shockarb show
python -m shockarb show -v

# Export CSVs for spreadsheet review
python -m shockarb export

# Switch to a different regime
python -m shockarb set-regime global_ukraine_shock
python -m shockarb build

# Override sticky regime for a single command
python -m shockarb score --regime gulf_war_recovery --date 1991-04-15

# Override data directory
python -m shockarb --data-dir /data/shockarb build
# OR: set SHOCK_ARB_DATA_DIR=C:\data\shockarb
```

---

## Regime Management

```bash
# List all available regimes
python -m shockarb list-regimes

# Show current sticky regime
python -m shockarb show-regime

# Set sticky regime (persists across sessions)
python -m shockarb set-regime <regime_name>

# Available regimes:
#   ukraine_shock           - US universe, Feb-Mar 2022
#   global_ukraine_shock    - Global universe, Feb-Mar 2022
#   gulf_war_recovery       - US recovery, Mar-Jun 1991
#   liberation_day_recovery - US universe, Apr-Jul 2025
```

---

## Adding and Removing Tickers

```bash
# Remove a ticker (e.g. accidentally added to the wrong model)
python -m shockarb remove-asset RTX --regime global_ukraine_shock --save

# Add a ticker
python -m shockarb add-asset SHOP --save
```

---

## Adding Tickers to an Existing Model

```bash
# Add one or more tickers without refitting
python -m shockarb add-asset SHOP COIN --save

# Explicit regime (if not using sticky)
python -m shockarb add-asset SHOP --regime ukraine_shock --save
```

Or programmatically:

```python
from shockarb.regimes import get_regime
import shockarb.pipeline as pipeline

regime = get_regime("ukraine_shock")
model  = pipeline.load_model(pipeline.find_latest_model("us"))
summary = pipeline.add_assets(["SHOP", "COIN"], model, regime.universe)
print(summary)
pipeline.save_model(model, "us")   # persists the new tickers
```

---

## Python — Minimal Workflow

```python
import shockarb.pipeline as pipeline
from shockarb.config import US_UNIVERSE

# Build
model = pipeline.build(US_UNIVERSE)

# Score
import pandas as pd
scores = model.score(
    pd.Series({"VOO": -0.015, "VDE": 0.030, "TLT": 0.008, "GLD": 0.012, "ITA": 0.020}),
    pd.Series({"V": -0.020, "MSFT": -0.018, "LMT": 0.025, "CVX": 0.035, "UNH": -0.005}),
)

# Top signals
print(scores[["delta_rel", "r_squared", "confidence_delta"]].head(10))

# Save / load
path = pipeline.save_model(model, "us")
model = pipeline.load_model(path)
model = pipeline.load_model(pipeline.find_latest_model("us"))
```

---

## Signal Interpretation

```
confidence_delta > +0.005  AND  r_squared > 0.50  →  Strong buy candidate
confidence_delta > +0.002  AND  r_squared > 0.30  →  Weak / speculative
confidence_delta < -0.002                          →  Avoid / consider short
r_squared < 0.30                                   →  Discard signal (bad fit)
```

**Rule of thumb:** Sort by `confidence_delta` descending, filter `r_squared > 0.50`, act on the top 5–10.

---

## Score Output Columns

| Column             | Quick meaning                                |
| ------------------ | -------------------------------------------- |
| `actual_return`    | What the stock did today                     |
| `expected_rel`     | What macro factors imply it should have done |
| `delta_rel`        | Gap (positive = undersold = potential buy)   |
| `r_squared`        | How well the model fits this stock           |
| `residual_vol`     | Unexplained volatility → use for stop sizing |
| `confidence_delta` | **delta × R² — the primary ranking signal**  |

---

## Diagnostics — What to Check

```python
print(model.diagnostics.summary())

# Key numbers to inspect:
# cumulative_variance > 0.70  → Good. Factors capture the market.
# cumulative_variance < 0.50  → Bad. Add more ETFs or increase n_components.
# r_squared range              → Stocks < 0.30 are unreliable signals.
```

---

## Custom Universe

```python
from shockarb.config import UniverseConfig, ExecutionConfig
import shockarb.pipeline as pipeline

universe = UniverseConfig(
    name="energy",
    market_etfs=["XLE", "XOP", "OIH", "BNO", "TLT", "GLD"],
    individual_stocks=["CVX", "XOM", "COP", "SLB", "HAL", "MPC"],
    n_components=2,
    start_date="2022-02-10",
    end_date="2022-03-31",
)
cfg = ExecutionConfig(data_dir="./data/energy", log_level="WARNING")
model = pipeline.build(universe, cfg)
```

---

## Operational / Intraday Use

```bash
# Run the live scanner (scores every 60 seconds)
python scripts/score_live.py --target us --interval 60

# Single pass
python scripts/score_live.py --target us --interval 0

# Cache the tape for offline replay
python scripts/score_live.py --target us --interval 0 --cache-tape

# Replay a saved tape
python scripts/score_live.py --target us --interval 0 --playback 20260301_095846
```

---

## Project Out-of-Sample Security

```python
import yfinance as yf

prices = yf.download("SHOP", start="2022-02-10", end="2022-03-31")["Adj Close"]
returns = prices.pct_change().dropna()
loadings = model.project_security("SHOP", returns)
print(loadings)  # Factor_1, Factor_2, ... betas
```

---

## Troubleshooting

| Symptom                                    | Likely cause                                        | Fix                                                                               |
| ------------------------------------------ | --------------------------------------------------- | --------------------------------------------------------------------------------- |
| All `r_squared` < 0.30                     | Too few ETFs or wrong sector mix                    | Add more ETFs covering the relevant sectors                                       |
| `cumulative_variance` < 0.50               | Factors don't span the crisis                       | Increase `n_components` or widen ETF selection                                    |
| Many signals but all similar stocks        | Sector concentration                                | Inspect `etf_basis` — one factor may dominate                                     |
| Many "ETF / Unknown" industries in reports | Ticker cache stubs not upgraded from reference CSVs | Run `python utils/maintain_ticker_cache.py --fix-stubs --sort`                    |
| Synthetic data warning                     | yfinance network failure                            | Check internet; check if tickers are still listed                                 |
| "No model found" on score                  | Forgot to run `build`                               | Run `python -m shockarb build` (or `set-regime` first if no sticky regime is set) |
| `Index mismatch` on `FactorModel()`        | ETF and stock return dates don't align              | `pipeline.build()` handles alignment automatically                                |

---

## Comparing Reports Across Regimes/Dates

```bash
# Compare a root .md report against an iran_shock verdicts CSV, print + save
python -m shockarb compare-reports \
    reports/stock_report_20260612_0856.md \
    reports/iran_shock/stock_report_20260612_0945_verdicts.csv \
    --out compare_verdicts.md

# Compare two .md reports (no --out: console only)
python -m shockarb compare-reports report_a.md report_b.md
```

**Inputs:** any mix of `stock_report_*.md` and `stock_report_*_verdicts.csv`
(2 or more). `.csv` comes from `stockfit --save-verdicts` and carries full
stats (R², Conf.Δ, Fwd P/E, ...) for **every** tier, including excluded.

**Which tickers show up in the Stats sections:** a ticker appears if it's
`act_on`/`watch` in *any* of the reports, or if its tier differs across
reports (flagged ⚠). A ticker excluded consistently in every report is
omitted — it's not "interesting" for the comparison.

**Reading the Reason column:** for excluded tickers, `reason` says *why* —
e.g. `"r²=0.063 below threshold (0.65) — model fit too weak"` (bad factor
fit) vs `"confidence_delta=+0.0092 below threshold (0.020)"` (good fit, weak
signal). These point to different conclusions about whether the other
report's regime fits the ticker better right now.

---

## Updating Analyst Targets

Analyst targets in `data/fundamentals.csv` can go stale or become inverted after stock splits. Use `get_analyst_targets.py` to refresh them from Finviz (no API key required):

```bash
# Fetch and automatically update data/fundamentals.csv (recommended)
python get_analyst_targets.py --tickers KLAC QCOM NOW MSFT --update-fundamentals

# Custom path for fundamentals.csv
python get_analyst_targets.py --tickers KLAC --update-fundamentals path/to/fundamentals.csv

# Fetch only (print + save to finviz_analyst_data.csv, no auto-update)
python get_analyst_targets.py --tickers KLAC QCOM NOW MSFT

# Explicit provider
python get_analyst_targets.py --tickers KLAC --provider yfinance --update-fundamentals

# Fetch targets for all tickers in fundamentals.csv
python get_analyst_targets.py --update-fundamentals
```

Output is printed to console and saved to `finviz_analyst_data.csv` (or `yfinance_analyst_data.csv` etc.) in the project root. With `--update-fundamentals`, the `Analyst Tgt` column in `data/fundamentals.csv` is patched automatically — no manual copy needed.

**When to run:** whenever the stock report flags a data quality issue (`analyst target below current price`) or after a stock split (targets will be pre-split until refreshed). KLAC did a 10-for-1 split on 2026-06-12 — all pre-split targets need dividing by 10.

---

## File Locations

```
data/                            # Input data and model artefacts (default ./data)
├── us_20260301_143022.json      # Saved model (load with pipeline.load_model())
├── us_etf_basis.csv             # ETF factor directions (human-readable)
├── us_stock_loadings.csv        # Stock loadings + R² + residual vol
├── live_alpha_us.csv            # Latest score output
├── market_snapshot.json         # Latest market data snapshot
├── news.txt                     # Headlines fetched by news_scanner.py
├── fundamentals.csv             # Fundamentals fetched by news_scanner.py
├── ticker_reference_cache.json  # Company name/industry cache (for markdown reports)
├── nyse_*.csv, nasdaq_*.csv     # Reference CSVs (download from exchanges)
├── shockarb.log                 # Execution log (rotating, 10MB)
├── cache/
│   ├── us_etf_ohlcv.parquet     # Cached ETF prices
│   ├── us_stock_ohlcv.parquet   # Cached stock prices
│   └── cache_metadata.json      # Cache inventory
└── backups/                     # Pre-mutation parquet backups (7-day retention)

reports/                         # Generated reports (default ./reports)
├── market_report.md             # Latest market fit report (overwritten each run)
├── market_report_YYYYMMDD_HHMM.md   # Timestamped archive (--timestamp flag)
├── stock_report.md              # Latest stock opportunity report
└── stock_report_YYYYMMDD_HHMM.md    # Timestamped archive (--timestamp flag)
```

To write reports to a different folder, pass `--reports-dir /your/path` to either CLI:

```bash
python -m marketfit report --reports-dir C:\Users\me\Desktop\reports
python -m stockfit report --reports-dir C:\Users\me\Desktop\reports
```

**Ticker cache maintenance:**

```bash
# Fix "Unknown" industries in markdown reports (update cache from reference CSVs)
python utils/maintain_ticker_cache.py --fix-stubs --sort
```

See [UTILS.md § maintain_ticker_cache.py](./UTILS.md#maintain_ticker_cachepy) for details.

---

## Adding a Custom Regime

Edit `shockarb/regimes.py`:

```python
from shockarb.config import UniverseConfig
from shockarb.regimes import HistoricFactorModel, REGIME_REGISTRY

TAIWAN_STRAIT_CRISIS = HistoricFactorModel(
    name="taiwan_strait_crisis",
    description="Taiwan Strait tensions (hypothetical)",
    narrative="...",
    universe=UniverseConfig(
        name="taiwan",
        market_etfs=["EWT", "KWEB", "TLT", "GLD", "XLE"],
        individual_stocks=["TSM", "AMAT", "LRCX", "ASML", "KLAC"],
        n_components=3,
        start_date="2024-01-01",
        end_date="2024-03-31",
    ),
    tags=("geopolitical", "semiconductor", "asia"),
    supersedes=None,
)

REGIME_REGISTRY["taiwan_strait_crisis"] = TAIWAN_STRAIT_CRISIS
```

Then:

```bash
python -m shockarb set-regime taiwan_strait_crisis
python -m shockarb build
python -m shockarb score
```

---

## Score History Archive

The `--save-recent` flag accumulates score output to `data/recent_scores/` as `YYYY-MM-DD_HHMMSS.parquet`. Multiple runs on the same day are preserved; only the latest per day is used by `load_window` and `available_days`. Enables regime health monitoring once enough days have accumulated (minimum: 5 days; meaningful: 20+ days).

```bash
# Archive today's scores (run every trading day)
python -m shockarb score --save-recent

# Combine with other flags
python -m shockarb score --save-recent --out results.csv
python -m shockarb score --save-recent --regime gulf_war_recovery
```

### Read the archive in Python

```python
from shockarb.score_history import ScoreArchive

archive = ScoreArchive("data")

# Last 30 data-days (count of trading days, not calendar span)
df = archive.load_window(days=30)
print(df.columns.tolist())
# ['date', 'ticker', 'actual', 'expected', 'delta', 'r2',
#  'conf_delta', 'regime', 'model_file', 'next_day_actual']

# How many days are in the archive?
print(archive.available_days())

# Manual purge (normally called automatically by --save-recent)
removed = archive.purge_stale(retention_days=90)
```

### Archive column quick reference

| Column            | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `actual`          | Observed return on the scoring date                          |
| `expected`        | Factor-implied return (no drift)                             |
| `delta`           | `expected − actual` — raw mispricing                         |
| `r2`              | Calibration R² — signal quality weight                       |
| `conf_delta`      | `delta × r2` — primary ranking signal                        |
| `next_day_actual` | Realized return the *following* day (backfilled by next run) |

### Gotchas

| Symptom                       | Cause                            | Fix                                                             |
| ----------------------------- | -------------------------------- | --------------------------------------------------------------- |
| `next_day_actual` all NaN     | Only one day in archive          | Normal — backfilled by next `--save-recent` run                 |
| Archive empty after scoring   | `--save-recent` flag not passed  | Re-run with `--save-recent`                                     |
| `--save-recent` with `--date` | Historical runs are not archived | Flag is silently ignored on `--date` runs; archive is live-only |
