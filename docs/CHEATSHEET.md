# ShockArb Cheatsheet

Quick reference for day-to-day use. Full details in [API.md](./API.md).

---

## CLI — The Three Daily Commands

```bash
# 1. Build (run once per event, or daily for live use)
python -m shockarb build --universe us

# 2. Score today's tape
python -m shockarb score --universe us

# 3. Score a historical date
python -m shockarb score --universe us --date 2022-03-01

# Save results to CSV
python -m shockarb score --universe us --output results.csv

# Show model quality (add -v for full factor tables)
python -m shockarb show --universe us
python -m shockarb show --universe us -v

# Export CSVs for spreadsheet review
python -m shockarb export --universe us

# Override data directory
python -m shockarb --data-dir /data/shockarb build --universe us
# OR: export SHOCK_ARB_DATA_DIR=/data/shockarb
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

| Column | Quick meaning |
|--------|--------------|
| `actual_return` | What the stock did today |
| `expected_rel` | What macro factors imply it should have done |
| `delta_rel` | Gap (positive = undersold = potential buy) |
| `r_squared` | How well the model fits this stock |
| `residual_vol` | Unexplained volatility → use for stop sizing |
| `confidence_delta` | **delta × R² — the primary ranking signal** |

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

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| All `r_squared` < 0.30 | Too few ETFs or wrong sector mix | Add more ETFs covering the relevant sectors |
| `cumulative_variance` < 0.50 | Factors don't span the crisis | Increase `n_components` or widen ETF selection |
| Many signals but all similar stocks | Sector concentration | Inspect `etf_basis` — one factor may dominate |
| Synthetic data warning | yfinance network failure | Check internet; check if tickers are still listed |
| "No model found" on score | Forgot to run `build` | Run `python -m shockarb build --universe us` first |
| `Index mismatch` on `FactorModel()` | ETF and stock return dates don't align | `pipeline.build()` handles alignment automatically |

---

## File Locations (default `./data`)

```
data/
├── us_20260301_143022.json      # Saved model (load with pipeline.load_model())
├── us_etf_basis.csv             # ETF factor directions (human-readable)
├── us_stock_loadings.csv        # Stock loadings + R² + residual vol
├── shockarb.log                 # Execution log (rotating, 10MB)
├── cache/
│   ├── us_etf_ohlcv.parquet     # Cached ETF prices
│   ├── us_stock_ohlcv.parquet   # Cached stock prices
│   └── cache_metadata.json      # Cache inventory
└── backups/                     # Pre-mutation parquet backups (7-day retention)
```

---

## Adding a Universe to the CLI

Edit `shockarb/cli.py`:

```python
from shockarb.config import UniverseConfig

UNIVERSES["taiwan"] = UniverseConfig(
    name="taiwan",
    market_etfs=["EWT", "KWEB", "TLT", "GLD", "XLE"],
    individual_stocks=["TSM", "AMAT", "LRCX", "ASML", "KLAC"],
    n_components=3,
    start_date="2022-02-10",
    end_date="2022-03-31",
)
```

Then:
```bash
python -m shockarb build --universe taiwan
python -m shockarb score --universe taiwan
```
