# ShockArb Factor Model

**A quantitative system for identifying stocks mispriced by geopolitical panic selling.**

ShockArb decomposes daily stock returns into a macro-factor-explained component and a residual. During a geopolitical shock — Russia-Ukraine, Middle East escalation, Taiwan Strait — the broad market, energy sector, and bond/gold complex all move in characteristic directions. Stocks whose actual returns differ significantly from what those factor moves imply are flagged as mispricing candidates.

---

## Quick Start

```bash
# Install
pip install -e .

# Build the US model (downloads ~35 days of price history, fits SVD, saves JSON)
python -m shockarb build --universe us

# Score today's live tape
python -m shockarb score --universe us

# Score a specific historical date
python -m shockarb score --universe us --date 2022-03-01

# Show model diagnostics
python -m shockarb show --universe us --verbose
```

---

## Installation

```bash
cd shockarb/
pip install -e .
```

Or without installing (add the repo root to `PYTHONPATH`):

```bash
export PYTHONPATH=/path/to/shockarb:$PYTHONPATH
python -m shockarb build --universe us
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.20 | SVD, OLS (lstsq) |
| pandas | >=1.3 | Return DataFrames, DatetimeIndex |
| yfinance | >=0.2 | Market data download |
| loguru | >=0.6 | Structured logging |
| pyarrow | >=8.0 | Parquet cache storage |

---

## Project Layout

```
shockarb/
├── shockarb/                  ← Python package
│   ├── __init__.py            # Public API exports
│   ├── __main__.py            # python -m shockarb entry point
│   ├── config.py              # UniverseConfig, ExecutionConfig, pre-built universes
│   ├── engine.py              # Pure math: SVD, OLS, FactorModel, FactorDiagnostics
│   ├── cache.py               # CacheManager: parquet-based OHLCV caching
│   ├── pipeline.py            # I/O layer: fetch, build, save, load, export
│   ├── report.py              # Terminal display: print_scores, print_live_alpha
│   └── cli.py                 # CLI: build / score / export / show subcommands
│
├── tests/
│   ├── conftest.py            # Shared fixtures (sample_etf_returns, fitted_model, …)
│   ├── unit/                  # Tests for individual modules in isolation
│   │   ├── test_config.py     # UniverseConfig, ExecutionConfig
│   │   ├── test_engine.py     # FactorModel, FactorDiagnostics
│   │   ├── test_cache.py      # CacheManager
│   │   ├── test_pipeline.py   # fetch_prices, prices_to_returns, save/load
│   │   ├── test_cli.py        # CLI argument parsing, command dispatch
│   │   └── test_report.py     # print_scores, print_model_state, print_live_alpha
│   └── integration/           # Multi-module and end-to-end tests
│       ├── test_pipeline.py   # Pipeline with real cache writes
│       ├── test_cli.py        # CLI commands against real model files
│       ├── test_end_to_end.py # Full build → score → save → load → score cycles
│       └── test_integration.py # Cross-module column contract, CLI → score chain
│
├── docs/
│   ├── README.md              # This file
│   ├── API.md                 # Complete function and class reference
│   └── CHEATSHEET.md          # Quick reference card
│
├── scripts/
│   ├── score_live.py          # Operational intraday scanner
│   ├── portfolio_sizer.py     # Dollar-denominated position sizing
│   ├── news_scanner.py        # Geopolitical event detection
│   ├── names.py               # Ticker → company name resolver
│   └── csv_to_md.py           # CSV → Markdown table formatter
│
├── examples/
│   └── basic_usage.py         # Five annotated usage examples
│
├── setup.py
└── pytest.ini
```

**Design principle:** `engine.py` contains zero I/O. It takes DataFrames in and returns DataFrames out. All network calls, file reads, and cache writes live in `pipeline.py` and `cache.py`. Swapping the data source (yfinance → Bloomberg, Snowflake) requires changes to `pipeline.py` only — the math is untouched.

---

## Mathematical Foundation

### Stage 1 — Factor Extraction (SVD)

Let **R_E** be the (T × N_etf) matrix of mean-centred daily ETF returns.

```
SVD: R_E = U Σ Vᵀ
```

Retain the top k rows of Vᵀ as factor directions. Factor return time series:

```
F = R_E @ Vᵀ[:k]ᵀ  →  shape (T × k)
```

**Typical factor interpretation during a geopolitical crisis:**
- **Factor 1:** Broad market direction (risk-on / risk-off)
- **Factor 2:** Energy vs. defensive assets (geopolitical shock axis)
- **Factor 3:** Growth vs. value rotation

### Stage 2 — Stock Projection (OLS)

Each stock's mean-centred returns are regressed on factor returns via least squares:

```
min ‖R_S − F @ Bᵀ‖   →   loadings B  shape (N_stock × k)
```

`numpy.linalg.lstsq` is used (numerically more stable than explicit pseudoinverse).

### Scoring

Given new ETF returns on day T+1:

```
f_today  =  Vᵀ[:k] @ (etf_returns_today − etf_mean_calib)
expected =  B @ f_today
delta    =  expected − actual
```

**Positive delta** means the stock fell more than macro factors imply — a potential buy.

**confidence_delta = delta_rel × R²** is the primary ranking signal.
It down-weights stocks where the factor model has low explanatory power.

---

## Interpreting Output

`model.score()` returns a DataFrame sorted descending by `confidence_delta`:

| Column | Meaning |
|--------|---------|
| `actual_return` | Observed return on the scoring day |
| `expected_rel` | Factor-implied return (no drift — pure structural signal) |
| `expected_abs` | Factor-implied return + calibration-window daily drift |
| `delta_rel` | `expected_rel − actual` — positive means undersold |
| `delta_abs` | `expected_abs − actual` — drift-adjusted version |
| `r_squared` | Calibration R² — how well factors explain this stock |
| `residual_vol` | Annualised unexplained volatility — use for position sizing |
| `confidence_delta` | `delta_rel × r_squared` — **primary ranking signal** |

### Decision Rules

| Signal | Criteria | Interpretation |
|--------|----------|----------------|
| **Strong buy** | `confidence_delta > 0.005` AND `r_squared > 0.50` | Model fits well, stock oversold vs. macro |
| **Weak signal** | Large delta BUT `r_squared < 0.30` | Model doesn't explain this stock — treat with skepticism |
| **Avoid / short** | Negative `confidence_delta` | Stock outperformed what macro predicts — not a bargain |

---

## Configuration

### Universe Configuration

```python
from shockarb.config import UniverseConfig

custom = UniverseConfig(
    name="my_universe",
    market_etfs=["SPY", "TLT", "GLD", "XLE", "XLF"],  # macro factor basis
    individual_stocks=["AAPL", "MSFT", "GOOGL", "CVX"],
    n_components=3,           # SVD factors to retain
    start_date="2022-02-10",  # calibration window start (inclusive)
    end_date="2022-03-31",    # calibration window end (exclusive)
)
```

**n_components guidance:**
- `2` — market + one sector axis (fast, less nuanced)
- `3` — market + sector + shock (recommended default)
- `4+` — risks overfitting on short (~35-day) windows

### Execution Configuration

```python
from shockarb.config import ExecutionConfig

cfg = ExecutionConfig(
    data_dir="/data/shockarb",   # default: ./data or $SHOCK_ARB_DATA_DIR
    log_to_file=True,            # write rotating log
    log_level="INFO",            # DEBUG / INFO / WARNING / ERROR
)
```

Set `SHOCK_ARB_DATA_DIR` to override `data_dir` without code changes:

```bash
export SHOCK_ARB_DATA_DIR=/data/shockarb
python -m shockarb build --universe us
```

### Pre-built Universes

| Constant | ETFs | Stocks | Focus |
|----------|------|--------|-------|
| `US_UNIVERSE` | 20 | 60+ | US large-cap crisis mispricing |
| `GLOBAL_UNIVERSE` | 20 | ~25 | Cross-border and European exposure |

---

## CLI Reference

```bash
python -m shockarb [--data-dir DIR] COMMAND [OPTIONS]
```

| Command | Key Options | Description |
|---------|-------------|-------------|
| `build` | `--universe us\|global` | Fetch data, fit model, save JSON |
| `score` | `--universe`, `--date YYYY-MM-DD`, `--top N`, `--output file.csv` | Score returns |
| `export` | `--universe` | Write ETF basis + stock loadings to CSV |
| `show` | `--universe`, `--verbose / -v` | Display model diagnostics |

---

## Data Files

All output goes to `data_dir` (default `./data`):

| File | Contents |
|------|----------|
| `{name}_{timestamp}.json` | Serialised model (scoring-ready) |
| `cache/{name}_etf_ohlcv.parquet` | Cached ETF OHLCV data |
| `cache/{name}_stock_ohlcv.parquet` | Cached stock OHLCV data |
| `cache/cache_metadata.json` | Cache inventory (tickers, date ranges) |
| `backups/` | Pre-mutation parquet backups (7-day retention) |
| `{name}_etf_basis.csv` | Factor directions per ETF |
| `{name}_stock_loadings.csv` | Stock loadings + R² + residual vol |
| `shockarb.log` | Rotating execution log |

---

## Running Tests

```bash
# Full suite
pytest

# Unit tests only (fast, offline)
pytest tests/unit/

# Integration tests (writes to temp dirs)
pytest tests/integration/

# With coverage
pytest --cov=shockarb --cov-report=term-missing

# A specific module
pytest tests/unit/test_engine.py -v
```

---

## Known Limitations

**Small calibration window** (~35 trading days). A single stock-specific event during calibration can contaminate factor loadings for that ticker. Inspect R² before trusting any signal.

**Regime non-stationarity.** Factor-stock relationships shift over a crisis. The model captures the average over the window — early panic days differ from recovery.

**Historical analogy risk.** Calibrated on Ukraine 2022. Energy/commodity transmission differs in Taiwan Strait, pandemic, or financial-system-stress events.

**Cannot distinguish rational re-pricing.** A stock falling 5% when factors imply 2% might be a mispricing candidate — or it might be accurately pricing in company-specific macro headwinds (rate exposure, supply-chain). The signal is a starting point, not a conclusion.

**No position sizing.** ShockArb generates ranked signals only. Production use requires stop-losses, correlation-aware sizing, and liquidity constraints.

---

## Extension Roadmap

Priority order from the original knowledge transfer document:

1. **Multi-window calibration** — average loadings across multiple historical crisis windows to reduce overfitting to one event
2. **R² filtering** — auto-exclude low-R² stocks from the scoring output entirely
3. **Exponential decay weighting** — upweight recent calibration days
4. **Sector-aware deduplication** — cap position concentration per GICS sector
5. **Regime classifier** — label calibration sub-windows (panic vs. recovery) and weight accordingly
6. **Bloomberg/IEX data source** — replace yfinance for production reliability

---

## Version History

| Version | Changes |
|---------|---------|
| 3.0.0 | Pipeline converted from class to module of functions. to_dict() stripped of raw return matrices. report.py integrated. _print_scores column name bug fixed. Tiered test structure (unit / integration). |
| 2.1.0 | CacheManager implementation. Tiered NaN strategy. FactorDiagnostics dataclass. |
| 2.0.0 | Original knowledge transfer implementation. |

---

## License

Proprietary. For authorised use only.
