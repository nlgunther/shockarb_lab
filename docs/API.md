# ShockArb API Reference

Complete reference for all public classes and functions. Organised by module.

---

## `shockarb.config`

### `UniverseConfig`

Immutable (frozen dataclass) specification of a trading universe.

```python
@dataclass(frozen=True)
class UniverseConfig:
    name: str
    market_etfs: List[str]
    individual_stocks: List[str]
    n_components: int
    start_date: str        # YYYY-MM-DD, inclusive
    end_date: str          # YYYY-MM-DD, exclusive (yfinance convention)
```

**Validation** (raises `ValueError` on construction):
- `market_etfs` must be non-empty
- `individual_stocks` must be non-empty
- `n_components` must be ≥ 1

**Pre-built constants:**

| Constant | `name` | ETFs | Stocks | n_components | Window |
|----------|--------|------|--------|-------------|--------|
| `US_UNIVERSE` | `"us"` | 20 | 60+ | 3 | 2022-02-10 → 2022-03-31 |
| `GLOBAL_UNIVERSE` | `"global"` | 20 | ~25 | 3 | 2022-02-10 → 2022-03-31 |

---

### `ExecutionConfig`

Mutable runtime settings. Pass an instance to any pipeline function to control paths and logging.

```python
@dataclass
class ExecutionConfig:
    data_dir: str          # default: $SHOCK_ARB_DATA_DIR or ./data
    log_to_file: bool      # default: True
    log_level: str         # default: "INFO"
```

**Methods:**

#### `resolve_path(filename: str) → str`

Return the absolute path `data_dir/filename`, creating `data_dir` if it does not exist.

```python
cfg = ExecutionConfig(data_dir="/data/shockarb")
path = cfg.resolve_path("us_model.json")
# → "/data/shockarb/us_model.json"
```

#### `configure_logger() → None`

Configure loguru for stdout (and optionally a rotating log file). Idempotent — safe to call multiple times.

---

## `shockarb.engine`

### `FactorDiagnostics`

Read-only quality metrics from a completed `FactorModel.fit()`.

```python
@dataclass
class FactorDiagnostics:
    n_observations: int
    n_etfs: int
    n_stocks: int
    n_factors: int
    explained_variance_ratio: NDArray     # shape (n_factors,)
    cumulative_variance: float
    stock_r_squared: pd.Series            # index = stock tickers
    residual_vol: pd.Series               # annualised, index = stock tickers
```

**Method:**

#### `summary() → str`

Return a compact multi-line string suitable for printing.

**Quality thresholds:**

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `cumulative_variance` | > 0.70 | Model captures most ETF co-movement |
| `stock_r_squared` | > 0.50 | Factor model explains the stock well |
| `stock_r_squared` | < 0.30 | Low fit — discount signals from this stock |

---

### `FactorModel`

Core two-stage structural factor model.

```python
model = FactorModel(etf_returns, stock_returns)
```

**Parameters:**
- `etf_returns: pd.DataFrame` — shape (T, N_etf), DatetimeIndex
- `stock_returns: pd.DataFrame` — shape (T, N_stock), must share the same index

Raises `ValueError` if indices do not match.

---

#### `fit(n_components: int = 3) → FactorModel`

Fit the SVD-based factor model. Returns `self` for method chaining.

```python
model = FactorModel(etf_returns, stock_returns).fit(n_components=3)
```

**Raises:**
- `ValueError` if `n_components ≥ min(T, N_etf)`

**After calling `fit()`**, the following attributes are available:

| Attribute | Type | Description |
|-----------|------|-------------|
| `loadings` | `pd.DataFrame` (N_stock × k) | OLS beta of each stock on each factor |
| `diagnostics` | `FactorDiagnostics` | Fit quality metrics |
| `etf_basis` *(property)* | `pd.DataFrame` (N_etf × k) | Factor direction vectors per ETF |
| `factor_returns` *(property)* | `pd.DataFrame` (T × k) | Historical factor return time series |

---

#### `score(today_etf_returns, today_stock_returns) → pd.DataFrame`

Score a single day's returns against the fitted model.

```python
scores = model.score(
    today_etf_returns=pd.Series({"VOO": -0.015, "VDE": 0.03, ...}),
    today_stock_returns=pd.Series({"V": -0.020, "MSFT": -0.018, ...}),
)
```

**Parameters:**
- `today_etf_returns: pd.Series` — indexed by ticker; missing tickers filled with 0
- `today_stock_returns: pd.Series` — only tickers present in both this Series and `loadings` are scored

**Returns:** `pd.DataFrame` sorted descending by `confidence_delta`

| Column | Dtype | Description |
|--------|-------|-------------|
| `actual_return` | float | Observed return |
| `expected_rel` | float | Factor-implied return (no drift) |
| `expected_abs` | float | Factor-implied return + calibration drift |
| `delta_rel` | float | `expected_rel − actual` (positive = undersold) |
| `delta_abs` | float | `expected_abs − actual` (drift-adjusted) |
| `r_squared` | float | Calibration fit quality [0, 1] |
| `residual_vol` | float | Annualised unexplained volatility |
| `confidence_delta` | float | `delta_rel × r_squared` — **primary signal** |

**Raises:** `RuntimeError` if model is not fitted.

---

#### `project_security(ticker, returns, min_overlap=0.8) → pd.Series`

Project a new ticker onto the existing factor basis without refitting.

```python
shop_returns = yf.download("SHOP", start=..., end=...)["Adj Close"].pct_change().dropna()
loadings = model.project_security("SHOP", shop_returns)
```

**Parameters:**
- `ticker: str` — display name for the security
- `returns: pd.Series` — daily returns with a DatetimeIndex
- `min_overlap: float` — minimum fraction of calibration dates with data (default 0.8)

**Returns:** `pd.Series` of factor loadings, length k, named `ticker`

**Raises:** `ValueError` if data coverage is below `min_overlap`

---

#### `to_dict() → dict`

Serialise the fitted model to a JSON-compatible dict. Does **not** include raw return matrices — only the minimal state needed for `score()`.

**Raises:** `RuntimeError` if not fitted.

---

#### `FactorModel.from_dict(d: dict) → FactorModel` *(classmethod)*

Reconstruct a scoring-ready model from a dict produced by `to_dict()`. The returned model can call `score()` and `project_security()` immediately. `etf_returns` and `stock_returns` are empty stubs (raw data is not persisted).

---

## `shockarb.pipeline`

All I/O operations. Import as a module and call functions directly:

```python
import shockarb.pipeline as pipeline
```

---

### `build(universe, exec_config=None) → FactorModel`

Full pipeline: fetch prices → compute returns → fit model.

```python
from shockarb.config import US_UNIVERSE
model = pipeline.build(US_UNIVERSE)
```

**Parameters:**
- `universe: UniverseConfig`
- `exec_config: ExecutionConfig` — optional; uses process default if omitted

---

### `fetch_prices(tickers, start, end, cache_name="prices", exec_config=None) → pd.DataFrame`

Download adjusted close prices, using a local parquet cache.

```python
prices = pipeline.fetch_prices(
    ["VOO", "TLT", "GLD"],
    start="2022-02-10",
    end="2022-03-31",
    cache_name="us_etf",
)
```

**Parameters:**
- `tickers: List[str]`
- `start, end: str` — YYYY-MM-DD; `end` is exclusive
- `cache_name: str` — logical cache key; each unique name gets its own parquet file
- `exec_config: ExecutionConfig` — optional

**Returns:** `pd.DataFrame` (dates × tickers), adjusted close prices

**Fallback:** If yfinance returns nothing (network outage, delisted tickers), falls back to synthetic crisis prices. This is logged at ERROR level and is clearly marked by all prices starting at exactly 100.

---

### `prices_to_returns(prices, min_coverage=0.8) → pd.DataFrame`

Convert a price matrix to daily returns with robust NaN handling.

```python
returns = pipeline.prices_to_returns(prices, min_coverage=0.8)
```

**Three-step NaN strategy:**
1. Drop tickers with fewer than `min_coverage` non-NaN rows
2. Forward-fill remaining gaps (foreign holiday misalignment)
3. Fill isolated post-pct_change NaNs with 0.0

---

### `fetch_live_returns(tickers, period="5d") → pd.Series`

Fetch the most recent day's closing returns. Uses a 5-day lookback so the calculation is robust to weekends and holidays.

```python
etf_returns = pipeline.fetch_live_returns(["VOO", "TLT", "GLD"])
```

**Raises:** `ValueError` if yfinance returns no data or returns are empty.

---

### `fetch_intraday_returns(tickers, max_retries=3, playback_path=None) → pd.Series | None`

Fetch intraday returns (prior close to current minute). Retries on transient failures. Returns `None` after all retries are exhausted.

```python
live = pipeline.fetch_intraday_returns(["VOO", "TLT"], max_retries=3)
```

Pass `playback_path` to replay a previously saved tape for offline debugging:

```python
live = pipeline.fetch_intraday_returns(["VOO"], playback_path="data/tape_20260301.parquet")
```

---

### `save_intraday_tape(tickers, path, full_tape=False) → pd.DataFrame | None`

Download and persist a 1-minute tape for later replay. By default stores only three rows (yesterday's close, today's open, current print). Pass `full_tape=True` for all ~600 rows.

---

### `save_model(model, name, exec_config=None) → str`

Serialise a fitted model to a timestamped JSON file.

```python
path = pipeline.save_model(model, "us")
# → "data/us_20260301_143022.json"
```

**Returns:** Absolute path to the saved file.

---

### `load_model(path) → FactorModel`

Load and return a scoring-ready model from a JSON file.

```python
model = pipeline.load_model("data/us_20260301_143022.json")
```

---

### `find_latest_model(name, exec_config=None) → str | None`

Return the path to the most recently saved model matching `name`, or `None` if none found.

```python
path = pipeline.find_latest_model("us")
```

---

### `export_csvs(model, name, exec_config=None) → tuple[str, str]`

Write ETF factor basis and stock loadings to human-readable CSV files. Stock loadings are augmented with R² and residual volatility and sorted by R² descending.

```python
basis_path, loadings_path = pipeline.export_csvs(model, "us")
```

**Returns:** `(etf_basis_path, stock_loadings_path)`

---

## `shockarb.cache`

### `CacheManager`

Intelligent parquet-based cache for yfinance OHLCV data. Handles ticker merging, date extension, atomic writes, and backup rotation automatically.

```python
from shockarb.cache import CacheManager
import yfinance as yf

mgr = CacheManager(
    cache_dir="data/cache",
    backup_dir="data/backups",
    downloader=yf.download,   # injectable for testing
)
```

---

#### `fetch_ohlcv(tickers, start, end, cache_name) → pd.DataFrame | None`

Fetch full OHLCV data, updating the cache only for what is not already stored.

```python
ohlcv = mgr.fetch_ohlcv(["VOO", "TLT"], "2022-02-10", "2022-03-31", "us_etf")
```

**Cache update logic (priority order):**
1. If no cache exists → download everything
2. If new tickers requested → download and merge them into existing date range
3. If date range extends into the recent past (<30 days ago) → download and append new dates
4. If full cache hit → return cached data, zero network calls

---

#### `extract_adj_close(ohlcv) → pd.DataFrame`

Extract the `Adj Close` price level from a MultiIndex OHLCV DataFrame. Falls back to `Close` if `Adj Close` is absent.

```python
prices = mgr.extract_adj_close(ohlcv)
# → pd.DataFrame, columns = ticker symbols, no MultiIndex
```

---

#### `get_cache_info() → dict`

Return the contents of the cache metadata sidecar JSON. Keys are parquet filenames; values contain ticker list, date range, row count, and last-updated timestamp.

---

## `shockarb.report`

Terminal presentation layer. All functions print to stdout and return `None`.

---

### `print_scores(scores, title, top_n=20, min_confidence=0.001, min_r_squared=0.3)`

Pretty-print a score DataFrame from `model.score()`.

```python
from shockarb.report import print_scores
print_scores(scores, "US | 2022-03-01", top_n=15, min_confidence=0.005)
```

**Parameters:**
- `scores: pd.DataFrame` — output of `model.score()`
- `title: str` — header label
- `top_n: int` — maximum actionable signals to display
- `min_confidence: float` — minimum `confidence_delta` threshold (default 0.1%)
- `min_r_squared: float` — minimum R² threshold (default 0.30)

---

### `print_model_state(json_path)`

Parse a saved JSON model file and render a full structural report: metadata, variance per factor, ETF basis vectors, stock loadings sorted by Factor 3.

```python
from shockarb.report import print_model_state
print_model_state("data/us_20260301_143022.json")
```

---

### `print_live_alpha(score_df, model_name="UNKNOWN", min_delta=0.005)`

Render a live alpha trading sheet, filtering to stocks with `delta_rel > min_delta`.

```python
from shockarb.report import print_live_alpha
print_live_alpha(scores, model_name="US", min_delta=0.005)
```

---

## `shockarb.cli`

The CLI is also accessible as a Python API if you need to script commands:

```python
from shockarb.cli import cmd_build, cmd_score, cmd_show, cmd_export, get_universe

class Args:
    universe = "us"
    data_dir = "/data/shockarb"
    no_log = False

cmd_build(Args())
```

### `get_universe(name: str) → UniverseConfig`

Look up a universe by name (case-insensitive). Raises `ValueError` for unknown names.

### `UNIVERSES: dict[str, UniverseConfig]`

Registry of all available universes. Add custom universes here to make them available via `--universe`.

```python
from shockarb.cli import UNIVERSES
from shockarb.config import UniverseConfig

UNIVERSES["my_universe"] = UniverseConfig(
    name="my_universe",
    market_etfs=["SPY", "TLT"],
    individual_stocks=["AAPL", "MSFT"],
    n_components=2,
    start_date="2022-02-10",
    end_date="2022-03-31",
)
```

---

## Column Reference

Columns returned by `model.score()` and written to CSV by `pipeline.export_csvs()`:

### Score output columns

| Column | Unit | Primary use |
|--------|------|-------------|
| `actual_return` | decimal (e.g. -0.02) | Reference |
| `expected_rel` | decimal | Ranking; no drift contamination |
| `expected_abs` | decimal | Position sizing when drift matters |
| `delta_rel` | decimal | Raw mispricing magnitude |
| `delta_abs` | decimal | Drift-adjusted mispricing |
| `r_squared` | [0, 1] | Signal quality gate |
| `residual_vol` | annualised decimal | Stop-loss width, position sizing |
| `confidence_delta` | decimal | **Primary sort and filter key** |

### Loadings CSV columns

| Column | Description |
|--------|-------------|
| `Factor_1` … `Factor_k` | OLS beta on each macro factor |
| `R_squared` | Calibration fit quality |
| `Residual_Vol` | Annualised unexplained return volatility |
