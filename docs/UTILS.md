# ShockArb Utils Reference

Five command-line scripts that form the post-scoring workflow. Each runs standalone from the project root after `pip install -e .`

```
utils/
├── daily_scanner.py     Score today's tape → export CSVs         (start here each day)
├── news_scanner.py      Fetch headlines for top targets
├── portfolio_sizer.py   Size a conviction-weighted trade ticket
├── csv_to_md.py         Convert a score CSV to a Markdown report
└── score_history.py     Score any historical date (backtesting)
```

All scripts accept `--help` for a full argument listing.

---

## Typical Daily Workflow

```
4:00 pm  python utils/daily_scanner.py              # score the close → data/live_alpha_*.csv
4:05 pm  python utils/news_scanner.py               # scan headlines for top signals
4:15 pm  python utils/portfolio_sizer.py            # size a trade ticket
         python utils/csv_to_md.py data/live_alpha_us.csv  # optional: shareable report
```

---

## daily_scanner.py

Loads the saved factor model(s), fetches today's closing Adj Close prices, computes daily returns, scores the tape, and writes the results to CSV. This is the entry point for every end-of-day run; all other utils consume the CSVs it produces.

**Output files**

| File | Contents |
|------|----------|
| `data/live_alpha_us.csv` | Scores for the US universe (if a US model exists) |
| `data/live_alpha_global.csv` | Scores for the Global universe (if a Global model exists) |

Each CSV has one row per stock with columns: `actual_return`, `expected_rel`, `delta_rel`, `r_squared`, `confidence_delta` (and others depending on model version).

**Usage**

```bash
# Scan both universes (default)
python utils/daily_scanner.py

# Scan one universe only
python utils/daily_scanner.py --universe us

# Custom data directory
python utils/daily_scanner.py --data-dir /path/to/data
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--universe` / `-u` | `us global` | One or more universe names to scan. Must match a saved model. |
| `--data-dir` | `./data` | Override data directory. Also accepts `$SHOCK_ARB_DATA_DIR`. |

**Notes**

- Fetches 5 days of history so prior-close is always available across weekends and single-day holidays.
- Prefers `Adj Close` over `Close` when yfinance returns a MultiIndex result.
- Skips a universe gracefully if no saved model is found (logs an error, continues).
- Prints next-step hints on completion.

---

## news_scanner.py

Fetches the three most recent Yahoo Finance headlines for each target. Useful for quickly checking whether a large delta is explained by a known catalyst (earnings miss, downgrade, FDA decision) or is a genuine unexplained dislocation.

Targets are selected in priority order:

1. `--tickers` — explicit list, ignores CSV entirely
2. `--csv` — top-N by `confidence_delta` from one or more score CSVs
3. *(default)* — top 10 from `data/live_alpha_us.csv`

**Usage**

```bash
# Default: top 10 from data/live_alpha_us.csv
python utils/news_scanner.py

# Top 5 from a specific CSV
python utils/news_scanner.py --csv data/live_alpha_us.csv --top 5

# Explicit tickers — no CSV required
python utils/news_scanner.py --tickers ROK CRM CPRT BSX

# Merge multiple universes, top 8 overall
python utils/news_scanner.py \
    --csv data/live_alpha_us.csv data/live_alpha_global.csv \
    --top 8

# Sort by a different column
python utils/news_scanner.py --csv data/live_alpha_us.csv --sort delta_rel
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `[]` | Path(s) to ShockArb score CSV files. Multiple files are merged before ranking. |
| `--top` | `10` | Number of targets to pull from CSV. Ignored when `--tickers` is used. |
| `--tickers` | — | Explicit ticker list. Overrides `--csv` entirely. |
| `--sort` | `confidence_delta` | CSV column to rank by. Falls back to `delta` if the specified column is absent. |

**Notes**

- Handles both the legacy flat yfinance news format and the newer nested-content format introduced around 2023. If a third, unrecognised format appears, the raw dict keys are printed for debugging.
- Network errors per ticker are caught and printed without aborting the scan.

---

## portfolio_sizer.py

Sizes a conviction-weighted trade ticket from one or more score CSVs. Allocates capital proportionally to `confidence_delta`, fetches live prices, and prints entry price, dollar allocation, share count, and take-profit target for each position.

**Usage**

```bash
# $100k across the top 5 US signals (default)
python utils/portfolio_sizer.py

# Explicit capital and position count
python utils/portfolio_sizer.py --csv data/live_alpha_us.csv --capital 50000 --top 3

# Merge US + Global into one ticket
python utils/portfolio_sizer.py \
    --csv data/live_alpha_us.csv data/live_alpha_global.csv \
    --capital 200000 --top 8
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `./data/live_alpha_us.csv` | Path(s) to score CSVs. Merged before ranking. |
| `--capital` | `100000` | Total dollar capital to allocate. |
| `--top` | `5` | Number of positions. |

**Output columns**

| Column | Description |
|--------|-------------|
| TICKER | Ticker symbol |
| WEIGHT | Conviction-weighted share of capital |
| ALLOCATION | Dollar amount allocated |
| CURRENT | Live price fetched from yfinance |
| TARGET | Take-profit limit price = `current × (1 + delta_rel)` |
| SHARES | Whole shares purchasable at current price |

**Required CSV columns:** `confidence_delta`, `delta_rel`.

**Notes**

- Allocation weight for each position = its `confidence_delta` / sum of all selected `confidence_delta` values.
- Take-profit target is the factor-model implied fair price, not a hard prediction. It represents where the stock *would* trade if the dislocation fully closed.
- Only stocks with `confidence_delta > 0` are considered. Negative signals are excluded.
- Tickers without a live price quote are skipped with a warning.

---

## csv_to_md.py

Converts a ShockArb score CSV into a formatted Markdown report suitable for sharing or archiving. Optionally resolves ticker symbols to full company names and industry classifications using local NYSE/NASDAQ reference CSVs.

**Usage**

```bash
# Basic conversion — includes company names by default
python utils/csv_to_md.py data/live_alpha_us.csv

# Save to a specific path
python utils/csv_to_md.py data/live_alpha_us.csv --out reports/2024-03-15.md

# Skip name resolution (faster, no reference CSVs needed)
python utils/csv_to_md.py data/live_alpha_us.csv --no-names

# Custom cache location
python utils/csv_to_md.py data/live_alpha_us.csv --cache data/my_cache.json
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `csv_file` | *(required)* | Path to the ShockArb score CSV. |
| `--out` | Same dir as CSV, `.md` extension | Output Markdown file path. |
| `--no-names` | `False` | Skip company name and industry resolution. |
| `--ref-dir` | `./data` | Directory containing NYSE/NASDAQ reference CSVs. |
| `--cache` | `./data/ticker_reference_cache.json` | Path to the ticker name cache JSON. |

**Name resolution**

When `--names` is set, the script scans `--ref-dir` for CSV files whose names contain `nyse` or `nasdaq` (case-insensitive), loads them in NYSE-first order, and performs a waterfall lookup: NYSE is checked first, then NASDAQ, then any other CSVs found. Each reference CSV must contain columns `Symbol`, `Name`, and `Industry`. Results are persisted to the cache JSON so subsequent runs avoid redundant file loads.

Reference file discovery is automatic — any file matching `*nyse*.csv` or `*nasdaq*.csv` in the directory is included. There is no need to hardcode filenames.

**Universe detection**

The universe label in the report header (`US`, `GLOBAL`, or `UNKNOWN`) is inferred from the CSV filename: files containing `_us` are labelled US, files containing `_global` are labelled GLOBAL.

**Column mapping**

| CSV column | Markdown header |
|------------|-----------------|
| `actual_return` | Actual Return |
| `expected_rel` | Expected (Relative) |
| `delta_rel` | Delta (Relative) |
| `r_squared` | R² |
| `confidence_delta` | Confidence Δ |
| *(others present)* | Included with formatted names |

Percentage columns are formatted as `±XX.XX%`. R² is formatted to 3 decimal places without a percent sign.

---

## score_history.py

Scores any historical trading date against a saved factor model. Equivalent to `python -m shockarb score --date`, but as a standalone script with a more explicit interface and support for `--top 0` to show all results.

**Usage**

```bash
# Score the day Russia invaded Ukraine
python utils/score_history.py --universe us --date 2022-02-24

# Score a Fed rate decision
python utils/score_history.py --universe us --date 2022-03-16

# Use a specific model file
python utils/score_history.py --universe us --date 2022-03-16 \
    --model data/model_us_20220401.json

# Show all results (not just top 20)
python utils/score_history.py --universe us --date 2022-02-24 --top 0
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--universe` / `-u` | `us` | Universe name. Must match a saved model. |
| `--date` / `-d` | *(required)* | Historical date in `YYYY-MM-DD` format. |
| `--model` / `-m` | *(latest)* | Explicit path to a model JSON. Uses the most-recently saved model by default. |
| `--data-dir` | `./data` | Override data directory. |
| `--top` / `-n` | `20` | Rows to display. Pass `0` to show all. |

**Date snapping**

If the requested date is a Saturday, Sunday, or market holiday, the script automatically snaps to the nearest prior valid trading day and logs a warning. A ±10-day download window ensures the prior close is always present even across long holiday weekends.

**Notes**

- Fetches with `auto_adjust=False` and prefers `Adj Close` when available.
- Dead tickers (all-NaN columns) are dropped before return computation.
- Output is rendered with `print_scores` from `shockarb.report`, identical to the live scoring display.

---

## shockarb/names.py — TickerReferenceResolver

This module lives in the `shockarb` package (not `utils/`) so it can be imported by `csv_to_md.py` and any other tooling. It is not a standalone script.

```python
from shockarb.names import TickerReferenceResolver

resolver = TickerReferenceResolver(
    file_paths=["data/nyse.csv", "data/nasdaq.csv"],
    cache_path="data/ticker_reference_cache.json",
)
result = resolver.get_reference(["AAPL", "MSFT", "VOO"])
# {"AAPL": {"Name": "Apple Inc.", "Industry": "..."}, ...}
```

**Constructor**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `file_paths` | `["./data/nyse.csv", "./data/nasdaq.csv"]` | Ordered list of reference files to search. `.csv` and `.parquet` are supported. Each must have `Symbol`, `Name`, `Industry` columns. |
| `cache_path` | `"./data/ticker_reference_cache.json"` | Path to the persistent JSON cache. Created on first write if absent. |

**`get_reference(tickers: list) → dict`**

Returns `{ticker: {"Name": str, "Industry": str}}` for all requested tickers.

Lookup order: JSON cache → reference files in `file_paths` order (waterfall, stops at first match) → fallback entry `{"Name": ticker, "Industry": "ETF / Unknown"}`. The cache is updated on disk after each call that resolved new tickers.

**Reference file format**

```
Symbol,Name,Industry
AAPL,Apple Inc. Common Stock,Electronic Computers
MSFT,Microsoft Corporation Common Stock,Computer Software: Prepackaged Software
```

Duplicate symbols are deduplicated (first occurrence wins). `Industry` NaN values are filled with `"Unknown"`.

**data/ticker_reference_cache.json**

A pre-populated cache is included in the repository. It covers all ETFs in the US universe (resolved as `ETF / Unknown` since exchange CSVs don't list ETFs) plus all stocks that have been looked up in previous runs. Add your own NYSE/NASDAQ reference CSVs to `data/` and run `csv_to_md.py --names` to extend it.
