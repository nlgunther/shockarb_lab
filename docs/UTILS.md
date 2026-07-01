# ShockArb Utils Reference

Command-line scripts that form the post-scoring workflow. Each runs standalone from the project root after `pip install -e .`

```
utils/
├── daily_scanner.py        Score today's tape → export CSVs         (start here each day)
├── news_scanner.py         Fetch headlines + fundamentals for top targets
├── fundamental_scanner.py  Fetch yfinance fundamentals table (imported by news_scanner);
│                           analyst targets overridden by data/analyst_overrides.csv
├── portfolio_sizer.py      Size a conviction-weighted trade ticket
├── price_trend.py          Trailing adj-close price history (reads shared parquet cache)
├── refresh_prices.py       Top-up the daily OHLCV parquet cache (gap-analysis tail-fetch)
├── eval_picks.py           Evaluate pick performance vs entry prices
├── csv_to_md.py            Convert a score CSV to a Markdown report
└── score_history.py        Score any historical date (backtesting)

get_analyst_targets.py      Fetch consensus analyst price targets from Finviz, yfinance,
                            FMP, Finnhub, or Alpha Vantage (project root, not utils/)
```

All scripts accept `--help` for a full argument listing.

---

## Typical Daily Workflow

```
4:00 pm  python -m shockarb score                   # score → data/live_alpha_us.csv (default)
     OR  python utils/daily_scanner.py              # score both US + Global universes
4:05 pm  python utils/news_scanner.py               # headlines + fundamentals → data/news.txt, data/fundamentals.csv
4:10 pm  python utils/price_trend.py --daily        # optional: 60-day adj-close matrix → data/price_trend_daily.csv
4:15 pm  python utils/portfolio_sizer.py            # trade ticket → data/portfolio_sizer.csv (default)
         python utils/csv_to_md.py data/live_alpha_us.csv  # optional: shareable markdown report
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
# Scan both US and Global regimes (if models exist)
python utils\daily_scanner.py

# Scan one regime only
python utils\daily_scanner.py --regime ukraine_shock

# Custom data directory
python utils\daily_scanner.py --data-dir /path/to/data

# NOTE: --universe flag is deprecated, use --regime instead
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--regime` / `-r` | `ukraine_shock global_ukraine_shock` | One or more regime names to scan. Must match a saved model. |
| `--data-dir` | `./data` | Override data directory. Also accepts `$SHOCK_ARB_DATA_DIR`. |
| `--out` / `-o` | *(data-dir)* | Output directory for `live_alpha_*.csv` and provenance files. Defaults to `data-dir`. |

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

**Output files** (written to `--out` directory, default `data/`)

| File | Contents |
|------|----------|
| `news.txt` | Full headlines output as printed to the terminal |
| `fundamentals.csv` | Fundamentals table: Price, Fwd P/E, TTM/Fwd EPS, Next Earnings, Ex-Div, Analyst Target |

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `[]` | Path(s) to ShockArb score CSV files. Multiple files are merged before ranking. |
| `--top` | `10` | Number of targets to pull from CSV. Ignored when `--tickers` is used. |
| `--tickers` | — | Explicit ticker list. Overrides `--csv` entirely. |
| `--sort` | `confidence_delta` | CSV column to rank by. Falls back to `delta` if the specified column is absent. |
| `--out` / `-o` | `./data` | Output directory for `news.txt` and `fundamentals.csv`. |
| `--no-out` / `-sout` | — | Suppress all file output. |

**Notes**

- Handles both the legacy flat yfinance news format and the newer nested-content format introduced around 2023. If a third, unrecognised format appears, the raw dict keys are printed for debugging.
- Network errors per ticker are caught and printed without aborting the scan.
- Fwd P/E values are cross-checked against `price / forwardEps`; values that diverge >25% are flagged with `?` to indicate a likely yfinance split-adjustment artifact.
- Ex-dividend dates older than 2 years are suppressed as `—` to avoid stale data for non-dividend payers.

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

# Size only the tickers the stock report flagged INCLUDE (bypasses CSV ranking)
python utils/portfolio_sizer.py --tickers AMAT ADI ETN --capital 10000
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `./data/live_alpha_us.csv` | Path(s) to score CSVs. Merged before ranking. |
| `--capital` | `100000` | Total dollar capital to allocate. |
| `--top` | `5` | Number of positions. Ignored when `--tickers` is set. |
| `--tickers` / `-t` | — | Size only these tickers. Bypasses CSV ranking, `--top`, and `--exclude`. Case-insensitive. Use this to act on the INCLUDE list from the stock report. |
| `--exclude` / `-e` | — | Tickers to exclude before ranking (e.g. `--exclude SNPS BSX`). Case-insensitive. Ignored when `--tickers` is set. |
| `--out` / `-o` | `./data/portfolio_sizer.csv` | Save ticket to CSV at this path. |
| `--no-out` / `-sout` | — | Suppress CSV output. |

**Output columns**

| Column | Description |
|--------|-------------|
| TICKER | Ticker symbol |
| WEIGHT | Conviction-weighted share of capital |
| ALLOCATION | Dollar amount allocated |
| CURRENT | Most recent adj_close from the shared parquet cache (same store as scoring pipeline) |
| TARGET | Take-profit limit price = `current × (1 + delta_rel)` |
| SHARES | Whole shares purchasable at current price |

**Required CSV columns:** `confidence_delta`, `delta_rel`.

**Notes**

- Allocation weight for each position = its `confidence_delta` / sum of all selected `confidence_delta` values.
- Take-profit target is the factor-model implied fair price, not a hard prediction. It represents where the stock *would* trade if the dislocation fully closed.
- Only stocks with `confidence_delta > 0` are considered. Negative signals are excluded.
- Tickers without a current price in the cache are skipped with a warning.
- Current prices are fetched via the DataCoordinator using a 7-day window, so results are correct across weekends and holidays. If prices were already cached by today's score run, no network call is made.

---

## price_trend.py

Prints a trailing price trend table for one or more tickers and optionally saves the full adj-close matrix to CSV. Uses the DataCoordinator's shared parquet cache — if prices are already cached from today's score run, no network call is made.

**Output files** (written to `data/`)

| File | Contents |
|------|----------|
| `data/price_trend.csv` (with `--csv`) | Per-ticker summary: Start, End, Chg_pct |
| `data/price_trend_daily.csv` (with `--daily`) | Full adj-close matrix (dates × tickers), suitable for upload to Claude |

**Usage**

```bash
# All tickers in live_alpha_us.csv, 60-day window
python utils/price_trend.py

# Specific tickers
python utils/price_trend.py --tickers MSFT BLK ORCL

# 30-day window
python utils/price_trend.py --tickers MSFT BLK --days 30

# Save full daily matrix for upload
python utils/price_trend.py --tickers MSFT BLK ORCL QCOM NOW --daily

# Save both summary and matrix
python utils/price_trend.py --csv --daily
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--tickers` | *(all from live_alpha_us.csv)* | One or more tickers. |
| `--days` | `60` | Trailing window in trading sessions. |
| `--csv` | `False` | Save per-ticker summary to `data/price_trend.csv`. |
| `--daily` | `False` | Save adj-close matrix to `data/price_trend_daily.csv`. |

**Notes**

- Prices come from the same DataCoordinator parquet cache as the scoring pipeline. After a score run, `price_trend.py` reads from cache at zero network cost.
- The adj-close matrix saved by `--daily` can be uploaded to Claude for context alongside a ShockArb score CSV.

---

## refresh_prices.py

Manually top-up the daily OHLCV parquet cache for specific tickers. The DataCoordinator's gap analysis means only missing dates are downloaded — already-current tickers generate zero network calls.

Run this when you want to ensure prices are cached before running `price_trend.py` or `portfolio_sizer.py` outside of a normal score workflow.

**Usage**

```bash
# Refresh specific tickers (last 30 days)
python utils/refresh_prices.py ETN HON ISRG

# Extend the window
python utils/refresh_prices.py ETN HON ISRG --days 90

# Refresh all tickers in live_alpha_us.csv
python utils/refresh_prices.py
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `tickers` | *(all from live_alpha_us.csv)* | Positional list of tickers to refresh. |
| `--days` | `30` | Calendar days of history to ensure in the cache. |

**Notes**

- Writes to `data/prices/daily/{TICKER}.parquet` — the same files that the scoring pipeline, `price_trend.py`, and `portfolio_sizer.py` all read from.
- Safe to run repeatedly; already-current files are skipped by the gap analyzer.
- Output is one line per ticker: `{TICKER}: cache up to date through YYYY-MM-DD`.

---

## get_analyst_targets.py

Fetches consensus analyst price targets for a list of tickers from your choice of data provider. Lives in the **project root** (not `utils/`). Finviz is the default — no API key required.

**Usage**

```bash
# Fetch targets for specific tickers (Finviz, no API key needed)
python get_analyst_targets.py --tickers NOW MSFT CRM INTU IDXX CPRT

# All tickers in fundamentals.csv (reads column 0 by default)
python get_analyst_targets.py

# Different provider
python get_analyst_targets.py --tickers KLAC QCOM --provider yfinance

# Different column index in the CSV
python get_analyst_targets.py --file data/fundamentals.csv --column 0
```

Output is printed to console and saved to `{provider}_analyst_data.csv` in the project root (e.g. `finviz_analyst_data.csv`). Copy the `Target_Consensus` values into the `Analyst Tgt` column of `data/fundamentals.csv`.

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--tickers` / `-t` | — | One or more tickers. Mutually exclusive with `--file`. |
| `--file` / `-f` | `fundamentals.csv` | CSV to read tickers from (if `--tickers` not given). |
| `--column` / `-c` | `0` | Zero-indexed column containing tickers in the CSV. |
| `--provider` / `-p` | `finviz` | Provider: `finviz`, `yfinance`, `fmp`, `finnhub`, `alpha_advantage`. |
| `--update-fundamentals` / `-u` | — | Patch `Analyst Tgt` in `data/fundamentals.csv` automatically. Optionally supply a path: `--update-fundamentals path/to/fundamentals.csv`. |

**Providers**

| Provider | API Key Required | Data returned |
|----------|-----------------|---------------|
| `finviz` | No | `Target_Consensus` (single consensus figure) |
| `yfinance` | No | Mean, median, high, low targets |
| `fmp` | `FMP_API_KEY` | Consensus, median, high, low |
| `finnhub` | `FINNHUB_API_KEY` | Mean, median, high, low (rate-limited: 60/min) |
| `alpha_advantage` | `AV_API_KEY` | EPS estimates (not price targets) |

**When to run:** after the stock report flags `analyst target below current price` (data quality exclusion), or after any stock split — targets become pre-split stale. Example: KLAC did a 10-for-1 split on 2026-06-12; its pre-split target of ~$1,855 must be divided by 10 (~$185) and updated in `fundamentals.csv`.

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
python utils\score_history.py --regime ukraine_shock --date 2022-02-24

# Score with the global regime
python utils\score_history.py --regime global_ukraine_shock --date 2022-02-24

# Score a Fed rate decision
python utils\score_history.py --regime ukraine_shock --date 2022-03-16

# Use a specific model file
python utils\score_history.py --regime ukraine_shock --date 2022-03-16 \
    --model data/model_ukraine_shock_20220401.json

# Show all results (not just top 20)
python utils\score_history.py --regime ukraine_shock --date 2022-02-24 --top 0

# NOTE: --universe flag is deprecated, use --regime instead
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--regime` / `-r` | `ukraine_shock` | Regime name. Must match a saved model. |
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

## maintain_ticker_cache.py

Maintains the ticker reference cache (`data/ticker_reference_cache.json`) used by `csv_to_md.py` to resolve company names and industry classifications. The cache populates lazily as tickers are looked up, but it can accumulate "stub" entries (fallback placeholders) when new tickers lack matching reference data.

**When to run this**

- After downloading fresh NYSE/NASDAQ reference CSVs (from NASDAQ or NYSE websites)
- When you notice many tickers showing "ETF / Unknown" in markdown reports despite being well-known companies
- Before a major reporting run to ensure all tickers have current data

**Usage**

```bash
# Full maintenance pass (recommended): update, fix stubs, and sort
python utils/maintain_ticker_cache.py --update --fix-stubs --sort

# Just fix stubs (upgrade placeholder entries from reference CSVs)
python utils/maintain_ticker_cache.py --fix-stubs

# Preview changes without writing to disk
python utils/maintain_ticker_cache.py --fix-stubs --dry-run

# Custom cache and reference directory
python utils/maintain_ticker_cache.py \
    --update --fix-stubs --sort \
    --cache data/my_cache.json \
    --ref-dir data/
```

**Operations**

| Operation | Description |
|-----------|-------------|
| `--update` | Add tickers present in NYSE/NASDAQ CSVs but missing from the cache. Does not touch existing cache entries. |
| `--fix-stubs` | Find stub entries (Name = ticker, Industry = "ETF / Unknown") and replace with real CSV data where available. This is the key operation for fixing "Unknown" industries. |
| `--sort` | Rewrite the cache in alphabetical key order for readability and consistency. |

All operations are applied in the order listed above, so `--fix-stubs` always runs against an already-updated cache.

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--cache` | `./data/ticker_reference_cache.json` | Path to the ticker reference cache JSON. |
| `--ref-dir` | `./data` | Directory containing NYSE/NASDAQ reference CSVs (any file with `nyse` or `nasdaq` in the name, case-insensitive). |
| `--update` | `False` | Add missing tickers from CSVs to the cache. |
| `--fix-stubs` | `False` | Upgrade stub entries with real CSV data. |
| `--sort` | `False` | Rewrite cache in alphabetical order. |
| `--dry-run` | `False` | Report changes without writing to disk. |

**Reference CSV format**

The script searches `--ref-dir` for files matching `*nyse*.csv` or `*nasdaq*.csv` (case-insensitive). Each file must contain columns:

```
Symbol,Name,Industry
AAPL,Apple Inc. Common Stock,Electronic Computers
MSFT,Microsoft Corporation Common Stock,Computer Software: Prepackaged Software
```

Duplicate symbols are deduplicated (NYSE wins over NASDAQ if both have the same symbol).

**Example workflow**

1. Download the latest NASDAQ and NYSE reference CSVs from the exchanges and save to `data/`.
2. Run the full maintenance pass:
   ```bash
   python utils/maintain_ticker_cache.py --update --fix-stubs --sort
   ```
3. Verify with a dry-run first if unsure:
   ```bash
   python utils/maintain_ticker_cache.py --fix-stubs --dry-run
   ```
4. Re-run `csv_to_md.py` on any stored CSVs to refresh their markdown reports with updated industry data.

**Notes**

- The cache is persistent and grows over time. It's safe to run `--fix-stubs --sort` regularly — operations are idempotent.
- Stubs are identified as entries where `Name == ticker` AND `Industry == "ETF / Unknown"`. Real entries are never overwritten by `--update` alone.
- ETFs (like VOO, TLT, GLD) intentionally remain as "ETF / Unknown" since they don't appear in stock exchange CSVs. This is expected behavior.

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
