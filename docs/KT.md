# ShockArb — Knowledge Transfer

*Updated after each session. Captures decisions and context not derivable from reading the code.  
For API details see API.md; for quick commands see CHEATSHEET.md.*

---

## What ShockArb Does

Identifies stocks mispriced by geopolitical panic selling. During a shock event the broad market, energy sector, and bond/gold complex all move in characteristic directions. ShockArb decomposes each stock's actual return into a macro-factor-explained part and a residual. A large positive residual (stock fell more than factors imply) is a mean-reversion candidate — the "ShockArb thesis".

**Primary signal:** `confidence_delta = delta_rel × r_squared`  
Sort descending, filter `r_squared > 0.50`, act on the top 5–10.

---

## Architecture — The Key Split

```
engine.py    — pure math, zero I/O. SVD + OLS. Takes DataFrames in, returns DataFrames out.
pipeline.py  — all I/O. fetch, cache, build, save, load, export, add_assets.
cli.py       — glue only. Parses args, calls pipeline/engine, formats output.
regimes.py   — regime catalogue. Adding a regime is adding a named dataclass — nothing else.
config.py    — UniverseConfig (what), ExecutionConfig (how).
cache.py     — parquet-based OHLCV caching via CacheManager.
report.py    — terminal display. print_scores, print_model_state, print_live_alpha.
store.py     — ShockArbStore: parquet file management for the datamgr coordinator.
```

**Why engine.py has zero I/O:** swapping the data source (yfinance → Bloomberg) means touching `pipeline.py` only. The math is untouched. This boundary has been deliberately enforced — don't put file reads or network calls in engine.py.

---

## The Build / Score Lifecycle

`build` is expensive (downloads ~35 days of calibration prices, fits SVD). Run it once per regime and save the result. `score` is cheap — it loads the frozen JSON and fetches only today's live prices.

```
build  →  save JSON  →  [days pass]  →  load JSON  →  fetch live prices  →  score
```

Model files live in `data_dir` (default `./data`, override with `SHOCK_ARB_DATA_DIR`):

```
data/
├── ukraine_shock_us_20260510_143022.json         ← frozen US model
├── global_ukraine_shock_global_20260510_143055.json
├── .shockarb_regime                              ← sticky regime (one line, regime name)
├── cache/                                        ← parquet OHLCV cache
└── backups/                                      ← pre-mutation parquet backups (7-day)
```

`find_latest_model(name)` picks the most-recent JSON matching the universe name. The sticky file is stored in `data_dir` (not the project root) so different data directories can have independent sticky regimes.

---

## Regimes

A regime is a `HistoricFactorModel`: a `UniverseConfig` (tickers + calibration window) plus narrative metadata. The registry is the single source of truth.

| Regime | Universe name | ETFs | Stocks | Factors | Window |
|--------|--------------|------|--------|---------|--------|
| `ukraine_shock` | `us` | 19 | 66 | 3 | 2022-02-10 → 2022-03-31 |
| `global_ukraine_shock` | `global` | 14 | 15 | 3 | 2022-02-10 → 2022-03-31 |
| `gulf_war_recovery` | `us_recovery` | 5 | 27 | 4 | 1991-03-01 → 1991-06-28 |
| `liberation_day_recovery` | `us_lib_day` | 19 | 66 | 3 | 2025-04-01 → 2025-07-31 |

**Why `global_ukraine_shock` uses universe name `"global"` not `"ukraine_shock_global"`:** `find_latest_model` searches by the universe `name` field in the JSON. Keeping it short and distinct avoids ambiguous glob matches.

**Why `gulf_war_recovery` has 4 factors:** 1991 had a distinct recovery dynamic (market + energy + defensive rotation + recovery axis) that 3 factors didn't capture cleanly in testing.

**Adding a new regime:** define a `HistoricFactorModel` in `regimes.py`, add it to `REGIME_REGISTRY`. The CLI and pipeline pick it up automatically — no other files change.

---

## The `add-asset` Workflow

Adds new tickers to an existing model **without refitting**. Projects the new stock onto the existing factor basis using OLS — faster than a full rebuild but slightly less accurate (the new ticker doesn't influence the factor directions).

```bash
python -m shockarb add-asset SHOP COIN --save
```

`pipeline.add_assets()` fetches both the ETF calibration prices (to reconstruct the factor return series from the stored `_Vt` / `_etf_mean`) and the new ticker prices, then calls `model.add_asset()` in-place. Saving with `--save` or `pipeline.save_model()` persists the change.

**When to use `add-asset` vs. full refit:**  
- `add-asset` — you want to score a new ticker quickly and don't need it to influence factor directions  
- Full refit (`build`) — you're adding a ticker that should shape the factor structure (e.g. a new ETF for the basis), or you want maximum accuracy

---

## DataCoordinator / datamgr

`datamgr/` is a provider-agnostic data management layer introduced in a later refactor. `pipeline.py` uses it via `_coordinator()` for all price fetching. The coordinator deduplicates requests, handles caching, and routes to the right provider (currently yfinance).

`datamgr/coordinator.py` is the entry point. `store.py` (in `shockarb/`) is the `ShockArbStore` implementation of the `DataStore` interface — it wraps the parquet files in `data_dir/cache/`.

---

## Test Suite

287 tests, all flat in `tests/`. Run with `pytest tests\ -q`.

Key fixture hierarchy in `conftest.py`:
- `sample_etf_returns` → 36 days × 5 ETFs, synthetic crisis structure
- `sample_stock_returns` → 36 days × 5 stocks, aligned
- `fitted_model` → `FactorModel` fitted with `n_components=2`, ready for scoring tests
- `mock_model` → 3-ETF / 2-stock model saved to `temp_dir` as `"us"`, for CLI/pipeline tests
- `InMemoryStore` → test double for `DataStore`, used by coordinator tests

---

## File Integrity

`MANIFEST.txt` tracks SHA-256 prefixes (16 hex chars) for all source and test files, plus a bundle hash (hash-of-hashes over sorted file hashes). `verify_install.py` reads it and flags drift. Hashes are computed with CRLF normalisation for Windows compatibility.

Regenerate after any code change:
```bash
python verify_install.py --regenerate
```

---

## Known Design Debt / Limitations

- **`gulf_war_recovery` tickers are placeholders.** The `check_1991.py` script was written to validate which 1991-era tickers yfinance actually has data for, but the regime hasn't been validated against real data yet. Treat it as a skeleton.
- **Small calibration window (~35 trading days).** A single stock-specific event during calibration can contaminate that ticker's loadings. Inspect R² before trusting any signal.
- **No position sizing.** ShockArb generates ranked signals only.
- **`liberation_day_recovery` end date is `2025-07-31`** — this window is still in the future as of the last session. Update it once the period is complete and you have a view on when normalization finished.

---

## Session Log

| Date | What changed |
|------|-------------|
| 2026-05-10 | Added `GLOBAL_UKRAINE_SHOCK` regime; fixed CLI `--universe global` deprecation warning; updated all four docs |
| 2026-05-10 | Added `FactorModel.add_asset()`, `pipeline.add_assets()`, `add-asset` CLI subcommand; fixed `to_dict()` to preserve `etf_mean` |
| 2026-05-11 | Fixed stale `.pyc` cache causing `SyntaxError` in `regimes.py`; updated all four docs to cover `add-asset`; created this KT |
| 2026-05-11 | Added `FactorModel.remove_asset()`, `remove-asset` CLI subcommand, 13 new tests; 300 tests passing |
