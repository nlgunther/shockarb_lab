"""
Canonical paths for the ShockArb pipeline.

All pipeline file paths — inputs and outputs — are defined here as absolute
pathlib.Path objects anchored to the project root via ``__file__``.  Import
from this module; do not define path literals elsewhere.  See docs/PATHS.md
for the full design rationale.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — always shockarb_lab/, regardless of working directory
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent   # utils/paths.py → utils/ → shockarb_lab/

# ---------------------------------------------------------------------------
# Directory roots
# ---------------------------------------------------------------------------
DATA    = _ROOT / "data"
REPORTS = _ROOT / "reports"

# ---------------------------------------------------------------------------
# Pipeline inputs
# ---------------------------------------------------------------------------
MARKET_SNAPSHOT = DATA / "market_snapshot.json"
LIVE_ALPHA_US   = DATA / "live_alpha_us.csv"
FUNDAMENTALS    = DATA / "fundamentals.csv"
NEWS            = DATA / "news.txt"

# ---------------------------------------------------------------------------
# Pipeline outputs
# ---------------------------------------------------------------------------
MARKET_REPORT          = REPORTS / "market_report.md"
MARKET_REPORT_INTRADAY = REPORTS / "market_report_intraday.md"
STOCK_REPORT           = REPORTS / "stock_report.md"
REPORTS_DIR            = REPORTS

# ---------------------------------------------------------------------------
# Price trend outputs
# ---------------------------------------------------------------------------
PRICE_TREND_DAILY   = DATA / "price_trend_daily.csv"   # adj-close matrix (dates × tickers)
PRICE_TREND_SUMMARY = DATA / "price_trend.csv"          # per-ticker summary (start/end/chg%)

# ---------------------------------------------------------------------------
# Watchlist — user-maintained ticker list for the watchlist_news scanner
# ---------------------------------------------------------------------------
WATCHLIST = DATA / "watchlist.txt"

# ---------------------------------------------------------------------------
# Sticky CLI state
# ---------------------------------------------------------------------------
# Mirrors shockarb/cli.py's .shockarb_regime sticky-file pattern.
STOCKFIT_RVOL_FILE = DATA / ".stockfit_rvol"

# ---------------------------------------------------------------------------
# ShockArb position tracking — portfolio_sizer.py --positions / --execute
# ---------------------------------------------------------------------------
# POSITION_MARK_OUT mirrors portfolio_sizer.py's own _DEFAULT_OUT: an
# ephemeral snapshot, overwritten on every --positions run.
# POSITION_LOG is the durable counterpart — append-only, never overwritten,
# written only when --execute is passed. This is the audit trail that lets
# you reconstruct "what did ShockArb say about this ticker the day I bought
# it" without re-deriving it from archived reports.
POSITION_MARK_OUT = DATA / "shockarb_position_mark.csv"
POSITION_LOG       = DATA / "shockarb_position_log.csv"


# ---------------------------------------------------------------------------
# Reference data (ticker → company name/sector/industry resolution)
# ---------------------------------------------------------------------------
# Filenames only (not full paths) — combined with a data directory by callers
# that support a configurable --data-dir, e.g. csv_to_md.py and
# shockarb.reference_sync. Single source of truth: edit here, nowhere else.
TICKER_CACHE_FILENAME = "ticker_reference_cache.json"
EXCHANGE_CSV_FILENAMES = [
    "nyse_1668526574444.csv",
    "nasdaq_1668526380140.csv",
]
