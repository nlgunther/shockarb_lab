"""
Canonical paths for the ShockArb pipeline.

All pipeline file paths — inputs and outputs — are defined here as relative
pathlib.Path objects. Import from this module; do not define path literals
elsewhere. See docs/PATHS.md for the full design rationale.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory roots  (relative to utils/ — the required working directory)
# ---------------------------------------------------------------------------
DATA    = Path("../data")
REPORTS = Path("../reports")

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
