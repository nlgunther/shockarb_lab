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
