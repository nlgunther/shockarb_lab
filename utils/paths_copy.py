"""
Canonical paths for the ShockArb pipeline.

All pipeline input and output paths are defined here. Individual modules
import from this file instead of carrying their own string literals.

To relocate the pipeline inputs folder (e.g. to data/pipeline_inputs/),
change _INPUTS — nothing else needs to touch.
"""

from pathlib import Path

_ROOT   = Path(__file__).resolve().parent.parent   # shockarb_lab/
_DATA   = _ROOT / "data"

# Pipeline inputs — all four files the report commands read.
# Currently mirrors data/ directly; see HIL_todo.md for the planned
# migration to data/pipeline_inputs/.
_INPUTS = _DATA

MARKET_SNAPSHOT = _INPUTS / "market_snapshot.json"
LIVE_ALPHA_US   = _INPUTS / "live_alpha_us.csv"
FUNDAMENTALS    = _INPUTS / "fundamentals.csv"
NEWS            = _INPUTS / "news.txt"

# Pipeline outputs
MARKET_REPORT     = _DATA / "market_report.md"
MARKET_REPORT_INTRADAY = _DATA / "market_report_intraday.md"
STOCK_REPORT      = _DATA / "stock_report.md"
REPORTS_DIR       = _DATA / "reports"
