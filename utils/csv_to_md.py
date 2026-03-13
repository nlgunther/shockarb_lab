"""
ShockArb Alpha Sheet → Markdown Report Converter.

Reads a ShockArb score CSV and produces a human-readable Markdown table,
optionally enriched with company names and industries from local exchange
reference files.

Name resolution is delegated to shockarb.names.TickerReferenceResolver,
which searches a JSON cache first, then the exchange CSVs in order.
See names.py for full lookup semantics.

Key file paths  (ALL filenames are constants — edit here, nowhere else)
-----------------------------------------------------------------------
  TICKER_CACHE   ticker_reference_cache.json   — JSON disk cache
  EXCHANGE_CSVS  nyse_1668526574444.csv        — NYSE reference (checked first)
                 nasdaq_1668526380140.csv       — NASDAQ reference

Each exchange CSV must contain at minimum: Symbol, Name, Industry.

Usage
-----
    # Basic — resolves names, writes report next to the CSV
    python utils/csv_to_md.py data/live_alpha_us.csv

    # Explicit output path
    python utils/csv_to_md.py data/live_alpha_us.csv --out reports/today.md

    # Skip name resolution (faster, no reference files needed)
    python utils/csv_to_md.py data/live_alpha_us.csv --no-names

    # Override data directory (cache and CSVs are looked up relative to it)
    python utils/csv_to_md.py data/live_alpha_us.csv --data-dir /mnt/shockarb/data
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Project root — ensures 'shockarb' is importable without pip install -e .
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from shockarb.names import TickerReferenceResolver
except ImportError:
    TickerReferenceResolver = None
    logger.warning("shockarb.names not importable — name resolution unavailable.")

# =============================================================================
# File paths — ALL filenames live here.  Change them in one place only.
# =============================================================================

_DEFAULT_DATA_DIR = "./data"

# JSON disk cache: ticker → {"Name": ..., "Industry": ...}
TICKER_CACHE = "ticker_reference_cache.json"

# Exchange reference CSVs searched in this order (NYSE before NASDAQ).
# Supports .csv and .parquet — see names.py for format requirements.
EXCHANGE_CSVS = [
    "nyse_1668526574444.csv",
    "nasdaq_1668526380140.csv",
]

# =============================================================================
# Column definitions — formatting rules and display names in one place.
# To add a new column: add it to _COLUMN_FORMAT and _DISPLAY_NAMES below.
# =============================================================================

# "pct"   → ±XX.XX%  |  "r2" → 0.NNN  |  None → omit from output
_COLUMN_FORMAT: dict[str, str] = {
    "actual_return":    "pct",
    "expected_return":  "pct",
    "expected_rel":     "pct",
    "expected_abs":     "pct",
    "delta":            "pct",
    "delta_rel":        "pct",
    "delta_abs":        "pct",
    "residual_vol":     "pct",
    "confidence_delta": "pct",
    "r_squared":        "r2",
}

_DISPLAY_NAMES: dict[str, str] = {
    "actual_return":    "Actual Return",
    "expected_return":  "Expected Return",
    "expected_rel":     "Expected (Relative)",
    "expected_abs":     "Expected (Absolute)",
    "delta":            "Delta",
    "delta_rel":        "Delta (Relative)",
    "delta_abs":        "Delta (Absolute)",
    "residual_vol":     "Residual Vol",
    "confidence_delta": "Confidence Δ",
    "r_squared":        "R²",
}

# Universe label derived from the CSV filename suffix.
_UNIVERSE_LABELS = {"_us": "US", "_global": "GLOBAL"}

# Markdown legend block — edit the text here if signal definitions change.
_REPORT_LEGEND = """\
> **How to read this report:**
> * **Delta (Relative):** Pure arbitrage signal — positive means the stock \
fell more than macro factors justified.
> * **R²:** Model's historical fit for this stock. Higher = more reliable baseline.
> * **Confidence Δ:** Conviction-weighted signal (Delta × R²)."""

# =============================================================================
# Formatting helpers
# =============================================================================

def _fmt_pct(val: float) -> str:
    """Decimal fraction → ±XX.XX%, or 'N/A' for missing values."""
    return "N/A" if pd.isna(val) else f"{val * 100:+.2f}%"


def _fmt_r2(val: float) -> str:
    """R-squared → 0.NNN, or 'N/A' for missing values."""
    return "N/A" if pd.isna(val) else f"{val:.3f}"


_FORMATTERS = {"pct": _fmt_pct, "r2": _fmt_r2}


def _format_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a display-ready DataFrame with formatted values and renamed headers.

    Only columns present in _COLUMN_FORMAT are included; all others (e.g.
    internal pipeline columns) are silently dropped.  Headers are renamed
    via _DISPLAY_NAMES.  Index is labelled 'Ticker'.
    """
    out = pd.DataFrame(index=df.index)
    for col, fmt_key in _COLUMN_FORMAT.items():
        if col in df.columns:
            out[col] = df[col].apply(_FORMATTERS[fmt_key])
    out = out.rename(columns=_DISPLAY_NAMES)
    out.index.name = "Ticker"
    return out


def _universe_label(filename: str) -> str:
    """Derive 'US' / 'GLOBAL' / 'UNKNOWN' from the score CSV filename."""
    fn = filename.lower()
    return next((label for suffix, label in _UNIVERSE_LABELS.items() if suffix in fn), "UNKNOWN")

# =============================================================================
# Report generator
# =============================================================================

def generate_markdown_report(
    csv_path: str,
    output_path: str | None = None,
    enrich_names: bool = True,
    data_dir: str = _DEFAULT_DATA_DIR,
) -> None:
    """
    Convert a ShockArb score CSV to a Markdown report.

    Parameters
    ----------
    csv_path     : score CSV produced by ``shockarb score --output``
    output_path  : destination .md file; defaults to csv_path with .md extension
    enrich_names : if True, prepend Company and Industry columns
    data_dir     : directory containing TICKER_CACHE and EXCHANGE_CSVS

    Files used (relative to data_dir)
    ----------------------------------
        {data_dir}/{TICKER_CACHE}       — JSON cache (read + updated)
        {data_dir}/{EXCHANGE_CSVS[0]}   — NYSE reference
        {data_dir}/{EXCHANGE_CSVS[1]}   — NASDAQ reference
    """
    logger.info(f"Loading alpha sheet: {csv_path}")
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as exc:
        logger.error(f"Cannot read {csv_path}: {exc}")
        return

    if "confidence_delta" in df.columns:
        df = df.sort_values("confidence_delta", ascending=False)

    display_df = _format_columns(df)

    # Prepend Company / Industry columns when name resolution is available.
    if enrich_names:
        if TickerReferenceResolver is None:
            logger.warning("Name resolution skipped — shockarb.names not importable.")
        else:
            resolver = TickerReferenceResolver(
                file_paths = [os.path.join(data_dir, f) for f in EXCHANGE_CSVS],
                cache_path = os.path.join(data_dir, TICKER_CACHE),
            )
            ref = resolver.get_reference(df.index.tolist())
            # Insert leftmost so Company/Industry appear before numeric columns.
            display_df.insert(0, "Industry", [ref[t]["Industry"] for t in df.index])
            display_df.insert(0, "Company",  [ref[t]["Name"]     for t in df.index])

    filename = os.path.basename(csv_path)
    md_content = "\n".join([
        f"# ⚡ ShockArb Alpha Report ({_universe_label(filename)})",
        f"**Source:** `{filename}`\n",
        _REPORT_LEGEND + "\n",
        "### Actionable Targets\n",
        display_df.to_markdown(),
    ])

    if not output_path:
        output_path = os.path.splitext(csv_path)[0] + ".md"

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.success(f"Report written: {output_path}")
    except Exception as exc:
        logger.error(f"Failed to write {output_path}: {exc}")

# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a ShockArb score CSV to a Markdown report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv_file",
        help="Path to the ShockArb score CSV")
    parser.add_argument("--out", default=None,
        help="Output .md path (default: same name/dir as CSV with .md extension)")
    parser.add_argument("--no-names", action="store_true",
        help="Skip company name and industry resolution")
    parser.add_argument("--data-dir", default=_DEFAULT_DATA_DIR,
        help=f"Directory containing reference CSVs and cache (default: {_DEFAULT_DATA_DIR})")
    args = parser.parse_args()

    generate_markdown_report(
        csv_path     = args.csv_file,
        output_path  = args.out,
        enrich_names = not args.no_names,
        data_dir     = args.data_dir,
    )
