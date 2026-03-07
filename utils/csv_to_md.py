"""
ShockArb Alpha Sheet → Markdown Report Converter.

Reads a ShockArb score CSV and produces a human-readable Markdown table,
optionally enriched with company names and industries from local exchange
reference files.

Usage examples
--------------
    # Basic conversion — includes company names by default
    python utils/csv_to_md.py data/live_alpha_us.csv

    # Save to a specific path
    python utils/csv_to_md.py data/live_alpha_us.csv --out reports/today.md

    # Skip name resolution (faster, no reference CSVs needed)
    python utils/csv_to_md.py data/live_alpha_us.csv --no-names

Reference CSV format
--------------------
Each file must contain at minimum three columns: Symbol, Name, Industry.
Files are searched in the order they appear in --ref-dir (NYSE then NASDAQ
by convention).  Results are cached in data/ticker_reference_cache.json so
subsequent runs are faster.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from loguru import logger

import sys
import pathlib

# Ensure the project root (parent of utils/) is on sys.path so
# 'shockarb' is importable without needing pip install -e .
_project_root = pathlib.Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from shockarb.names import TickerReferenceResolver
except ImportError:
    TickerReferenceResolver = None
    logger.warning("Could not load TickerReferenceResolver — name resolution unavailable.")


# =============================================================================
# Helpers
# =============================================================================

def _fmt_pct(val: float) -> str:
    """Format a decimal fraction as ±XX.XX%."""
    if pd.isna(val):
        return "N/A"
    return f"{val * 100:+.2f}%"


def _find_reference_csvs(ref_dir: str) -> list[str]:
    """
    Return NYSE and NASDAQ reference CSVs from ref_dir, NYSE first.
    Only files whose names contain 'nyse' or 'nasdaq' are included —
    other CSVs in the directory (e.g. live_alpha_us.csv) are ignored.
    """
    if not os.path.isdir(ref_dir):
        return []
    files = [f for f in os.listdir(ref_dir) if f.lower().endswith(".csv")]
    nyse   = sorted(f for f in files if "nyse"   in f.lower())
    nasdaq = sorted(f for f in files if "nasdaq" in f.lower())
    return [os.path.join(ref_dir, f) for f in nyse + nasdaq]


# =============================================================================
# Report generator
# =============================================================================

PCT_COLS = [
    "actual_return", "expected_return", "expected_rel", "expected_abs",
    "delta", "delta_rel", "delta_abs", "residual_vol", "confidence_delta",
]

RENAME_MAP = {
    "actual_return":   "Actual Return",
    "expected_return": "Expected Return",
    "expected_rel":    "Expected (Relative)",
    "expected_abs":    "Expected (Absolute)",
    "delta":           "Delta",
    "delta_rel":       "Delta (Relative)",
    "delta_abs":       "Delta (Absolute)",
    "r_squared":       "R²",
    "residual_vol":    "Residual Vol",
    "confidence_delta":"Confidence Δ",
}


def generate_markdown_report(
    csv_path: str,
    output_path: str | None = None,
    resolve_names: bool = True,
    ref_dir: str = "./data",
    cache_path: str = "./data/ticker_reference_cache.json",
) -> None:
    """
    Convert a ShockArb score CSV to a Markdown report.

    Parameters
    ----------
    csv_path : str
        Path to the score CSV produced by ``shockarb score --output``.
    output_path : str, optional
        Destination .md file.  Defaults to csv_path with .md extension.
    resolve_names : bool
        If True, look up company names and industries from ref_dir CSVs.
    ref_dir : str
        Directory containing exchange reference CSVs (Symbol, Name, Industry).
    cache_path : str
        Path to the JSON cache for resolved tickers.
    """
    logger.info(f"Loading alpha sheet: {csv_path}")
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as exc:
        logger.error(f"Cannot read {csv_path}: {exc}")
        return

    if "confidence_delta" in df.columns:
        df = df.sort_values("confidence_delta", ascending=False)

    md_df = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------
    # Optional name/industry injection
    # ------------------------------------------------------------------
    if resolve_names:
        if TickerReferenceResolver is None:
            logger.error("Name resolution skipped: shockarb package not importable.")
        else:
            ref_files = _find_reference_csvs(ref_dir)
            if not ref_files:
                logger.warning(
                    f"No NYSE/NASDAQ reference CSVs found in '{ref_dir}' — "
                    "names will fall back to ticker_reference_cache.json only."
                )
            else:
                logger.info(f"Resolving names from {len(ref_files)} file(s) in {ref_dir}")
            resolver = TickerReferenceResolver(
                file_paths=ref_files,
                cache_path=cache_path,
            )
            ref_map = resolver.get_reference(df.index.tolist())
            md_df["Company"] = df.index.map(lambda t: ref_map.get(t, {}).get("Name", t))
            md_df["Industry"] = df.index.map(lambda t: ref_map.get(t, {}).get("Industry", "—"))

    # ------------------------------------------------------------------
    # Format columns
    # ------------------------------------------------------------------
    for col in PCT_COLS:
        if col in df.columns:
            md_df[col] = df[col].apply(_fmt_pct)

    if "r_squared" in df.columns:
        md_df["r_squared"] = df["r_squared"].apply(lambda x: f"{x:.3f}")

    md_df = md_df.rename(columns=RENAME_MAP)
    md_df.index.name = "Ticker"

    # ------------------------------------------------------------------
    # Universe label from filename
    # ------------------------------------------------------------------
    filename = os.path.basename(csv_path)
    fn_lower = filename.lower()
    if "_global" in fn_lower:
        universe = "GLOBAL"
    elif "_us" in fn_lower:
        universe = "US"
    else:
        universe = "UNKNOWN"

    # ------------------------------------------------------------------
    # Assemble Markdown
    # ------------------------------------------------------------------
    lines = [
        f"# ⚡ ShockArb Alpha Report ({universe})",
        f"**Source:** `{filename}`\n",
        "> **How to read this report:**",
        "> * **Delta (Relative):** Pure arbitrage signal — positive means the stock fell "
        "more than macro factors justified.",
        "> * **R²:** Model's historical fit for this stock. Higher = more reliable baseline.",
        "> * **Confidence Δ:** Conviction-weighted signal (Delta × R²).\n",
        "### Actionable Targets\n",
        md_df.to_markdown(),
    ]

    md_content = "\n".join(lines)

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
    parser.add_argument("csv_file", help="Path to the ShockArb score CSV")
    parser.add_argument("--out", default=None, help="Output .md path (default: same dir as CSV)")
    parser.add_argument(
        "--no-names", action="store_true",
        help="Skip company name and industry resolution (faster, no reference CSVs needed)",
    )
    parser.add_argument(
        "--ref-dir", default="./data",
        help="Directory containing NYSE/NASDAQ reference CSVs (default: ./data)",
    )
    parser.add_argument(
        "--cache", default="./data/ticker_reference_cache.json",
        help="Path to the ticker name cache JSON (default: ./data/ticker_reference_cache.json)",
    )
    args = parser.parse_args()

    generate_markdown_report(args.csv_file, args.out, not args.no_names, args.ref_dir, args.cache)
