"""
maintain_ticker_cache.py — ShockArb ticker reference cache maintenance.

Three operations, run independently or together:

  --update      Add tickers that are missing from the cache but present in
                the reference CSVs (NYSE / NASDAQ).

  --fix-stubs   Find stub entries (Name == ticker symbol, Industry == "ETF /
                Unknown") and replace them with real data from the CSVs if
                available.

  --sort        Rewrite the cache in alphabetical key order.

Operations are applied in the order above when multiple flags are given,
so --fix-stubs always runs against an already-updated cache, and --sort
is always the final write.

Usage examples
--------------
    # Full maintenance pass (recommended daily):
    python utils/maintain_ticker_cache.py --update --fix-stubs --sort

    # Just sort the cache:
    python utils/maintain_ticker_cache.py --sort

    # Report stubs without modifying the cache:
    python utils/maintain_ticker_cache.py --fix-stubs --dry-run

    # Point at non-default paths:
    python utils/maintain_ticker_cache.py --update --fix-stubs --sort \\
        --cache data/ticker_reference_cache.json \\
        --ref-dir data/
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Optional

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Make 'shockarb' importable when run directly from the repo root
# ---------------------------------------------------------------------------
_project_root = pathlib.Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ===========================================================================
# Helpers
# ===========================================================================

def _find_reference_csvs(ref_dir: str) -> list[str]:
    """Return NYSE then NASDAQ CSVs from ref_dir (same logic as csv_to_md.py)."""
    if not os.path.isdir(ref_dir):
        logger.error(f"Reference directory not found: {ref_dir}")
        return []
    files = [f for f in os.listdir(ref_dir) if f.lower().endswith(".csv")]
    nyse   = sorted(f for f in files if "nyse"   in f.lower())
    nasdaq = sorted(f for f in files if "nasdaq" in f.lower())
    paths  = [os.path.join(ref_dir, f) for f in nyse + nasdaq]
    if not paths:
        logger.warning(f"No NYSE/NASDAQ CSVs found in '{ref_dir}'.")
    return paths


def _load_cache(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        logger.warning(f"Cache not found at {cache_path} — starting with empty cache.")
        return {}
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error(f"Cannot read cache ({exc}) — aborting to avoid data loss.")
        sys.exit(1)


def _save_cache(cache: dict, cache_path: str, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] Would write cache — skipping disk write.")
        return
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=4)
    logger.success(f"Cache written: {cache_path}  ({len(cache):,} entries)")


def _load_ref_df(csv_path: str) -> Optional[pd.DataFrame]:
    """Load a reference CSV into a Symbol-indexed DataFrame."""
    if not os.path.exists(csv_path):
        logger.warning(f"Reference file not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["Symbol", "Name", "Industry"])
        df["Industry"] = df["Industry"].fillna("Unknown")
        df = df.drop_duplicates(subset=["Symbol"]).set_index("Symbol")
        logger.debug(f"Loaded {len(df):,} rows from {os.path.basename(csv_path)}")
        return df
    except Exception as exc:
        logger.error(f"Failed to load {csv_path}: {exc}")
        return None


def _is_stub(ticker: str, entry: dict) -> bool:
    """
    Return True if the cache entry is a stub — i.e. the resolver fell back to
    the default placeholder rather than finding real data:

        {"Name": "<same as ticker>", "Industry": "ETF / Unknown"}
    """
    return (
        entry.get("Name") == ticker
        and entry.get("Industry") == "ETF / Unknown"
    )


# ===========================================================================
# Operations
# ===========================================================================

def run_update(cache: dict, ref_csvs: list[str], dry_run: bool) -> int:
    """
    Add tickers that are absent from the cache but present in the CSVs.

    This does NOT touch existing cache entries (even stubs) — use
    run_fix_stubs() for that.  Returns the count of new entries added.
    """
    missing_from_cache = set()   # We can only know what's missing once we
                                 # load the CSVs, so we report what we added.
    added = 0

    for csv_path in ref_csvs:
        df = _load_ref_df(csv_path)
        if df is None:
            continue

        for symbol in df.index:
            if symbol not in cache:
                entry = {
                    "Name":     str(df.loc[symbol, "Name"]),
                    "Industry": str(df.loc[symbol, "Industry"]),
                }
                if not dry_run:
                    cache[symbol] = entry
                logger.info(f"[update] Added: {symbol} → {entry['Name']} ({entry['Industry']})")
                added += 1

    if added:
        logger.success(f"[update] {added} new ticker(s) added to cache.")
    else:
        logger.info("[update] Cache already contains all tickers found in reference CSVs.")
    return added


def run_fix_stubs(cache: dict, ref_csvs: list[str], dry_run: bool) -> int:
    """
    Find stub entries and replace them with real CSV data where available.

    A stub is any entry where Name == the ticker symbol and Industry ==
    "ETF / Unknown" — the fallback placeholder written by TickerReferenceResolver
    when a ticker is not found in any reference file.

    Returns the count of stubs that were upgraded.
    """
    # Identify all stubs first
    stubs = {t: e for t, e in cache.items() if _is_stub(t, e)}

    if not stubs:
        logger.info("[fix-stubs] No stub entries found in cache.")
        return 0

    logger.info(f"[fix-stubs] Found {len(stubs)} stub(s): {sorted(stubs)}")

    # Build a combined lookup from all reference CSVs (NYSE wins over NASDAQ
    # for the same symbol, matching the waterfall order in the resolver).
    combined: dict[str, dict] = {}
    for csv_path in ref_csvs:
        df = _load_ref_df(csv_path)
        if df is None:
            continue
        for symbol in df.index:
            if symbol not in combined:   # first file wins (NYSE before NASDAQ)
                combined[symbol] = {
                    "Name":     str(df.loc[symbol, "Name"]),
                    "Industry": str(df.loc[symbol, "Industry"]),
                }

    upgraded = 0
    still_stubs = []

    for ticker in sorted(stubs):
        if ticker in combined:
            new_entry = combined[ticker]
            logger.info(
                f"[fix-stubs] Upgrading {ticker}: "
                f'"{stubs[ticker]["Name"]}" → "{new_entry["Name"]}" '
                f'({new_entry["Industry"]})'
            )
            if not dry_run:
                cache[ticker] = new_entry
            upgraded += 1
        else:
            still_stubs.append(ticker)

    if upgraded:
        logger.success(f"[fix-stubs] {upgraded} stub(s) upgraded with real data.")
    if still_stubs:
        logger.warning(
            f"[fix-stubs] {len(still_stubs)} stub(s) not found in any reference CSV "
            f"(leaving unchanged): {still_stubs}"
        )

    return upgraded


def run_sort(cache: dict, dry_run: bool) -> None:
    """Rewrite the cache dict in alphabetical key order (in-place)."""
    before = list(cache.keys())
    sorted_cache = dict(sorted(cache.items()))
    cache.clear()
    cache.update(sorted_cache)
    after = list(cache.keys())

    if before == after:
        logger.info("[sort] Cache was already sorted — no change.")
    else:
        logger.success(f"[sort] Cache sorted alphabetically ({len(cache):,} entries).")


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShockArb ticker reference cache maintenance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache", default="./data/ticker_reference_cache.json",
        help="Path to the ticker reference cache JSON (default: ./data/ticker_reference_cache.json)",
    )
    parser.add_argument(
        "--ref-dir", default="./data",
        help="Directory containing NYSE/NASDAQ reference CSVs (default: ./data)",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Add tickers missing from cache but present in the reference CSVs.",
    )
    parser.add_argument(
        "--fix-stubs", action="store_true",
        help="Find stub entries and replace with real CSV data where available.",
    )
    parser.add_argument(
        "--sort", action="store_true",
        help="Rewrite cache in alphabetical key order.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing to disk.",
    )
    args = parser.parse_args()

    if not any([args.update, args.fix_stubs, args.sort]):
        parser.error("Specify at least one of --update, --fix-stubs, --sort.")

    if args.dry_run:
        logger.info("*** DRY RUN — no files will be modified ***")

    cache    = _load_cache(args.cache)
    ref_csvs = _find_reference_csvs(args.ref_dir) if (args.update or args.fix_stubs) else []

    logger.info(f"Cache loaded: {len(cache):,} entries from {args.cache}")

    dirty = False

    if args.update:
        added = run_update(cache, ref_csvs, args.dry_run)
        dirty = dirty or added > 0

    if args.fix_stubs:
        upgraded = run_fix_stubs(cache, ref_csvs, args.dry_run)
        dirty = dirty or upgraded > 0

    if args.sort:
        run_sort(cache, args.dry_run)
        dirty = True  # always write after sort so ordering is guaranteed on disk

    if dirty:
        _save_cache(cache, args.cache, args.dry_run)
    else:
        logger.info("No changes — cache not rewritten.")


if __name__ == "__main__":
    main()
