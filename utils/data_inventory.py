"""
Data Inventory — report the actual contents of every parquet in the store.

Reads parquet files directly (not the manifest) and reports:
  - True date spans with gap detection
  - Row counts per span
  - Column names
  - Comparison against manifest date_range (flags mismatches)

Run from the shockarb_lab root:
    python utils/data_inventory.py
    python utils/data_inventory.py --ticker VOO        # single ticker
    python utils/data_inventory.py --data-dir data     # custom data dir
    python utils/data_inventory.py --gaps-only          # only show tickers with gaps
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# ── Gap detection ────────────────────────────────────────────────────────────

def find_spans(index: pd.DatetimeIndex, max_gap_bdays: int = 5) -> List[dict]:
    """
    Split a DatetimeIndex into contiguous spans.

    A new span starts when the gap between consecutive dates exceeds
    max_gap_bdays business days.  Default 5 allows for a full week of
    holidays without splitting (e.g. Christmas week).

    Returns list of {"start": date, "end": date, "rows": int}.
    """
    if len(index) == 0:
        return []

    dates = sorted(index)
    spans = []
    span_start = dates[0]
    prev = dates[0]

    for dt in dates[1:]:
        bdays_gap = len(pd.bdate_range(prev, dt)) - 1
        if bdays_gap > max_gap_bdays:
            spans.append({
                "start": span_start.date() if hasattr(span_start, "date") else span_start,
                "end":   prev.date() if hasattr(prev, "date") else prev,
                "rows":  len([d for d in dates if span_start <= d <= prev]),
            })
            span_start = dt
        prev = dt

    # Final span
    spans.append({
        "start": span_start.date() if hasattr(span_start, "date") else span_start,
        "end":   prev.date() if hasattr(prev, "date") else prev,
        "rows":  len([d for d in dates if span_start <= d <= prev]),
    })
    return spans


# ── Inventory ────────────────────────────────────────────────────────────────

def inventory_daily(
    data_dir: str,
    ticker_filter: Optional[str] = None,
    gaps_only: bool = False,
) -> List[dict]:
    """
    Inventory all daily parquet files.

    Returns a list of dicts, one per ticker, with:
      ticker, path, columns, total_rows, spans[], manifest_range, mismatch
    """
    daily_dir = Path(data_dir) / "prices" / "daily"
    manifest_path = Path(data_dir) / "manifest.json"

    manifest = {}
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            pass

    if not daily_dir.exists():
        return []

    results = []

    parquets = sorted(daily_dir.glob("*.parquet"))
    for path in parquets:
        ticker = path.stem
        if ticker_filter and ticker.upper() != ticker_filter.upper():
            continue

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            results.append({
                "ticker": ticker, "path": str(path), "error": str(exc),
                "columns": [], "total_rows": 0, "spans": [],
                "manifest_range": None, "mismatch": False,
            })
            continue

        columns = list(df.columns)
        spans = find_spans(df.index)
        has_gaps = len(spans) > 1

        if gaps_only and not has_gaps:
            continue

        # Manifest comparison
        manifest_key = f"daily/{ticker}"
        manifest_entry = manifest.get(manifest_key, {})
        manifest_range = manifest_entry.get("date_range")
        manifest_rows = manifest_entry.get("rows")

        mismatch = False
        mismatch_details = []

        if manifest_range and len(manifest_range) == 2:
            actual_start = str(spans[0]["start"]) if spans else None
            actual_end = str(spans[-1]["end"]) if spans else None
            if actual_start != manifest_range[0]:
                mismatch = True
                mismatch_details.append(
                    f"start: manifest={manifest_range[0]} actual={actual_start}"
                )
            if actual_end != manifest_range[1]:
                mismatch = True
                mismatch_details.append(
                    f"end: manifest={manifest_range[1]} actual={actual_end}"
                )
        if manifest_rows and manifest_rows != len(df):
            mismatch = True
            mismatch_details.append(
                f"rows: manifest={manifest_rows} actual={len(df)}"
            )

        results.append({
            "ticker": ticker,
            "path": str(path),
            "columns": columns,
            "total_rows": len(df),
            "spans": spans,
            "has_gaps": has_gaps,
            "manifest_range": manifest_range,
            "manifest_rows": manifest_rows,
            "mismatch": mismatch,
            "mismatch_details": mismatch_details,
        })

    return results


def inventory_intraday(data_dir: str) -> List[dict]:
    """Inventory intraday parquet files."""
    intra_dir = Path(data_dir) / "prices" / "intraday"
    if not intra_dir.exists():
        return []

    results = []
    for path in sorted(intra_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(path)
            results.append({
                "file": path.name,
                "rows": len(df),
                "first": str(df.index.min()),
                "last": str(df.index.max()),
            })
        except Exception as exc:
            results.append({"file": path.name, "error": str(exc)})

    return results


# ── Display ──────────────────────────────────────────────────────────────────

def print_inventory(results: List[dict], show_columns: bool = False) -> None:
    if not results:
        print("  No daily data found.")
        return

    # Summary stats
    total_tickers = len(results)
    tickers_with_gaps = sum(1 for r in results if r.get("has_gaps"))
    tickers_with_mismatch = sum(1 for r in results if r.get("mismatch"))
    total_rows = sum(r.get("total_rows", 0) for r in results)

    print(f"  Tickers: {total_tickers}  |  Total rows: {total_rows:,}")
    if tickers_with_gaps:
        print(f"  ⚠️  Tickers with gaps: {tickers_with_gaps}")
    if tickers_with_mismatch:
        print(f"  ⚠️  Manifest mismatches: {tickers_with_mismatch}")

    # Per-ticker detail
    for r in results:
        ticker = r["ticker"]

        if "error" in r:
            print(f"\n  {ticker}: ❌ {r['error']}")
            continue

        spans = r["spans"]
        span_strs = []
        for s in spans:
            span_strs.append(f"{s['start']} -> {s['end']} ({s['rows']} rows)")

        gap_marker = " ⚠️  GAP" if r.get("has_gaps") else ""
        print(f"\n  {ticker}:{gap_marker}")
        for i, ss in enumerate(span_strs):
            prefix = "    " if i == 0 else "    ~~gap~~  "
            print(f"{prefix}{ss}")

        if show_columns:
            print(f"    Fields: {', '.join(r['columns'])}")

        if r.get("mismatch"):
            for detail in r.get("mismatch_details", []):
                print(f"    ❌ Manifest mismatch: {detail}")


def print_intraday(results: List[dict]) -> None:
    if not results:
        print("  No intraday data found.")
        return

    for r in results:
        if "error" in r:
            print(f"  {r['file']}: ❌ {r['error']}")
        else:
            print(f"  {r['file']:40s}  {r['rows']:4d} bars  {r['first']} -> {r['last']}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Report the actual contents of every parquet in the store."
    )
    parser.add_argument("--data-dir", default="data",
                        help="Data directory (default: data)")
    parser.add_argument("--ticker", default=None,
                        help="Show only this ticker")
    parser.add_argument("--gaps-only", action="store_true",
                        help="Only show tickers that have date gaps")
    parser.add_argument("--fields", action="store_true",
                        help="Show column names for each ticker")
    parser.add_argument("--intraday", action="store_true",
                        help="Also show intraday files")
    args = parser.parse_args()

    print("=" * 72)
    print("  DATA INVENTORY")
    print(f"  {datetime.now().isoformat()}")
    print("=" * 72)

    # Daily
    print("\n" + "-" * 72)
    print("  DAILY DATA")
    print("-" * 72)

    results = inventory_daily(args.data_dir, args.ticker, args.gaps_only)
    print_inventory(results, show_columns=args.fields)

    # Intraday
    if args.intraday:
        print("\n" + "-" * 72)
        print("  INTRADAY DATA")
        print("-" * 72)
        intra = inventory_intraday(args.data_dir)
        print_intraday(intra)

    print("\n" + "=" * 72)
    print("  END INVENTORY")
    print("=" * 72)


if __name__ == "__main__":
    main()
