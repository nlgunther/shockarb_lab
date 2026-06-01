"""
Backfill historical score CSVs for performance analysis.

Scores one date per week from START_DATE to END_DATE using the active regime,
saving each to data/backfill/alpha_YYYY-MM-DD.csv.

Usage
-----
    python utils/backfill_scores.py
    python utils/backfill_scores.py --start 2025-06-01 --end 2026-05-30
    python utils/backfill_scores.py --weeks 2          # every 2 weeks instead of 1
    python utils/backfill_scores.py --dry-run          # print dates without scoring
    python utils/backfill_scores.py --prefetch-only    # download prices then exit

Notes
-----
    Price data is downloaded automatically by each score call via the datamgr
    coordinator. On a cold cache the first call pulls the full date range for
    all tickers (slow, ~1–2 min). Subsequent calls are fast cache hits.

    Use --prefetch-only to warm the cache in one shot before the scoring loop,
    which avoids repeated yfinance round-trips and reduces rate-limit risk.
    Prefetch works by scoring the last date in the range (which forces the
    coordinator to fetch the entire window) and discarding the output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


DEFAULT_START = date(2024, 12, 6)
DEFAULT_END   = date(2026, 5, 30)


def date_range(start: date, end: date, step_weeks: int = 1):
    d = start
    while d <= end:
        yield d
        d += timedelta(weeks=step_weeks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start",    default=DEFAULT_START.isoformat(),
                        help=f"First date to score (default: {DEFAULT_START})")
    parser.add_argument("--end",      default=DEFAULT_END.isoformat(),
                        help=f"Last date to score (default: {DEFAULT_END})")
    parser.add_argument("--weeks",    type=int, default=1,
                        help="Step size in weeks (default: 1)")
    parser.add_argument("--out-dir",  default="data/backfill_us",
                        help="Output directory (default: data/backfill_us)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print dates without running shockarb")
    parser.add_argument("--prefetch-only", action="store_true",
                        help="Warm the price cache for the full date range, then exit")
    args = parser.parse_args()

    start   = date.fromisoformat(args.start)
    end     = date.fromisoformat(args.end)
    out_dir = Path(args.out_dir)

    # ── Prefetch: score the last date to pull the full window into cache ──────
    if args.prefetch_only:
        print(f"Prefetching prices for {start} → {end} by scoring {end} ...")
        cmd = [sys.executable, "-m", "shockarb", "score",
               "--date", end.isoformat(), "--out", str(out_dir / "_prefetch.csv"),
               "--no-log"]
        out_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd)
        prefetch_file = out_dir / "_prefetch.csv"
        if prefetch_file.exists():
            prefetch_file.unlink()
        sys.exit(0 if result.returncode == 0 else 1)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    dates = list(date_range(start, end, args.weeks))
    print(f"Scoring {len(dates)} dates from {start} to {end} "
          f"(every {args.weeks} week(s)) → {out_dir}/\n")

    ok = skipped = failed = 0

    for i, d in enumerate(dates, 1):
        out_path = out_dir / f"alpha_{d.isoformat()}.csv"
        status_prefix = f"[{i:3}/{len(dates)}]  {d}"

        if out_path.exists():
            print(f"{status_prefix}  skip (already exists)")
            skipped += 1
            continue

        if args.dry_run:
            print(f"{status_prefix}  (dry run)")
            continue

        cmd = [
            sys.executable, "-m", "shockarb",
            "score",
            "--date", d.isoformat(),
            "--out",  str(out_path),
            "--no-log",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and out_path.exists():
            print(f"{status_prefix}  ok  → {out_path.name}")
            ok += 1
        else:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            print(f"{status_prefix}  FAILED  ({err})")
            failed += 1

    print(f"\nDone.  ok={ok}  skipped={skipped}  failed={failed}")
    if failed:
        print("Re-run to retry failed dates — existing files are skipped automatically.")


if __name__ == "__main__":
    main()
