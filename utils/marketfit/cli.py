"""
marketfit CLI — generate and display a local ShockArb market report.

Commands
--------
  report    Read market_snapshot.json, evaluate conditions, write market_report.md

Usage
-----
    python -m marketfit report
    python -m marketfit report --snapshot data/market_snapshot.json --out data/market_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from marketfit import features, rules, report


_DEFAULT_SNAPSHOT = "./data/market_snapshot.json"
_DEFAULT_OUT      = "./data/market_report.md"
_STALE_HOURS      = 6


def _is_stale(snapshot: dict) -> bool:
    fetched_at = snapshot.get("fetched_at")
    if not fetched_at:
        return True
    try:
        fetched = datetime.fromisoformat(fetched_at)
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
        return age_hours > _STALE_HOURS
    except Exception:
        return True


def cmd_report(args) -> None:
    """Generate a market report from a local snapshot."""
    snap_path = args.snapshot

    if not os.path.exists(snap_path):
        print(f"\n❌  Snapshot not found: {snap_path}")
        print("    Run first:  python utils/market_data.py")
        sys.exit(1)

    with open(snap_path, encoding="utf-8") as f:
        snapshot = json.load(f)

    stale = _is_stale(snapshot)
    if stale:
        logger.warning(f"Snapshot is more than {_STALE_HOURS}h old — data may be stale.")

    feats   = features.extract(snapshot)
    verdict = rules.evaluate(feats)
    md      = report.build(snapshot, verdict, stale=stale)

    # Print to terminal
    print(md)

    # Save to file
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Path(out_path).write_text(md, encoding="utf-8")
    logger.success(f"Report saved: {out_path}")
    print(f"\n📁  Saved to: {out_path}")
    print(f"    Snapshot:  {snapshot.get('fetched_at_local', 'unknown')}")
    print(f"    Verdict:   {verdict.overall}  (score {verdict.score}/11)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="marketfit",
        description="Local ShockArb market report — no LLM required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # report
    p = sub.add_parser("report", help="Generate market report from local snapshot")
    p.add_argument(
        "--snapshot", "-s", default=_DEFAULT_SNAPSHOT,
        help=f"Path to market_snapshot.json (default: {_DEFAULT_SNAPSHOT})",
    )
    p.add_argument(
        "--out", "-o", default=_DEFAULT_OUT,
        help=f"Output .md path (default: {_DEFAULT_OUT})",
    )
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
