"""
marketfit CLI — generate and display a local ShockArb market report.

Commands
--------
  report    Read market_snapshot.json, evaluate conditions, write market_report.md

Usage
-----
    cd utils && python -m marketfit report
    cd utils && python -m marketfit report --llm          # enhanced report via LLM
    cd utils && python -m marketfit report --snapshot /path/to/snapshot.json

    Input data resolves to ../data/ (relative to utils/).
    Reports are written to ../reports/ by default; override with --reports-dir.
    Intraday snapshots (mode=intraday) default to market_report_intraday.md.
    Always run from the utils/ directory: cd utils && python -m marketfit report

LLM mode
--------
    Set GOOGLE_API_KEY (Gemini) or ANTHROPIC_API_KEY (Anthropic, preferred).
    Loads ../data/live_alpha_us.csv, ../data/news.txt, ../data/fundamentals.csv
    if present, and generates the full <!-- LEARN --> enhanced report.
    See docs/ENVIRONMENT_VARIABLES.md for all env vars.
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


_DATA_DIR             = "../data"
_DEFAULT_SNAPSHOT     = "../data/market_snapshot.json"
_DEFAULT_REPORTS_DIR  = "../reports"
_STALE_HOURS          = 6


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


def _resolve_out(args_out: str | None, reports_dir: str, snapshot: dict, timestamp: bool = False) -> str:
    """
    Resolve output path.

    Priority: explicit --out > --timestamp auto-name > mode-based default.
    --timestamp appends the snapshot fetch time so reports are never overwritten:
        market_report_20260604_1455.md  (daily)
        market_report_intraday_20260604_0932.md  (intraday)
    --reports-dir changes the output folder (default: ../reports).
    """
    if args_out is not None:
        return args_out   # explicit path always wins

    mode = snapshot.get("mode", "daily")
    base = "market_report_intraday" if mode == "intraday" else "market_report"

    if timestamp:
        ts = snapshot.get("fetched_at_local", "").replace(" ", "_").replace(":", "")
        return f"{reports_dir}/{base}_{ts}.md"

    suffix = "_intraday" if mode == "intraday" else ""
    return f"{reports_dir}/market_report{suffix}.md"


def _load_picks(path: str):
    """Load live_alpha_us.csv as a DataFrame, or None if absent."""
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        # Normalise column names (shockarb score output uses various conventions)
        df.columns = [c.strip() for c in df.columns]
        if "ticker" in df.columns and "Ticker" not in df.columns:
            df.rename(columns={"ticker": "Ticker"}, inplace=True)
        return df
    except Exception as exc:
        logger.warning(f"Could not load picks from {path}: {exc}")
        return None


def _load_fundamentals(path: str):
    """Load fundamentals.csv as a DataFrame, or None if absent."""
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    except Exception as exc:
        logger.warning(f"Could not load fundamentals from {path}: {exc}")
        return None


def _load_news(path: str) -> dict[str, list[str]]:
    """
    Parse news.txt into {ticker: [headline, ...]} dict.
    Same format as catalyst_feed.txt — [TICKER] blocks separated by dashes.
    """
    if not os.path.exists(path):
        return {}
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        chunks  = content.split("---" * 29)   # 87-dash separator
        result  = {}
        for chunk in chunks:
            file_lines = [l.strip() for l in chunk.strip().split("\n") if l.strip()]
            if not file_lines or not file_lines[0].startswith("["):
                continue
            ticker = file_lines[0].split("]")[0].replace("[","").strip().upper()
            headlines = []
            for i, line in enumerate(file_lines):
                if line.startswith(">") and i + 1 < len(file_lines):
                    nxt = file_lines[i + 1]
                    if not nxt.startswith(">") and not nxt.startswith("["):
                        headlines.append(nxt)
            result[ticker] = headlines
        return result
    except Exception as exc:
        logger.warning(f"Could not parse news from {path}: {exc}")
        return {}


def cmd_report(args) -> None:
    """Generate a market report from a local snapshot."""
    if not os.path.exists(args.snapshot):
        print(f"\n\u274c  Snapshot not found: {args.snapshot}")
        print("    Run first:  python utils/market_data.py")
        sys.exit(1)

    with open(args.snapshot, encoding="utf-8") as f:
        snapshot = json.load(f)

    if "baseline_date" not in snapshot:
        logger.warning(
            "Snapshot predates baseline_date schema. "
            "Re-run `python utils/market_data.py` for a dated baseline in the report."
        )

    stale   = _is_stale(snapshot)
    if stale:
        logger.warning(f"Snapshot is more than {_STALE_HOURS}h old — data may be stale.")

    feats   = features.extract(snapshot)
    verdict = rules.evaluate(feats)

    if args.llm:
        md = _build_with_llm(snapshot, verdict, stale)
    else:
        md = report.build(snapshot, verdict, stale=stale)

    out_path = _resolve_out(args.out, args.reports_dir, snapshot, timestamp=args.timestamp)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        Path(out_path).write_text(md, encoding="utf-8")
        save_ok = True
    except Exception as exc:
        logger.error(f"Failed to save report: {exc}")
        save_ok = False

    print(md)

    if save_ok:
        print(f"\n\U0001f4c1  Saved to: {out_path}")
    else:
        print(f"\n\u274c  Save failed — check path and permissions: {out_path}")
    print(f"    Snapshot:  {snapshot.get('fetched_at_local', 'unknown')}")
    print(f"    Baseline:  {snapshot.get('baseline_date', '(unknown — re-run market_data.py)')}")
    print(f"    Mode:      {snapshot.get('mode', 'daily')}")
    llm_note = " | LLM: enabled" if args.llm else ""
    print(f"    Verdict:   {verdict.overall}  (score {verdict.score}/11){llm_note}")


def _build_with_llm(snapshot: dict, verdict, stale: bool) -> str:
    """
    Generate the enhanced report using the LLM client.
    Falls back to basic report.build() if the LLM call fails or no key is set.
    """
    from marketfit.llm import MarketfitLLMClient

    try:
        client = MarketfitLLMClient.from_env()
    except RuntimeError as exc:
        logger.error(f"LLM not available: {exc}")
        logger.info("Falling back to basic report (no LLM).")
        return report.build(snapshot, verdict, stale=stale)

    picks_df        = _load_picks(_DATA_DIR + "/live_alpha_us.csv")
    fundamentals_df = _load_fundamentals(_DATA_DIR + "/fundamentals.csv")
    news_dict       = _load_news(_DATA_DIR + "/news.txt")

    if picks_df is not None:
        logger.info(f"Loaded {len(picks_df)} picks from live_alpha_us.csv")
    if fundamentals_df is not None:
        logger.info(f"Loaded {len(fundamentals_df)} fundamentals rows")
    if news_dict:
        logger.info(f"Loaded news for {len(news_dict)} tickers")

    narratives = client.generate_narratives(
        snapshot, verdict,
        picks_df=picks_df,
        news_dict=news_dict,
        fundamentals_df=fundamentals_df,
    )

    if not narratives:
        logger.warning("LLM returned no narratives — falling back to basic report.")
        return report.build(snapshot, verdict, stale=stale)

    return report.build_enhanced(
        snapshot, verdict, narratives,
        picks_df=picks_df,
        news_dict=news_dict,
        fundamentals_df=fundamentals_df,
        stale=stale,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="marketfit",
        description="Local ShockArb market report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", help="Generate market report from local snapshot")
    p.add_argument("--snapshot", "-s", default=_DEFAULT_SNAPSHOT,
                   help=f"Path to market_snapshot.json (default: {_DEFAULT_SNAPSHOT})")
    p.add_argument("--reports-dir", default=_DEFAULT_REPORTS_DIR,
                   help=f"Directory for report output (default: {_DEFAULT_REPORTS_DIR})")
    p.add_argument("--out", "-o", default=None,
                   help="Exact output .md path; overrides --reports-dir (default: auto)")
    p.add_argument("--llm", action="store_true",
                   help="Generate enhanced report with LLM narratives (requires GOOGLE_API_KEY or ANTHROPIC_API_KEY)")
    p.add_argument("--timestamp", action="store_true",
                   help="Save as market_report_YYYYMMDD_HHMM.md — never overwrites; builds archive for LLM training")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
