"""
stockfit CLI — generate and save a ShockArb stock opportunity report.

Commands
--------
  report    Read pipeline outputs, evaluate signals, write stock_report.md

Usage
-----
    cd utils && python -m stockfit report
    cd utils && python -m stockfit report --llm
    cd utils && python -m stockfit report --llm --timestamp
    cd utils && python -m stockfit report --min-r2 0.70 --min-confidence 0.025

    Input data resolves to ../data/ (relative to utils/).
    Reports are written to ../reports/ by default; override with --reports-dir.
    Always run from the utils/ directory: cd utils && python -m stockfit report

Options
-------
  --llm             Generate enhanced report with LLM narratives
                    (requires ANTHROPIC_API_KEY or GOOGLE_API_KEY)
  --timestamp       Save as stock_report_YYYYMMDD_HHMM.md — never overwrites;
                    builds archive for LLM training
  --reports-dir     Directory for report output (default ../reports)
  --earnings-window Days ahead to treat earnings as imminent (default 14)
  --min-r2          Minimum r² threshold (default 0.65)
  --min-confidence  Minimum confidence_delta threshold (default 0.020)
  --min-upside      Minimum analyst upside fraction (default 0.05 = 5%)
  --scores          Path to live_alpha_us.csv (default ../data/live_alpha_us.csv)
  --fundamentals    Path to fundamentals.csv  (default ../data/fundamentals.csv)
  --news            Path to news.txt          (default ../data/news.txt)
  --out             Exact output .md path; overrides --reports-dir (default: auto-resolved)

See docs/ENVIRONMENT_VARIABLES.md for LLM env vars.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from stockfit import features, rules, report

# All paths centralised in paths.py. See docs/PATHS.md for design rationale.
from paths import LIVE_ALPHA_US, FUNDAMENTALS, NEWS, REPORTS_DIR


def _check_cwd() -> None:
    """Exit with a clear error if not run from the utils/ directory."""
    if not Path("../data").is_dir():
        print(
            "\n❌  This command must be run from the utils/ directory.\n"
            "\n"
            "    Correct usage:\n"
            "        cd <project_root>/utils\n"
            "        python -m stockfit report\n"
            "\n"
            f"    Current directory: {Path.cwd()}\n"
        )
        sys.exit(1)


def _resolve_out(args_out: str | None, reports_dir: Path, timestamp: bool) -> Path:
    """
    Resolve output path.

    Priority: explicit --out > --timestamp auto-name > default.
    --timestamp appends YYYYMMDD_HHMM so reports are never overwritten:
        stock_report_20260606_1551.md
    --reports-dir changes the output folder (default: ../reports).
    """
    if args_out is not None:
        return Path(args_out)

    if timestamp:
        now = datetime.now(timezone.utc)
        ts  = now.strftime("%Y%m%d_%H%M")
        return reports_dir / f"stock_report_{ts}.md"

    return reports_dir / "stock_report.md"


def _check_inputs(scores: str, fundamentals: str, news: str) -> None:
    """Warn on missing input files; exit if scores (primary input) is missing."""
    if not os.path.exists(scores):
        print(f"\n\u274c  Scores file not found: {scores}")
        print("    Run first:  shockarb score")
        sys.exit(1)
    for path, label in [(fundamentals, "fundamentals.csv"), (news, "news.txt")]:
        if not os.path.exists(path):
            logger.warning(
                f"{label} not found at {path} — {label.split('.')[0]} data will be absent. "
                f"Run `python utils/news_scanner.py` to generate it."
            )


def cmd_report(args) -> None:
    """Generate a stock opportunity report from pipeline outputs."""
    _check_inputs(args.scores, args.fundamentals, args.news)

    feats    = features.extract_all(args.scores, args.fundamentals, args.news,
                                    earnings_window=args.earnings_window)
    verdicts = rules.evaluate_all(
        feats,
        min_r2           = args.min_r2,
        min_conf_delta   = args.min_confidence,
        min_upside       = args.min_upside,
        earnings_exclude = not args.include_earnings,
    )

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    thresholds = {
        "min_r2":           args.min_r2,
        "min_conf_delta":   args.min_confidence,
        "min_upside":       args.min_upside,
    }

    if args.llm:
        md = _build_with_llm(verdicts, now_str, thresholds)
    else:
        md = report.build(verdicts, date_str=now_str, thresholds=thresholds)

    out_path = _resolve_out(args.out, args.reports_dir, timestamp=args.timestamp)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        Path(out_path).write_text(md, encoding="utf-8")
        save_ok = True
    except Exception as exc:
        logger.error(f"Failed to save report: {exc}")
        save_ok = False

    print(md)

    include_n = sum(1 for v in verdicts if v.tier == "INCLUDE")
    watch_n   = sum(1 for v in verdicts if v.tier == "WATCH")
    exclude_n = sum(1 for v in verdicts if v.tier == "EXCLUDE")

    if save_ok:
        print(f"\n\U0001f4c1  Saved to: {out_path}")
    else:
        print(f"\n\u274c  Save failed — check path and permissions: {out_path}")

    llm_note = " | LLM: enabled" if args.llm else ""
    print(
        f"    Tickers: {include_n} INCLUDE | {watch_n} WATCH | {exclude_n} EXCLUDE{llm_note}"
    )
    print(
        f"    Thresholds: r2>={args.min_r2:.2f} | conf.D>={args.min_confidence:.3f} "
        f"| upside>={args.min_upside * 100:.0f}%"
    )


def _build_with_llm(
    verdicts:   list,
    date_str:   str,
    thresholds: dict,
) -> str:
    """Generate the enhanced report via LLM; fall back to basic if unavailable."""
    from stockfit.llm import StockfitLLMClient

    try:
        client = StockfitLLMClient.from_env()
    except RuntimeError as exc:
        logger.error(f"LLM not available: {exc}")
        logger.info("Falling back to basic report (no LLM).")
        return report.build(verdicts, date_str=date_str, thresholds=thresholds)

    narratives = client.generate_narratives(verdicts)

    if not narratives:
        logger.warning("LLM returned no narratives — falling back to basic report.")
        return report.build(verdicts, date_str=date_str, thresholds=thresholds)

    return report.build_enhanced(
        verdicts, narratives,
        date_str=date_str,
        thresholds=thresholds,
    )


def main() -> None:
    _check_cwd()
    parser = argparse.ArgumentParser(
        prog="stockfit",
        description="ShockArb stock opportunity report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", help="Generate stock report from pipeline outputs")
    p.add_argument("--scores", default=LIVE_ALPHA_US,
                   help=f"Path to live_alpha_us.csv (default: {LIVE_ALPHA_US})")
    p.add_argument("--fundamentals", default=FUNDAMENTALS,
                   help=f"Path to fundamentals.csv (default: {FUNDAMENTALS})")
    p.add_argument("--news", default=NEWS,
                   help=f"Path to news.txt (default: {NEWS})")
    p.add_argument("--reports-dir", default=REPORTS_DIR,
                   help=f"Directory for report output (default: {REPORTS_DIR})")
    p.add_argument("--out", "-o", default=None,
                   help="Exact output .md path; overrides --reports-dir (default: auto)")
    p.add_argument("--llm", action="store_true",
                   help="Enhanced report with LLM narratives (requires API key)")
    p.add_argument("--timestamp", action="store_true",
                   help="Save as stock_report_YYYYMMDD_HHMM.md — never overwrites")
    p.add_argument("--earnings-window", type=int, default=14,
                   help="Days ahead to treat earnings as imminent and exclude (default: 14)")
    p.add_argument("--min-r2", type=float, default=0.65,
                   help="Minimum r2 threshold (default: 0.65)")
    p.add_argument("--min-confidence", type=float, default=0.020,
                   help="Minimum confidence_delta threshold (default: 0.020)")
    p.add_argument("--min-upside", type=float, default=0.05,
                   help="Minimum analyst upside fraction (default: 0.05)")
    p.add_argument("--include-earnings", action="store_true",
                   help="Do not exclude tickers with imminent earnings (default: exclude)")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
    