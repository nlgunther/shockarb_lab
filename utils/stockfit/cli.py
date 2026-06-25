"""
stockfit CLI — generate and save a ShockArb stock opportunity report.

Commands
--------
  report    Read pipeline outputs, evaluate signals, write stock_report.md
  set-rvol  Persist the sticky RVOL display setting (on/off) for future `report` runs
  show-rvol Show the current sticky RVOL display setting

Usage
-----
    cd utils && python -m stockfit report
    cd utils && python -m stockfit report --llm
    cd utils && python -m stockfit report --llm --timestamp
    cd utils && python -m stockfit report --min-r2 0.70 --min-confidence 0.025
    cd utils && python -m stockfit report --rvol
    cd utils && python -m stockfit report --intraday
    cd utils && python -m stockfit set-rvol on
    cd utils && python -m stockfit show-rvol

    Input data resolves to ../data/ (relative to utils/).
    Reports are written to ../reports/ by default; override with --reports-dir.
    Always run from the utils/ directory: cd utils && python -m stockfit report

Options
-------
  --no-llm          Disable LLM narratives; produce rules-based report only
                    (LLM is ON by default; requires ANTHROPIC_API_KEY or GOOGLE_API_KEY)
  --timestamp       Save as stock_report_YYYYMMDD_HHMM.md — never overwrites;
                    builds archive for LLM training
  --reports-dir     Directory for report output (default ../reports)
  --earnings-window Days ahead to treat earnings as imminent (default 14)
  --include-earnings Do not exclude tickers with imminent earnings (default: exclude)
  --min-r2          Minimum r² threshold (default 0.65)
  --min-confidence  Minimum confidence_delta threshold (default 0.020)
  --min-upside      Minimum analyst upside fraction (default 0.05 = 5%)
  --scores          Path to live_alpha_us.csv (default ../data/live_alpha_us.csv)
  --fundamentals    Path to fundamentals.csv  (default ../data/fundamentals.csv)
  --news            Path to news.txt          (default ../data/news.txt)
  --out             Exact output .md path; overrides --reports-dir (default: auto-resolved)
  --rvol            Show RVOL (relative volume) column (overrides sticky setting)
  --no-rvol         Hide RVOL column (overrides sticky setting)
  --intraday        Fetch live current prices and show intraday % change vs.
                    cached close (network call, off by default)
  --update-reference-data
                    Sync NYSE/NASDAQ reference CSVs (data/nyse_*.csv,
                    data/nasdaq_*.csv) from LondonMarket/Global-Stock-Symbols
                    before generating the report (network call, off by
                    default). Updates existing rows, adds new symbols, and
                    clears ticker_reference_cache.json entries for anything
                    changed. See shockarb/reference_sync.py.
  --save-verdicts   Also write <report>_verdicts.csv alongside the .md
                    report, with full per-ticker stats (r², conf.Δ, upside,
                    price, target, fwd P/E, rvol, intraday, cluster,
                    warnings) for ALL tiers including EXCLUDE — a durable
                    record that survives live_alpha_us.csv being overwritten.

RVOL display
------------
  RVOL = latest cached day's volume / trailing average volume (5-20 day
  dynamic window), shown as e.g. "2.3x (10d)". Informational only — it does
  not affect scoring, ranking, or filtering. Off by default; persist a
  preference with `set-rvol on`/`set-rvol off`, or override per-run with
  --rvol/--no-rvol. See docs/KT.md ("RVOL (relative volume) display") for
  the full design.

See docs/ENVIRONMENT_VARIABLES.md for LLM env vars.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from stockfit import features, rules, report

# All paths centralised in paths.py. See docs/PATHS.md for design rationale.
from paths import (
    DATA,
    EXCHANGE_CSV_FILENAMES,
    FUNDAMENTALS,
    LIVE_ALPHA_US,
    NEWS,
    REPORTS_DIR,
    STOCKFIT_RVOL_FILE,
    TICKER_CACHE_FILENAME,
)


# ---------------------------------------------------------------------------
# Sticky RVOL setting (mirrors shockarb/cli.py's .shockarb_regime pattern)
# ---------------------------------------------------------------------------

def _get_sticky_rvol() -> bool | None:
    """Read the sticky RVOL setting from .stockfit_rvol. None if unset/invalid."""
    if not STOCKFIT_RVOL_FILE.exists():
        return None
    value = STOCKFIT_RVOL_FILE.read_text(encoding="utf-8").strip().lower()
    if value == "on":
        return True
    if value == "off":
        return False
    return None


def _set_sticky_rvol(enabled: bool) -> None:
    """Write the sticky RVOL setting to .stockfit_rvol."""
    STOCKFIT_RVOL_FILE.write_text("on" if enabled else "off", encoding="utf-8")


def _resolve_rvol(args) -> bool:
    """
    Resolve whether to compute/display RVOL.

    Priority: --rvol flag > --no-rvol flag > sticky file > default off.
    """
    if getattr(args, "rvol", False):
        return True
    if getattr(args, "no_rvol", False):
        return False
    sticky = _get_sticky_rvol()
    return sticky if sticky is not None else False


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

    reports_dir = Path(reports_dir)   # accept str or Path from argparse / tests
    if timestamp:
        now = datetime.now(timezone.utc)
        ts  = now.strftime("%Y%m%d_%H%M")
        return reports_dir / f"stock_report_{ts}.md"

    return reports_dir / "stock_report.md"


def _verdicts_path(out_path: Path) -> Path:
    """
    Derive the --save-verdicts CSV path from the report's .md path.

    Example
    -------
        _verdicts_path(Path("reports/stock_report_20260612_0800.md"))
        # → Path("reports/stock_report_20260612_0800_verdicts.csv")
    """
    return out_path.with_name(out_path.stem + "_verdicts.csv")


def _check_inputs(scores: str, fundamentals: str, news: str) -> None:
    """Warn on missing input files; exit if scores (primary input) is missing."""
    if not os.path.exists(scores):
        print(f"\n❌  Scores file not found: {scores}")
        print("    Run first:  shockarb score")
        sys.exit(1)
    for path, label in [(fundamentals, "fundamentals.csv"), (news, "news.txt")]:
        if not os.path.exists(path):
            logger.warning(
                f"{label} not found at {path} — {label.split('.')[0]} data will be absent. "
                f"Run `python utils/news_scanner.py` to generate it."
            )


def _update_reference_data() -> None:
    """Sync NYSE/NASDAQ reference CSVs from LondonMarket/Global-Stock-Symbols."""
    from shockarb.reference_sync import sync_reference_data

    stats = sync_reference_data(
        data_dir   = str(DATA),
        files      = EXCHANGE_CSV_FILENAMES,
        cache_path = str(DATA / TICKER_CACHE_FILENAME),
    )
    for filename, result in stats.items():
        print(f"    Reference data — {filename}: {result.updated} updated, "
              f"{result.added} added (total {result.total})")
    if not stats:
        print("    Reference data — no files synced (see warnings above)")


def cmd_report(args) -> None:
    """Generate a stock opportunity report from pipeline outputs."""
    _check_inputs(args.scores, args.fundamentals, args.news)

    if args.update_reference_data:
        _update_reference_data()

    compute_rvol = _resolve_rvol(args)
    feats    = features.extract_all(args.scores, args.fundamentals, args.news,
                                    earnings_window=args.earnings_window,
                                    compute_rvol=compute_rvol,
                                    compute_intraday=args.intraday)
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
    scores_name = Path(args.scores).name
    source_str  = f"{scores_name} + fundamentals.csv + news.txt"

    if args.llm:
        md = _build_with_llm(verdicts, now_str, thresholds, source_str)
    else:
        md = report.build(verdicts, date_str=now_str, thresholds=thresholds, source=source_str)

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
        print(f"\n❌  Save failed — check path and permissions: {out_path}")

    if args.save_verdicts:
        verdicts_path = _verdicts_path(Path(out_path))
        try:
            rows = rules.verdicts_to_rows(verdicts)
            with open(verdicts_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rules.VERDICT_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"Saved full verdicts ({len(rows)} tickers, all tiers) to {verdicts_path}")
            print(f"\U0001f4c1  Verdicts saved to: {verdicts_path}")
        except Exception as exc:
            logger.error(f"Failed to save verdicts CSV: {exc}")
            print(f"❌  Verdicts save failed — check path and permissions: {verdicts_path}")

    llm_note = " | LLM: enabled" if args.llm else " | LLM: disabled (--no-llm)"
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
    source:     str = "live_alpha_us.csv + fundamentals.csv + news.txt",
) -> str:
    """Generate the enhanced report via LLM; fall back to basic if unavailable."""
    from stockfit.llm import StockfitLLMClient

    try:
        client = StockfitLLMClient.from_env()
    except RuntimeError as exc:
        logger.error(f"LLM not available: {exc}")
        logger.info("Falling back to basic report (no LLM).")
        return report.build(verdicts, date_str=date_str, thresholds=thresholds, source=source)

    narratives = client.generate_narratives(verdicts)

    if not narratives:
        logger.warning("LLM returned no narratives — falling back to basic report.")
        return report.build(verdicts, date_str=date_str, thresholds=thresholds, source=source)

    return report.build_enhanced(
        verdicts, narratives,
        date_str=date_str,
        thresholds=thresholds,
        source=source,
    )


def cmd_set_rvol(args) -> None:
    """Persist the sticky RVOL display setting for future `report` runs."""
    enabled = args.state == "on"
    _set_sticky_rvol(enabled)
    print(f"RVOL display: {args.state} (sticky — applies to future `report` runs "
          f"unless overridden with --rvol/--no-rvol)")


def cmd_show_rvol(args) -> None:
    """Print the current sticky RVOL display setting."""
    sticky = _get_sticky_rvol()
    if sticky is None:
        print("RVOL display: off (default — no sticky setting saved)")
    else:
        print(f"RVOL display: {'on' if sticky else 'off'} (sticky)")


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
    p.add_argument("--no-llm", action="store_false", dest="llm",
                   help="Disable LLM narratives; use rules-based report only")
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
    p.add_argument("--rvol", action="store_true",
                   help="Show RVOL (relative volume) column (overrides sticky setting)")
    p.add_argument("--no-rvol", action="store_true",
                   help="Hide RVOL column (overrides sticky setting)")
    p.add_argument("--intraday", action="store_true",
                   help="Fetch live current prices and show intraday %% change vs. "
                        "cached close (network call, off by default)")
    p.add_argument("--update-reference-data", action="store_true",
                   help="Sync NYSE/NASDAQ reference CSVs from "
                        "LondonMarket/Global-Stock-Symbols before generating the report "
                        "(network call, off by default)")
    p.add_argument("--save-verdicts", action="store_true",
                   help="Also save full verdicts (all tiers, all stats) to a "
                        "<report>_verdicts.csv alongside the .md report")
    p.set_defaults(func=cmd_report, llm=True)

    p_set_rvol = sub.add_parser("set-rvol", help="Set sticky RVOL display on/off for future report runs")
    p_set_rvol.add_argument("state", choices=["on", "off"])
    p_set_rvol.set_defaults(func=cmd_set_rvol)

    p_show_rvol = sub.add_parser("show-rvol", help="Show the current sticky RVOL display setting")
    p_show_rvol.set_defaults(func=cmd_show_rvol)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
