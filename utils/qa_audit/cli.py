"""
qa_audit CLI — ShockArb health-check.

Usage
-----
    python -m qa_audit run
    python -m qa_audit run --sample-n 5 --sample-mode random
    python -m qa_audit run --no-llm                    # stats_checks only, no API key needed
    python -m qa_audit run --out reports/qa_20260819.md

Run from the project root with utils/ on PYTHONPATH, same as stockfit/marketfit:
    set PYTHONPATH=%cd%\\utils;%PYTHONPATH%
    python -m qa_audit run
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from loguru import logger

# qa_audit needs shockarb.* (project root package) alongside stockfit.*/
# news_flags (utils-level) — same project-root sys.path trick market_data.py
# already uses, since this package straddles both halves of the codebase.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from paths import LIVE_ALPHA_US, FUNDAMENTALS, NEWS, DATA, REPORTS_DIR  # noqa: E402


def cmd_run(args: argparse.Namespace) -> None:
    from stockfit.features import extract_all
    from stockfit.rules import evaluate_all, MIN_R2, MIN_R2_WATCH, MIN_CONF_DELTA, MIN_ANALYST_UPSIDE

    from qa_audit.stats_checks import run_all_checks
    from qa_audit.llm_client import LLMClient, LLMUnavailableError
    from qa_audit.llm_validator import (
        run_validation_batch, summarize_concordance, build_market_context,
    )
    from qa_audit.report import build_qa_report

    logger.info("Extracting features and computing verdicts...")
    features = extract_all(
        scores_path=args.scores, fundamentals_path=args.fundamentals, news_path=args.news,
    )
    verdicts = evaluate_all(
        features, min_r2=args.min_r2, min_r2_watch=args.min_r2_watch,
        min_conf_delta=args.min_confidence, min_upside=args.min_upside,
    )
    n_include = sum(1 for v in verdicts if v.tier == "INCLUDE")
    n_lowconf = sum(1 for v in verdicts if v.tier == "LOW_CONFIDENCE")
    n_watch   = sum(1 for v in verdicts if v.tier == "WATCH")
    logger.info(
        f"Universe: {len(features)} tickers — {n_include} INCLUDE, "
        f"{n_lowconf} LOW_CONFIDENCE, {n_watch} WATCH"
    )

    archive = None
    try:
        from shockarb.score_history import ScoreArchive
        archive = ScoreArchive(args.data_dir)
    except Exception as exc:
        logger.warning(f"Score archive unavailable — pick-count-vs-history check will be skipped: {exc}")

    logger.info("Running deterministic checks...")
    stats_results = run_all_checks(features, verdicts, data_dir=args.data_dir, archive=archive)
    for r in stats_results:
        level = {"PASS": "info", "WARN": "warning", "FAIL": "error"}[r.status]
        getattr(logger, level)(f"[{r.status}] {r.name}: {r.message}")

    validation_results = []
    concordance = None
    if not args.no_llm:
        try:
            client = LLMClient.from_env()
            logger.info(f"Running LLM validation ({client.backend}, sample n={args.sample_n}, mode={args.sample_mode})...")
            features_by_ticker = {f["ticker"]: f for f in features}
            company_names = _resolve_names([v.ticker for v in verdicts], args.data_dir)
            market_context = build_market_context(args.data_dir)
            validation_results = run_validation_batch(
                client, verdicts, features_by_ticker, company_names, market_context,
                n=args.sample_n, mode=args.sample_mode, seed=args.seed,
            )
            concordance = summarize_concordance(validation_results)
            logger.info(f"Concordance: {concordance}")
        except LLMUnavailableError as exc:
            logger.warning(f"Skipping LLM validation: {exc}")
    else:
        logger.info("--no-llm passed — skipping LLM validation layer.")

    md = build_qa_report(
        stats_results, validation_results, concordance,
        universe_size=len(features), n_include=n_include, n_lowconf=n_lowconf, n_watch=n_watch,
    )

    out_path = args.out or os.path.join(
        args.reports_dir, f"qa_audit_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"\n📁 Saved to: {out_path}")


def _resolve_names(tickers: list[str], data_dir: str) -> dict[str, dict]:
    """Best-effort company-name resolution; falls back to {ticker: ticker} on any failure."""
    try:
        from shockarb.names import TickerReferenceResolver
        from paths import TICKER_CACHE_FILENAME, EXCHANGE_CSV_FILENAMES
        file_paths = [os.path.join(data_dir, f) for f in EXCHANGE_CSV_FILENAMES]
        cache_path = os.path.join(data_dir, TICKER_CACHE_FILENAME)
        resolver = TickerReferenceResolver(file_paths=file_paths, cache_path=cache_path)
        return resolver.get_reference(tickers)
    except Exception as exc:
        logger.warning(f"Name resolution unavailable, falling back to raw tickers: {exc}")
        return {t: {"Name": t, "Industry": "Unknown"} for t in tickers}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="qa_audit",
        description="ShockArb QA health-check — deterministic sanity checks plus an independent LLM validation sample.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="Run the full QA audit")
    p.add_argument("--scores", default=str(LIVE_ALPHA_US), help="Path to live_alpha_us.csv")
    p.add_argument("--fundamentals", default=str(FUNDAMENTALS), help="Path to fundamentals.csv")
    p.add_argument("--news", default=str(NEWS), help="Path to news.txt")
    p.add_argument("--data-dir", default=str(DATA), help="Data directory (parquet cache, score archive)")
    p.add_argument("--reports-dir", default=str(REPORTS_DIR), help="Directory for report output")
    p.add_argument("--out", "-o", default=None, help="Exact output .md path (default: auto-timestamped)")
    p.add_argument("--min-r2", type=float, default=None, help="Override r2 gate (default: stockfit's own default)")
    p.add_argument("--min-r2-watch", type=float, default=None,
                   help="Override the Lower-Confidence tier's r2 floor (default: stockfit's own default, 0.45)")
    p.add_argument("--min-confidence", type=float, default=None, help="Override confidence_delta gate")
    p.add_argument("--min-upside", type=float, default=None, help="Override analyst upside gate")
    p.add_argument("--sample-n", type=int, default=3, help="Number of picks to LLM-validate (default: 3)")
    p.add_argument("--sample-mode", choices=["random", "stratified"], default="stratified",
                    help="'stratified' always includes the top confidence_delta pick (default); 'random' is uniform")
    p.add_argument("--seed", type=int, default=None, help="Random seed for sample selection (reproducibility)")
    p.add_argument("--no-llm", action="store_true", help="Skip the LLM validation layer (stats_checks only)")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args()

    # argparse defaults of None for the gate overrides should fall through to
    # rules.py's own defaults rather than passing None into evaluate_all().
    from stockfit.rules import MIN_R2, MIN_R2_WATCH, MIN_CONF_DELTA, MIN_ANALYST_UPSIDE
    if args.min_r2 is None:
        args.min_r2 = MIN_R2
    if args.min_r2_watch is None:
        args.min_r2_watch = MIN_R2_WATCH
    if args.min_confidence is None:
        args.min_confidence = MIN_CONF_DELTA
    if args.min_upside is None:
        args.min_upside = MIN_ANALYST_UPSIDE

    args.func(args)


if __name__ == "__main__":
    main()
