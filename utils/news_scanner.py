"""
ShockArb News / Catalyst Scanner.

Fetches recent Yahoo Finance headlines for top arbitrage targets and prints
a fundamentals summary table. Output is saved to data/ by default.

Targets can be supplied in three ways (in priority order):

  1. --tickers AAPL MSFT ROK      explicit list (ignores CSV)
  2. --csv path/to/alpha.csv      top-N by confidence_delta from a score CSV
  3. (default)                    data/live_alpha_us.csv

Usage examples
--------------
    # Top 10 from the default US alpha sheet (saves to data/)
    python utils/news_scanner.py

    # Top 5 from a specific CSV
    python utils/news_scanner.py --csv data/live_alpha_us.csv --top 5

    # Explicit tickers, no CSV needed
    python utils/news_scanner.py --tickers ROK CRM CPRT

    # Merge multiple universes
    python utils/news_scanner.py --csv data/live_alpha_us.csv data/live_alpha_global.csv --top 10

    # Custom output directory
    python utils/news_scanner.py --out reports/

    # Suppress file output
    python utils/news_scanner.py --no-out
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd
import yfinance as yf
from loguru import logger

from fundamental_scanner import fetch_fundamentals, print_fundamentals
from paths import FUNDAMENTALS as _FUNDAMENTALS_PATH
from paths import LIVE_ALPHA_US as _LIVE_ALPHA_US_PATH


_DEFAULT_OUT_DIR = str(_FUNDAMENTALS_PATH.parent)


# =============================================================================
# Article parsing
# =============================================================================

def _extract_article(article: dict) -> tuple[str, str, int | None]:
    """
    Parse a yfinance news article dict into (title, publisher, unix_timestamp).

    Handles both the legacy flat format and the newer nested-content format.
    Returns a debug string as title if neither format is recognised.
    """
    # Format 1: legacy flat structure
    if "title" in article and "publisher" in article:
        return (
            article.get("title", ""),
            article.get("publisher", ""),
            article.get("providerPublishTime"),
        )

    # Format 2: nested content dict (introduced ~2023)
    if "content" in article and isinstance(article["content"], dict):
        content   = article["content"]
        title     = content.get("title", "Unknown Title")
        publisher = content.get("provider", {}).get("displayName", "Unknown Publisher")
        pub_time  = content.get("pubDate")

        if isinstance(pub_time, str):
            try:
                dt = datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
                return title, publisher, int(dt.timestamp())
            except ValueError:
                pass
        return title, publisher, pub_time

    # Format 3: unrecognised — surface keys for debugging
    return f"[Unknown format — keys: {list(article.keys())}]", "N/A", None


# =============================================================================
# Core scanner
# =============================================================================

def scan_news(
    csv_paths: list[str] | None = None,
    top_n: int = 10,
    explicit_tickers: list[str] | None = None,
    sort_col: str = "confidence_delta",
    out_dir: str | None = _DEFAULT_OUT_DIR,
) -> None:
    """
    Print recent headlines and a fundamentals table for the top arbitrage targets.

    Parameters
    ----------
    csv_paths : list of str, optional
        Paths to ShockArb score CSV files. Used when explicit_tickers is None.
    top_n : int
        Number of targets to pull from the CSV.
    explicit_tickers : list of str, optional
        If supplied, skip CSV loading entirely and scan exactly these tickers.
    sort_col : str
        Column to sort by when selecting top-N from CSV.
    out_dir : str or None
        Directory to write output files. Two files are written:
            {out_dir}/news.txt        — full headlines output
            {out_dir}/fundamentals.csv — fundamentals table as CSV
        Pass None to suppress file output.
    """
    print(f"\n{'='*95}")
    print("📰  SHOCKARB CATALYST SCANNER")
    print(f"{'='*95}")
    print("Reviewing for earnings reports, downgrades, or fundamental impairments...\n")

    targets: list[dict] = []

    # ------------------------------------------------------------------
    # Route A: explicit ticker list
    # ------------------------------------------------------------------
    if explicit_tickers:
        for t in explicit_tickers:
            targets.append({"ticker": t.strip().upper(), "signal": "N/A (manual)"})

    # ------------------------------------------------------------------
    # Route B: load from CSV(s)
    # ------------------------------------------------------------------
    else:
        dfs = []
        for path in (csv_paths or []):
            if not os.path.exists(path):
                logger.warning(f"CSV not found: {path}")
                continue
            try:
                df = pd.read_csv(path)
                if "Ticker" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "Ticker"})
                dfs.append(df)
            except Exception as exc:
                logger.error(f"Failed to read {path}: {exc}")

        if not dfs:
            logger.error("No valid CSVs loaded and no --tickers provided.")
            return

        master = pd.concat(dfs, ignore_index=True)

        if sort_col not in master.columns:
            fallback = "delta" if "delta" in master.columns else master.columns[-1]
            logger.warning(f"Column '{sort_col}' not found; sorting by '{fallback}'.")
            sort_col = fallback

        buys = master[master[sort_col] > 0].sort_values(sort_col, ascending=False).head(top_n)

        if buys.empty:
            logger.warning("No positive-signal targets found in the supplied CSVs.")
            return

        for _, row in buys.iterrows():
            targets.append({
                "ticker": str(row["Ticker"]).strip().upper(),
                "signal": f"{row[sort_col]:+.2%}",
            })

    # ------------------------------------------------------------------
    # Fetch and print headlines; capture for file output
    # ------------------------------------------------------------------
    headlines_lines: list[str] = []

    def _emit(line: str = "") -> None:
        print(line)
        headlines_lines.append(line)

    for item in targets:
        ticker = item["ticker"]
        _emit(f"[{ticker:<6}]  signal: {item['signal']}")

        try:
            news = yf.Ticker(ticker).news
            if not news:
                _emit("   > No recent news on Yahoo Finance.")
            else:
                for article in news[:3]:
                    title, publisher, ts = _extract_article(article)
                    date_str = (
                        datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                        if isinstance(ts, (int, float))
                        else "Unknown date"
                    )
                    _emit(f"   > {date_str}  |  {publisher}")
                    _emit(f"     {title}")
        except Exception as exc:
            _emit(f"   > Error fetching news: {exc}")

        _emit("-" * 95)

    # ------------------------------------------------------------------
    # Fundamentals table
    # ------------------------------------------------------------------
    all_tickers = [item["ticker"] for item in targets]
    fund_rows: list[dict] = []
    if all_tickers:
        fund_rows = fetch_fundamentals(all_tickers)
        print_fundamentals(fund_rows)

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------
    if out_dir and (headlines_lines or fund_rows):
        os.makedirs(out_dir, exist_ok=True)

        news_path = os.path.join(out_dir, "news.txt")
        with open(news_path, "w", encoding="utf-8") as f:
            f.write("\n".join(headlines_lines) + "\n")
        logger.success(f"Headlines saved: {news_path}")

        if fund_rows:
            fund_path = os.path.join(out_dir, "fundamentals.csv")
            # Drop internal _cached_at key before saving
            clean = [{k: v for k, v in r.items() if k != "_cached_at"} for r in fund_rows]
            pd.DataFrame(clean).to_csv(fund_path, index=False)
            logger.success(f"Fundamentals saved: {fund_path}")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan Yahoo Finance headlines for ShockArb targets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv", nargs="+", default=[],
        help="Path(s) to ShockArb score CSV files",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of top targets to pull from CSV (default: 10)",
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Explicit ticker list — overrides CSV entirely",
    )
    parser.add_argument(
        "--sort", default="confidence_delta",
        help="CSV column to sort by (default: confidence_delta)",
    )
    parser.add_argument(
        "--out", "-o", default=_DEFAULT_OUT_DIR, metavar="DIR",
        help=f"Output directory for news.txt and fundamentals.csv (default: {_DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--no-out", "-sout", action="store_true",
        help="Suppress file output entirely",
    )
    args = parser.parse_args()

    # Default CSV if nothing else provided
    # Absolute, via paths.py — cwd-independent, unlike a "./data/..." literal
    # (this used to break whenever the script was run from utils/, as the
    # docs instruct, rather than from the project root).
    if not args.tickers and not args.csv:
        args.csv = [str(_LIVE_ALPHA_US_PATH)]

    out_dir = None if args.no_out else args.out

    scan_news(args.csv, args.top, args.tickers, args.sort, out_dir)
