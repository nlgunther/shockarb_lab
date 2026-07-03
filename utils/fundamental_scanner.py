"""
ShockArb Fundamental Scanner.

Fetches key fundamental data for a list of tickers via yfinance and prints
a compact summary table. Designed as a post-scan overlay for news_scanner.py.

Public API
----------
    from fundamental_scanner import fetch_fundamentals, print_fundamentals

    rows = fetch_fundamentals(["BLK", "TXN", "PH"])
    print_fundamentals(rows)

Data pulled per ticker
----------------------
    Price           Last closing price
    Fwd P/E         Forward price/earnings ratio (flagged with '?' if inconsistent)
    TTM EPS         Trailing twelve-month EPS (GAAP)
    Fwd EPS         Forward EPS consensus estimate
    Next Earnings   Expected earnings announcement date
    Est. EPS        Consensus EPS estimate for next quarter
    Ex-Div          Next ex-dividend date (suppressed if > 2 years old)
    Div Amt         Next dividend amount
    Analyst Tgt     Mean analyst 12-month price target

Analyst target priority
-----------------------
    1. yfinance ``targetMeanPrice``  — always fetched
    2. ``data/analyst_overrides.csv`` — manual per-ticker overrides; always wins.

    Edit ``data/analyst_overrides.csv`` to pin a target for any ticker::

        Ticker,Analyst Tgt
        KLAC,1855.00
        QCOM,185.00

    Blank or non-numeric rows are silently skipped.
    Override file path can be changed via the ``overrides_path`` argument.
"""

from __future__ import annotations

import csv as _csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from paths import DATA as _PROJECT_DATA_DIR

# Suppress noisy per-ticker debug lines — callers see INFO and above only.
logger.disable("fundamental_scanner")

# Default paths — both honour SHOCK_ARB_DATA_DIR if set.
# Falls back to paths.py's project-root-anchored DATA dir (not a "./data"
# literal) so this resolves correctly regardless of cwd — matching the
# pattern already used in market_data.py. A bare "./data" default here used
# to silently resolve to utils/data/ (and load no overrides at all) whenever
# this was run from utils/, as the docs instruct, rather than the project root.
_DATA_DIR          = Path(os.environ.get("SHOCK_ARB_DATA_DIR", str(_PROJECT_DATA_DIR)))
_DEFAULT_CACHE     = _DATA_DIR / "fundamentals_cache.json"
_DEFAULT_OVERRIDES = _DATA_DIR / "analyst_overrides.csv"

# Ex-dividend dates older than this many days are almost certainly stale yfinance
# artifacts (e.g. pre-suspension data for companies that no longer pay dividends).
_MAX_EX_DIV_AGE_DAYS = 730

# Fields pulled from yf.Ticker.info — mapped to display names
_INFO_FIELDS: dict[str, str] = {
    "currentPrice":       "Price",
    "forwardPE":          "Fwd P/E",
    "trailingEps":        "TTM EPS",
    "forwardEps":         "Fwd EPS",
    "targetMeanPrice":    "Target",
}


def _safe_get(d: dict, key: str, fmt: str = "") -> str:
    """Return formatted value from dict, or '—' if missing/None."""
    val = d.get(key)
    if val is None or val != val:  # catches None and NaN
        return "—"
    try:
        return format(val, fmt) if fmt else str(val)
    except (ValueError, TypeError):
        return str(val)


def _next_earnings(ticker_obj: yf.Ticker) -> tuple[str, str]:
    """
    Return (date_str, est_eps_str) for the next earnings event.

    earnings_dates is a DataFrame indexed by date (descending), with columns
    'EPS Estimate' and 'Reported EPS'. Future dates have Reported EPS = NaN.
    """
    try:
        df = ticker_obj.earnings_dates
        if df is None or df.empty:
            return "—", "—"
        future = df[df["Reported EPS"].isna()]
        if future.empty:
            return "—", "—"
        next_date = future.index.max()
        date_str  = pd.Timestamp(next_date).strftime("%Y-%m-%d")
        est       = future.loc[next_date, "EPS Estimate"]
        est_str   = f"${est:.2f}" if pd.notna(est) else "—"
        return date_str, est_str
    except Exception:
        return "—", "—"


def _next_dividend(info: dict) -> tuple[str, str]:
    """
    Return (ex_date_str, amount_str) from ticker info dict.

    Ex-dividend dates older than _MAX_EX_DIV_AGE_DAYS are suppressed as '—'
    to avoid displaying stale yfinance artifacts for non-dividend payers.
    """
    ex_ts = info.get("exDividendDate")
    amt   = info.get("dividendRate")

    if ex_ts:
        ex_date  = datetime.fromtimestamp(ex_ts)
        age_days = (datetime.now() - ex_date).days
        ex_str   = "—" if age_days > _MAX_EX_DIV_AGE_DAYS else ex_date.strftime("%Y-%m-%d")
    else:
        ex_str = "—"

    amt_str = f"${amt:.2f}" if amt else "—"
    return ex_str, amt_str


def _validated_pe(info: dict) -> str:
    """
    Return a display string for Forward P/E, flagged with '?' if the reported
    value is inconsistent with price / forwardEps (>25% relative difference).

    yfinance sometimes returns a forwardPE computed against an unadjusted EPS
    while currentPrice is split-adjusted, producing absurdly low multiples.
    The cross-check catches this without needing an external data source.
    """
    reported_pe = info.get("forwardPE")
    price       = info.get("currentPrice")
    fwd_eps     = info.get("forwardEps")

    if reported_pe is None or reported_pe != reported_pe:
        return "—"

    if price and fwd_eps and fwd_eps != 0:
        computed_pe = price / fwd_eps
        if abs(computed_pe - reported_pe) / abs(computed_pe) > 0.25:
            return f"{reported_pe:.1f}?"

    try:
        return format(reported_pe, ".1f")
    except (ValueError, TypeError):
        return str(reported_pe)


def _load_cache(cache_path: Path) -> dict:
    """Load the fundamentals cache from disk. Returns {} if absent or corrupt."""
    try:
        if cache_path.exists():
            return json.loads(cache_path.read_text())
    except Exception:
        pass
    return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    """Write the fundamentals cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))


def _load_overrides(path: Path) -> dict[str, float]:
    """
    Load per-ticker analyst target overrides from a CSV file.

    CSV format (header required)::

        Ticker,Analyst Tgt
        KLAC,1855.00
        QCOM,185.00

    Rows with a blank or non-numeric ``Analyst Tgt`` are silently skipped.
    Missing file is silently ignored (returns ``{}``).

    Returns
    -------
    dict mapping ticker (upper) → float target
    """
    result: dict[str, float] = {}
    try:
        text = path.read_text(encoding="utf-8")
        for row in _csv.DictReader(text.splitlines()):
            ticker = row.get("Ticker", "").strip().upper()
            raw    = row.get("Analyst Tgt", "").strip().replace("$", "").replace(",", "")
            if ticker and raw:
                try:
                    result[ticker] = float(raw)
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning(f"[Overrides] Failed to load {path}: {exc}")
    return result


def fetch_fundamentals(
    tickers: list[str],
    cache_path: Path | str = _DEFAULT_CACHE,
    overrides_path: Path | None = None,
) -> list[dict]:
    """
    Fetch fundamental data for each ticker, write to cache, and return row dicts.

    Each dict contains display-ready strings suitable for tabular printing.
    Missing fields are represented as '—' rather than raising. Results are
    written to ``cache_path`` (default: data/fundamentals_cache.json) keyed by
    ticker so that ``load_cached_fundamentals`` can serve them without a
    network call.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to fetch.
    cache_path : Path or str, optional
        Where to persist the cache. Defaults to ``_DEFAULT_CACHE``.
    overrides_path : Path, optional
        Path to ``analyst_overrides.csv``. Defaults to ``_DEFAULT_OVERRIDES``.
        Overrides are applied last and always win over yfinance values.
        Pass a non-existent path to disable.

    Returns
    -------
    list of dict
        One dict per ticker with keys matching the table columns.

    Example
    -------
        rows = fetch_fundamentals(["BLK", "TXN"])
        print_fundamentals(rows)
    """
    cache_path     = Path(cache_path)
    overrides_path = Path(overrides_path) if overrides_path else _DEFAULT_OVERRIDES
    overrides      = _load_overrides(overrides_path)
    cache = _load_cache(cache_path)
    rows = []
    for symbol in tickers:
        try:
            t    = yf.Ticker(symbol)
            info = t.info or {}

            earn_date, earn_est = _next_earnings(t)
            ex_date, div_amt    = _next_dividend(info)

            yf_target = _safe_get(info, "targetMeanPrice", ".2f")
            override  = overrides.get(symbol.upper())
            tgt       = f"{override:.2f}" if override is not None else yf_target

            row = {
                "Ticker":        symbol,
                "Price":         _safe_get(info, "currentPrice",    ".2f"),
                "Fwd P/E":       _validated_pe(info),
                "TTM EPS":       _safe_get(info, "trailingEps",     ".2f"),
                "Fwd EPS":       _safe_get(info, "forwardEps",      ".2f"),
                "Next Earnings": earn_date,
                "Est. EPS":      earn_est,
                "Ex-Div":        ex_date,
                "Div Amt":       div_amt,
                "Analyst Tgt":   tgt,
            }
            rows.append(row)
            cache[symbol] = {**row, "_cached_at": datetime.now().isoformat(timespec="seconds")}
            logger.debug(f"[Fundamentals] {symbol} ok")
        except Exception as exc:
            logger.warning(f"[Fundamentals] {symbol} failed: {exc}")
            rows.append({"Ticker": symbol, **{k: "ERR" for k in (
                "Price", "Fwd P/E", "TTM EPS", "Fwd EPS",
                "Next Earnings", "Est. EPS", "Ex-Div", "Div Amt", "Analyst Tgt",
            )}})
    _save_cache(cache, cache_path)
    return rows


def load_cached_fundamentals(
    tickers: list[str],
    cache_path: Path | str = _DEFAULT_CACHE,
) -> list[dict]:
    """
    Return fundamentals from the on-disk cache without any network calls.

    Tickers absent from the cache are included as rows of '—' values with a
    note in the Ticker field so they are visible in the table.

    Parameters
    ----------
    tickers : list of str
    cache_path : Path or str, optional

    Returns
    -------
    list of dict — same shape as fetch_fundamentals output, plus '_cached_at'.
    """
    cache = _load_cache(Path(cache_path))
    rows = []
    for symbol in tickers:
        if symbol in cache:
            rows.append(dict(cache[symbol]))
        else:
            rows.append({"Ticker": symbol, **{k: "—" for k in (
                "Price", "Fwd P/E", "TTM EPS", "Fwd EPS",
                "Next Earnings", "Est. EPS", "Ex-Div", "Div Amt", "Analyst Tgt",
            )}, "_cached_at": None})
    return rows


def print_fundamentals(rows: list[dict]) -> None:
    """
    Print a fixed-width fundamental summary table to stdout.

    Parameters
    ----------
    rows : list of dict
        Output of fetch_fundamentals().

    Example
    -------
        print_fundamentals(fetch_fundamentals(["BLK", "TXN", "PH"]))
    """
    if not rows:
        return

    cols = [
        ("Ticker",        6),
        ("Price",         8),
        ("Fwd P/E",       7),
        ("TTM EPS",       8),
        ("Fwd EPS",       8),
        ("Next Earnings", 14),
        ("Est. EPS",      9),
        ("Ex-Div",        10),
        ("Div Amt",       8),
        ("Analyst Tgt",   11),
    ]

    sep   = "  "
    width = sum(w for _, w in cols) + len(sep) * (len(cols) - 1)

    cached_at_vals = [r.get("_cached_at") for r in rows if r.get("_cached_at")]
    cache_note = f"  [cached — oldest: {min(cached_at_vals)}]" if cached_at_vals else ""

    print(f"\n{'='*width}")
    print(f"  FUNDAMENTAL OVERVIEW{cache_note}")
    print(f"{'='*width}")
    header = sep.join(f"{name:<{w}}" for name, w in cols)
    print(f"  {header}")
    print(f"  {'-'*width}")

    for row in rows:
        line = sep.join(
            f"{str(row.get(name, '—')):<{w}}"
            for name, w in cols
        )
        print(f"  {line}")

    print(f"{'='*width}\n")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch and display fundamental data for one or more tickers.",
        epilog=(
            "Examples:\n"
            "  python utils/fundamental_scanner.py TXN CPRT PH\n"
            "  python utils/fundamental_scanner.py TXN --cached\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbol(s)")
    parser.add_argument(
        "--cached", action="store_true",
        help="Display last-cached values without downloading",
    )
    args = parser.parse_args()

    if args.cached:
        print_fundamentals(load_cached_fundamentals(args.tickers))
    else:
        print_fundamentals(fetch_fundamentals(args.tickers))
