"""
ShockArb Market Data Fetcher.

Pulls a market snapshot from yfinance and writes it to data/market_snapshot.json.
Run this once before asking Claude for a market report — it takes ~30 seconds and
does all the network work so the report itself is instant.

Usage
-----
    python utils/market_data.py                    # → data/market_snapshot.json
    python utils/market_data.py --out data/snap.json

Schedule (optional)
-------------------
    # Windows Task Scheduler: run at 4:15pm ET daily
    # Or add to your shockarb post-close workflow
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import yfinance as yf
from loguru import logger


_DEFAULT_OUT = "./data/market_snapshot.json"

# ---------------------------------------------------------------------------
# Ticker universe
# ---------------------------------------------------------------------------

_US_BROAD = {
    "SPY":  ("S&P 500",       "us_broad"),
    "QQQ":  ("Nasdaq 100",    "us_broad"),
    "IWM":  ("Russell 2000",  "us_broad"),
    "DIA":  ("Dow Jones",     "us_broad"),
}

_US_SECTORS = {
    "XLK":  ("Tech",              "us_sector"),
    "XLF":  ("Financials",        "us_sector"),
    "XLE":  ("Energy",            "us_sector"),
    "XLV":  ("Health Care",       "us_sector"),
    "XLI":  ("Industrials",       "us_sector"),
    "XLY":  ("Consumer Disc.",    "us_sector"),
    "XLP":  ("Consumer Staples",  "us_sector"),
    "XLU":  ("Utilities",         "us_sector"),
    "XLRE": ("Real Estate",       "us_sector"),
    "XLB":  ("Materials",         "us_sector"),
    "XLC":  ("Comm. Services",    "us_sector"),
}

_BONDS = {
    "TLT":  ("20yr Treasury",     "bond"),
    "IEF":  ("7-10yr Treasury",   "bond"),
    "HYG":  ("High Yield",        "bond"),
    "LQD":  ("Inv. Grade Corp.",  "bond"),
}

_RISK = {
    "^VIX": ("VIX",   "risk"),
    "GLD":  ("Gold",  "risk"),
    "USO":  ("Oil",   "risk"),
}

# Overseas indices — yfinance tickers
_OVERSEAS = {
    "^FTSE":    ("FTSE 100 (London)",       "overseas"),
    "^GDAXI":   ("DAX (Frankfurt)",         "overseas"),
    "^FCHI":    ("CAC 40 (Paris)",          "overseas"),
    "^STOXX50E":("Euro Stoxx 50",           "overseas"),
    "^N225":    ("Nikkei 225 (Tokyo)",      "overseas"),
    "^HSI":     ("Hang Seng (HK)",          "overseas"),
    "000001.SS":("Shanghai Composite",      "overseas"),
    "^BSESN":   ("BSE Sensex (Mumbai)",     "overseas"),
    "^AXJO":    ("ASX 200 (Sydney)",        "overseas"),
    "^BVSP":    ("Bovespa (São Paulo)",     "overseas"),
}

ALL_TICKERS: dict[str, tuple[str, str]] = {
    **_US_BROAD,
    **_US_SECTORS,
    **_BONDS,
    **_RISK,
    **_OVERSEAS,
}


def _fetch_one(ticker: str, label: str, group: str) -> dict:
    """Fetch latest close + prior close for one ticker."""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if len(hist) < 2:
            raise ValueError("insufficient history")
        prev  = float(hist["Close"].iloc[-2])
        close = float(hist["Close"].iloc[-1])
        chg   = (close - prev) / prev
        last_date = hist.index[-1].strftime("%Y-%m-%d")
        return {
            "ticker":    ticker,
            "label":     label,
            "group":     group,
            "close":     round(close, 4),
            "prev":      round(prev, 4),
            "chg_pct":   round(chg * 100, 4),
            "last_date": last_date,
            "status":    "ok",
        }
    except Exception as exc:
        logger.warning(f"{ticker}: {exc}")
        return {
            "ticker":  ticker,
            "label":   label,
            "group":   group,
            "close":   None,
            "prev":    None,
            "chg_pct": None,
            "last_date": None,
            "status":  "error",
        }


def fetch_snapshot(out_path: str = _DEFAULT_OUT) -> dict:
    """
    Fetch a full market snapshot and write it to out_path.

    Returns the snapshot dict.
    """
    logger.info(f"Fetching market snapshot ({len(ALL_TICKERS)} tickers)…")
    data = []
    for i, (ticker, (label, group)) in enumerate(ALL_TICKERS.items(), 1):
        logger.info(f"  [{i:2d}/{len(ALL_TICKERS)}] {ticker:<12} {label}")
        data.append(_fetch_one(ticker, label, group))

    snapshot = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "fetched_at_local": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tickers": data,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    logger.success(f"Snapshot saved: {out_path}  ({len(data)} tickers)")
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch market snapshot for ShockArb market report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out", "-o", default=_DEFAULT_OUT,
        help=f"Output path (default: {_DEFAULT_OUT})",
    )
    args = parser.parse_args()
    fetch_snapshot(args.out)
