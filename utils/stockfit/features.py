"""
stockfit.features — Extract a per-ticker feature dict from ShockArb pipeline outputs.

Pure module: reads three file paths, returns a list of named feature dicts.
No network calls, no side effects.

Feature spec (single source of truth — rules.py and report.py both consume this)
----------------------------------------------------------------------------------
  ticker            str   — ticker symbol
  r_squared         float — R² of the factor model fit (0–1)
  confidence_delta  float — primary ShockArb signal (delta_rel × r²)
  delta_rel         float — return unexplained by macro factors
  actual_return     float — stock’s raw return over the window
  expected_rel      float — model-predicted return (relative)
  residual_vol      float — volatility of the residual series (noise floor)
  price             float | None — last close price
  analyst_target    float | None — consensus analyst price target
  analyst_upside    float | None — (target − price) / price  (signed)
  fwd_pe            float | None — forward P/E
  ttm_eps           float | None — trailing twelve months EPS
  fwd_eps           float | None — forward EPS estimate
  ex_div            str   | None — next ex-dividend date
  div_amt           float | None — dividend amount
  next_earnings     str   | None — next earnings date ('' if blank)
  news_headlines    list[str]    — up to 3 news headlines for this ticker
  target_below_price bool        — True when analyst_target < price (data quality flag)
  earnings_imminent bool         — True when next_earnings is non-empty
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from paths import LIVE_ALPHA_US, FUNDAMENTALS, NEWS  # noqa: E402


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _load_scores(path: str) -> list[dict[str, Any]]:
    """
    Load live_alpha_us.csv into a list of dicts with numeric fields cast to float.

    Expected columns (shockarb score output):
      Ticker, actual_return, expected_rel, expected_abs, delta_rel, delta_abs,
      r_squared, residual_vol, confidence_delta
    """
    rows = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            entry: dict[str, Any] = {}
            # Ticker column may be empty string (index column) or named 'Ticker'/'ticker'
            raw = {k.strip(): v.strip() for k, v in row.items()}
            ticker = (
                raw.get("Ticker") or raw.get("ticker") or raw.get("")
            )
            if not ticker:
                continue
            entry["ticker"] = ticker.upper()
            for col in ("actual_return", "expected_rel", "expected_abs",
                        "delta_rel", "delta_abs", "r_squared",
                        "residual_vol", "confidence_delta"):
                try:
                    entry[col] = float(raw.get(col, "nan"))
                except (ValueError, TypeError):
                    entry[col] = float("nan")
            rows.append(entry)
    except FileNotFoundError:
        pass
    return rows


def _load_fundamentals(path: str) -> dict[str, dict[str, Any]]:
    """
    Load fundamentals.csv into {ticker: {field: value}} dict.

    Expected columns:
      Ticker, Price, Fwd P/E, TTM EPS, Fwd EPS, Next Earnings,
      Est. EPS, Ex-Div, Div Amt, Analyst Tgt
    """
    result: dict[str, dict[str, Any]] = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            raw = {k.strip(): v.strip() for k, v in row.items()}
            ticker = raw.get("Ticker", "").upper()
            if not ticker:
                continue

            def _f(key: str) -> float | None:
                v = raw.get(key, "")
                try:
                    return float(v.replace("$", "").replace(",", "")) if v and v != "—" else None
                except (ValueError, TypeError):
                    return None

            result[ticker] = {
                "price":         _f("Price"),
                "fwd_pe":        _f("Fwd P/E"),
                "ttm_eps":       _f("TTM EPS"),
                "fwd_eps":       _f("Fwd EPS"),
                "analyst_target": _f("Analyst Tgt"),
                "div_amt":       _f("Div Amt"),
                "next_earnings": raw.get("Next Earnings", "").strip("—").strip() or None,
                "ex_div":        raw.get("Ex-Div", "").strip("—").strip() or None,
            }
    except FileNotFoundError:
        pass
    return result


def _load_news(path: str) -> dict[str, list[str]]:
    """
    Parse news.txt into {ticker: [headline, ...]} dict.
    Same separator format as catalyst_feed.txt — 87-dash blocks.
    """
    result: dict[str, list[str]] = {}
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        chunks  = content.split("---" * 29)   # 87-dash separator
        for chunk in chunks:
            file_lines = [l.strip() for l in chunk.strip().split("\n") if l.strip()]
            if not file_lines or not file_lines[0].startswith("["):
                continue
            ticker = file_lines[0].split("]")[0].replace("[", "").strip().upper()
            headlines = []
            for i, line in enumerate(file_lines):
                if line.startswith(">") and i + 1 < len(file_lines):
                    nxt = file_lines[i + 1]
                    if not nxt.startswith(">") and not nxt.startswith("["):
                        headlines.append(nxt)
            result[ticker] = headlines[:3]
    except FileNotFoundError:
        pass
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_all(
    scores_path:       str = str(LIVE_ALPHA_US),
    fundamentals_path: str = str(FUNDAMENTALS),
    news_path:         str = str(NEWS),
) -> list[dict[str, Any]]:
    """
    Extract per-ticker feature dicts from the three pipeline output files.

    Parameters
    ----------
    scores_path       : path to live_alpha_us.csv
    fundamentals_path : path to fundamentals.csv
    news_path         : path to news.txt

    Returns
    -------
    list of feature dicts, one per ticker, sorted by confidence_delta descending.

    Example
    -------
        from stockfit.features import extract_all
        candidates = extract_all()
        for c in candidates:
            print(c["ticker"], c["confidence_delta"], c["analyst_upside"])
    """
    scores = _load_scores(scores_path)
    fundamentals = _load_fundamentals(fundamentals_path)
    news = _load_news(news_path)

    results = []
    for row in scores:
        ticker = row["ticker"]
        fund   = fundamentals.get(ticker, {})

        price  = fund.get("price")
        target = fund.get("analyst_target")

        analyst_upside = None
        if price and target and price > 0:
            analyst_upside = (target - price) / price

        target_below_price = bool(
            price is not None and target is not None and target < price
        )

        earnings_imminent = bool(fund.get("next_earnings"))

        results.append({
            # Signal features
            "ticker":           ticker,
            "r_squared":        row.get("r_squared", float("nan")),
            "confidence_delta": row.get("confidence_delta", float("nan")),
            "delta_rel":        row.get("delta_rel", float("nan")),
            "actual_return":    row.get("actual_return", float("nan")),
            "expected_rel":     row.get("expected_rel", float("nan")),
            "residual_vol":     row.get("residual_vol", float("nan")),
            # Fundamental features
            "price":            price,
            "analyst_target":   target,
            "analyst_upside":   analyst_upside,
            "fwd_pe":           fund.get("fwd_pe"),
            "ttm_eps":          fund.get("ttm_eps"),
            "fwd_eps":          fund.get("fwd_eps"),
            "ex_div":           fund.get("ex_div"),
            "div_amt":          fund.get("div_amt"),
            "next_earnings":    fund.get("next_earnings"),
            # Catalyst features
            "news_headlines":   news.get(ticker, []),
            # Derived flags
            "target_below_price": target_below_price,
            "earnings_imminent":  earnings_imminent,
        })

    results.sort(key=lambda d: d.get("confidence_delta") or float("-inf"), reverse=True)
    return results
