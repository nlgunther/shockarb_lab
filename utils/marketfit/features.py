"""
marketfit.features — Extract a feature vector from a market snapshot.

Pure module: reads a dict (already loaded from market_snapshot.json),
returns a flat dict of named floats. No I/O, no side effects.

Feature spec (single source of truth — rules.py and future model.py both consume this)
---------------------------------------------------------------------------------------
  vix_level          VIX closing level
  vix_chg            VIX % change (positive = fear rising)
  breadth            (# sectors up − # sectors down) / 11  ∈ [-1, +1]
  sector_dispersion  max(sector chg_pct) − min(sector chg_pct)  [pp]
  tech_rel           XLK chg_pct − SPY chg_pct  (positive = tech outperforming)
  qqq_rel            QQQ chg_pct − SPY chg_pct
  tlt_chg            20yr Treasury % change (positive = bonds bid = risk-off)
  hyg_chg            High Yield % change (positive = credit risk-on)
  gold_chg           Gold % change
  oil_chg            Oil (USO) % change
  spy_chg            S&P 500 % change
  iwm_rel            IWM chg_pct − SPY chg_pct  (small-cap vs large-cap)
  overseas_breadth   mean % change across overseas tickers (NaN-safe)

Missing tickers (status=="error") are treated as NaN and noted in `missing`.
"""

from __future__ import annotations

from typing import Any


# Ticker keys used for feature extraction
_SPY   = "SPY"
_QQQ   = "QQQ"
_IWM   = "IWM"
_XLK   = "XLK"
_VIX   = "^VIX"
_TLT   = "TLT"
_HYG   = "HYG"
_GLD   = "GLD"
_USO   = "USO"

_SECTOR_TICKERS = {"XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC"}
_OVERSEAS_GROUP = "overseas"


def extract(snapshot: dict[str, Any]) -> dict[str, float | None]:
    """
    Extract named features from a loaded market snapshot dict.

    Parameters
    ----------
    snapshot : dict
        Parsed content of market_snapshot.json.

    Returns
    -------
    dict with keys matching the feature spec above, plus:
        missing : list[str]  — tickers where status=="error" or chg_pct is None

    Example
    -------
        import json
        snap = json.loads(Path("data/market_snapshot.json").read_text())
        feats = extract(snap)
        print(feats["vix_level"], feats["breadth"])
    """
    by_ticker: dict[str, dict] = {t["ticker"]: t for t in snapshot.get("tickers", [])}

    missing: list[str] = [
        t["ticker"] for t in snapshot.get("tickers", [])
        if t.get("status") == "error" or t.get("chg_pct") is None
    ]

    def chg(ticker: str) -> float | None:
        row = by_ticker.get(ticker)
        if row is None or row.get("status") == "error":
            return None
        return row.get("chg_pct")

    spy_chg = chg(_SPY)

    # Sector breadth: fraction of 11 SPDR sectors that are positive
    sector_chgs = [chg(t) for t in _SECTOR_TICKERS]
    sector_chgs_valid = [c for c in sector_chgs if c is not None]
    breadth = (
        (sum(1 for c in sector_chgs_valid if c > 0) - sum(1 for c in sector_chgs_valid if c < 0))
        / len(sector_chgs_valid)
        if sector_chgs_valid else None
    )
    dispersion = (max(sector_chgs_valid) - min(sector_chgs_valid)) if len(sector_chgs_valid) >= 2 else None

    def rel(ticker: str) -> float | None:
        """Ticker chg minus SPY chg — measures relative performance."""
        t_chg = chg(ticker)
        if t_chg is None or spy_chg is None:
            return None
        return t_chg - spy_chg

    overseas_chgs = [
        t.get("chg_pct") for t in snapshot.get("tickers", [])
        if t.get("group") == _OVERSEAS_GROUP and t.get("chg_pct") is not None
    ]
    overseas_breadth = sum(overseas_chgs) / len(overseas_chgs) if overseas_chgs else None

    vix_row = by_ticker.get(_VIX, {})
    vix_level = vix_row.get("close") if vix_row.get("status") != "error" else None
    vix_chg   = chg(_VIX)

    return {
        "vix_level":         vix_level,
        "vix_chg":           vix_chg,
        "breadth":           breadth,
        "sector_dispersion": dispersion,
        "tech_rel":          rel(_XLK),
        "qqq_rel":           rel(_QQQ),
        "tlt_chg":           chg(_TLT),
        "hyg_chg":           chg(_HYG),
        "gold_chg":          chg(_GLD),
        "oil_chg":           chg(_USO),
        "spy_chg":           spy_chg,
        "iwm_rel":           rel(_IWM),
        "overseas_breadth":  overseas_breadth,
        "missing":           missing,
    }
