"""
stockfit.rules — Deterministic per-ticker verdict from feature dicts.

Pure module: takes the output of features.extract_all() and returns a list
of StockVerdict objects. No I/O, no randomness.

Verdict tiers
-------------
  INCLUDE  — meets all quality gates; act on this signal
  WATCH    — signal is valid but secondary concern warrants caution; monitor
  EXCLUDE  — disqualified; do not trade (reason documented)

Threshold defaults (override via evaluate_all kwargs)
-------------------------------------------------------
  MIN_R2              0.65   — model must explain ≥65% of variance
  MIN_CONF_DELTA      0.020  — signal strength floor
  MIN_ANALYST_UPSIDE  0.05   — analyst must see ≥5% upside from current price
  EARNINGS_EXCLUDE    True   — exclude any ticker with next_earnings set

Data quality rules
------------------
  target_below_price  → EXCLUDE (analyst target corrupted; do not trade)
  analyst_target None → WATCH   (insufficient fundamental data)

Cluster caveat
--------------
  When multiple INCLUDE candidates are in the same sector cluster
  (e.g. ≥3 semiconductor equipment names), the evaluate_all() result
  notes the cluster risk. It does NOT automatically exclude them — that
  judgment belongs to the human or LLM layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_R2             = 0.65
MIN_CONF_DELTA     = 0.020
MIN_ANALYST_UPSIDE = 0.05


# ---------------------------------------------------------------------------
# Sector cluster map (used for cluster-risk annotation)
# ---------------------------------------------------------------------------

_SEMI_EQUIPMENT = {"KLAC", "LRCX", "AMAT", "ONTO", "BRCM", "NVDA"}
_SEMI_DESIGN    = {"QCOM", "ADI", "TXN", "MCHP", "ON", "MRVL"}
_MEGA_CAP_TECH  = {"AAPL", "MSFT", "GOOGL", "META", "AMZN"}

_CLUSTER_MAP: dict[str, str] = {
    **{t: "Semi Equipment" for t in _SEMI_EQUIPMENT},
    **{t: "Semi Design"    for t in _SEMI_DESIGN},
    **{t: "Mega-Cap Tech"  for t in _MEGA_CAP_TECH},
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class StockVerdict:
    """
    Per-ticker verdict from the rules engine.

    Example
    -------
        feats = features.extract_all()
        verdicts = rules.evaluate_all(feats)
        for v in verdicts:
            print(v.ticker, v.tier, v.reason)
            if v.tier == "INCLUDE":
                print(v.as_markdown_row())
    """
    ticker:      str
    tier:        str           # INCLUDE / WATCH / EXCLUDE
    reason:      str           # human-readable reason for tier assignment
    r_squared:   float
    confidence_delta: float
    analyst_upside:   float | None
    price:            float | None
    analyst_target:   float | None
    fwd_pe:           float | None
    news_headlines:   list[str]
    cluster:          str | None   # cluster label if ticker is in a known cluster
    warnings:         list[str] = field(default_factory=list)
    rvol:             float | None = None   # relative volume (informational only)
    rvol_window:      int | None = None     # trailing window (days) used for rvol
    intraday_price:   float | None = None   # live current price (informational only)
    intraday_chg_pct: float | None = None   # (intraday_price - price) / price

    def as_markdown_row(self) -> str:
        """Single table row for the stock report."""
        upside   = f"{self.analyst_upside * 100:+.1f}%" if self.analyst_upside is not None else "—"
        price    = f"\\${self.price:,.2f}" if self.price else "—"
        target   = f"\\${self.analyst_target:,.2f}" if self.analyst_target else "—"
        pe       = f"{self.fwd_pe:.1f}x" if self.fwd_pe else "—"
        rvol     = f"{self.rvol:.1f}x ({self.rvol_window}d)" if self.rvol is not None else "—"
        intraday = f"{self.intraday_chg_pct * 100:+.2f}%" if self.intraday_chg_pct is not None else "—"
        return (
            f"| {self.ticker} | {self.r_squared:.3f} | "
            f"{self.confidence_delta:+.4f} | {price} | {target} | {upside} | {pe} | {rvol} | {intraday} |"
        )


# ---------------------------------------------------------------------------
# Single-ticker evaluation
# ---------------------------------------------------------------------------

def _evaluate_one(
    feats:            dict[str, Any],
    min_r2:           float,
    min_conf_delta:   float,
    min_upside:       float,
    earnings_exclude: bool,
) -> StockVerdict:
    ticker  = feats["ticker"]
    r2      = feats.get("r_squared", 0.0)
    cd      = feats.get("confidence_delta", 0.0)
    upside  = feats.get("analyst_upside")
    cluster = _CLUSTER_MAP.get(ticker)
    warnings: list[str] = []

    # --- Data quality gates (hard EXCLUDE) ---
    if feats.get("target_below_price"):
        return StockVerdict(
            ticker=ticker, tier="EXCLUDE",
            reason="Analyst target below current price — data quality issue; do not act until resolved",
            r_squared=r2, confidence_delta=cd, analyst_upside=upside,
            price=feats.get("price"), analyst_target=feats.get("analyst_target"),
            fwd_pe=feats.get("fwd_pe"), news_headlines=feats.get("news_headlines", []),
            cluster=cluster,
            rvol=feats.get("rvol"), rvol_window=feats.get("rvol_window"),
            intraday_price=feats.get("intraday_price"),
            intraday_chg_pct=feats.get("intraday_chg_pct"),
        )

    # --- Earnings imminent (hard EXCLUDE) ---
    if earnings_exclude and feats.get("earnings_imminent"):
        return StockVerdict(
            ticker=ticker, tier="EXCLUDE",
            reason=f"Earnings imminent ({feats.get('next_earnings')}) — event risk closes the reversion window",
            r_squared=r2, confidence_delta=cd, analyst_upside=upside,
            price=feats.get("price"), analyst_target=feats.get("analyst_target"),
            fwd_pe=feats.get("fwd_pe"), news_headlines=feats.get("news_headlines", []),
            cluster=cluster,
            rvol=feats.get("rvol"), rvol_window=feats.get("rvol_window"),
            intraday_price=feats.get("intraday_price"),
            intraday_chg_pct=feats.get("intraday_chg_pct"),
        )

    # --- Signal strength gates ---
    if r2 < min_r2:
        return StockVerdict(
            ticker=ticker, tier="EXCLUDE",
            reason=f"r²={r2:.3f} below threshold ({min_r2:.2f}) — model fit too weak",
            r_squared=r2, confidence_delta=cd, analyst_upside=upside,
            price=feats.get("price"), analyst_target=feats.get("analyst_target"),
            fwd_pe=feats.get("fwd_pe"), news_headlines=feats.get("news_headlines", []),
            cluster=cluster,
            rvol=feats.get("rvol"), rvol_window=feats.get("rvol_window"),
            intraday_price=feats.get("intraday_price"),
            intraday_chg_pct=feats.get("intraday_chg_pct"),
        )

    if cd < min_conf_delta:
        return StockVerdict(
            ticker=ticker, tier="EXCLUDE",
            reason=f"confidence_delta={cd:+.4f} below threshold ({min_conf_delta:.3f})",
            r_squared=r2, confidence_delta=cd, analyst_upside=upside,
            price=feats.get("price"), analyst_target=feats.get("analyst_target"),
            fwd_pe=feats.get("fwd_pe"), news_headlines=feats.get("news_headlines", []),
            cluster=cluster,
            rvol=feats.get("rvol"), rvol_window=feats.get("rvol_window"),
            intraday_price=feats.get("intraday_price"),
            intraday_chg_pct=feats.get("intraday_chg_pct"),
        )

    # --- Analyst upside gate ---
    if upside is None:
        warnings.append("No analyst target — fundamental check not possible")
        tier   = "WATCH"
        reason = "Signal passes quality gates but analyst target unavailable"
        return StockVerdict(
            ticker=ticker, tier=tier, reason=reason,
            r_squared=r2, confidence_delta=cd, analyst_upside=upside,
            price=feats.get("price"), analyst_target=feats.get("analyst_target"),
            fwd_pe=feats.get("fwd_pe"), news_headlines=feats.get("news_headlines", []),
            cluster=cluster, warnings=warnings,
            rvol=feats.get("rvol"), rvol_window=feats.get("rvol_window"),
            intraday_price=feats.get("intraday_price"),
            intraday_chg_pct=feats.get("intraday_chg_pct"),
        )

    if upside < min_upside:
        return StockVerdict(
            ticker=ticker, tier="WATCH",
            reason=f"Analyst upside {upside * 100:+.1f}% is thin (threshold: {min_upside * 100:.0f}%) — "
                   "signal valid but reward insufficient",
            r_squared=r2, confidence_delta=cd, analyst_upside=upside,
            price=feats.get("price"), analyst_target=feats.get("analyst_target"),
            fwd_pe=feats.get("fwd_pe"), news_headlines=feats.get("news_headlines", []),
            cluster=cluster, warnings=warnings,
            rvol=feats.get("rvol"), rvol_window=feats.get("rvol_window"),
            intraday_price=feats.get("intraday_price"),
            intraday_chg_pct=feats.get("intraday_chg_pct"),
        )

    # --- INCLUDE ---
    return StockVerdict(
        ticker=ticker, tier="INCLUDE",
        reason=f"r²={r2:.3f}, conf.Δ={cd:+.4f}, analyst upside {upside * 100:+.1f}% — all gates pass",
        r_squared=r2, confidence_delta=cd, analyst_upside=upside,
        price=feats.get("price"), analyst_target=feats.get("analyst_target"),
        fwd_pe=feats.get("fwd_pe"), news_headlines=feats.get("news_headlines", []),
        cluster=cluster, warnings=warnings,
        rvol=feats.get("rvol"), rvol_window=feats.get("rvol_window"),
        intraday_price=feats.get("intraday_price"),
        intraday_chg_pct=feats.get("intraday_chg_pct"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_all(
    feature_list:     list[dict[str, Any]],
    min_r2:           float = MIN_R2,
    min_conf_delta:   float = MIN_CONF_DELTA,
    min_upside:       float = MIN_ANALYST_UPSIDE,
    earnings_exclude: bool  = True,
) -> list[StockVerdict]:
    """
    Evaluate all tickers and return a sorted verdict list.

    Sorting: INCLUDE (by confidence_delta desc) → WATCH (by confidence_delta desc) → EXCLUDE.
    Cluster-risk annotations are added to INCLUDE verdicts when ≥2 names share a cluster.

    Parameters
    ----------
    feature_list     : output of features.extract_all()
    min_r2           : minimum R² (default 0.65)
    min_conf_delta   : minimum confidence_delta (default 0.020)
    min_upside       : minimum analyst upside fraction (default 0.05 = 5%)
    earnings_exclude : exclude tickers with imminent earnings (default True)

    Returns
    -------
    list[StockVerdict] — all tickers, sorted INCLUDE → WATCH → EXCLUDE.

    Example
    -------
        feats = features.extract_all()
        verdicts = rules.evaluate_all(feats)
        include = [v for v in verdicts if v.tier == "INCLUDE"]
        print(f"{len(include)} candidates to act on")
    """
    verdicts = [
        _evaluate_one(f, min_r2, min_conf_delta, min_upside, earnings_exclude)
        for f in feature_list
    ]

    # Annotate cluster risk for INCLUDE tickers
    include_verdicts = [v for v in verdicts if v.tier == "INCLUDE"]
    cluster_counts: dict[str, int] = {}
    for v in include_verdicts:
        if v.cluster:
            cluster_counts[v.cluster] = cluster_counts.get(v.cluster, 0) + 1

    for v in include_verdicts:
        if v.cluster and cluster_counts[v.cluster] >= 2:
            count = cluster_counts[v.cluster]
            v.warnings.append(
                f"Cluster risk: {count} names from '{v.cluster}' cluster in INCLUDE tier — "
                "consider limiting to 1–2 names from this group"
            )

    # Sort: INCLUDE → WATCH → EXCLUDE, each by confidence_delta desc
    _order = {"INCLUDE": 0, "WATCH": 1, "EXCLUDE": 2}
    verdicts.sort(key=lambda v: (_order[v.tier], -(v.confidence_delta or 0)))
    return verdicts
