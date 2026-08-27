"""
stockfit.rules — Deterministic per-ticker verdict from feature dicts.

Pure module: takes the output of features.extract_all() and returns a list
of StockVerdict objects. No I/O, no randomness.

Verdict tiers
-------------
  INCLUDE        — meets all quality gates including the full r² bar; act on this signal
  LOW_CONFIDENCE — conf.Δ and analyst upside both clear their gates, but r² sits in the
                   band between MIN_R2_WATCH and MIN_R2: a real, statistically
                   meaningful factor-model fit (see HIL_todo.md R2-GATE-NEAR-MISS,
                   2026-08-21) but weaker than the action bar — review before acting,
                   don't auto-trade
  WATCH          — signal is valid but secondary concern (thin upside / missing
                   analyst target) warrants caution; monitor
  EXCLUDE        — disqualified; do not trade (reason documented)

Threshold defaults (override via evaluate_all kwargs)
-------------------------------------------------------
  MIN_R2              0.65   — model must explain ≥65% of variance to INCLUDE
  MIN_R2_WATCH        0.45   — floor for LOW_CONFIDENCE; below this, r² is treated as
                                too weak a fit to trust at all (still statistically
                                "significant" at n=46/k=3, but not economically
                                meaningful — see the 2026-08-21 r²-cutoff analysis)
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
MIN_R2_WATCH       = 0.45
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
    min_r2_watch:     float,
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
    # Below MIN_R2_WATCH the factor-model fit is too weak to trust at all (it may
    # still be "statistically significant" in a bare F-test sense, but not
    # economically meaningful — see the r²-cutoff analysis in HIL_todo.md,
    # R2-GATE-NEAR-MISS, 2026-08-21). Between MIN_R2_WATCH and MIN_R2, a ticker
    # can still qualify for the LOW_CONFIDENCE tier below if conf.Δ and upside
    # both clear their normal gates.
    if r2 < min_r2_watch:
        return StockVerdict(
            ticker=ticker, tier="EXCLUDE",
            reason=f"r²={r2:.3f} below threshold ({min_r2_watch:.2f}) — model fit too weak",
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

    # --- INCLUDE vs. LOW_CONFIDENCE ---
    # conf.Δ and upside both cleared their gates above; r² decides which of the
    # two action tiers this lands in.
    if r2 >= min_r2:
        tier   = "INCLUDE"
        reason = f"r²={r2:.3f}, conf.Δ={cd:+.4f}, analyst upside {upside * 100:+.1f}% — all gates pass"
    else:
        tier = "LOW_CONFIDENCE"
        reason = (
            f"r²={r2:.3f} is below the {min_r2:.2f} action threshold but above the "
            f"{min_r2_watch:.2f} floor — conf.Δ={cd:+.4f} and analyst upside "
            f"{upside * 100:+.1f}% both clear their gates; weaker model fit means "
            "lower confidence this is a macro-driven dislocation rather than noise"
        )
    return StockVerdict(
        ticker=ticker, tier=tier,
        reason=reason,
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
    min_r2_watch:     float = MIN_R2_WATCH,
    min_conf_delta:   float = MIN_CONF_DELTA,
    min_upside:       float = MIN_ANALYST_UPSIDE,
    earnings_exclude: bool  = True,
) -> list[StockVerdict]:
    """
    Evaluate all tickers and return a sorted verdict list.

    Sorting: INCLUDE → LOW_CONFIDENCE → WATCH → EXCLUDE, each by confidence_delta desc.
    Cluster-risk annotations are added within INCLUDE and, separately, within
    LOW_CONFIDENCE when ≥2 names in that same tier share a cluster — the two
    tiers are counted independently so a lower-confidence pick never inflates
    an action-tier cluster warning or vice versa.

    Parameters
    ----------
    feature_list     : output of features.extract_all()
    min_r2           : minimum R² to reach INCLUDE (default 0.65)
    min_r2_watch     : minimum R² to reach LOW_CONFIDENCE; below this, EXCLUDE
                        regardless of conf.Δ/upside (default 0.45)
    min_conf_delta   : minimum confidence_delta (default 0.020)
    min_upside       : minimum analyst upside fraction (default 0.05 = 5%)
    earnings_exclude : exclude tickers with imminent earnings (default True)

    Returns
    -------
    list[StockVerdict] — all tickers, sorted INCLUDE → LOW_CONFIDENCE → WATCH → EXCLUDE.

    Example
    -------
        feats = features.extract_all()
        verdicts = rules.evaluate_all(feats)
        include = [v for v in verdicts if v.tier == "INCLUDE"]
        print(f"{len(include)} candidates to act on")
    """
    verdicts = [
        _evaluate_one(f, min_r2, min_r2_watch, min_conf_delta, min_upside, earnings_exclude)
        for f in feature_list
    ]

    # Annotate cluster risk within each actionable tier separately.
    for tier_name in ("INCLUDE", "LOW_CONFIDENCE"):
        tier_verdicts = [v for v in verdicts if v.tier == tier_name]
        cluster_counts: dict[str, int] = {}
        for v in tier_verdicts:
            if v.cluster:
                cluster_counts[v.cluster] = cluster_counts.get(v.cluster, 0) + 1

        for v in tier_verdicts:
            if v.cluster and cluster_counts[v.cluster] >= 2:
                count = cluster_counts[v.cluster]
                v.warnings.append(
                    f"Cluster risk: {count} names from '{v.cluster}' cluster in "
                    f"{tier_name} tier — consider limiting to 1–2 names from this group"
                )

    # Sort: INCLUDE → LOW_CONFIDENCE → WATCH → EXCLUDE, each by confidence_delta desc
    _order = {"INCLUDE": 0, "LOW_CONFIDENCE": 1, "WATCH": 2, "EXCLUDE": 3}
    verdicts.sort(key=lambda v: (_order[v.tier], -(v.confidence_delta or 0)))
    return verdicts


# ---------------------------------------------------------------------------
# CSV serialization (--save-verdicts)
# ---------------------------------------------------------------------------

VERDICT_CSV_FIELDS = [
    "ticker", "tier", "reason", "r_squared", "confidence_delta",
    "analyst_upside", "price", "analyst_target", "fwd_pe", "cluster",
    "rvol", "rvol_window", "intraday_price", "intraday_chg_pct",
    "news_headlines", "warnings",
]


def verdicts_to_rows(verdicts: list[StockVerdict]) -> list[dict[str, Any]]:
    """
    Flatten verdicts (all tiers) into CSV-ready row dicts.

    List fields (news_headlines, warnings) are joined with "; " so the
    result writes cleanly with csv.DictWriter. Column order is
    VERDICT_CSV_FIELDS.

    Example
    -------
        rows = verdicts_to_rows(verdicts)
        with open("verdicts.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=VERDICT_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    """
    rows = []
    for v in verdicts:
        row = {
            "ticker":           v.ticker,
            "tier":             v.tier,
            "reason":           v.reason,
            "r_squared":        v.r_squared,
            "confidence_delta": v.confidence_delta,
            "analyst_upside":   v.analyst_upside,
            "price":            v.price,
            "analyst_target":   v.analyst_target,
            "fwd_pe":           v.fwd_pe,
            "cluster":          v.cluster,
            "rvol":             v.rvol,
            "rvol_window":      v.rvol_window,
            "intraday_price":   v.intraday_price,
            "intraday_chg_pct": v.intraday_chg_pct,
            "news_headlines":   "; ".join(v.news_headlines),
            "warnings":         "; ".join(v.warnings),
        }
        rows.append(row)
    return rows
