"""
marketfit.rules — Deterministic ShockArb-fit condition checks and verdict.

Pure module: takes a feature dict (from features.extract()) and returns a
structured verdict. No I/O, no randomness, no sklearn dependency.

This is the always-available failover that produces reliable output even when
the learned model has insufficient training data or fails to load.

Verdict levels
--------------
  GOOD     — conditions favour running ShockArb (panic, dispersion, negative breadth)
  CAUTION  — mixed signals; run scanner with elevated thresholds
  POOR     — melt-up / complacency; idiosyncratic dislocations unlikely

Condition labels (match the market-report skill's table exactly)
----------------------------------------------------------------
  breadth_status       POSITIVE / MIXED / NEGATIVE
  vix_status           LOW / MODERATE / ELEVATED
  dispersion_status    WIDE / MODERATE / NARROW
  trend_status         MELT-UP / RECOVERY / CHOPPY / SHOCK
  tech_status          HIGH / MODERATE / LOW   (tech outperformance)
  bond_status          RISK-ON / NEUTRAL / RISK-OFF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Thresholds — single source of truth, documented here
# ---------------------------------------------------------------------------

# VIX level bands
_VIX_LOW      = 15.0
_VIX_ELEVATED = 20.0

# Sector dispersion (pp spread between best and worst SPDR sector)
_DISP_WIDE    = 3.0   # pp
_DISP_NARROW  = 1.5   # pp

# Breadth: fraction of sectors positive − fraction negative, ∈ [-1, +1]
_BREADTH_POS  =  0.3   # majority positive
_BREADTH_NEG  = -0.3   # majority negative

# Tech relative performance vs SPY (pp)
_TECH_HIGH    =  1.0   # tech outperforming SPY by > 1 pp
_TECH_LOW     = -1.0   # tech underperforming SPY by > 1 pp

# TLT change: rising bonds = risk-off = better for ShockArb
_TLT_RISKOFF  =  0.3   # pp
_TLT_RISKON   = -0.3   # pp


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    """
    Full ShockArb-fit verdict for one day's market snapshot.

    Example
    -------
        feats = features.extract(snapshot)
        v = evaluate(feats)
        print(v.overall)            # "POOR"
        print(v.recommendation)     # one-sentence action
        print(v.as_markdown_table()) # condition-check table
    """
    breadth_status:    str
    vix_status:        str
    dispersion_status: str
    trend_status:      str
    tech_status:       str
    bond_status:       str
    overall:           str         # GOOD / CAUTION / POOR
    score:             int         # 0–6 internal score (for future calibration)
    recommendation:    str
    notes:             dict[str, str] = field(default_factory=dict)
    source:            str = "rules"

    def as_markdown_table(self) -> str:
        rows = [
            ("**Breadth**",          self.breadth_status,    self.notes.get("breadth", "")),
            ("**Volatility (VIX)**", self.vix_status,        self.notes.get("vix", "")),
            ("**Sector dispersion**",self.dispersion_status, self.notes.get("dispersion", "")),
            ("**Market trend**",     self.trend_status,      self.notes.get("trend", "")),
            ("**Tech concentration**",self.tech_status,      self.notes.get("tech", "")),
            ("**Bond signal**",      self.bond_status,       self.notes.get("bond", "")),
        ]
        header = "| Condition | Status | Notes |\n|-----------|--------|-------|"
        body   = "\n".join(f"| {c} | {s} | {n} |" for c, s, n in rows)
        return f"{header}\n{body}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "breadth_status":    self.breadth_status,
            "vix_status":        self.vix_status,
            "dispersion_status": self.dispersion_status,
            "trend_status":      self.trend_status,
            "tech_status":       self.tech_status,
            "bond_status":       self.bond_status,
            "overall":           self.overall,
            "score":             self.score,
            "recommendation":    self.recommendation,
            "source":            self.source,
        }


# ---------------------------------------------------------------------------
# Condition classifiers (each returns a label + a short note)
# ---------------------------------------------------------------------------

def _breadth(feats: dict) -> tuple[str, str]:
    b = feats.get("breadth")
    if b is None:
        return "MIXED", "breadth unavailable"
    if b >= _BREADTH_POS:
        return "POSITIVE", f"sectors net positive ({b:+.2f})"
    if b <= _BREADTH_NEG:
        return "NEGATIVE", f"sectors net negative ({b:+.2f}) — more dislocation candidates"
    return "MIXED", f"balanced ({b:+.2f})"


def _vix(feats: dict) -> tuple[str, str]:
    level = feats.get("vix_level")
    chg   = feats.get("vix_chg")
    chg_str = f", {chg:+.1f}% today" if chg is not None else ""
    if level is None:
        return "MODERATE", "VIX unavailable"
    if level >= _VIX_ELEVATED:
        return "ELEVATED", f"VIX {level:.1f}{chg_str} — fear elevated, good for ShockArb"
    if level <= _VIX_LOW:
        return "LOW", f"VIX {level:.1f}{chg_str} — complacency, dislocations compressed"
    return "MODERATE", f"VIX {level:.1f}{chg_str}"


def _dispersion(feats: dict) -> tuple[str, str]:
    d = feats.get("sector_dispersion")
    if d is None:
        return "MODERATE", "dispersion unavailable"
    if d >= _DISP_WIDE:
        return "WIDE", f"{d:.1f} pp spread — sectors diverging, candidates available"
    if d <= _DISP_NARROW:
        return "NARROW", f"{d:.1f} pp spread — uniform move, few idiosyncratic names"
    return "MODERATE", f"{d:.1f} pp spread"


def _tech(feats: dict) -> tuple[str, str]:
    t = feats.get("tech_rel")
    if t is None:
        return "MODERATE", "tech relative unavailable"
    if t >= _TECH_HIGH:
        return "HIGH", f"tech outperforming SPY by {t:+.1f} pp — melt-up risk"
    if t <= _TECH_LOW:
        return "LOW", f"tech underperforming SPY by {t:+.1f} pp — rotation out of growth"
    return "MODERATE", f"tech vs SPY: {t:+.1f} pp"


def _bond(feats: dict) -> tuple[str, str]:
    tlt = feats.get("tlt_chg")
    if tlt is None:
        return "NEUTRAL", "TLT unavailable"
    if tlt >= _TLT_RISKOFF:
        return "RISK-OFF", f"TLT {tlt:+.2f}% — bonds bid, risk-off favours ShockArb"
    if tlt <= _TLT_RISKON:
        return "RISK-ON", f"TLT {tlt:+.2f}% — bonds selling, risk-on environment"
    return "NEUTRAL", f"TLT {tlt:+.2f}%"


def _trend(feats: dict) -> str:
    """
    Infer broad market trend from the feature combination.

    SHOCK      — VIX elevated AND broad market down significantly
    MELT-UP    — broad market up, positive breadth, low VIX
    RECOVERY   — broad market recovering after prior selloff (mixed breadth, moderate VIX)
    CHOPPY     — flat or mixed with no clear direction
    """
    spy   = feats.get("spy_chg") or 0.0
    vix   = feats.get("vix_level") or 17.0
    vix_c = feats.get("vix_chg")  or 0.0
    b     = feats.get("breadth")  or 0.0

    if vix >= _VIX_ELEVATED and spy < -0.5 and vix_c > 5:
        return "SHOCK"
    if spy > 0.3 and b >= _BREADTH_POS and vix <= _VIX_LOW:
        return "MELT-UP"
    if spy > 0.1 and b > 0 and vix < _VIX_ELEVATED:
        return "RECOVERY"
    return "CHOPPY"


# ---------------------------------------------------------------------------
# Scoring and overall verdict
# ---------------------------------------------------------------------------

# Each condition contributes to a 0–6 score.
# Higher score = better ShockArb conditions.

_CONDITION_SCORES = {
    # breadth
    "NEGATIVE": 2, "MIXED": 1, "POSITIVE": 0,
    # vix
    "ELEVATED": 2, "MODERATE": 1, "LOW": 0,
    # dispersion
    "WIDE": 2, "NARROW": 0,
    # trend
    "SHOCK": 3, "CHOPPY": 2, "RECOVERY": 1, "MELT-UP": 0,
    # tech (low outperformance = better for ShockArb)
    "LOW": 1, "HIGH": 0,
    # bond
    "RISK-OFF": 1, "NEUTRAL": 0, "RISK-ON": 0,
}


def _score(breadth_s, vix_s, disp_s, trend_s, tech_s, bond_s) -> int:
    return (
        _CONDITION_SCORES.get(breadth_s, 0)
        + _CONDITION_SCORES.get(vix_s, 0)
        + _CONDITION_SCORES.get(disp_s, 0)
        + _CONDITION_SCORES.get(trend_s, 0)
        + _CONDITION_SCORES.get(tech_s, 0)
        + _CONDITION_SCORES.get(bond_s, 0)
    )


def _overall_and_rec(score: int, trend: str) -> tuple[str, str]:
    if score >= 7 or trend == "SHOCK":
        return (
            "GOOD",
            "Conditions favour running ShockArb — run the scanner and focus on "
            "names with r² > 0.50 and confidence_delta > 0.003.",
        )
    if score >= 4:
        return (
            "CAUTION",
            "Mixed conditions — run the scanner but apply elevated thresholds "
            "(r² > 0.55, confidence_delta > 0.003) and avoid clusters of correlated names.",
        )
    return (
        "POOR",
        "Conditions are not favourable — hold off until VIX > 20 or a pullback day "
        "with negative breadth before deploying ShockArb capital.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(feats: dict[str, Any]) -> Verdict:
    """
    Evaluate ShockArb-fit conditions from an extracted feature dict.

    Parameters
    ----------
    feats : dict
        Output of marketfit.features.extract().

    Returns
    -------
    Verdict dataclass with condition labels, overall fit, and recommendation.

    Example
    -------
        import json
        from marketfit import features, rules
        snap  = json.loads(Path("data/market_snapshot.json").read_text())
        feats = features.extract(snap)
        v     = rules.evaluate(feats)
        print(v.overall)
        print(v.as_markdown_table())
    """
    breadth_s,  breadth_note  = _breadth(feats)
    vix_s,      vix_note      = _vix(feats)
    disp_s,     disp_note     = _dispersion(feats)
    tech_s,     tech_note     = _tech(feats)
    bond_s,     bond_note     = _bond(feats)
    trend_s                   = _trend(feats)

    s = _score(breadth_s, vix_s, disp_s, trend_s, tech_s, bond_s)
    overall, rec = _overall_and_rec(s, trend_s)

    return Verdict(
        breadth_status    = breadth_s,
        vix_status        = vix_s,
        dispersion_status = disp_s,
        trend_status      = trend_s,
        tech_status       = tech_s,
        bond_status       = bond_s,
        overall           = overall,
        score             = s,
        recommendation    = rec,
        notes             = {
            "breadth":    breadth_note,
            "vix":        vix_note,
            "dispersion": disp_note,
            "trend":      "",
            "tech":       tech_note,
            "bond":       bond_note,
        },
        source = "rules",
    )
