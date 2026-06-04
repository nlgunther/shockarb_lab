"""
Tests for utils/marketfit — features, rules, report.

All tests use synthetic fixtures — no network calls, no file I/O.

Coverage
--------
  TestFeatureExtraction  — features.extract() from synthetic snapshot
  TestRulesEngine        — rules.evaluate() verdicts on known market scenarios
  TestRulesFailover      — model.is_usable() always False until ML implemented
  TestReportBuild        — report.build() produces correct Markdown structure
"""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from marketfit import features, rules, report, model


# =============================================================================
# Fixtures — synthetic market snapshots for known scenarios
# =============================================================================

def _make_snapshot(
    spy_chg: float = 0.0,
    vix_level: float = 17.0,
    vix_chg: float = 0.0,
    tlt_chg: float = 0.0,
    sector_chgs: dict | None = None,
    overseas_chgs: list[float] | None = None,
) -> dict:
    """Build a minimal synthetic market snapshot dict."""
    sectors = {
        "XLK": 0.0, "XLF": 0.0, "XLE": 0.0, "XLV": 0.0, "XLI": 0.0,
        "XLY": 0.0, "XLP": 0.0, "XLU": 0.0, "XLRE": 0.0, "XLB": 0.0, "XLC": 0.0,
    }
    if sector_chgs:
        sectors.update(sector_chgs)

    tickers = [
        {"ticker": "SPY",  "label": "S&P 500",     "group": "us_broad",  "close": 700.0 + spy_chg, "prev": 700.0, "chg_pct": spy_chg,  "status": "ok"},
        {"ticker": "QQQ",  "label": "Nasdaq 100",   "group": "us_broad",  "close": 500.0, "prev": 500.0, "chg_pct": spy_chg * 0.9, "status": "ok"},
        {"ticker": "IWM",  "label": "Russell 2000", "group": "us_broad",  "close": 200.0, "prev": 200.0, "chg_pct": spy_chg * 1.1, "status": "ok"},
        {"ticker": "DIA",  "label": "Dow Jones",    "group": "us_broad",  "close": 400.0, "prev": 400.0, "chg_pct": spy_chg * 0.8, "status": "ok"},
        {"ticker": "^VIX", "label": "VIX",          "group": "risk",      "close": vix_level, "prev": vix_level - vix_chg * vix_level / 100, "chg_pct": vix_chg, "status": "ok"},
        {"ticker": "GLD",  "label": "Gold",         "group": "risk",      "close": 300.0, "prev": 300.0, "chg_pct": 0.0, "status": "ok"},
        {"ticker": "USO",  "label": "Oil",          "group": "risk",      "close": 70.0,  "prev": 70.0,  "chg_pct": 0.0, "status": "ok"},
        {"ticker": "TLT",  "label": "20yr Treasury","group": "bond",      "close": 85.0,  "prev": 85.0,  "chg_pct": tlt_chg, "status": "ok"},
        {"ticker": "IEF",  "label": "7-10yr Tsy",   "group": "bond",      "close": 95.0,  "prev": 95.0,  "chg_pct": 0.0, "status": "ok"},
        {"ticker": "HYG",  "label": "High Yield",   "group": "bond",      "close": 79.0,  "prev": 79.0,  "chg_pct": 0.0, "status": "ok"},
        {"ticker": "LQD",  "label": "IG Corp",      "group": "bond",      "close": 108.0, "prev": 108.0, "chg_pct": 0.0, "status": "ok"},
    ]
    for ticker, chg in sectors.items():
        label = ticker
        tickers.append({
            "ticker": ticker, "label": label, "group": "us_sector",
            "close": 100.0 + chg, "prev": 100.0, "chg_pct": chg, "status": "ok",
        })
    if overseas_chgs is not None:
        for i, chg in enumerate(overseas_chgs):
            tickers.append({
                "ticker": f"^OV{i}", "label": f"Overseas {i}", "group": "overseas",
                "close": 1000.0, "prev": 1000.0, "chg_pct": chg, "status": "ok",
            })
    return {
        "fetched_at": "2026-06-03T21:00:00+00:00",
        "fetched_at_local": "2026-06-03 17:00",
        "tickers": tickers,
    }


# =============================================================================
# TestFeatureExtraction
# =============================================================================

class TestFeatureExtraction:
    """features.extract() computes correct values from synthetic snapshot."""

    def test_vix_level_extracted(self):
        snap = _make_snapshot(vix_level=18.5)
        f = features.extract(snap)
        assert f["vix_level"] == pytest.approx(18.5)

    def test_vix_chg_extracted(self):
        snap = _make_snapshot(vix_chg=3.0)
        f = features.extract(snap)
        assert f["vix_chg"] == pytest.approx(3.0)

    def test_spy_chg_extracted(self):
        snap = _make_snapshot(spy_chg=0.5)
        f = features.extract(snap)
        assert f["spy_chg"] == pytest.approx(0.5)

    def test_tlt_chg_extracted(self):
        snap = _make_snapshot(tlt_chg=0.4)
        f = features.extract(snap)
        assert f["tlt_chg"] == pytest.approx(0.4)

    def test_breadth_all_positive(self):
        """All sectors up → breadth = +1.0."""
        chgs = {t: 0.5 for t in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLRE","XLB","XLC"]}
        snap = _make_snapshot(sector_chgs=chgs)
        f = features.extract(snap)
        assert f["breadth"] == pytest.approx(1.0)

    def test_breadth_all_negative(self):
        """All sectors down → breadth = -1.0."""
        chgs = {t: -0.5 for t in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLRE","XLB","XLC"]}
        snap = _make_snapshot(sector_chgs=chgs)
        f = features.extract(snap)
        assert f["breadth"] == pytest.approx(-1.0)

    def test_sector_dispersion(self):
        """Dispersion = max - min sector chg."""
        chgs = {"XLK": 3.0, "XLF": -1.0}
        snap = _make_snapshot(sector_chgs=chgs)
        f = features.extract(snap)
        assert f["sector_dispersion"] == pytest.approx(4.0)

    def test_tech_rel_vs_spy(self):
        """tech_rel = XLK chg - SPY chg."""
        snap = _make_snapshot(spy_chg=0.5, sector_chgs={"XLK": 2.0})
        f = features.extract(snap)
        assert f["tech_rel"] == pytest.approx(1.5)

    def test_missing_ticker_returns_none(self):
        """Error-status tickers produce None features, not crashes."""
        snap = _make_snapshot()
        # Mark VIX as error
        for t in snap["tickers"]:
            if t["ticker"] == "^VIX":
                t["status"] = "error"
        f = features.extract(snap)
        assert f["vix_level"] is None
        assert f["vix_chg"] is None
        assert "^VIX" in f["missing"]

    def test_overseas_breadth(self):
        snap = _make_snapshot(overseas_chgs=[1.0, -0.5, 0.5])
        f = features.extract(snap)
        assert f["overseas_breadth"] == pytest.approx((1.0 - 0.5 + 0.5) / 3)

    def test_missing_list_empty_on_clean_snapshot(self):
        snap = _make_snapshot()
        f = features.extract(snap)
        assert f["missing"] == []


# =============================================================================
# TestRulesEngine — known market scenarios produce expected verdicts
# =============================================================================

class TestRulesEngine:
    """rules.evaluate() produces correct verdicts on hand-crafted scenarios."""

    def _eval(self, **kwargs) -> rules.Verdict:
        snap = _make_snapshot(**kwargs)
        feats = features.extract(snap)
        return rules.evaluate(feats)

    def test_shock_scenario_is_good(self):
        """VIX >20, SPY -2%, VIX rising → SHOCK trend → GOOD verdict."""
        v = self._eval(
            spy_chg=-2.0, vix_level=22.0, vix_chg=15.0, tlt_chg=0.8,
            sector_chgs={t: -1.5 for t in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLRE","XLB","XLC"]},
        )
        assert v.overall == "GOOD"
        assert v.trend_status == "SHOCK"
        assert v.vix_status == "ELEVATED"

    def test_meltup_scenario_is_poor(self):
        """SPY +1%, VIX <15, all sectors up, tech outperforming → POOR."""
        v = self._eval(
            spy_chg=1.0, vix_level=13.0, vix_chg=-3.0, tlt_chg=-0.2,
            sector_chgs={t: 0.8 for t in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLRE","XLB","XLC"]},
        )
        assert v.overall == "POOR"
        assert v.trend_status == "MELT-UP"
        assert v.vix_status == "LOW"

    def test_caution_mixed_conditions(self):
        """Flat market, moderate VIX, mixed breadth → CAUTION."""
        v = self._eval(spy_chg=0.1, vix_level=17.0, vix_chg=0.5)
        assert v.overall == "CAUTION"

    def test_vix_bands(self):
        assert rules.evaluate(features.extract(_make_snapshot(vix_level=14.0))).vix_status == "LOW"
        assert rules.evaluate(features.extract(_make_snapshot(vix_level=17.5))).vix_status == "MODERATE"
        assert rules.evaluate(features.extract(_make_snapshot(vix_level=21.0))).vix_status == "ELEVATED"

    def test_dispersion_bands(self):
        wide   = _make_snapshot(sector_chgs={"XLK": 3.5, "XLE": -1.0})
        narrow = _make_snapshot(sector_chgs={"XLK": 0.5, "XLE":  0.3})
        assert rules.evaluate(features.extract(wide)).dispersion_status   == "WIDE"
        assert rules.evaluate(features.extract(narrow)).dispersion_status == "NARROW"

    def test_bond_signal(self):
        riskoff = rules.evaluate(features.extract(_make_snapshot(tlt_chg=0.5)))
        riskon  = rules.evaluate(features.extract(_make_snapshot(tlt_chg=-0.5)))
        assert riskoff.bond_status == "RISK-OFF"
        assert riskon.bond_status  == "RISK-ON"

    def test_tech_concentration(self):
        high = _make_snapshot(spy_chg=0.0, sector_chgs={"XLK": 2.0})
        low  = _make_snapshot(spy_chg=0.0, sector_chgs={"XLK": -2.0})
        assert rules.evaluate(features.extract(high)).tech_status == "HIGH"
        assert rules.evaluate(features.extract(low)).tech_status  == "LOW"

    def test_verdict_has_recommendation(self):
        v = self._eval()
        assert v.recommendation
        assert len(v.recommendation) > 10

    def test_verdict_score_is_non_negative(self):
        v = self._eval()
        assert v.score >= 0

    def test_source_is_rules(self):
        v = self._eval()
        assert v.source == "rules"


# =============================================================================
# TestRulesFailover — ML stub always returns unusable
# =============================================================================

class TestRulesFailover:
    """model.is_usable() returns False until ML is implemented."""

    def test_is_usable_always_false(self):
        assert model.is_usable("any/path.joblib") is False

    def test_predict_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            model.predict({}, "any/path.joblib")

    def test_train_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            model.train("training.parquet", "model.joblib")


# =============================================================================
# TestReportBuild — Markdown output structure
# =============================================================================

class TestReportBuild:
    """report.build() produces correct Markdown structure."""

    def _build(self, **kwargs) -> str:
        snap = _make_snapshot(**kwargs)
        feats = features.extract(snap)
        v = rules.evaluate(feats)
        return report.build(snap, v)

    def test_contains_broad_market_header(self):
        md = self._build()
        assert "## 🇺🇸 Broad Market" in md

    def test_contains_sectors_header(self):
        md = self._build()
        assert "## 🏭 Sectors" in md

    def test_contains_bonds_header(self):
        md = self._build()
        assert "## 💵 Bonds & Rates" in md

    def test_contains_overseas_header(self):
        md = self._build()
        assert "## 🌍 Overseas Markets" in md

    def test_contains_risk_gauges_header(self):
        md = self._build()
        assert "## 📉 Risk Gauges" in md

    def test_contains_shockarb_fit_header(self):
        md = self._build()
        assert "## 🎯 ShockArb Fit Analysis" in md

    def test_contains_overall_verdict(self):
        md = self._build()
        assert "Overall Fit:" in md
        assert any(v in md for v in ["GOOD", "CAUTION", "POOR"])

    def test_contains_recommendation(self):
        md = self._build()
        assert "### Recommendation" in md
        assert "> " in md   # blockquote

    def test_stale_warning_shown_when_stale(self):
        snap = _make_snapshot()
        feats = features.extract(snap)
        v = rules.evaluate(feats)
        md = report.build(snap, v, stale=True)
        assert "Stale data" in md

    def test_no_stale_warning_when_fresh(self):
        snap = _make_snapshot()
        feats = features.extract(snap)
        v = rules.evaluate(feats)
        md = report.build(snap, v, stale=False)
        assert "Stale data" not in md

    def test_source_noted_in_footer(self):
        md = self._build()
        assert "rules" in md
