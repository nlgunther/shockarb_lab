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
    baseline_date: str = "2026-06-03",
    mode: str = "daily",
    fetched_at_local: str | None = None,
    fetched_at_utc: str | None = None,
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
        "fetched_at":       fetched_at_utc or "2026-06-03T21:00:00+00:00",
        "fetched_at_local": fetched_at_local or "2026-06-03 17:00",
        "baseline_date":    baseline_date,
        "mode":             mode,
        "tickers":          tickers,
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


# =============================================================================
# TestReportBaseline — baseline date + mode display in header
# =============================================================================

class TestReportBaseline:
    """report.build() surfaces baseline_date and mode correctly in the header."""

    def _build(self, **kwargs) -> str:
        snap  = _make_snapshot(**kwargs)
        feats = features.extract(snap)
        v     = rules.evaluate(feats)
        return report.build(snap, v)

    def test_daily_after_hours_shows_baseline_date(self):
        """After-hours daily: baseline date appears in header, no warnings."""
        md = self._build(baseline_date="2026-06-03", mode="daily",
                         fetched_at_local="2026-06-03 17:00")
        assert "2026-06-03" in md
        assert "Daily close" in md

    def test_intraday_mode_shows_intraday_label(self):
        """--intraday run: header says Intraday explicitly."""
        md = self._build(baseline_date="2026-06-03", mode="intraday",
                         fetched_at_local="2026-06-04 10:30")
        assert "Intraday" in md
        assert "2026-06-03" in md

    def test_daily_during_market_hours_shows_warning(self):
        """Daily mode fetched at 11:00 ET (market open): show open-market warning."""
        md = self._build(baseline_date="2026-06-03", mode="daily",
                         fetched_at_utc="2026-06-04T15:00:00+00:00")
        assert "Market open" in md
        assert "--intraday" in md

    def test_daily_at_market_open_boundary(self):
        """09:30 ET is market open — warning should fire."""
        md = self._build(mode="daily", fetched_at_utc="2026-06-04T13:30:00+00:00")
        assert "Market open" in md

    def test_daily_before_open_no_warning(self):
        """08:00 ET — premarket, no open-market warning."""
        md = self._build(mode="daily", fetched_at_utc="2026-06-04T12:00:00+00:00")
        assert "Market open" not in md

    def test_daily_at_close_boundary_no_warning(self):
        """16:00 ET exactly — market closed, no warning."""
        md = self._build(mode="daily", fetched_at_utc="2026-06-04T20:00:00+00:00")
        assert "Market open" not in md

    def test_baseline_date_in_footer(self):
        """baseline_date appears in report footer."""
        md = self._build(baseline_date="2026-05-30")
        assert "2026-05-30" in md

    def test_mode_in_footer(self):
        """mode appears in report footer."""
        md = self._build(mode="intraday")
        assert "intraday" in md

    def test_missing_baseline_date_does_not_crash(self):
        """Snapshots without baseline_date (old format) still render."""
        snap = _make_snapshot()
        del snap["baseline_date"]
        feats = features.extract(snap)
        v = rules.evaluate(feats)
        md = report.build(snap, v)
        assert "Overall Fit" in md


# =============================================================================
# TestCliResolveOut — _resolve_out() filename logic
# =============================================================================

class TestCliResolveOut:
    """marketfit.cli._resolve_out() picks the right default output filename."""

    def setup_method(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
        from marketfit.cli import _resolve_out, _DEFAULT_OUT_DAILY, _DEFAULT_OUT_INTRA
        self._resolve_out     = _resolve_out
        self._default_daily   = _DEFAULT_OUT_DAILY
        self._default_intraday = _DEFAULT_OUT_INTRA

    def test_daily_snapshot_uses_daily_default(self):
        snap = {"mode": "daily"}
        assert self._resolve_out(self._default_daily, snap) == self._default_daily

    def test_intraday_snapshot_uses_intraday_default(self):
        snap = {"mode": "intraday"}
        assert self._resolve_out(self._default_daily, snap) == self._default_intraday

    def test_explicit_out_always_honoured(self):
        """User-supplied --out overrides mode-based default."""
        snap = {"mode": "intraday"}
        assert self._resolve_out("/tmp/custom.md", snap) == "/tmp/custom.md"

    def test_missing_mode_falls_back_to_daily(self):
        """Old snapshots without 'mode' field use the daily default."""
        snap = {}
        assert self._resolve_out(self._default_daily, snap) == self._default_daily


# =============================================================================
# TestCliTimestamp — _resolve_out timestamp routing
# =============================================================================

class TestCliTimestamp:
    """_resolve_out() with timestamp=True produces timestamped filenames."""

    def setup_method(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
        from marketfit.cli import _resolve_out, _DEFAULT_OUT_DAILY
        self._resolve_out   = _resolve_out
        self._default_daily = _DEFAULT_OUT_DAILY

    def _snap(self, mode="daily", fetched_at_local="2026-06-04 14:55"):
        return {"mode": mode, "fetched_at_local": fetched_at_local}

    def test_timestamp_daily_includes_datestamp(self):
        out = self._resolve_out(self._default_daily, self._snap(), timestamp=True)
        assert "market_report_" in out
        assert "2026-06-04" in out
        assert out.endswith(".md")

    def test_timestamp_intraday_includes_intraday_in_name(self):
        out = self._resolve_out(self._default_daily, self._snap(mode="intraday"), timestamp=True)
        assert "market_report_intraday_" in out

    def test_timestamp_colons_stripped_from_time(self):
        """14:55 → 1455 in filename (colons invalid on Windows paths)."""
        out = self._resolve_out(self._default_daily, self._snap(fetched_at_local="2026-06-04 14:55"), timestamp=True)
        assert "14:55" not in out
        assert "1455" in out

    def test_explicit_out_ignores_timestamp(self):
        """User-supplied --out is always honoured, even with --timestamp."""
        out = self._resolve_out("/tmp/my_report.md", self._snap(), timestamp=True)
        assert out == "/tmp/my_report.md"

    def test_no_timestamp_returns_default(self):
        out = self._resolve_out(self._default_daily, self._snap(), timestamp=False)
        assert out == self._default_daily


# =============================================================================
# TestReportEnhanced — build_enhanced() structure
# =============================================================================

class TestReportEnhanced:
    """build_enhanced() produces correct <!-- LEARN --> structure."""

    def _build(self, narratives=None, **snap_kwargs):
        snap  = _make_snapshot(**snap_kwargs)
        feats = features.extract(snap)
        v     = rules.evaluate(feats)
        return report.build_enhanced(snap, v, narratives or {})

    def test_contains_learn_tags(self):
        md = self._build()
        assert "<!-- LEARN " in md
        assert "<!-- /LEARN -->" in md

    def test_executive_summary_tag_present(self):
        md = self._build()
        assert 'section="executive_summary"' in md

    def test_all_core_sections_present(self):
        md = self._build()
        for section in ["broad_market_interpretation", "sector_rotation_story",
                        "bond_signal_interpretation", "overseas_read",
                        "risk_gauge_read", "shockarb_fit_analysis"]:
            assert f'section="{section}"' in md, f"missing section: {section}"

    def test_narrative_text_injected(self):
        """When narratives dict is populated, the text appears in the output."""
        narr = {"executive_summary": "Markets surged on strong jobs data."}
        md = self._build(narratives=narr)
        assert "Markets surged on strong jobs data." in md

    def test_fallback_placeholder_when_no_narratives(self):
        """Empty narratives dict produces placeholder text, not a crash."""
        md = self._build(narratives={})
        assert "*(narrative not generated)*" in md

    def test_inputs_attribute_populated(self):
        """inputs= attributes are not empty strings."""
        md = self._build()
        import re
        inputs_matches = re.findall(r'inputs="([^"]+)"', md)
        assert len(inputs_matches) > 0
        assert all(len(v) > 0 for v in inputs_matches)

    def test_llm_noted_in_footer(self):
        md = self._build()
        assert "LLM: enabled" in md

    def test_does_not_crash_with_shock_scenario(self):
        """Shock conditions produce a valid enhanced report."""
        md = self._build(spy_chg=-2.0, vix_level=22.0, vix_chg=15.0, tlt_chg=0.8)
        assert "<!-- LEARN " in md
        assert "GOOD" in md
