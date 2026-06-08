"""
Tests for utils/stockfit — features, rules, report, cli.

All tests use synthetic fixtures — no network calls, no real file I/O
(tmp_path used for CLI loader tests).

Coverage
--------
  TestFeatureExtraction    — features.extract_all() from synthetic CSV / news inputs
  TestRulesEngine          — rules.evaluate_all() verdicts on known signal scenarios
  TestClusterAnnotation    — cluster-risk warnings fire when ≥2 INCLUDE names share a cluster
  TestReportBuild          — report.build() produces correct Markdown structure
  TestReportEnhanced       — report.build_enhanced() injects LEARN tags and LLM narratives
  TestCliResolveOut        — cli._resolve_out() timestamp/default filename logic
  TestCliLoaders           — _load_scores(), _load_fundamentals(), _load_news() parsing
"""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from stockfit import features, rules, report
from stockfit.rules import StockVerdict


# =============================================================================
# Helpers — synthetic data builders
# =============================================================================

def _score_row(
    ticker: str = "ETN",
    r2: float = 0.70,
    conf_delta: float = 0.030,
    delta_rel: float = 0.040,
    actual_return: float = -0.05,
    expected_rel: float = -0.01,
    residual_vol: float = 0.25,
) -> dict:
    return {
        "ticker":           ticker,
        "r_squared":        r2,
        "confidence_delta": conf_delta,
        "delta_rel":        delta_rel,
        "actual_return":    actual_return,
        "expected_rel":     expected_rel,
        "residual_vol":     residual_vol,
    }


def _fund_row(
    ticker: str = "ETN",
    price: float = 396.0,
    analyst_target: float = 452.0,
    fwd_pe: float = 25.0,
    ttm_eps: float = 10.0,
    fwd_eps: float = 15.0,
    next_earnings: str = "",
    ex_div: str = "2026-05-07",
    div_amt: float = 4.40,
) -> dict:
    return {
        "price":          price,
        "analyst_target": analyst_target,
        "fwd_pe":         fwd_pe,
        "ttm_eps":        ttm_eps,
        "fwd_eps":        fwd_eps,
        "next_earnings":  next_earnings or None,
        "ex_div":         ex_div,
        "div_amt":        div_amt,
    }


def _make_features(
    ticker: str = "ETN",
    r2: float = 0.70,
    conf_delta: float = 0.030,
    price: float = 396.0,
    analyst_target: float = 452.0,
    next_earnings: str | None = None,
    news: list[str] | None = None,
    target_below_price: bool = False,
) -> dict:
    """Build a synthetic per-ticker feature dict directly."""
    upside = (analyst_target - price) / price if price and analyst_target else None
    if target_below_price:
        analyst_target = price * 0.95
        upside = (analyst_target - price) / price
    return {
        "ticker":             ticker,
        "r_squared":          r2,
        "confidence_delta":   conf_delta,
        "delta_rel":          0.04,
        "actual_return":      -0.05,
        "expected_rel":       -0.01,
        "residual_vol":       0.25,
        "price":              price,
        "analyst_target":     analyst_target,
        "analyst_upside":     upside,
        "fwd_pe":             25.0,
        "ttm_eps":            10.0,
        "fwd_eps":            15.0,
        "ex_div":             "2026-05-07",
        "div_amt":            4.40,
        "next_earnings":      next_earnings,
        "news_headlines":     news or [],
        "target_below_price": target_below_price,
        "earnings_imminent":  bool(next_earnings),
    }


# =============================================================================
# TestFeatureExtraction
# =============================================================================

class TestFeatureExtraction:
    """features.extract_all() parses synthetic CSV + news correctly."""

    def _write_scores_csv(self, tmp_path, rows: list[dict]) -> str:
        path = tmp_path / "live_alpha_us.csv"
        header = ",actual_return,expected_rel,expected_abs,delta_rel,delta_abs,r_squared,residual_vol,confidence_delta"
        lines = [header]
        for r in rows:
            lines.append(
                f"{r['ticker']},{r['actual_return']},{r['expected_rel']},0,{r['delta_rel']},0,"
                f"{r['r_squared']},{r['residual_vol']},{r['confidence_delta']}"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)

    def _write_fundamentals_csv(self, tmp_path, rows: list[dict]) -> str:
        path = tmp_path / "fundamentals.csv"
        header = "Ticker,Price,Fwd P/E,TTM EPS,Fwd EPS,Next Earnings,Est. EPS,Ex-Div,Div Amt,Analyst Tgt"
        lines = [header]
        for r in rows:
            ne = r.get("next_earnings", "—") or "—"
            lines.append(
                f"{r['ticker']},{r['price']},{r.get('fwd_pe', 25.0)},{r.get('ttm_eps', 10.0)},"
                f"{r.get('fwd_eps', 15.0)},{ne},—,2026-05-07,$4.40,{r['analyst_target']}"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)

    def _write_news_txt(self, tmp_path, ticker_headlines: dict[str, list[str]]) -> str:
        path = tmp_path / "news.txt"
        sep = "-" * 87
        blocks = []
        for ticker, headlines in ticker_headlines.items():
            block = f"[{ticker}  ]  signal: +3.00%\n"
            for h in headlines:
                block += f"   > 2026-06-05 09:00  |  Source\n     {h}\n"
            blocks.append(block)
        path.write_text(sep.join(blocks), encoding="utf-8")
        return str(path)

    def test_ticker_extracted_from_csv(self, tmp_path):
        scores_path = self._write_scores_csv(tmp_path, [_score_row("ETN")])
        fund_path   = self._write_fundamentals_csv(tmp_path, [{"ticker": "ETN", "price": 396.0, "analyst_target": 452.0}])
        result = features.extract_all(str(scores_path), str(fund_path), str(tmp_path / "missing.txt"))
        assert len(result) == 1
        assert result[0]["ticker"] == "ETN"

    def test_r2_parsed(self, tmp_path):
        scores_path = self._write_scores_csv(tmp_path, [_score_row("ETN", r2=0.693)])
        fund_path   = self._write_fundamentals_csv(tmp_path, [{"ticker": "ETN", "price": 396.0, "analyst_target": 452.0}])
        result = features.extract_all(str(scores_path), str(fund_path), str(tmp_path / "missing.txt"))
        assert result[0]["r_squared"] == pytest.approx(0.693)

    def test_confidence_delta_parsed(self, tmp_path):
        scores_path = self._write_scores_csv(tmp_path, [_score_row("ETN", conf_delta=0.0283)])
        fund_path   = self._write_fundamentals_csv(tmp_path, [{"ticker": "ETN", "price": 396.0, "analyst_target": 452.0}])
        result = features.extract_all(str(scores_path), str(fund_path), str(tmp_path / "missing.txt"))
        assert result[0]["confidence_delta"] == pytest.approx(0.0283, abs=1e-5)

    def test_analyst_upside_computed(self, tmp_path):
        scores_path = self._write_scores_csv(tmp_path, [_score_row("ETN")])
        fund_path   = self._write_fundamentals_csv(tmp_path, [{"ticker": "ETN", "price": 400.0, "analyst_target": 440.0}])
        result = features.extract_all(str(scores_path), str(fund_path), str(tmp_path / "missing.txt"))
        assert result[0]["analyst_upside"] == pytest.approx(0.10)

    def test_target_below_price_flag(self, tmp_path):
        scores_path = self._write_scores_csv(tmp_path, [_score_row("KLAC")])
        fund_path   = self._write_fundamentals_csv(tmp_path, [{"ticker": "KLAC", "price": 1929.0, "analyst_target": 1855.0}])
        result = features.extract_all(str(scores_path), str(fund_path), str(tmp_path / "missing.txt"))
        assert result[0]["target_below_price"] is True

    def test_earnings_imminent_within_window(self, tmp_path):
        """Earnings 3 days out → imminent under default 14-day window."""
        from datetime import date, timedelta
        soon = (date.today() + timedelta(days=3)).strftime("%Y-%m-%d")
        scores_path = self._write_scores_csv(tmp_path, [_score_row("ORCL")])
        fund_path   = self._write_fundamentals_csv(tmp_path, [
            {"ticker": "ORCL", "price": 213.0, "analyst_target": 251.0, "next_earnings": soon}
        ])
        result = features.extract_all(str(scores_path), str(fund_path), str(tmp_path / "missing.txt"))
        assert result[0]["earnings_imminent"] is True
        assert result[0]["next_earnings"] == soon

    def test_earnings_not_imminent_outside_window(self, tmp_path):
        """Earnings 60 days out → NOT imminent under default 14-day window."""
        from datetime import date, timedelta
        far = (date.today() + timedelta(days=60)).strftime("%Y-%m-%d")
        scores_path = self._write_scores_csv(tmp_path, [_score_row("AMAT")])
        fund_path   = self._write_fundamentals_csv(tmp_path, [
            {"ticker": "AMAT", "price": 453.0, "analyst_target": 511.0, "next_earnings": far}
        ])
        result = features.extract_all(str(scores_path), str(fund_path), str(tmp_path / "missing.txt"))
        assert result[0]["earnings_imminent"] is False

    def test_earnings_window_respected(self, tmp_path):
        """earnings_window=30 flags a date 20 days out; default 14 does not."""
        from datetime import date, timedelta
        target = (date.today() + timedelta(days=20)).strftime("%Y-%m-%d")
        scores_path = self._write_scores_csv(tmp_path, [_score_row("ETN")])
        fund_path   = self._write_fundamentals_csv(tmp_path, [
            {"ticker": "ETN", "price": 396.0, "analyst_target": 452.0, "next_earnings": target}
        ])
        result_default = features.extract_all(str(scores_path), str(fund_path),
                                               str(tmp_path / "missing.txt"), earnings_window=14)
        result_wide    = features.extract_all(str(scores_path), str(fund_path),
                                               str(tmp_path / "missing.txt"), earnings_window=30)
        assert result_default[0]["earnings_imminent"] is False
        assert result_wide[0]["earnings_imminent"] is True

    def test_news_headlines_loaded(self, tmp_path):
        scores_path = self._write_scores_csv(tmp_path, [_score_row("ETN")])
        fund_path   = self._write_fundamentals_csv(tmp_path, [{"ticker": "ETN", "price": 396.0, "analyst_target": 452.0}])
        news_path   = self._write_news_txt(tmp_path, {"ETN": ["Eaton at all-time high", "Dividend hike expected"]})
        result = features.extract_all(str(scores_path), str(fund_path), str(news_path))
        assert "Eaton at all-time high" in result[0]["news_headlines"]

    def test_sorted_by_confidence_delta_descending(self, tmp_path):
        rows = [_score_row("LOW", conf_delta=0.01), _score_row("HIGH", conf_delta=0.05)]
        scores_path = self._write_scores_csv(tmp_path, rows)
        fund_path   = self._write_fundamentals_csv(tmp_path, [
            {"ticker": "LOW", "price": 100.0, "analyst_target": 110.0},
            {"ticker": "HIGH", "price": 100.0, "analyst_target": 115.0},
        ])
        result = features.extract_all(str(scores_path), str(fund_path), str(tmp_path / "missing.txt"))
        assert result[0]["ticker"] == "HIGH"
        assert result[1]["ticker"] == "LOW"

    def test_missing_fundamentals_file_does_not_crash(self, tmp_path):
        scores_path = self._write_scores_csv(tmp_path, [_score_row("ETN")])
        result = features.extract_all(str(scores_path), str(tmp_path / "missing_funds.csv"), str(tmp_path / "missing.txt"))
        assert len(result) == 1
        assert result[0]["analyst_upside"] is None

    def test_missing_scores_file_returns_empty(self, tmp_path):
        result = features.extract_all(str(tmp_path / "missing.csv"), str(tmp_path / "f.csv"), str(tmp_path / "n.txt"))
        assert result == []


# =============================================================================
# TestRulesEngine
# =============================================================================

class TestRulesEngine:
    """rules.evaluate_all() produces correct tiers on hand-crafted feature dicts."""

    def _eval(self, feats_list: list[dict]) -> list[StockVerdict]:
        return rules.evaluate_all(feats_list)

    def test_strong_signal_passes_all_gates(self):
        f = _make_features("ETN", r2=0.693, conf_delta=0.028, price=396.0, analyst_target=452.0)
        v = self._eval([f])
        assert v[0].tier == "INCLUDE"

    def test_low_r2_excluded(self):
        f = _make_features("CDNS", r2=0.50, conf_delta=0.030)
        v = self._eval([f])
        assert v[0].tier == "EXCLUDE"
        assert "r²" in v[0].reason

    def test_low_confidence_delta_excluded(self):
        f = _make_features("HON", r2=0.70, conf_delta=0.001)
        v = self._eval([f])
        assert v[0].tier == "EXCLUDE"
        assert "confidence_delta" in v[0].reason

    def test_thin_upside_is_watch(self):
        f = _make_features("LRCX", r2=0.774, conf_delta=0.056, price=303.0, analyst_target=316.0)
        v = self._eval([f])
        assert v[0].tier == "WATCH"
        assert "thin" in v[0].reason

    def test_target_below_price_excluded(self):
        f = _make_features("KLAC", r2=0.746, conf_delta=0.059, target_below_price=True)
        v = self._eval([f])
        assert v[0].tier == "EXCLUDE"
        assert "data quality" in v[0].reason.lower()

    def test_earnings_imminent_excluded(self):
        f = _make_features("ORCL", r2=0.70, conf_delta=0.05, next_earnings="2026-06-10")
        v = self._eval([f])
        assert v[0].tier == "EXCLUDE"
        assert "Earnings imminent" in v[0].reason

    def test_earnings_included_when_flag_off(self):
        f = _make_features("ORCL", r2=0.70, conf_delta=0.05, next_earnings="2026-06-10")
        v = rules.evaluate_all([f], earnings_exclude=False)
        # passes gates if upside is sufficient
        assert v[0].tier in ("INCLUDE", "WATCH")

    def test_no_analyst_target_is_watch(self):
        f = _make_features("XYZ", r2=0.70, conf_delta=0.03)
        f["analyst_target"] = None
        f["analyst_upside"] = None
        v = self._eval([f])
        assert v[0].tier == "WATCH"
        assert "analyst target unavailable" in v[0].reason.lower()

    def test_sort_order_include_before_watch_before_exclude(self):
        feats = [
            _make_features("EXC", r2=0.50,  conf_delta=0.03),                   # EXCLUDE r2
            _make_features("WAT", r2=0.70,  conf_delta=0.03, price=300.0, analyst_target=312.0),  # WATCH thin upside
            _make_features("INC", r2=0.70,  conf_delta=0.03, price=300.0, analyst_target=340.0),  # INCLUDE
        ]
        verdicts = self._eval(feats)
        tiers = [v.tier for v in verdicts]
        assert tiers.index("INCLUDE") < tiers.index("WATCH") < tiers.index("EXCLUDE")

    def test_threshold_overrides_respected(self):
        """Higher --min-r2 threshold excludes a ticker that would otherwise pass."""
        f = _make_features("ETN", r2=0.65, conf_delta=0.03, price=396.0, analyst_target=452.0)
        v_default = rules.evaluate_all([f], min_r2=0.65)
        v_strict  = rules.evaluate_all([f], min_r2=0.70)
        assert v_default[0].tier == "INCLUDE"
        assert v_strict[0].tier  == "EXCLUDE"

    def test_verdict_has_reason(self):
        f = _make_features("ETN")
        v = self._eval([f])
        assert v[0].reason and len(v[0].reason) > 5

    def test_verdict_r2_field(self):
        f = _make_features("ETN", r2=0.693)
        v = self._eval([f])
        assert v[0].r_squared == pytest.approx(0.693)

    def test_verdict_confidence_delta_field(self):
        f = _make_features("ETN", conf_delta=0.028)
        v = self._eval([f])
        assert v[0].confidence_delta == pytest.approx(0.028)


# =============================================================================
# TestClusterAnnotation
# =============================================================================

class TestClusterAnnotation:
    """Cluster-risk warnings fire when ≥2 INCLUDE names share a sector cluster."""

    def test_single_semi_equipment_no_cluster_warning(self):
        """AMAT alone in INCLUDE — no cluster warning."""
        f = _make_features("AMAT", r2=0.70, conf_delta=0.05, price=453.0, analyst_target=511.0)
        v = rules.evaluate_all([f])
        assert not any("Cluster risk" in w for w in v[0].warnings)

    def test_two_semi_equipment_triggers_cluster_warning(self):
        """AMAT + KLAC-like name both in INCLUDE → cluster warning on both."""
        feats = [
            _make_features("AMAT", r2=0.70, conf_delta=0.05, price=453.0, analyst_target=511.0),
            _make_features("KLAC", r2=0.70, conf_delta=0.05, price=1800.0, analyst_target=2050.0),
        ]
        # Force KLAC target above price
        feats[1]["target_below_price"] = False
        feats[1]["analyst_upside"] = (2050.0 - 1800.0) / 1800.0
        feats[1]["analyst_target"] = 2050.0
        verdicts = rules.evaluate_all(feats)
        include_verdicts = [v for v in verdicts if v.tier == "INCLUDE"]
        assert len(include_verdicts) == 2
        for v in include_verdicts:
            assert any("Cluster risk" in w for w in v.warnings), f"No cluster warning on {v.ticker}"

    def test_cluster_count_in_warning_message(self):
        feats = [
            _make_features("AMAT", r2=0.70, conf_delta=0.05, price=453.0, analyst_target=511.0),
            _make_features("KLAC", r2=0.70, conf_delta=0.05, price=1800.0, analyst_target=2050.0),
        ]
        feats[1]["target_below_price"] = False
        feats[1]["analyst_upside"] = (2050.0 - 1800.0) / 1800.0
        feats[1]["analyst_target"] = 2050.0
        verdicts = rules.evaluate_all(feats)
        include = [v for v in verdicts if v.tier == "INCLUDE"]
        warning_text = " ".join(w for v in include for w in v.warnings)
        assert "2" in warning_text   # count mentioned

    def test_different_clusters_no_cross_cluster_warning(self):
        """AMAT (Semi Equipment) and ADI (Semi Design) in INCLUDE — no cluster warning."""
        feats = [
            _make_features("AMAT", r2=0.70, conf_delta=0.05, price=453.0, analyst_target=511.0),
            _make_features("ADI",  r2=0.70, conf_delta=0.03, price=401.0, analyst_target=451.0),
        ]
        verdicts = rules.evaluate_all(feats)
        for v in verdicts:
            assert not any("Cluster risk" in w for w in v.warnings)


# =============================================================================
# TestReportBuild
# =============================================================================

class TestReportBuild:
    """report.build() produces correct Markdown structure."""

    def _verdicts(self, features_list: list[dict]) -> list[StockVerdict]:
        return rules.evaluate_all(features_list)

    def _build(self, features_list: list[dict], **kwargs) -> str:
        v = self._verdicts(features_list)
        return report.build(v, **kwargs)

    def test_include_header_present(self):
        md = self._build([_make_features("ETN")])
        assert "✅ Act on These" in md

    def test_watch_header_present(self):
        md = self._build([_make_features("ETN")])
        assert "⚠️ Watch" in md

    def test_exclude_header_present(self):
        md = self._build([_make_features("ETN")])
        assert "❌ Excluded" in md

    def test_include_count_in_header(self):
        feats = [
            _make_features("ETN", r2=0.70, conf_delta=0.03, price=396.0, analyst_target=452.0),
            _make_features("ADI", r2=0.70, conf_delta=0.03, price=401.0, analyst_target=451.0),
        ]
        md = self._build(feats)
        assert "2 candidates" in md

    def test_ticker_appears_in_include_section(self):
        md = self._build([_make_features("ETN")])
        assert "ETN" in md

    def test_exclude_reason_in_table(self):
        f = _make_features("CDNS", r2=0.50, conf_delta=0.03)
        md = self._build([f])
        assert "CDNS" in md
        assert "r²" in md

    def test_data_quality_flag_section(self):
        f = _make_features("KLAC", target_below_price=True)
        md = self._build([f])
        assert "Data Quality Flags" in md
        assert "KLAC" in md

    def test_no_data_quality_section_when_no_flags(self):
        f = _make_features("ETN")
        md = self._build([f])
        assert "Data Quality Flags" not in md

    def test_learn_tags_in_include_block(self):
        f = _make_features("ETN")
        md = self._build([f])
        assert "<!-- LEARN " in md
        assert "<!-- /LEARN -->" in md

    def test_learn_tag_has_ticker_attribute(self):
        f = _make_features("ETN")
        md = self._build([f])
        assert 'ticker="ETN"' in md

    def test_learn_tag_has_r2_in_inputs(self):
        f = _make_features("ETN", r2=0.693)
        md = self._build([f])
        assert "r2=0.693" in md

    def test_learn_tag_has_upside_in_inputs(self):
        f = _make_features("ETN", price=396.0, analyst_target=452.0)
        md = self._build([f])
        assert "upside=" in md

    def test_threshold_summary_in_header(self):
        md = self._build([_make_features("ETN")])
        assert "r² ≥" in md
        assert "conf.Δ ≥" in md

    def test_stale_warning_shown(self):
        md = self._build([_make_features("ETN")], stale=True)
        assert "stale" in md.lower()

    def test_no_stale_warning_when_fresh(self):
        md = self._build([_make_features("ETN")], stale=False)
        assert "stale" not in md.lower()

    def test_all_tickers_appear_somewhere(self):
        feats = [
            _make_features("ETN"),
            _make_features("ADI"),
            _make_features("CDNS", r2=0.50),
        ]
        md = self._build(feats)
        for t in ("ETN", "ADI", "CDNS"):
            assert t in md

    def test_custom_thresholds_shown_in_header(self):
        f = _make_features("ETN")
        v = rules.evaluate_all([f])
        md = report.build(v, thresholds={"min_r2": 0.70, "min_conf_delta": 0.025, "min_upside": 0.08})
        assert "0.70" in md
        assert "0.025" in md
        assert "8%" in md

    def test_watch_reason_appears(self):
        f = _make_features("LRCX", r2=0.774, conf_delta=0.056, price=303.0, analyst_target=316.0)
        md = self._build([f])
        assert "thin" in md.lower()

    def test_empty_verdicts_does_not_crash(self):
        md = report.build([])
        assert "✅ Act on These" in md
        assert "No tickers pass" in md


# =============================================================================
# TestReportEnhanced
# =============================================================================

class TestReportEnhanced:
    """report.build_enhanced() injects LLM narratives and LEARN tags correctly."""

    def _build(self, narratives: dict, features_list: list[dict] | None = None) -> str:
        feats = features_list or [_make_features("ETN")]
        v = rules.evaluate_all(feats)
        return report.build_enhanced(v, narratives)

    def test_executive_summary_injected(self):
        narr = {"executive_summary": "Three tickers pass all gates today."}
        md = self._build(narr)
        assert "Three tickers pass all gates today." in md

    def test_executive_summary_learn_tag_present(self):
        narr = {"executive_summary": "Summary text."}
        md = self._build(narr)
        assert 'section="executive_summary"' in md

    def test_picks_analysis_narrative_injected(self):
        narr = {
            "picks_analysis": {"ETN": "Eaton fell with industrials but AI data center demand is intact."}
        }
        md = self._build(narr)
        assert "Eaton fell with industrials" in md

    def test_watch_list_notes_injected(self):
        feats = [_make_features("LRCX", r2=0.774, conf_delta=0.056, price=303.0, analyst_target=316.0)]
        narr = {"watch_list_notes": "LRCX has high r² but thin upside; wait for stabilisation."}
        md = self._build(narr, features_list=feats)
        assert "wait for stabilisation" in md

    def test_risk_factors_injected(self):
        narr = {"risk_factors": "Fed rate hike fears could continue compressing semis."}
        md = self._build(narr)
        assert "Fed rate hike fears" in md

    def test_learn_tags_present_for_include_ticker(self):
        narr = {"picks_analysis": {"ETN": "Narrative text."}}
        md = self._build(narr)
        assert "<!-- LEARN " in md
        assert "<!-- /LEARN -->" in md

    def test_empty_narratives_falls_back_to_basic(self):
        """Empty narratives dict produces a valid basic report, no crash."""
        md = self._build({})
        assert "✅ Act on These" in md

    def test_missing_picks_analysis_key_does_not_crash(self):
        narr = {"executive_summary": "Some summary."}
        md = self._build(narr)
        assert "Some summary." in md

    def test_include_table_present(self):
        narr = {}
        md = self._build(narr)
        assert "| Ticker |" in md

    def test_risk_factors_section_absent_when_not_provided(self):
        md = self._build({"executive_summary": "X"})
        assert "Risk Factors" not in md


# =============================================================================
# TestCliResolveOut
# =============================================================================

class TestCliResolveOut:
    """cli._resolve_out() filename logic."""

    def setup_method(self):
        from stockfit.cli import _resolve_out, _DEFAULT_REPORTS_DIR
        self._resolve_out  = _resolve_out
        self._reports_dir  = _DEFAULT_REPORTS_DIR

    def test_no_timestamp_returns_default(self):
        out = self._resolve_out(None, self._reports_dir, timestamp=False)
        assert out == f"{self._reports_dir}/stock_report.md"

    def test_timestamp_produces_datestamped_name(self):
        out = self._resolve_out(None, self._reports_dir, timestamp=True)
        assert "stock_report_" in out
        import re
        assert re.search(r"\d{8}_\d{4}", out), f"No YYYYMMDD_HHMM in: {out}"

    def test_timestamp_output_ends_with_md(self):
        out = self._resolve_out(None, self._reports_dir, timestamp=True)
        assert out.endswith(".md")

    def test_explicit_out_always_honoured(self):
        assert self._resolve_out("/tmp/custom.md", self._reports_dir, timestamp=True) == "/tmp/custom.md"

    def test_explicit_out_without_timestamp(self):
        assert self._resolve_out("/tmp/out.md", self._reports_dir, timestamp=False) == "/tmp/out.md"

    def test_custom_reports_dir_is_used(self):
        out = self._resolve_out(None, "/my/reports", timestamp=False)
        assert out == "/my/reports/stock_report.md"

    def test_custom_reports_dir_used_in_timestamp_path(self):
        out = self._resolve_out(None, "/my/reports", timestamp=True)
        assert out.startswith("/my/reports/stock_report_")


# =============================================================================
# TestCliLoaders — _load_scores, _load_fundamentals, _load_news
# =============================================================================

class TestCliLoaders:
    """CLI loader functions parse files correctly and fail gracefully."""

    def setup_method(self):
        from stockfit.features import _load_scores, _load_fundamentals, _load_news
        self._load_scores       = _load_scores
        self._load_fundamentals = _load_fundamentals
        self._load_news         = _load_news

    def test_load_scores_returns_list(self, tmp_path):
        csv = tmp_path / "scores.csv"
        csv.write_text(
            ",actual_return,expected_rel,expected_abs,delta_rel,delta_abs,r_squared,residual_vol,confidence_delta\n"
            "ETN,-0.05,-0.01,0,0.04,0,0.693,0.25,0.028\n", encoding="utf-8")
        rows = self._load_scores(str(csv))
        assert len(rows) == 1
        assert rows[0]["ticker"] == "ETN"
        assert rows[0]["r_squared"] == pytest.approx(0.693)

    def test_load_scores_missing_file_returns_empty(self, tmp_path):
        rows = self._load_scores(str(tmp_path / "nonexistent.csv"))
        assert rows == []

    def test_load_fundamentals_returns_dict_keyed_by_ticker(self, tmp_path):
        csv = tmp_path / "fundamentals.csv"
        csv.write_text(
            "Ticker,Price,Fwd P/E,TTM EPS,Fwd EPS,Next Earnings,Est. EPS,Ex-Div,Div Amt,Analyst Tgt\n"
            "ETN,395.94,25.2,10.22,15.72,—,—,2026-05-07,$4.40,451.73\n", encoding="utf-8")
        result = self._load_fundamentals(str(csv))
        assert "ETN" in result
        assert result["ETN"]["price"] == pytest.approx(395.94)
        assert result["ETN"]["analyst_target"] == pytest.approx(451.73)

    def test_load_fundamentals_missing_file_returns_empty(self, tmp_path):
        result = self._load_fundamentals(str(tmp_path / "missing.csv"))
        assert result == {}

    def test_load_fundamentals_target_parsed_without_dollar_sign(self, tmp_path):
        """Analyst Tgt column may or may not have a $ prefix."""
        csv = tmp_path / "fundamentals.csv"
        csv.write_text(
            "Ticker,Price,Fwd P/E,TTM EPS,Fwd EPS,Next Earnings,Est. EPS,Ex-Div,Div Amt,Analyst Tgt\n"
            "ETN,395.94,25.2,10.22,15.72,—,—,2026-05-07,$4.40,$451.73\n", encoding="utf-8")
        result = self._load_fundamentals(str(csv))
        assert result["ETN"]["analyst_target"] == pytest.approx(451.73)

    def test_load_news_returns_dict_keyed_by_ticker(self, tmp_path):
        sep = "-" * 87
        news_txt = (
            "[ETN  ]  signal: +2.83%\n"
            "   > 2026-06-05 08:15  |  Zacks\n"
            "     Data Center Power Demands Push This Dividend Aristocrat to All Time Highs\n"
            f"{sep}\n"
            "[ADI  ]  signal: +3.03%\n"
            "   > 2026-06-05 09:42  |  Insider Monkey\n"
            "     JPMorgan Raises its Price Target on Analog Devices (ADI)\n"
        )
        path = tmp_path / "news.txt"
        path.write_text(news_txt, encoding="utf-8")
        result = self._load_news(str(path))
        assert "ETN" in result
        assert "ADI" in result
        # join headlines to avoid any() pytest-rewriting confusion
        etn_text = " ".join(result["ETN"])
        adi_text = " ".join(result["ADI"])
        assert "Data Center Power Demands" in etn_text
        assert "JPMorgan Raises" in adi_text

    def test_load_news_missing_file_returns_empty(self, tmp_path):
        result = self._load_news(str(tmp_path / "missing.txt"))
        assert result == {}

    def test_load_news_max_three_headlines(self, tmp_path):
        sep = "-" * 87
        block = "[ETN  ]  signal: +2.83%\n"
        for i in range(5):
            block += f"   > 2026-06-05 0{i}:00  |  Source\n     Headline {i}\n"
        path = tmp_path / "news.txt"
        path.write_text(block, encoding="utf-8")
        result = self._load_news(str(path))
        assert len(result.get("ETN", [])) <= 3
