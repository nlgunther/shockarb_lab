"""
Tests for utils/qa_audit — deterministic stats_checks, LLM validator
(with an injected fake client, never a real API call), and report
assembly.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import pandas as pd
import pytest

from qa_audit.stats_checks import (
    CheckResult, check_cache_date_alignment, check_return_magnitude_outliers,
    check_r2_distribution, check_pick_count_vs_history, check_upside_sanity,
    check_cluster_concentration, check_tier_count_sanity, run_all_checks,
)
from qa_audit.llm_client import LLMClient, LLMUnavailableError
from qa_audit.llm_validator import (
    select_sample, build_user_prompt, build_market_context,
    _parse_validation_response, validate_pick, run_validation_batch,
    summarize_concordance, ValidationResult, VALIDATION_SYSTEM_PROMPT,
)
from qa_audit.report import build_qa_report
from stockfit.rules import StockVerdict


# =============================================================================
# Helpers
# =============================================================================

def _verdict(
    ticker="ETN", tier="INCLUDE", r_squared=0.75, confidence_delta=0.03,
    analyst_upside=0.10, price=100.0, analyst_target=110.0, cluster=None,
    news_headlines=None, reason="Passes all gates",
) -> StockVerdict:
    return StockVerdict(
        ticker=ticker, tier=tier, reason=reason, r_squared=r_squared,
        confidence_delta=confidence_delta, analyst_upside=analyst_upside,
        price=price, analyst_target=analyst_target, fwd_pe=20.0,
        news_headlines=news_headlines or [], cluster=cluster,
    )


def _feature(ticker="ETN", actual_return=-0.02, expected_rel=0.01, r_squared=0.75,
             confidence_delta=0.03) -> dict:
    return {
        "ticker": ticker, "actual_return": actual_return, "expected_rel": expected_rel,
        "expected_abs": expected_rel, "delta_rel": expected_rel - actual_return,
        "delta_abs": expected_rel - actual_return, "r_squared": r_squared,
        "residual_vol": 0.2, "confidence_delta": confidence_delta,
    }


def _write_daily_parquet(data_dir, ticker: str, dates: list[str], closes: list[float]) -> None:
    path = os.path.join(data_dir, "prices", "daily")
    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame({"adj_close": closes}, index=pd.to_datetime(dates))
    df.to_parquet(os.path.join(path, f"{ticker}.parquet"))


# =============================================================================
# stats_checks
# =============================================================================

class TestCacheDateAlignment:

    def test_all_tickers_aligned_passes(self, tmp_path):
        d = str(tmp_path)
        _write_daily_parquet(d, "AAA", ["2026-08-13", "2026-08-14"], [10.0, 11.0])
        _write_daily_parquet(d, "BBB", ["2026-08-13", "2026-08-14"], [20.0, 21.0])
        result = check_cache_date_alignment(["AAA", "BBB"], d)
        assert result.status == "PASS"

    def test_misaligned_ticker_fails(self, tmp_path):
        d = str(tmp_path)
        _write_daily_parquet(d, "AAA", ["2026-08-13", "2026-08-14"], [10.0, 11.0])
        _write_daily_parquet(d, "BBB", ["2026-08-10", "2026-08-11"], [20.0, 21.0])  # behind
        result = check_cache_date_alignment(["AAA", "BBB"], d)
        assert result.status == "FAIL"
        assert "BBB" in result.details["behind"]

    def test_missing_ticker_fails(self, tmp_path):
        d = str(tmp_path)
        _write_daily_parquet(d, "AAA", ["2026-08-13", "2026-08-14"], [10.0, 11.0])
        result = check_cache_date_alignment(["AAA", "ZZZ"], d)
        assert result.status == "FAIL"
        assert "ZZZ" in result.details["missing"]


class TestReturnMagnitudeOutliers:

    def test_no_outliers_passes(self):
        feats = [_feature(actual_return=0.01), _feature("BBB", actual_return=-0.02)]
        assert check_return_magnitude_outliers(feats, threshold=0.15).status == "PASS"

    def test_large_move_flagged(self):
        feats = [_feature(actual_return=0.25)]
        result = check_return_magnitude_outliers(feats, threshold=0.15)
        assert result.status == "WARN"
        assert "ETN" in result.details["outliers"]

    def test_nan_return_ignored_not_crash(self):
        feats = [_feature(actual_return=float("nan"))]
        result = check_return_magnitude_outliers(feats, threshold=0.15)
        assert result.status == "PASS"


class TestR2Distribution:

    def test_normal_spread_passes(self):
        feats = [_feature(f"T{i}", r_squared=r) for i, r in enumerate([0.3, 0.5, 0.7, 0.9, 0.4, 0.6])]
        assert check_r2_distribution(feats).status == "PASS"

    def test_degenerate_uniform_r2_fails(self):
        feats = [_feature(f"T{i}", r_squared=0.700001) for i in range(10)]
        result = check_r2_distribution(feats)
        assert result.status == "FAIL"

    def test_too_few_values_warns(self):
        feats = [_feature("A", r_squared=0.5)]
        assert check_r2_distribution(feats).status == "WARN"


class TestPickCountVsHistory:

    class _FakeArchive:
        def __init__(self, df):
            self._df = df
        def load_window(self, days=30):
            return self._df

    def test_no_archive_warns(self):
        result = check_pick_count_vs_history(3, 0, 2, archive=None)
        assert result.status == "WARN"

    def test_empty_history_warns(self):
        archive = self._FakeArchive(pd.DataFrame())
        result = check_pick_count_vs_history(3, 0, 2, archive=archive)
        assert result.status == "WARN"

    def test_typical_count_passes(self):
        rows = []
        for day in range(10):
            for i in range(3):  # 3 "passing" tickers/day -> matches today's total
                rows.append({"date": f"2026-08-{day+1:02d}", "ticker": f"T{i}",
                             "r2": 0.8, "conf_delta": 0.03, "regime": "ukraine_shock"})
        archive = self._FakeArchive(pd.DataFrame(rows))
        result = check_pick_count_vs_history(2, 0, 1, archive=archive, regime_name="ukraine_shock")
        assert result.status == "PASS"

    def test_outlier_count_warns(self):
        rows = []
        for day in range(10):
            for i in range(2):  # historically ~2/day
                rows.append({"date": f"2026-08-{day+1:02d}", "ticker": f"T{i}",
                             "r2": 0.8, "conf_delta": 0.03, "regime": "ukraine_shock"})
        archive = self._FakeArchive(pd.DataFrame(rows))
        # today: 15 total, way above historical mean of 2
        result = check_pick_count_vs_history(10, 0, 5, archive=archive, regime_name="ukraine_shock")
        assert result.status == "WARN"

    def test_lowconf_counts_toward_total(self):
        rows = []
        for day in range(10):
            for i in range(3):
                rows.append({"date": f"2026-08-{day+1:02d}", "ticker": f"T{i}",
                             "r2": 0.8, "conf_delta": 0.03, "regime": "ukraine_shock"})
        archive = self._FakeArchive(pd.DataFrame(rows))
        # 1 INCLUDE + 2 LOW_CONFIDENCE + 0 WATCH = 3, matches historical mean of 3
        result = check_pick_count_vs_history(1, 2, 0, archive=archive, regime_name="ukraine_shock")
        assert result.status == "PASS"
        assert result.details["today_total"] == 3


class TestUpsideSanity:

    def test_reasonable_upside_passes(self):
        verdicts = [_verdict(analyst_upside=0.15)]
        assert check_upside_sanity(verdicts).status == "PASS"

    def test_absurd_upside_flagged(self):
        verdicts = [_verdict(ticker="QCOM", tier="INCLUDE", analyst_upside=1.5)]
        result = check_upside_sanity(verdicts)
        assert result.status == "WARN"
        assert "QCOM" in result.details["flagged"]

    def test_exclude_tier_not_checked(self):
        verdicts = [_verdict(tier="EXCLUDE", analyst_upside=5.0)]
        assert check_upside_sanity(verdicts).status == "PASS"

    def test_low_confidence_tier_checked(self):
        verdicts = [_verdict(ticker="ISRG", tier="LOW_CONFIDENCE", analyst_upside=1.5)]
        result = check_upside_sanity(verdicts)
        assert result.status == "WARN"
        assert "ISRG" in result.details["flagged"]


class TestClusterConcentration:

    def test_spread_picks_pass(self):
        verdicts = [
            _verdict("A", cluster="Tech"), _verdict("B", cluster="Industrials"),
            _verdict("C", cluster="Energy"),
        ]
        assert check_cluster_concentration(verdicts).status == "PASS"

    def test_concentrated_picks_warn(self):
        verdicts = [
            _verdict("A", cluster="Semis"), _verdict("B", cluster="Semis"),
            _verdict("C", cluster="Semis"), _verdict("D", cluster="Energy"),
        ]
        result = check_cluster_concentration(verdicts)
        assert result.status == "WARN"
        assert result.details["worst_cluster"] == "Semis"

    def test_no_picks_passes(self):
        assert check_cluster_concentration([_verdict(tier="EXCLUDE")]).status == "PASS"


class TestTierCountSanity:

    def test_normal_split_passes(self):
        verdicts = [_verdict(tier="INCLUDE"), _verdict(tier="WATCH"), _verdict(tier="EXCLUDE")]
        assert check_tier_count_sanity(verdicts).status == "PASS"

    def test_all_include_fails(self):
        verdicts = [_verdict(tier="INCLUDE") for _ in range(5)]
        assert check_tier_count_sanity(verdicts).status == "FAIL"

    def test_all_low_confidence_fails(self):
        verdicts = [_verdict(tier="LOW_CONFIDENCE") for _ in range(5)]
        assert check_tier_count_sanity(verdicts).status == "FAIL"

    def test_normal_split_with_low_confidence_passes(self):
        verdicts = [_verdict(tier="INCLUDE"), _verdict(tier="LOW_CONFIDENCE"),
                    _verdict(tier="WATCH"), _verdict(tier="EXCLUDE")]
        assert check_tier_count_sanity(verdicts).status == "PASS"

    def test_empty_fails(self):
        assert check_tier_count_sanity([]).status == "FAIL"


class TestRunAllChecks:

    def test_returns_seven_results(self, tmp_path):
        d = str(tmp_path)
        _write_daily_parquet(d, "ETN", ["2026-08-13", "2026-08-14"], [100.0, 98.0])
        feats = [_feature("ETN")]
        verdicts = [_verdict("ETN")]
        results = run_all_checks(feats, verdicts, data_dir=d)
        assert len(results) == 7
        assert all(isinstance(r, CheckResult) for r in results)


# =============================================================================
# llm_validator
# =============================================================================

class TestSelectSample:

    def test_stratified_includes_top_conf_delta(self):
        verdicts = [
            _verdict("LOW", confidence_delta=0.02, tier="WATCH"),
            _verdict("HIGH", confidence_delta=0.09, tier="INCLUDE"),
            _verdict("MID", confidence_delta=0.05, tier="INCLUDE"),
        ]
        sample = select_sample(verdicts, n=2, mode="stratified", seed=1)
        tickers = {v.ticker for v in sample}
        assert "HIGH" in tickers
        assert len(sample) == 2

    def test_random_mode_respects_n(self):
        verdicts = [_verdict(f"T{i}", tier="INCLUDE") for i in range(10)]
        sample = select_sample(verdicts, n=4, mode="random", seed=1)
        assert len(sample) == 4

    def test_excludes_tier(self):
        verdicts = [_verdict("A", tier="EXCLUDE"), _verdict("B", tier="INCLUDE")]
        sample = select_sample(verdicts, n=5, mode="random", seed=1)
        assert [v.ticker for v in sample] == ["B"]

    def test_includes_low_confidence_tier(self):
        verdicts = [_verdict("A", tier="EXCLUDE"), _verdict("B", tier="LOW_CONFIDENCE")]
        sample = select_sample(verdicts, n=5, mode="random", seed=1)
        assert [v.ticker for v in sample] == ["B"]

    def test_seed_reproducible(self):
        verdicts = [_verdict(f"T{i}", tier="INCLUDE") for i in range(10)]
        s1 = select_sample(verdicts, n=3, mode="random", seed=42)
        s2 = select_sample(verdicts, n=3, mode="random", seed=42)
        assert [v.ticker for v in s1] == [v.ticker for v in s2]

    def test_empty_candidates_returns_empty(self):
        assert select_sample([_verdict(tier="EXCLUDE")], n=3) == []

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            select_sample([_verdict(tier="INCLUDE")], n=1, mode="bogus")


class TestBuildUserPrompt:

    def test_includes_key_numbers_and_ticker(self):
        prompt = build_user_prompt(
            ticker="KLAC", company_name="KLA Corporation", industry="Semi Equipment",
            actual_return=-0.0086, expected_return=0.0086, r_squared=0.849,
            confidence_delta=0.0284, price=205.76, analyst_target=305.50,
            analyst_upside=0.485, news_headlines=["[RATING] Tough month for KLAC"],
            market_context={"spy_trailing_return": 0.03, "spy_trailing_days": 5,
                             "dispersion_status": "MODERATE"},
            shockarb_tier="INCLUDE", shockarb_reason="Passes all gates",
        )
        assert "KLAC" in prompt
        assert "KLA Corporation" in prompt
        assert "205.76" in prompt
        assert "305.50" in prompt
        assert "48.5%" in prompt
        assert "Tough month for KLAC" in prompt
        assert "3.0%" in prompt or "+3.0%" in prompt

    def test_missing_price_renders_not_available_not_none(self):
        prompt = build_user_prompt(
            ticker="XYZ", company_name="XYZ Corp", industry="Unknown",
            actual_return=0.01, expected_return=0.02, r_squared=0.7,
            confidence_delta=0.02, price=None, analyst_target=None, analyst_upside=None,
            news_headlines=[], market_context={}, shockarb_tier="WATCH", shockarb_reason="thin data",
        )
        assert "None" not in prompt
        assert "not available" in prompt

    def test_no_headlines_says_so(self):
        prompt = build_user_prompt(
            ticker="XYZ", company_name="XYZ Corp", industry="Unknown",
            actual_return=0.01, expected_return=0.02, r_squared=0.7,
            confidence_delta=0.02, price=100.0, analyst_target=110.0, analyst_upside=0.1,
            news_headlines=[], market_context={}, shockarb_tier="WATCH", shockarb_reason="x",
        )
        assert "none attached" in prompt.lower()


class TestParseValidationResponse:

    def test_clean_json(self):
        text = '{"verdict": "AGREE", "confidence": 0.8, "reasoning": "ok"}'
        assert _parse_validation_response(text)["verdict"] == "AGREE"

    def test_fenced_json(self):
        text = '```json\n{"verdict": "DISAGREE"}\n```'
        assert _parse_validation_response(text)["verdict"] == "DISAGREE"

    def test_preamble_before_json(self):
        text = 'Here is my analysis:\n{"verdict": "UNCERTAIN"}'
        assert _parse_validation_response(text)["verdict"] == "UNCERTAIN"

    def test_garbage_returns_empty_dict(self):
        assert _parse_validation_response("not json at all") == {}


class TestValidatePick:

    def test_well_formed_response_parsed(self):
        canned = json.dumps({
            "verdict": "AGREE", "confidence": 0.75, "reasoning": "Looks like a real dislocation.",
            "red_flags": [], "supporting_points": ["No negative news found"],
            "would_need_to_know": ["Confirm live analyst consensus"],
        })
        client = LLMClient(backend="anthropic", api_key="x", model="x",
                            call_fn=lambda s, u: canned)
        result = validate_pick(
            client, _verdict("KLAC"), company_name="KLA Corporation", industry="Semis",
            actual_return=-0.01, expected_return=0.01, market_context={},
        )
        assert result.llm_verdict == "AGREE"
        assert result.llm_confidence == 0.75
        assert result.concordant is True

    def test_malformed_response_becomes_error_not_crash(self):
        client = LLMClient(backend="anthropic", api_key="x", model="x",
                            call_fn=lambda s, u: "garbage")
        result = validate_pick(
            client, _verdict("KLAC"), company_name="KLA Corporation", industry="Semis",
            actual_return=-0.01, expected_return=0.01, market_context={},
        )
        assert result.llm_verdict == "ERROR"

    def test_llm_exception_becomes_error_not_crash(self):
        def _raise(s, u):
            raise RuntimeError("API down")
        client = LLMClient(backend="anthropic", api_key="x", model="x", call_fn=_raise)
        result = validate_pick(
            client, _verdict("KLAC"), company_name="KLA Corporation", industry="Semis",
            actual_return=-0.01, expected_return=0.01, market_context={},
        )
        assert result.llm_verdict == "ERROR"
        assert "API down" in result.llm_reasoning

    def test_disagree_not_concordant(self):
        canned = json.dumps({"verdict": "DISAGREE", "confidence": 0.6, "reasoning": "x"})
        client = LLMClient(backend="anthropic", api_key="x", model="x", call_fn=lambda s, u: canned)
        result = validate_pick(
            client, _verdict("KLAC"), company_name="KLA", industry="Semis",
            actual_return=-0.01, expected_return=0.01, market_context={},
        )
        assert result.concordant is False


class TestRunValidationBatchAndConcordance:

    def test_batch_runs_and_summarizes(self):
        responses = {
            "HIGH": json.dumps({"verdict": "AGREE", "confidence": 0.8, "reasoning": "x"}),
            "MID":  json.dumps({"verdict": "DISAGREE", "confidence": 0.7, "reasoning": "y"}),
        }
        def _call(system, user):
            for ticker, resp in responses.items():
                if f"TICKER: {ticker}" in user:
                    return resp
            return json.dumps({"verdict": "UNCERTAIN"})

        client = LLMClient(backend="anthropic", api_key="x", model="x", call_fn=_call)
        verdicts = [
            _verdict("HIGH", confidence_delta=0.09, tier="INCLUDE"),
            _verdict("MID", confidence_delta=0.05, tier="INCLUDE"),
        ]
        features_by_ticker = {"HIGH": _feature("HIGH"), "MID": _feature("MID")}
        names = {"HIGH": {"Name": "High Corp", "Industry": "Tech"},
                 "MID": {"Name": "Mid Corp", "Industry": "Tech"}}

        results = run_validation_batch(
            client, verdicts, features_by_ticker, names, market_context={},
            n=2, mode="stratified", seed=1,
        )
        summary = summarize_concordance(results)
        assert summary["n"] == 2
        assert summary["agree"] == 1
        assert summary["disagree"] == 1
        assert "MID" in summary["disagreements"]


class TestBuildMarketContext:

    def test_computes_trailing_return(self, tmp_path):
        d = str(tmp_path)
        dates = pd.bdate_range("2026-08-10", periods=6).strftime("%Y-%m-%d").tolist()
        closes = [100, 101, 102, 103, 104, 106]  # 6% over 5 sessions
        _write_daily_parquet(d, "SPY", dates, closes)
        ctx = build_market_context(d, trailing_days=5)
        assert ctx["spy_trailing_return"] == pytest.approx(0.06)

    def test_missing_files_degrade_gracefully(self, tmp_path):
        ctx = build_market_context(str(tmp_path))
        assert ctx["spy_trailing_return"] is None
        assert ctx["dispersion_status"] is None


class TestLLMClientFromEnv:

    def test_no_keys_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(LLMUnavailableError):
            LLMClient.from_env()

    def test_anthropic_preferred(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
        client = LLMClient.from_env()
        assert client.backend == "anthropic"
        assert client.api_key == "a-key"

    def test_gemini_fallback(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
        client = LLMClient.from_env()
        assert client.backend == "gemini"

    def test_model_override(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
        monkeypatch.setenv("SHOCKARB_LLM_MODEL", "claude-custom")
        client = LLMClient.from_env()
        assert client.model == "claude-custom"


# =============================================================================
# report
# =============================================================================

class TestBuildQaReport:

    def test_renders_stats_and_no_llm_note(self):
        stats = [CheckResult(name="cache_date_alignment", status="PASS", message="all good")]
        md = build_qa_report(stats, [], concordance=None, universe_size=10, n_include=1, n_watch=2)
        assert "ShockArb QA Health Check" in md
        assert "cache_date_alignment" in md
        assert "No LLM validation was run" in md

    def test_renders_validation_results(self):
        stats = [CheckResult(name="x", status="PASS", message="ok")]
        val = ValidationResult(
            ticker="KLAC", shockarb_tier="INCLUDE", llm_verdict="AGREE",
            llm_confidence=0.8, llm_reasoning="Looks fine.",
            red_flags=[], supporting_points=["p1"], would_need_to_know=["check target"],
        )
        concordance = summarize_concordance([val])
        md = build_qa_report(stats, [val], concordance, universe_size=5, n_include=1, n_watch=0)
        assert "KLAC" in md
        assert "AGREE" in md
        assert "check target" in md

    def test_failed_check_surfaces_in_bottom_line(self):
        stats = [CheckResult(name="cache_date_alignment", status="FAIL", message="bad")]
        md = build_qa_report(stats, [], None, universe_size=1, n_include=0, n_watch=0)
        assert "do not trust today's picks" in md.lower()
