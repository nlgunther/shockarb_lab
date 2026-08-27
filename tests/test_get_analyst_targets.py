"""
Tests for get_analyst_targets.py.

Covers the pieces testable without network access or a GOOGLE_API_KEY:
  - `_extract_json`      — JSON-parsing helper used by GeminiSearchProvider
  - `_format_llm_note`   — comparison-note text for combined output
  - `_combine_llm_results` — merges LLM cross-check columns onto out_df
  - `_fetch_llm_estimates` — orchestration loop, with GeminiSearchProvider mocked
"""

from unittest.mock import patch

import pandas as pd
import pytest

from get_analyst_targets import (
    _combine_llm_results,
    _extract_json,
    _fetch_llm_estimates,
    _format_llm_note,
    _is_permission_denied,
)


class TestIsPermissionDenied:
    def test_true_for_403(self):
        exc = Exception(
            "403 PERMISSION_DENIED. {'error': {'code': 403, "
            "'message': 'Lightning dunning decision is deny for project: "
            "projects/617619678027', 'status': 'PERMISSION_DENIED'}}"
        )
        assert _is_permission_denied(exc) is True

    def test_false_for_429(self):
        assert _is_permission_denied(Exception("429 RESOURCE_EXHAUSTED.")) is False

    def test_false_for_503(self):
        assert _is_permission_denied(Exception("503 UNAVAILABLE.")) is False

    def test_false_for_message_with_no_leading_code(self):
        assert _is_permission_denied(Exception("connection reset by peer")) is False


class TestExtractJson:
    def test_parses_clean_json(self):
        text = '{"target_mean": 277.34, "estimates": []}'
        assert _extract_json(text) == {"target_mean": 277.34, "estimates": []}

    def test_parses_json_wrapped_in_markdown_fence(self):
        text = 'Here you go:\n```json\n{"target_mean": 277.34}\n```'
        assert _extract_json(text) == {"target_mean": 277.34}

    def test_parses_json_with_leading_prose(self):
        text = 'Based on my search, here are the estimates: {"target_mean": 195.0}'
        assert _extract_json(text) == {"target_mean": 195.0}

    def test_returns_none_when_no_json_object_present(self):
        assert _extract_json("I could not find any analyst estimates.") is None

    def test_returns_none_on_malformed_json(self):
        assert _extract_json('{"target_mean": 277.34,}') is None

    def test_parses_nested_estimates_list(self):
        text = (
            '{"target_mean": 277.34, "estimates": '
            '[{"firm": "Cantor Fitzgerald", "target": 325.0, "date": "2026-06-15"}]}'
        )
        parsed = _extract_json(text)
        assert parsed["estimates"][0]["firm"] == "Cantor Fitzgerald"
        assert parsed["estimates"][0]["date"] == "2026-06-15"

    def test_does_not_splice_across_multiple_brace_groups(self):
        """
        Regression test: a naive greedy `\\{.*\\}` regex would span from the
        first "{" to the very last "}" here, producing an unparseable mix of
        both objects. Brace-counting should instead return the first
        *balanced* object intact.
        """
        text = 'Example format: {"a": 1}. Actual answer: {"target_mean": 277.34}'
        assert _extract_json(text) == {"a": 1}


class TestFormatLlmNote:
    def test_both_sources_present_shows_delta(self):
        llm_result = {
            "Target_Mean": 277.34,
            "Num_Analysts": 4,
            "Estimates_JSON": '[{"firm": "Cantor Fitzgerald", "target": 325.0, "date": "2026-06-15"}]',
        }
        note = _format_llm_note("finviz", 214.21, llm_result)
        assert "$214.21" in note
        assert "$277.34" in note
        assert "n=4" in note
        assert "2026-06-15" in note
        assert "+29.5%" in note  # (277.34 - 214.21) / 214.21 * 100

    def test_no_main_value_still_reports_llm(self):
        note = _format_llm_note("finviz", None, {"Target_Mean": 195.0, "Estimates_JSON": "[]"})
        assert "No finviz target" in note
        assert "$195.00" in note

    def test_llm_failure_reports_reason_and_falls_back(self):
        note = _format_llm_note("finviz", 214.21, "GOOGLE_API_KEY environment variable is not set.")
        assert "LLM cross-check unavailable" in note
        assert "finviz value only" in note

    def test_llm_failure_with_no_main_value_either(self):
        note = _format_llm_note("finviz", None, "daily quota exhausted")
        assert "no target available" in note


class TestCombineLlmResults:
    def test_merges_llm_columns_only_for_requested_tickers(self):
        out_df = pd.DataFrame([
            {"Symbol": "KLAC", "Target_Consensus": 214.21},
            {"Symbol": "ADI", "Target_Consensus": 250.0},
        ])
        llm_results = {"KLAC": {"Target_Mean": 277.34, "Target_High": 325.0, "Target_Low": 150.0,
                                 "Num_Analysts": 4, "Estimates_JSON": "[]", "Sources_JSON": "[]"}}
        combined = _combine_llm_results(out_df, ["KLAC"], llm_results, "finviz")

        klac = combined[combined["Symbol"] == "KLAC"].iloc[0]
        adi = combined[combined["Symbol"] == "ADI"].iloc[0]
        assert klac["LLM_Target_Mean"] == 277.34
        assert klac["Note"] is not None
        assert pd.isna(adi["LLM_Target_Mean"])
        assert adi["Note"] is None

    def test_adds_row_for_llm_ticker_not_in_main_results(self):
        out_df = pd.DataFrame([{"Symbol": "ADI", "Target_Consensus": 250.0}])
        llm_results = {"NEWCO": {"Target_Mean": 50.0, "Estimates_JSON": "[]"}}
        combined = _combine_llm_results(out_df, ["NEWCO"], llm_results, "finviz")

        assert len(combined) == 2
        newco = combined[combined["Symbol"] == "NEWCO"].iloc[0]
        assert newco["LLM_Target_Mean"] == 50.0
        assert pd.isna(newco["Target_Consensus"])

    def test_failed_llm_ticker_keeps_main_value_and_notes_reason(self):
        out_df = pd.DataFrame([{"Symbol": "KLAC", "Target_Consensus": 214.21}])
        llm_results = {"KLAC": "daily quota exhausted"}
        combined = _combine_llm_results(out_df, ["KLAC"], llm_results, "finviz")

        row = combined.iloc[0]
        assert row["Target_Consensus"] == 214.21
        assert pd.isna(row["LLM_Target_Mean"])
        assert "unavailable" in row["Note"]


class TestFetchLlmEstimates:
    def test_missing_api_key_returns_reason_for_every_ticker(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        results = _fetch_llm_estimates(["KLAC", "ADI"])
        assert "GOOGLE_API_KEY" in results["KLAC"]
        assert "GOOGLE_API_KEY" in results["ADI"]

    def test_quota_exhaustion_labels_remaining_tickers_without_more_calls(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")

        class _FakeProvider:
            def __init__(self):
                self.calls = []

            def fetch_target(self, symbol):
                self.calls.append(symbol)
                if symbol == "B":
                    raise PermissionError("daily call limit reached")
                return {"Symbol": symbol, "Target_Mean": 100.0}

        fake = _FakeProvider()
        with patch("get_analyst_targets.GeminiSearchProvider", return_value=fake):
            results = _fetch_llm_estimates(["A", "B", "C"])

        assert results["A"] == {"Symbol": "A", "Target_Mean": 100.0}
        assert "daily call limit reached" in results["B"]
        assert "daily call limit reached" in results["C"]
        assert fake.calls == ["A", "B"]  # C never attempted — budget already known exhausted
