"""
Tests for utils/news_flags.py.

Covers the two bugs it fixes:
  1. cross_attach_headlines() — a headline about ticker B, filed only under
     ticker A's news block in news.txt, never reached B's LLM narrative.
  2. flagged_headlines_missing_from_narrative() — a [TAG]-prefixed headline
     was correctly attached but the LLM narrative built its case around a
     different, more flattering headline instead (observed on HON, 2026-08-12;
     see HIL_todo.md CPRT-MISSING-FUNDAMENTAL-CONTEXT).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from news_flags import cross_attach_headlines, flagged_headlines_missing_from_narrative


class TestCrossAttachHeadlines:

    def test_headline_mentioning_other_ticker_is_copied(self):
        news = {
            "HON": ["[GUIDANCE] Honeywell Slides 5% As Growth Guidance Disappoints"],
            "GE":  ["GE partners with Honeywell (HON) on new aerospace deal"],
        }
        result = cross_attach_headlines(news)
        assert any("HON" in h for h in result["HON"])
        # GE's headline mentions HON, so HON should also see it
        assert "GE partners with Honeywell (HON) on new aerospace deal" in result["HON"]

    def test_unrelated_headline_not_copied(self):
        news = {
            "AAPL": ["Apple unveils new iPhone lineup"],
            "MSFT": ["Microsoft Azure revenue grows 30%"],
        }
        result = cross_attach_headlines(news)
        assert result["AAPL"] == ["Apple unveils new iPhone lineup"]
        assert result["MSFT"] == ["Microsoft Azure revenue grows 30%"]

    def test_single_ticker_returns_unchanged(self):
        news = {"HON": ["Some headline"]}
        assert cross_attach_headlines(news) == news

    def test_empty_dict_returns_unchanged(self):
        assert cross_attach_headlines({}) == {}

    def test_short_ticker_symbol_skipped(self):
        """Tickers under 3 chars are too noisy to whole-word match reliably."""
        news = {
            "GE": ["General Electric reports earnings"],
            "F":  ["Ford (F) posts strong sales"],  # "F" too short to cross-attach
        }
        result = cross_attach_headlines(news)
        assert result["GE"] == ["General Electric reports earnings"]

    def test_does_not_duplicate_existing_headline(self):
        news = {
            "HON": ["Honeywell and GE (GE) announce joint venture"],
            "GE":  ["Honeywell and GE (GE) announce joint venture"],
        }
        result = cross_attach_headlines(news)
        assert result["GE"].count("Honeywell and GE (GE) announce joint venture") == 1

    def test_respects_max_headlines_cap(self):
        news = {"HON": [f"[RATING] Headline mentioning GE #{i}" for i in range(10)]}
        news["GE"] = []
        result = cross_attach_headlines(news)
        assert len(result["GE"]) <= 5

    def test_original_dict_not_mutated(self):
        news = {
            "HON": ["Honeywell and GE (GE) deal"],
            "GE":  [],
        }
        cross_attach_headlines(news)
        assert news["GE"] == []


class TestFlaggedHeadlinesMissingFromNarrative:

    def test_tagged_headline_absent_from_narrative_is_flagged(self):
        headlines = ["[GUIDANCE] Honeywell Slides 5% As Growth Guidance Disappoints"]
        narrative = "Honeywell looks undervalued after the Aerospace spinoff."
        missing = flagged_headlines_missing_from_narrative(headlines, narrative)
        assert len(missing) == 1
        assert "GUIDANCE" in missing[0]

    def test_tagged_headline_covered_by_narrative_not_flagged(self):
        headlines = ["[GUIDANCE] Honeywell Slides 5% As Growth Guidance Disappoints"]
        narrative = "Honeywell's growth guidance disappointed investors this week."
        assert flagged_headlines_missing_from_narrative(headlines, narrative) == []

    def test_untagged_headline_never_flagged(self):
        headlines = ["Honeywell announces new product line"]
        narrative = "Honeywell looks like a solid pick."
        assert flagged_headlines_missing_from_narrative(headlines, narrative) == []

    def test_empty_headlines_returns_empty(self):
        assert flagged_headlines_missing_from_narrative([], "some narrative") == []

    def test_empty_narrative_returns_empty(self):
        headlines = ["[LEGAL] Company faces antitrust probe"]
        assert flagged_headlines_missing_from_narrative(headlines, "") == []

    def test_multiple_tags_only_missing_ones_flagged(self):
        headlines = [
            "[RATING] Analyst upgrades to Buy",
            "[LEGAL] Company faces antitrust probe",
        ]
        narrative = "Analysts upgraded the stock to Buy this week on strong fundamentals."
        missing = flagged_headlines_missing_from_narrative(headlines, narrative)
        assert len(missing) == 1
        assert "LEGAL" in missing[0]
