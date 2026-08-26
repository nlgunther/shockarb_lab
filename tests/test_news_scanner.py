"""
Tests for utils/news_scanner.py severity tagging and headline ranking.

All tests are pure-function/unit level — no yfinance network calls.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from news_scanner import _MAX_HEADLINES, _flag_severity


class TestFlagSeverity:
    def test_rating_action_downgrade(self):
        assert _flag_severity("KeyBanc downgrades Salesforce to Sector Weight") == "RATING"

    def test_rating_action_upgrade(self):
        assert _flag_severity("Analyst upgrades stock to Outperform") == "RATING"

    def test_leadership_change(self):
        title = "Copart CEO Jeff Liaw steps down, Jay Adair named successor"
        assert _flag_severity(title) == "LEADERSHIP"

    def test_guidance_warning(self):
        title = "Lam Research shares fall on cooling demand and margin concerns"
        assert _flag_severity(title) == "GUIDANCE"

    def test_legal_investigation(self):
        assert _flag_severity("Company faces SEC investigation over disclosures") == "LEGAL"

    def test_generic_headline_not_flagged(self):
        title = "Copart posts quarterly revenue in line with estimates"
        assert _flag_severity(title) is None

    def test_case_insensitive(self):
        assert _flag_severity("ANALYST DOWNGRADE hits shares") == "RATING"

    def test_first_matching_category_wins(self):
        # Contains both a RATING keyword ("downgrade") and a GUIDANCE keyword
        # ("margin") — RATING is checked first in _SEVERITY_KEYWORDS.
        title = "Downgrade cites margin pressure ahead"
        assert _flag_severity(title) == "RATING"


class TestHeadlineRanking:
    """
    Exercises the same sort key scan_news() uses to rank parsed articles,
    without going through the network-calling scan_news() itself.
    """

    def _rank(self, articles: list[tuple[str, str, int | None]]) -> list[str]:
        ranked = sorted(
            articles,
            key=lambda a: (_flag_severity(a[0]) is None, -(a[2] or 0)),
        )
        return [title for title, _, _ in ranked]

    def test_flagged_headline_beats_newer_unflagged(self):
        articles = [
            ("Company opens new distribution center", "Wire", 200),
            ("Analyst downgrade cites weak demand", "Wire", 100),
        ]
        assert self._rank(articles)[0] == "Analyst downgrade cites weak demand"

    def test_ties_broken_by_recency(self):
        articles = [
            ("CEO resigns effective immediately", "Wire", 100),
            ("CFO appointed as new leader", "Wire", 200),
        ]
        assert self._rank(articles)[0] == "CFO appointed as new leader"

    def test_missing_timestamp_sorts_last_within_group(self):
        articles = [
            ("Generic headline with a timestamp", "Wire", 100),
            ("Generic headline, no timestamp", "Wire", None),
        ]
        assert self._rank(articles)[0] == "Generic headline with a timestamp"

    def test_max_headlines_constant_is_five(self):
        # Guards against silent drift from the mirrored cap in
        # utils/stockfit/features.py::_load_news.
        assert _MAX_HEADLINES == 5
