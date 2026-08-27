"""
Tests for utils/trading_calendar.py — prior_trading_day(), session_label(),
et_datetime(), market_open_at_fetch().

Root-caused bug (2026-08-07): a premarket market report's LLM narrative
asserted "US equities closed lower today" when it was actually pre-open —
the data described the prior completed session. These tests cover the
calendar-gap cases that made the naive "prior day" framing wrong (Monday,
post-weekend reports) and the weekend market-open-banner bug already
tracked in HIL_todo.md.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from trading_calendar import prior_trading_day, session_label, et_datetime, market_open_at_fetch


# =============================================================================
# prior_trading_day()
# =============================================================================

class TestPriorTradingDay:
    def test_midweek_goes_back_one_day(self):
        """Thursday -> Wednesday."""
        assert prior_trading_day(date(2026, 8, 6)) == date(2026, 8, 5)

    def test_monday_skips_weekend_to_friday(self):
        """Monday -> Friday, not Sunday."""
        assert prior_trading_day(date(2026, 8, 10)) == date(2026, 8, 7)

    def test_tuesday_after_monday_is_normal(self):
        assert prior_trading_day(date(2026, 8, 11)) == date(2026, 8, 10)

    def test_custom_calendar_can_skip_a_holiday(self):
        """A custom is_trading_day predicate can layer in holiday awareness."""
        def no_aug_6(d: date) -> bool:
            return d.weekday() < 5 and d != date(2026, 8, 6)

        assert prior_trading_day(date(2026, 8, 7), is_trading_day=no_aug_6) == date(2026, 8, 5)


# =============================================================================
# session_label()
# =============================================================================

class TestSessionLabel:
    def test_same_day_is_today(self):
        assert session_label(date(2026, 8, 6), date(2026, 8, 6)) == "today"

    def test_normal_midweek_gap_is_yesterday(self):
        assert session_label(date(2026, 8, 6), date(2026, 8, 7)) == "yesterday"

    def test_friday_on_monday_report_is_yesterday_not_three_days_ago(self):
        """The case that broke the naive 'prior day' heuristic."""
        assert session_label(date(2026, 8, 7), date(2026, 8, 10)) == "yesterday"

    def test_two_session_gap_is_flagged_stale(self):
        """Data 2+ sessions behind is the real MARKET-REPORT-STALE-LABEL bug,
        not a normal calendar gap — must not be silently phrased as 'yesterday'."""
        result = session_label(date(2026, 8, 4), date(2026, 8, 7))
        assert "STALE" in result
        assert "2026-08-04" in result


# =============================================================================
# et_datetime()
# =============================================================================

class TestEtDatetime:
    def test_summer_is_utc_minus_4(self):
        utc_dt = datetime(2026, 8, 7, 11, 46, tzinfo=timezone.utc)
        assert et_datetime(utc_dt) == datetime(2026, 8, 7, 7, 46)

    def test_winter_is_utc_minus_5(self):
        utc_dt = datetime(2026, 1, 7, 11, 46, tzinfo=timezone.utc)
        assert et_datetime(utc_dt) == datetime(2026, 1, 7, 6, 46)

    def test_naive_input_treated_as_utc(self):
        utc_dt = datetime(2026, 8, 7, 11, 46)
        assert et_datetime(utc_dt) == datetime(2026, 8, 7, 7, 46)


# =============================================================================
# market_open_at_fetch()
# =============================================================================

class TestMarketOpenAtFetch:
    def test_open_during_trading_hours(self):
        assert market_open_at_fetch({"fetched_at": "2026-08-07T15:00:00+00:00"}) is True

    def test_closed_before_open(self):
        assert market_open_at_fetch({"fetched_at": "2026-08-07T11:46:00+00:00"}) is False

    def test_closed_after_close(self):
        assert market_open_at_fetch({"fetched_at": "2026-08-07T21:00:00+00:00"}) is False

    def test_closed_on_saturday_even_during_trading_hours(self):
        """2026-08-08 is a Saturday — regression test for the banner bug
        flagged in HIL_todo.md (MARKET-REPORT-STALE-LABEL, 2026-07-13)."""
        assert market_open_at_fetch({"fetched_at": "2026-08-08T15:00:00+00:00"}) is False

    def test_closed_on_sunday(self):
        """2026-08-09 is a Sunday."""
        assert market_open_at_fetch({"fetched_at": "2026-08-09T15:00:00+00:00"}) is False

    def test_missing_fetched_at_defaults_closed(self):
        assert market_open_at_fetch({}) is False
