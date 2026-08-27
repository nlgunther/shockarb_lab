"""
trading_calendar — small, dependency-free helpers for reasoning about which
NYSE trading session a piece of market data describes, and whether that
session has closed yet.

No holiday awareness by default (weekends only) — prior_trading_day()'s
`is_trading_day` parameter exists specifically so a real calendar (e.g.
pandas_market_calendars) can be plugged in later without touching any call
site. Not needed today: this module never has to guess which prior calendar
day was a holiday, because the trading-day dates it compares against always
come from real fetched data (see market_data.py's session_date/baseline_date),
which by construction only ever lands on days the market actually traded.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable


def prior_trading_day(d: date, is_trading_day: Callable[[date], bool] | None = None) -> date:
    """
    Most recent trading day strictly before `d`.

    `is_trading_day` defaults to a plain Mon-Fri check (no holiday
    awareness). Swap in a real NYSE calendar later by passing a different
    predicate — nothing else about the call site changes.

    Example:
        prior_trading_day(date(2026, 8, 10))   # Monday
        # -> date(2026, 8, 7)                   # Friday
    """
    is_trading_day = is_trading_day or (lambda d: d.weekday() < 5)
    cur = d - timedelta(days=1)
    while not is_trading_day(cur):
        cur -= timedelta(days=1)
    return cur


def session_label(session_date: date, today: date) -> str:
    """
    Describe `session_date` in plain English relative to `today`.

    Returns "today" or "yesterday" for the two expected cases — "yesterday"
    correctly means "Friday" when today is Monday, matching how market
    reports actually talk (nobody says "Friday's close" on a Monday).

    Anything else means the data is stale by more than one session, which
    is flagged explicitly rather than guessed at with a plausible-sounding
    "N days ago" phrase — that condition usually means the multi-session-
    skip bug tracked as MARKET-REPORT-STALE-LABEL in HIL_todo.md, not an
    ordinary calendar gap, and deserves a loud flag rather than smooth prose.

    Example:
        session_label(date(2026, 8, 6), date(2026, 8, 7))
        # -> "yesterday"
        session_label(date(2026, 8, 7), date(2026, 8, 10))
        # -> "yesterday"   (Friday, described the following Monday)
    """
    if session_date == today:
        return "today"
    if session_date == prior_trading_day(today):
        return "yesterday"
    return (
        f"STALE (data from {session_date.isoformat()}, "
        f"expected {prior_trading_day(today).isoformat()})"
    )


def et_datetime(utc_dt: datetime) -> datetime:
    """
    Convert a UTC datetime to naive US/Eastern wall-clock time.

    Approximates DST as "March-October = EDT (UTC-4), else EST (UTC-5)" —
    off by a few days around the actual transition dates in early March/
    November, a simplification already accepted elsewhere in this codebase;
    good enough for premarket/after-hours labeling, not for anything that
    needs the exact DST boundary.

    Example:
        et_datetime(datetime(2026, 8, 7, 11, 46, tzinfo=timezone.utc))
        # -> datetime(2026, 8, 7, 7, 46)   # EDT, UTC-4
    """
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    et_offset = -4 if 3 <= utc_dt.month <= 10 else -5
    return (utc_dt + timedelta(hours=et_offset)).replace(tzinfo=None)


def market_open_at_fetch(snapshot: dict) -> bool:
    """
    Return True if the snapshot was fetched during NYSE trading hours (ET).

    Uses fetched_at (UTC ISO string) — unambiguous regardless of the machine
    timezone that produced fetched_at_local. NYSE: Mon-Fri 09:30-16:00 ET.

    Weekend fetches always return False regardless of time-of-day — fixes
    the "⚠️ Market open" banner incorrectly firing on Saturday/Sunday fetches
    (HIL_todo.md, MARKET-REPORT-STALE-LABEL, 2026-07-13 update).

    Example:
        market_open_at_fetch({"fetched_at": "2026-08-07T15:00:00+00:00"})
        # -> True   (11:00 ET, Friday)
    """
    fetched_at = snapshot.get("fetched_at")
    if not fetched_at:
        return False
    try:
        utc_dt = datetime.fromisoformat(fetched_at)
        et_dt  = et_datetime(utc_dt)
        if et_dt.weekday() >= 5:
            return False
        minutes_et = et_dt.hour * 60 + et_dt.minute
        return 9 * 60 + 30 <= minutes_et < 16 * 60
    except Exception:
        return False
