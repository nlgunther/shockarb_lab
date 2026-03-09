"""
datamgr.requests — DataRequest dataclass and Frequency constants.

DataRequest is the single object that callers (pipeline, scanners, backtest)
use to express what data they need.  The coordinator accepts a list of
DataRequest objects and satisfies them efficiently.

Frequency constants
-------------------
Use Frequency.DAILY, Frequency.INTRADAY_15M, etc. — never bare strings.
A typo in a free string silently runs the wrong gap-analysis logic.
Frequency.validate() raises immediately on an unknown value.

Example
-------
    from datamgr.requests import DataRequest, Frequency

    req = DataRequest(
        tickers   = ("VOO", "TLT", "VGT"),
        start     = "2022-02-10",
        end       = "2022-03-31",
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = "pipeline.build",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Frequency constants
# =============================================================================

class Frequency:
    """
    Allowed frequency identifiers for DataRequest.

    Values double as yfinance interval strings where applicable:
      INTRADAY_15M = "15m"  →  passed directly to yf.download(interval=...)
      DAILY        = "daily" →  yf.download uses start/end, no interval param

    Add new constants here as needed.  The _VALID set must be updated to match.
    """

    DAILY        = "daily"
    INTRADAY_15M = "15m"
    INTRADAY_1M  = "1m"

    _VALID = {"daily", "15m", "1m"}

    @classmethod
    def validate(cls, freq: str) -> str:
        """
        Return *freq* unchanged if valid, else raise ValueError immediately.

        Call this in DataCoordinator.__init__ on every incoming DataRequest
        so typos surface at registration time, not silently during gap analysis.
        """
        if freq not in cls._VALID:
            raise ValueError(
                f"Unknown frequency {freq!r}. "
                f"Use a Frequency constant: {sorted(cls._VALID)}"
            )
        return freq


# =============================================================================
# DataRequest
# =============================================================================

@dataclass(frozen=True)
class DataRequest:
    """
    An immutable specification of a caller's data need.

    Frozen so it is hashable and safe to use as a dict key.  The coordinator
    accepts a list of these and satisfies them with minimal downloads.

    Attributes
    ----------
    tickers   : tuple of str
        Tickers needed.  Tuple (not list) to preserve hashability.
    start     : str
        YYYY-MM-DD, inclusive.
    end       : str
        YYYY-MM-DD, exclusive (yfinance convention).
    frequency : str
        Use Frequency constants.  Validated on construction.
    retention : str
        "permanent" — daily calibration data, never swept.
        "ephemeral" — intraday data, swept after SWEEP_STALE_DAYS.
    requester : str
        Audit label identifying the caller, e.g. "pipeline.build",
        "daily_scanner", "backtest".  Used in logs and manifest entries.
    trade_date : str, optional
        YYYY-MM-DD trading date for intraday requests.  Not used for daily.

    Notes
    -----
    tickers is typed as tuple[str, ...] to enforce hashability.  Callers
    passing a list should convert: DataRequest(tickers=tuple(my_list), ...).
    """

    tickers    : tuple[str, ...]
    start      : str
    end        : str
    frequency  : str
    retention  : str
    requester  : str
    trade_date : Optional[str] = None

    def __post_init__(self):
        Frequency.validate(self.frequency)
        if self.retention not in {"permanent", "ephemeral"}:
            raise ValueError(
                f"retention must be 'permanent' or 'ephemeral', got {self.retention!r}"
            )
        if not self.tickers:
            raise ValueError("DataRequest.tickers cannot be empty")
        if not self.requester:
            raise ValueError("DataRequest.requester cannot be empty")
