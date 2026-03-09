"""
datamgr.coordinator — DataCoordinator: central request registry and orchestrator.

Phase 1 implementation
----------------------
This is the Phase 1 coordinator.  Its job is to establish the contract and
wire callers correctly — not yet to deduplicate or batch.

What Phase 1 does:
  - Accepts DataRequest objects from callers via register().
  - Validates frequency at registration time (fast-fail on typos).
  - Delegates each request to the existing DataStore methods unchanged.
  - Returns per-requester DataFrames from fulfill().

What Phase 1 does NOT do (deferred to later phases):
  - _merge_requests(): deduplicate overlapping requests (Phase 2)
  - _gap_analyse():    per-ticker coverage check across all tickers (Phase 2)
  - _batch_download(): one provider call per frequency cluster (Phase 3)
  - _validate() / WAL: data integrity layer (Phase 4)

Design constraints:
  - datamgr never imports from shockarb.
  - The coordinator speaks to the store via the DataStore interface only.
  - Callers never call the store or provider directly.
  - fulfill() is idempotent: safe to call multiple times.

Phase 2 upgrade path
--------------------
_merge_requests() and _gap_analyse() will be inserted into fulfill() between
registration and dispatch.  The public API (register / fulfill) does not change.
The generic grouping key is (frequency, retention) — written that way now so
Phase 2 is additive.

Example (Phase 1)
-----------------
    from datamgr.coordinator import DataCoordinator
    from datamgr.requests import DataRequest, Frequency

    coordinator = DataCoordinator(store)

    coordinator.register(DataRequest(
        tickers   = tuple(universe.market_etfs),
        start     = universe.start_date,
        end       = universe.end_date,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = "pipeline.build.etf",
    ))

    results = coordinator.fulfill()
    etf_prices = results["pipeline.build.etf"]
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from loguru import logger

from datamgr.interfaces import DataStore
from datamgr.requests import DataRequest, Frequency


class DataCoordinator:
    """
    Central registry for DataRequest objects.

    Callers register their data needs; fulfill() satisfies all of them,
    returning a dict keyed by requester label.

    Parameters
    ----------
    store : DataStore
        The concrete store implementation (e.g. ParquetStore wrapping
        shockarb's existing DataStore).  Injected for testability.
    """

    def __init__(self, store: DataStore) -> None:
        self._store: DataStore = store
        self._requests: List[DataRequest] = []

    # =========================================================================
    # Public API
    # =========================================================================

    def register(self, request: DataRequest) -> None:
        """
        Register a DataRequest.

        Frequency is validated here — ProviderError raised immediately on
        an unknown value so the bug surfaces at call-site, not at fulfill() time.

        Parameters
        ----------
        request : DataRequest
            Caller's data specification.  tickers must be a tuple.
        """
        # Frequency already validated in DataRequest.__post_init__, but we
        # log it here for the audit trail.
        logger.debug(
            f"[Coordinator] Registered: {request.requester!r} "
            f"{len(request.tickers)} tickers "
            f"{request.start} → {request.end} "
            f"({request.frequency}, {request.retention})"
        )
        self._requests.append(request)

    def fulfill(self) -> Dict[str, pd.DataFrame]:
        """
        Satisfy all registered requests and return results keyed by requester.

        Phase 1 behaviour: delegates each request directly to the store.
        No deduplication, no batching — that is Phase 2/3.

        Returns
        -------
        dict of str → DataFrame
            Keys are requester labels.  Values are DataFrames as returned
            by the store for that request's frequency:
              - DAILY        → (dates × tickers) adj_close DataFrame
              - INTRADAY_15M → MultiIndex (field, ticker) DataFrame

        Notes
        -----
        If multiple requests share the same requester label, only the last
        result is retained in the output dict.  Use distinct requester labels
        (e.g. "pipeline.build.etf" vs "pipeline.build.stock") to avoid this.
        """
        if not self._requests:
            logger.warning("[Coordinator] fulfill() called with no registered requests.")
            return {}

        results: Dict[str, pd.DataFrame] = {}

        for req in self._requests:
            logger.info(
                f"[Coordinator] Fulfilling: {req.requester!r} "
                f"{len(req.tickers)} tickers "
                f"{req.start} → {req.end} ({req.frequency})"
            )
            df = self._dispatch(req)
            results[req.requester] = df

        logger.info(f"[Coordinator] fulfill() complete: {len(results)} result(s).")
        return results

    def clear(self) -> None:
        """
        Clear all registered requests.

        Call this between logical runs if the coordinator instance is reused
        (e.g. in a long-running process that rebuilds the model daily).
        """
        self._requests.clear()
        logger.debug("[Coordinator] Request registry cleared.")

    # =========================================================================
    # Internal dispatch (Phase 1)
    # =========================================================================

    def _dispatch(self, req: DataRequest) -> pd.DataFrame:
        """
        Route a single DataRequest to the appropriate store method.

        Phase 1: direct delegation.
        Phase 2: this becomes the post-gap-analysis commit path.
        """
        if req.frequency == Frequency.DAILY:
            return self._dispatch_daily(req)
        elif req.frequency in (Frequency.INTRADAY_15M, Frequency.INTRADAY_1M):
            return self._dispatch_intraday(req)
        else:
            # Frequency.validate() in DataRequest.__post_init__ should have
            # caught this already; guard here for defensive completeness.
            raise ValueError(f"Unhandled frequency: {req.frequency!r}")

    def _dispatch_daily(self, req: DataRequest) -> pd.DataFrame:
        """Delegate a DAILY request to the store's fetch_daily path."""
        try:
            # store.fetch_daily returns a (dates × tickers) adj_close DataFrame.
            # We call through the DataStore interface; the concrete store
            # (shockarb.store.DataStore / future ParquetStore) handles caching.
            df = self._store.fetch_daily(
                tickers=list(req.tickers),
                start=req.start,
                end=req.end,
            )
            if df is None or df.empty:
                logger.warning(
                    f"[Coordinator] Empty result for {req.requester!r} "
                    f"({req.start} → {req.end})"
                )
                return pd.DataFrame()
            return df
        except Exception as exc:
            logger.error(
                f"[Coordinator] fetch_daily failed for {req.requester!r}: {exc}"
            )
            return pd.DataFrame()

    def _dispatch_intraday(self, req: DataRequest) -> pd.DataFrame:
        """Delegate an intraday request to the store's fetch_intraday path."""
        from datetime import date as date_type
        import datetime

        trade_date = (
            date_type.fromisoformat(req.trade_date)
            if req.trade_date
            else datetime.date.today()
        )
        try:
            df = self._store.fetch_intraday(
                tickers=list(req.tickers),
                trade_date=trade_date,
            )
            if df is None or df.empty:
                logger.warning(
                    f"[Coordinator] Empty intraday result for {req.requester!r} "
                    f"(trade_date={trade_date})"
                )
                return pd.DataFrame()
            return df
        except Exception as exc:
            logger.error(
                f"[Coordinator] fetch_intraday failed for {req.requester!r}: {exc}"
            )
            return pd.DataFrame()

    # =========================================================================
    # Phase 2 stubs — not yet implemented
    # =========================================================================

    def _merge_requests(self) -> List[DataRequest]:
        """
        [Phase 2] Merge overlapping requests to minimise downloads.

        Groups by (frequency, retention).  Within each group, takes the union
        of tickers and the widest date range.  Returns one merged DataRequest
        per group.

        Written generically from the start so scenarios 2 & 3 (backtest
        warm-up overlap, multi-universe dedup) are additive when needed.
        """
        raise NotImplementedError("Phase 2")

    def _gap_analyse(self, merged: List[DataRequest]):
        """
        [Phase 2] For each merged request, determine what is missing from the store.

        For each ticker, calls store.coverage(key):
          - None               → full download needed
          - (earliest, latest) → tail download needed if latest < request.end
          - Fully covered      → no download needed

        Returns a list of gap requests to be satisfied by _batch_download().
        """
        raise NotImplementedError("Phase 2")
