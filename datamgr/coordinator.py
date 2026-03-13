"""
datamgr.coordinator — DataCoordinator: central request registry and orchestrator.

Phase 2 implementation
----------------------
Phase 2 adds gap analysis and request deduplication to fulfill():

    1. _merge_requests()  — group by (frequency, retention), union tickers,
                            widen date range.  One merged request per group.
    2. _gap_analyse()     — for each ticker in a merged request, call
                            store.coverage(key).  Emit only the missing spans.
    3. _download_and_commit() — one provider.fetch() per span cluster, then
                            merge with cached data and write to store.
    4. _read_for_request() — slice per-requester results from the store.

What does NOT change vs Phase 1:
    - Public API: register() / fulfill() / clear() signatures unchanged.
    - DataRequest, Frequency, DataStore interfaces unchanged.
    - store.py (shockarb.store.DataStore) unchanged.
    - All Phase 1 tests still pass.

Phase 3/4 upgrade path:
    - _download_and_commit() <- Phase 3 clusters by contiguous date ranges for
                               true single-call batching across all tickers.
    - validate() / WAL       <- Phase 4 inserts between download and write.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from datamgr.interfaces import DataProvider, DataStore
from datamgr.requests import DataRequest, Frequency


class DataCoordinator:
    """
    Central registry for DataRequest objects.

    Callers register their data needs; fulfill() satisfies all of them,
    returning a dict keyed by requester label.

    Parameters
    ----------
    store    : DataStore
        Concrete store (e.g. ParquetStore).  Injected for testability.
    provider : DataProvider, optional
        Concrete provider (e.g. YFinanceProvider).  Injected for testability.
        Required for any cache miss.  If None and a miss occurs, RuntimeError.
    """

    def __init__(
        self,
        store: DataStore,
        provider: Optional[DataProvider] = None,
    ) -> None:
        self._store    = store
        self._provider = provider
        self._requests: List[DataRequest] = []
        # In-memory cache for intraday fetches (keyed by requester label).
        # Populated during fulfill(); read by _read_intraday().
        # TODO Step 5: intraday caching — cache the open on the first call
        #   of the day, then fetch only the latest bar on subsequent calls.
        self._intraday_results: Dict[str, pd.DataFrame] = {}

    # =========================================================================
    # Public API
    # =========================================================================

    def register(self, request: DataRequest) -> None:
        """Register a DataRequest for fulfillment."""
        logger.debug(
            f"[Coordinator] Registered: {request.requester!r} "
            f"{len(request.tickers)} tickers "
            f"{request.start} -> {request.end} "
            f"({request.frequency}, {request.retention})"
        )
        self._requests.append(request)

    def fulfill(self) -> Dict[str, pd.DataFrame]:
        """
        Satisfy all registered requests.

        Steps:
          1. Merge overlapping requests (union tickers, widen date range).
          2. Gap-analyse each merged request against the store.
          3. Download missing spans and commit to the store.
          4. Read and return per-requester slices.

        Returns
        -------
        dict of str -> DataFrame  keyed by requester label.
        """
        if not self._requests:
            logger.warning("[Coordinator] fulfill() called with no registered requests.")
            return {}

        # Steps 2-3
        merged = self._merge_requests()
        for merged_req in merged:
            if merged_req.frequency != Frequency.DAILY:
                # Intraday: fetch via provider, skip gap analysis.
                # The cache flag on the original requests controls whether
                # data is committed to the store (Step 2b: always False).
                self._fetch_intraday(merged_req)
                continue
            gaps = self._gap_analyse(merged_req)
            if not gaps:
                logger.debug(
                    f"[Coordinator] Full cache hit — "
                    f"{len(merged_req.tickers)} tickers ({merged_req.frequency})"
                )
            else:
                self._download_and_commit(merged_req, gaps)
        # Step 4
        results: Dict[str, pd.DataFrame] = {}
        for req in self._requests:
            logger.info(
                f"[Coordinator] Slicing: {req.requester!r} "
                f"{len(req.tickers)} tickers "
                f"{req.start} -> {req.end} ({req.frequency})"
            )
            results[req.requester] = self._read_for_request(req)
        logger.info(f"[Coordinator] fulfill() complete: {len(results)} result(s).")
        return results
    
    def clear(self) -> None:
        """Clear all registered requests and in-memory intraday results."""
        self._requests.clear()
        self._intraday_results.clear()
        logger.debug("[Coordinator] Request registry cleared.")

    # =========================================================================
    # Step 1: merge
    # =========================================================================

    def _merge_requests(self) -> List[DataRequest]:
        """
        Group requests by (frequency, retention).
        Within each group: union tickers, widen date range.
        Returns one merged DataRequest per group.
        """
        groups: Dict[Tuple[str, str], List[DataRequest]] = defaultdict(list)
        for req in self._requests:
            groups[(req.frequency, req.retention)].append(req)

        merged: List[DataRequest] = []
        for (frequency, retention), reqs in groups.items():
            all_tickers = tuple(sorted(set(t for r in reqs for t in r.tickers)))
            earliest    = min(r.start for r in reqs)
            latest      = max(r.end   for r in reqs)
            # If any request in the group opts out of caching, the merged
            # request inherits cache=False.  This is conservative: we never
            # silently commit data that a caller asked not to cache.
            should_cache = all(r.cache for r in reqs)
            merged.append(DataRequest(
                tickers   = all_tickers,
                start     = earliest,
                end       = latest,
                frequency = frequency,
                retention = retention,
                requester = f"_merged_{frequency}_{retention}",
                cache     = should_cache,
            ))
            logger.debug(
                f"[Coordinator] Merged {len(reqs)} request(s) into "
                f"{len(all_tickers)} tickers "
                f"{earliest} -> {latest} ({frequency})"
            )

        return merged

    # =========================================================================
    # Step 2: gap analysis
    # =========================================================================

    _OVERLAP_ROWS = 10  # business days of overlap for restatement detection

    def _gap_analyse(
        self,
        req: DataRequest,
    ) -> Dict[str, Tuple[str, str]]:
        """
        Determine which tickers are missing data and what span to download.

        For each ticker, calls store.coverage(key):
          - None               -> full download: (req.start, req.end)
          - (start, end) where end < req.end
                               -> tail download: (end - OVERLAP_ROWS BDays, req.end)
          - Fully covered      -> skipped

        Returns dict of ticker -> (gap_start, gap_end).

        Note on post-close staleness
        ----------------------------
        The redundant-download problem (yfinance hasn't published today's
        daily bar yet) is handled by _market_is_open()'s grace period
        (extended to 17:00 ET), which routes to the intraday path during
        the window where the daily bar is unavailable.  Gap analysis itself
        does NOT apply any grace — if the cache is behind, it fetches.
        """
        gaps: Dict[str, Tuple[str, str]] = {}

        for ticker in req.tickers:
            key      = f"{req.frequency}/{ticker}"
            coverage = self._store.coverage(key)

            if coverage is None:
                logger.debug(f"[GapAnalyse] MISS  {ticker}: {req.start} -> {req.end}")
                gaps[ticker] = (req.start, req.end)
                continue

            cached_start, cached_end = coverage
            cached_end_ts = pd.Timestamp(cached_end)
            req_end_ts    = pd.Timestamp(req.end)

            if cached_end_ts >= req_end_ts:
                logger.debug(f"[GapAnalyse] HIT   {ticker}: covered through {cached_end}")
                continue

            # Tail miss — step back for restatement overlap
            overlap_start = (
                cached_end_ts - pd.tseries.offsets.BDay(self._OVERLAP_ROWS)
            ).strftime("%Y-%m-%d")
            logger.debug(
                f"[GapAnalyse] TAIL  {ticker}: "
                f"{overlap_start} -> {req.end} (cached through {cached_end})"
            )
            gaps[ticker] = (overlap_start, req.end)

        return gaps

    # =========================================================================
    # Step 3: download + commit
    # =========================================================================

    def _download_and_commit(
        self,
        merged_req: DataRequest,
        gaps: Dict[str, Tuple[str, str]],
    ) -> None:
        """
        Batch download missing spans and commit each ticker to the store.

        Groups tickers by identical (gap_start, gap_end) span so that
        tickers with the same gap are fetched in one provider call.
        """
        if self._provider is None:
            raise RuntimeError(
                "[Coordinator] Provider required for cache miss but none was injected. "
                "Pass provider= to DataCoordinator()."
            )

        # Group by span for batching
        span_groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for ticker, span in gaps.items():
            span_groups[span].append(ticker)

        for (gap_start, gap_end), tickers in span_groups.items():
            logger.info(
                f"[Coordinator] Fetching {len(tickers)} ticker(s) "
                f"{gap_start} -> {gap_end} ({merged_req.frequency})"
            )
            try:
                raw = self._provider.fetch(
                    tickers   = tickers,
                    start     = gap_start,
                    end       = gap_end,
                    frequency = merged_req.frequency,
                )
            except Exception as exc:
                logger.error(f"[Coordinator] Provider fetch failed: {exc}")
                continue

            if raw is None or raw.empty:
                logger.warning(
                    f"[Coordinator] Provider returned empty for "
                    f"{tickers} {gap_start}->{gap_end}"
                )
                continue

            for ticker in tickers:
                self._commit_ticker(ticker, raw, merged_req)

    def _commit_ticker(
        self,
        ticker: str,
        raw: pd.DataFrame,
        req: DataRequest,
    ) -> None:
        """
        Extract one ticker from a batch result and merge+write to the store.

        Merges with any cached data (keeping new rows for overlapping dates
        so adj_factor restatements are applied), then writes the combined
        result.
        """
        # Extract ticker slice from MultiIndex batch result
        if isinstance(raw.columns, pd.MultiIndex):
            ticker_cols = [c for c in raw.columns if c[1] == ticker]
            if not ticker_cols:
                logger.warning(f"[Coordinator] No data for {ticker} in provider result")
                return
            new_df = raw[ticker_cols].copy()
            new_df.columns = pd.Index([c[0] for c in ticker_cols])  # flatten to field names
        else:
            new_df = raw.copy()

        key      = f"{req.frequency}/{ticker}"
        existing = self._store.read(key, start="1900-01-01", end="2100-01-01")

        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df])
            combined = (combined[~combined.index.duplicated(keep="last")]
                        .sort_index())
        else:
            combined = new_df

        self._store.write(key, combined, meta={
            "ticker":    ticker,
            "frequency": req.frequency,
            "retention": req.retention,
        })
        logger.debug(f"[Coordinator] Committed {key}: {len(combined)} rows")

    # =========================================================================
    # Step 3b: intraday fetch (provider-backed, optionally no-commit)
    # =========================================================================

    def _fetch_intraday(self, merged_req: DataRequest) -> None:
        """
        Fetch intraday data via the provider and stash in _intraday_results.

        The merged request's cache flag determines whether data is also
        committed to the store.  In Step 2b, callers pass cache=False so
        intraday data is returned to the caller without polluting the
        daily store with partial-day bars.

        The result is keyed by the merged request's requester label and
        picked up by _read_intraday() during the Step 4 slice phase.
        """
        if self._provider is None:
            raise RuntimeError(
                "[Coordinator] Provider required for intraday fetch but none was injected. "
                "Pass provider= to DataCoordinator()."
            )

        tickers = list(merged_req.tickers)
        logger.info(
            f"[Coordinator] Intraday fetch: {len(tickers)} ticker(s) "
            f"({merged_req.frequency})"
        )

        try:
            raw = self._provider.fetch(
                tickers   = tickers,
                start     = merged_req.start,
                end       = merged_req.end,
                frequency = merged_req.frequency,
            )
        except Exception as exc:
            logger.error(f"[Coordinator] Intraday provider fetch failed: {exc}")
            return

        if raw is None or raw.empty:
            logger.warning(
                f"[Coordinator] Intraday provider returned empty for "
                f"{len(tickers)} tickers"
            )
            return

        # Stash the raw result for _read_intraday() to slice
        self._intraday_results[merged_req.requester] = raw

        # Honour the cache flag — commit to store only if requested.
        if merged_req.cache:
            for ticker in tickers:
                self._commit_ticker(ticker, raw, merged_req)
        else:
            logger.debug(
                f"[Coordinator] Intraday fetch complete — cache=False, "
                f"skipping store commit ({len(tickers)} tickers)"
            )

    # =========================================================================
    # Step 4: read slices for callers
    # =========================================================================

    def _read_for_request(self, req: DataRequest) -> pd.DataFrame:
        """Route to daily or intraday reader."""
        if req.frequency == Frequency.DAILY:
            return self._read_daily(req)
        return self._read_intraday(req)

    def _read_daily(self, req: DataRequest) -> pd.DataFrame:
        """
        Assemble a (dates × tickers) adj_close DataFrame from the store.
        Robust to flat adj_close frames, ticker-named frames, and
        single-column frames of any column name.
        """
        frames = []
        for ticker in req.tickers:
            key = f"daily/{ticker}"
            try:
                df = self._store.read(key, req.start, req.end)
            except Exception as exc:
                logger.warning("[Coordinator] read(%s) raised: %s", key, exc)
                continue
            if df is None or df.empty:
                continue
            # Normalise to a single Series named after the ticker
            if "adj_close" in df.columns:
                s = df["adj_close"].rename(ticker)
            elif ticker in df.columns:
                s = df[ticker].rename(ticker)
            else:
                # Single-column frame of unknown name — assume it is adj_close
                s = df.iloc[:, 0].rename(ticker)
            frames.append(s)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)
    
    def _read_intraday(self, req: DataRequest) -> pd.DataFrame:
        """
        Return intraday data from the in-memory results stashed by _fetch_intraday().

        The merged request's requester label is a synthetic key like
        '_merged_15m_ephemeral'.  We locate the stashed result by scanning
        _intraday_results for a key whose frequency and retention match
        the original request.
        """
        # Find the merged key that covers this request's frequency+retention
        merged_key = f"_merged_{req.frequency}_{req.retention}"
        raw = self._intraday_results.get(merged_key)
        if raw is None or raw.empty:
            logger.warning(
                f"[Coordinator] No intraday data available for {req.requester!r}"
            )
            return pd.DataFrame()
        return raw
