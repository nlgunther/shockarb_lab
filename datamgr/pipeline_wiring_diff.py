"""
Phase 1 pipeline.py changes — summary and diff.

Only two functions change in pipeline.py for Phase 1.
Everything else is untouched.

CHANGES
-------

1. New import at top of pipeline.py (add after existing shockarb imports):

    from datamgr.coordinator import DataCoordinator
    from datamgr.requests import DataRequest, Frequency
    from datamgr.stores.parquet import ParquetStore

2. New helper _coordinator() (add alongside _cache_manager()):

    def _coordinator(exec_cfg: ExecutionConfig) -> DataCoordinator:
        \"\"\"Build a DataCoordinator backed by a DataStore for exec_cfg's data dir.\"\"\"
        from shockarb.store import DataStore as ShockArbStore
        inner = ShockArbStore(exec_cfg.data_dir)
        store = ParquetStore(inner)
        return DataCoordinator(store)

3. build() — replace the two fetch_prices() calls with coordinator registration:

BEFORE:
    etf_returns = prices_to_returns(
        fetch_prices(
            universe.market_etfs,
            universe.start_date,
            universe.end_date,
            cache_name=f"{universe.name}_etf",
            exec_config=exec_cfg,
        )
    )
    stock_returns = prices_to_returns(
        fetch_prices(
            universe.individual_stocks,
            universe.start_date,
            universe.end_date,
            cache_name=f"{universe.name}_stock",
            exec_config=exec_cfg,
        )
    )

AFTER:
    coordinator = _coordinator(exec_cfg)

    coordinator.register(DataRequest(
        tickers   = tuple(universe.market_etfs),
        start     = universe.start_date,
        end       = universe.end_date,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = f"{universe.name}.etf",
    ))
    coordinator.register(DataRequest(
        tickers   = tuple(universe.individual_stocks),
        start     = universe.start_date,
        end       = universe.end_date,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = f"{universe.name}.stock",
    ))

    results = coordinator.fulfill()

    etf_returns    = prices_to_returns(results[f"{universe.name}.etf"])
    stock_returns  = prices_to_returns(results[f"{universe.name}.stock"])

NOTES
-----
- fetch_prices() is NOT removed yet.  It is still used by any caller that
  hasn't been migrated.  It can be deprecated and removed in Phase 2 once
  all callers are wired through the coordinator.

- The daily_scanner and adaptive_scanner register their own DataRequests
  in the same pattern.  Example for daily_scanner:

    coordinator.register(DataRequest(
        tickers   = tuple(all_tickers),
        start     = start_date,
        end       = end_date,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = "daily_scanner",
    ))
    results = coordinator.fulfill()
    prices  = results["daily_scanner"]

- For intraday, the scanner registers with frequency=Frequency.INTRADAY_15M
  and retention="ephemeral".  trade_date is set to today's date string.

    coordinator.register(DataRequest(
        tickers    = tuple(all_tickers),
        start      = today_str,
        end        = today_str,
        frequency  = Frequency.INTRADAY_15M,
        retention  = "ephemeral",
        requester  = "daily_scanner.intraday",
        trade_date = today_str,
    ))
"""

# This file is documentation only — not imported by any module.
# Apply the changes above to pipeline.py manually or via str_replace.
