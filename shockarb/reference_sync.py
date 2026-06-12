"""
shockarb.reference_sync — Refresh local exchange reference CSVs from the
LondonMarket/Global-Stock-Symbols GitHub repo.

Each local reference CSV (filenames: paths.EXCHANGE_CSV_FILENAMES) is merged
against the matching remote file at:
    https://raw.githubusercontent.com/LondonMarket/Global-Stock-Symbols/main/<filename>

Merge rule (per symbol, keyed on the "Symbol" column)
------------------------------------------------------
  present in both   → local row replaced with the remote row (full row)
  remote only       → appended as a new row
  local only        → left untouched

After a sync, ticker_reference_cache.json entries for changed symbols are
cleared via shockarb.names.clear_cache_entries() so TickerReferenceResolver
picks up the new Name/Sector/Industry on next lookup instead of serving a
stale cached value.

Off by default — wired to `stockfit report --update-reference-data`. Network
or parse errors for a given file are logged and skipped; the existing local
file is left as-is and the run continues.

Usage
-----
    from shockarb.reference_sync import sync_reference_data
    from paths import DATA, EXCHANGE_CSV_FILENAMES, TICKER_CACHE_FILENAME

    stats = sync_reference_data(
        data_dir   = str(DATA),
        files      = EXCHANGE_CSV_FILENAMES,
        cache_path = str(DATA / TICKER_CACHE_FILENAME),
    )
    for filename, result in stats.items():
        print(f"{filename}: {result.updated} updated, {result.added} added")
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import pandas as pd
import requests
from loguru import logger

from shockarb.names import clear_cache_entries

__all__ = ["sync_reference_data", "SyncResult"]

REMOTE_BASE_URL = "https://raw.githubusercontent.com/LondonMarket/Global-Stock-Symbols/main/"
_REQUEST_TIMEOUT = 15


@dataclass
class SyncResult:
    """Per-file sync outcome."""
    updated: int   # rows replaced (symbol existed locally, remote row differed)
    added:   int   # rows appended (symbol only present in remote)
    total:   int   # row count of the local file after merging


def sync_reference_data(
    data_dir:   str,
    files:      list[str],
    cache_path: str,
    timeout:    int = _REQUEST_TIMEOUT,
) -> dict[str, SyncResult]:
    """
    Sync each reference CSV in *files* against its remote counterpart.

    Parameters
    ----------
    data_dir   : directory containing the local reference CSVs
    files      : filenames to sync (e.g. paths.EXCHANGE_CSV_FILENAMES)
    cache_path : path to ticker_reference_cache.json (entries for changed
                 symbols are cleared so they re-resolve with fresh data)
    timeout    : per-file HTTP timeout in seconds

    Returns
    -------
    dict[filename, SyncResult] — only successfully-synced files are included.
    Files that fail to fetch, parse, or read are logged and omitted.
    """
    results: dict[str, SyncResult] = {}
    for filename in files:
        result = _sync_one(data_dir, filename, cache_path, timeout)
        if result is not None:
            results[filename] = result
    return results


def _sync_one(data_dir: str, filename: str, cache_path: str, timeout: int) -> SyncResult | None:
    local_path = os.path.join(data_dir, filename)
    remote_url = REMOTE_BASE_URL + filename

    try:
        response = requests.get(remote_url, timeout=timeout)
        response.raise_for_status()
        remote_df = pd.read_csv(io.StringIO(response.text), dtype=str)
    except Exception as exc:
        logger.warning(f"[reference_sync] Could not fetch/parse {remote_url}: {exc}")
        return None

    try:
        local_df = pd.read_csv(local_path, dtype=str)
    except Exception as exc:
        logger.warning(f"[reference_sync] Could not read local {local_path}: {exc}")
        return None

    if "Symbol" not in remote_df.columns or "Symbol" not in local_df.columns:
        logger.warning(f"[reference_sync] {filename}: missing 'Symbol' column — skipping")
        return None

    local_df = local_df.set_index("Symbol")
    remote_df = remote_df.set_index("Symbol")

    updated, added, changed_symbols = _merge_rows(local_df, remote_df)

    if changed_symbols:
        local_df.to_csv(local_path, index_label="Symbol")
        cleared = clear_cache_entries(cache_path, changed_symbols)
        logger.info(
            f"[reference_sync] {filename}: {updated} updated, {added} added, "
            f"{cleared} cache entries cleared"
        )

    return SyncResult(updated=updated, added=added, total=len(local_df))


def _merge_rows(local_df: pd.DataFrame, remote_df: pd.DataFrame) -> tuple[int, int, list[str]]:
    """
    Apply the merge rule in place on *local_df* (indexed by Symbol).

    Returns (updated_count, added_count, changed_symbols).
    """
    updated = 0
    added = 0
    changed: list[str] = []

    for symbol, remote_row in remote_df.iterrows():
        remote_row = remote_row.reindex(local_df.columns)
        if symbol in local_df.index:
            if not local_df.loc[symbol].equals(remote_row):
                local_df.loc[symbol] = remote_row
                updated += 1
                changed.append(symbol)
        else:
            local_df.loc[symbol] = remote_row
            added += 1
            changed.append(symbol)

    return updated, added, changed
